"""
LoRA 微调 Stable Diffusion v1.5

核心原理：
    扩散模型训练 = 学习去噪。给干净图片加噪声，让 UNet 预测噪声，用 MSE 损失更新参数。
    LoRA 只训练注意力层的低秩旁路（约占总参数 0.37%），冻结其余全部参数。

训练流程：
    1. 加载 SD 模型，拆出 VAE / Text Encoder / UNet
    2. 冻结所有基座参数，只在 UNet 的注意力层加 LoRA
    3. 每步：图片→VAE编码→加噪→UNet预测噪声→MSE损失→更新LoRA
    4. 保存 LoRA 权重，推理时加载到基座模型上

Usage:
    python train_lora.py --data_dir ./my_images --prompt "a photo of sks dog" --epochs 100
"""
import argparse
import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from diffusers import StableDiffusionPipeline, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model
import json
from tqdm import tqdm

# 将 sd_inference 目录加入路径，复用模型下载工具函数
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'sd_inference'))
from utils import get_model_path, SD_V15_MODELSCOPE


# ============================================================
# 第一部分：数据集
# ============================================================
class ImageDataset(Dataset):
    """
    训练数据集。每条数据包含：
      - pixel_values: 图片张量 [3, H, W]，像素范围 [-1, 1]
      - input_ids:    文字 token 序列 [77]，CLIP tokenizer 编码

    数据来源：metadata.jsonl（BLIP2 自动生成的题注）
      每行格式：{"file_name": "图片路径", "text": "图片描述"}
    """

    def __init__(self, data_dir, tokenizer, resolution=512):
        self.tokenizer = tokenizer
        self.resolution = resolution

        # 读取 metadata.jsonl，每行是一张图片的路径和文字描述
        meta_path = os.path.join(data_dir, 'metadata.jsonl')
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                self.items = [json.loads(line) for line in f]
        else:
            # 没有 metadata 时，扫描目录下所有图片，用固定描述 "a photo"
            valid_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
            self.items = []
            for f in sorted(os.listdir(data_dir)):
                if os.path.splitext(f)[1].lower() in valid_ext:
                    self.items.append({
                        'file_name': os.path.join(data_dir, f),
                        'text': 'a photo'
                    })

        print(f"Dataset: {len(self.items)} images")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        """
        返回一条训练数据。处理流程：
          图片：加载 → 缩放 → 转 tensor → 归一化到 [-1, 1]
          文字：CLIP tokenizer 编码 → token ID 序列
        """
        item = self.items[idx]

        # ---- 图片预处理 ----
        # 加载图片并转为 RGB（排除 RGBA 灰度等格式）
        image = Image.open(item['file_name']).convert('RGB')
        # 缩放到训练分辨率（256×256 或 512×512），LANCZOS 是高质量缩放算法
        image = image.resize((self.resolution, self.resolution), Image.LANCZOS)
        # PIL Image → 展平为像素列表 → 转为 float32 tensor
        # getdata() 返回 [(R,G,B), (R,G,B), ...]，list() 展平后长度 = H*W*3
        image = torch.tensor(list(image.getdata()), dtype=torch.float32)
        # 重组成 [H, W, 3] 的矩阵
        image = image.view(self.resolution, self.resolution, 3)
        # permute(2,0,1)：从 [H,W,C] 转为 [C,H,W]（PyTorch 要求通道在前）
        # / 127.5 - 1.0：像素从 [0,255] 归一化到 [-1,1]
        #   SD 的 VAE 训练时输入就是 [-1,1]，不归一化会导致生成质量下降
        image = image.permute(2, 0, 1) / 127.5 - 1.0

        # ---- 文字编码 ----
        # CLIP tokenizer 把文字切分成 token，转为数字 ID
        # 例如 "a sunset" → [49406, 320, 14158, 49407, 0, 0, ..., 0]
        tokens = self.tokenizer(
            item['text'],                          # BLIP2 生成的文字描述
            padding='max_length',                  # 不足 77 个 token 的用 0 填充
            max_length=self.tokenizer.model_max_length,  # CLIP 最大长度 = 77
            truncation=True,                       # 超过 77 个 token 的截断
            return_tensors='pt',                   # 返回 PyTorch tensor
        )

        return {
            'pixel_values': image,                 # [3, H, W] 图片张量
            'input_ids': tokens.input_ids.squeeze(0),  # [77] token ID 序列
        }


# ============================================================
# 第二部分：训练主函数
# ============================================================
def main():
    # ---- 命令行参数 ----
    parser = argparse.ArgumentParser(description='LoRA fine-tuning for SD')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='训练数据目录（需包含 metadata.jsonl）')
    parser.add_argument('--model', type=str, default=SD_V15_MODELSCOPE,
                        help='基座模型 ID（ModelScope）')
    parser.add_argument('--hf', action='store_true',
                        help='用 HuggingFace 而不是 ModelScope 下载模型')
    parser.add_argument('--prompt', type=str, default=None,
                        help='覆盖所有图片的文字描述（不设置则用 metadata.jsonl 中的描述）')
    parser.add_argument('--epochs', type=int, default=20,
                        help='训练轮数，把所有图片看多少遍')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率，每步参数更新的幅度')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='每步同时处理的图片数量')
    parser.add_argument('--rank', type=int, default=16,
                        help='LoRA 秩，控制低秩矩阵的维度，越大表达力越强')
    parser.add_argument('--resolution', type=int, default=256,
                        help='训练图片分辨率，越小越快')
    parser.add_argument('--save_dir', type=str, default='checkpoints/lora',
                        help='LoRA 权重保存目录')
    parser.add_argument('--save_every', type=int, default=10,
                        help='每 N 个 epoch 保存一次检查点')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = 'cuda'

    # ============================================================
    # 步骤 1：加载 Stable Diffusion 模型
    # ============================================================
    # SD 模型由 4 个组件组成：
    #   tokenizer      — CLIP tokenizer，把文字切分成 token
    #   text_encoder   — CLIP Text Encoder，把 token 转成 768 维语义向量
    #   vae            — VAE 编码器/解码器，图片和潜变量之间的转换
    #   unet           — UNet 去噪网络，在潜空间预测噪声（唯一需要训练的部分）
    model_path = get_model_path(args.model, use_modelscope=not args.hf)
    print(f"Loading model from: {model_path}")

    # from_pretrained 会从本地缓存或 ModelScope 下载模型
    # torch_dtype=torch.float16 用半精度加载，节省一半显存
    pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)

    # 从 pipeline 中拆出各个组件，方便单独控制
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder.to(device)  # 文字→语义向量
    vae = pipe.vae.to(device)                    # 图片→潜变量
    unet = pipe.unet.to(device)                  # 潜变量去噪

    # ============================================================
    # 步骤 2：冻结基座模型
    # ============================================================
    # requires_grad_(False) 表示不计算梯度，参数不会被更新
    # 这样只训练 LoRA 的少量参数（约 320 万），而不是全部 8.6 亿参数
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    # ============================================================
    # 步骤 3：在 UNet 上应用 LoRA
    # ============================================================
    # LoRA 原理：原权重 W 不动，加一个低秩旁路 ΔW = B × A
    #   W_new = W + ΔW = W + (alpha/r) × B × A
    #   其中 A: [d, rank], B: [rank, d]，rank << d
    #   只训练 A 和 B，参数量远小于原始权重
    lora_config = LoraConfig(
        r=args.rank,                      # 低秩矩阵的维度，16 表示 A 是 [d,16], B 是 [16,d]
        lora_alpha=args.rank * 2,         # 缩放系数，alpha/rank = 32/16 = 2
                                          # 控制 LoRA 对原权重的影响程度
        target_modules=[                  # 在 UNet 的哪些层加 LoRA
            'to_k',                       # 注意力的 Key 投影层
            'to_q',                       # 注意力的 Query 投影层
            'to_v',                       # 注意力的 Value 投影层
            'to_out.0',                   # 注意力输出的线性层
        ],
        lora_dropout=0.05,                # 训练时随机丢弃 5%，防止过拟合
    )
    # get_peft_model 会扫描 UNet，在 target_modules 指定的层旁边插入 LoRA 旁路
    unet = get_peft_model(unet, lora_config)
    # 打印可训练参数量，应该看到约 0.37%（320 万 / 8.6 亿）
    unet.print_trainable_parameters()

    # ============================================================
    # 步骤 4：准备数据
    # ============================================================
    dataset = ImageDataset(args.data_dir, tokenizer, args.resolution)

    # 如果指定了 --prompt，覆盖所有图片的文字描述
    # 适用于训练特定概念，如 "a photo of sks dog"
    if args.prompt:
        for item in dataset.items:
            item['text'] = args.prompt

    # DataLoader 参数说明：
    #   batch_size=4   每步取 4 张图一起训练
    #   shuffle=True   每个 epoch 随机打乱顺序，避免过拟合
    #   num_workers=4  4 个子进程并行加载数据，加速 IO
    #   pin_memory=True  数据锁页内存，加速 CPU→GPU 传输
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=4, pin_memory=True)

    # ============================================================
    # 步骤 5：优化器和噪声调度器
    # ============================================================
    # AdamW8bit：8-bit 量化版 AdamW，显存占用减少约 30%
    # 如果没安装 bitsandbytes，回退到标准 AdamW
    try:
        from bitsandbytes import AdamW8bit
        optimizer = AdamW8bit(unet.parameters(), lr=args.lr)
        print("Using AdamW8bit optimizer")
    except ImportError:
        optimizer = torch.optim.AdamW(unet.parameters(), lr=args.lr)
        print("bitsandbytes not found, using standard AdamW")

    # DDPM 调度器：控制加噪和去噪的过程
    # 定义了 1000 个时间步的噪声调度（β 从 0.0001 线性增到 0.02）
    noise_scheduler = DDPMScheduler.from_pretrained(model_path, subfolder='scheduler')

    # ============================================================
    # 步骤 6：训练循环
    # ============================================================
    # 总步数 = (图片数 / batch_size) × epochs
    # 例如 15531 张图 / 4 batch × 20 epochs = 77655 步
    print(f"Training for {args.epochs} epochs, {len(dataloader)} steps/epoch...")

    for epoch in range(args.epochs):
        unet.train()        # 切换到训练模式（启用 dropout）
        total_loss = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            # ---- 6a. 数据搬到 GPU ----
            # pixel_values: [B, 3, H, W] 图片张量，转为 float16 节省显存
            pixel_values = batch['pixel_values'].to(device, dtype=torch.float16)
            # input_ids: [B, 77] 文字 token 序列
            input_ids = batch['input_ids'].to(device)

            # ---- 6b. 编码到潜空间（不计算梯度）----
            with torch.no_grad():
                # VAE 编码：512×512×3 图片 → 64×64×4 潜变量（8 倍压缩）
                # .latent_dist.sample() 从正态分布采样
                # * 0.18215 是 SD 训练时的缩放因子，让潜变量方差接近 1
                latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215

                # CLIP 编码：文字 token → 77×768 语义向量
                # [0] 取 last_hidden_state，忽略 pooler_output
                encoder_hidden_states = text_encoder(input_ids)[0]

            # ---- 6c. 扩散过程：加噪 ----
            # 生成和潜变量同形状的随机噪声（这是模型要学习预测的目标）
            noise = torch.randn_like(latents)

            # 随机选时间步 t（0~999），每个样本独立选
            # t 越小噪声越少，t 越大噪声越多
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],), device=device
            ).long()

            # 按时间步 t 给潜变量加噪
            # 公式：x_t = √(ᾱ_t)·x_0 + √(1-ᾱ_t)·ε
            #   t 小时 ᾱ_t≈1，x_t ≈ x_0（几乎没噪声）
            #   t 大时 ᾱ_t≈0，x_t ≈ ε（几乎全是噪声）
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # ---- 6d. UNet 预测噪声 ----
            # UNet 输入：带噪潜变量 + 时间步 + 文字语义向量
            # UNet 输出：预测的噪声 noise_pred
            # 文字通过 cross-attention 注入 UNet：
            #   Q 来自图像特征，K/V 来自文字向量
            #   这样 UNet 就知道"按什么文字描述去噪"
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            # ---- 6e. 计算损失并更新 ----
            # MSE 损失：预测噪声 vs 真实噪声 的均方误差
            # 损失越小，UNet 去噪越准，生成质量越高
            loss = torch.nn.functional.mse_loss(noise_pred, noise)

            optimizer.zero_grad()   # 清空上一步的梯度
            loss.backward()         # 反向传播，计算梯度（只对 LoRA 参数）
            optimizer.step()        # 更新 LoRA 参数

            total_loss += loss.item()

        # 打印每个 epoch 的平均损失
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{args.epochs}: Loss = {avg_loss:.4f}")

        # 定期保存检查点（只保存 LoRA 权重，体积很小）
        if (epoch + 1) % args.save_every == 0:
            ckpt_dir = os.path.join(args.save_dir, f'epoch_{epoch+1}')
            unet.save_pretrained(ckpt_dir)
            print(f"Saved checkpoint: {ckpt_dir}")

    # ============================================================
    # 步骤 7：保存最终 LoRA 权重
    # ============================================================
    # save_pretrained 只保存 LoRA 的 A 和 B 矩阵（几十 MB），不是完整模型（几 GB）
    final_dir = os.path.join(args.save_dir, 'final')
    unet.save_pretrained(final_dir)
    print(f"\nTraining complete! LoRA saved to: {final_dir}")
    print(f"\nUse: python inference_lora.py --lora_path {final_dir} --prompt 'your prompt'")


if __name__ == '__main__':
    main()
