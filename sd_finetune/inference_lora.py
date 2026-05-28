"""
LoRA 微调模型推理

推理流程（和训练相反）：
    训练：干净图片 → 加噪 → UNet 预测噪声 → 更新参数
    推理：纯噪声 → UNet 逐步去噪 → 还原图片

具体步骤：
    1. 从纯随机噪声开始（timestep=999）
    2. UNet 预测噪声，减去噪声得到略干净的图（timestep=998）
    3. 重复 30 步，逐步去噪
    4. 最终得到清晰图片

文字通过 cross-attention 控制生成内容：
    "a cat" → UNet 去噪时偏向生成猫
    "a dog" → UNet 去噪时偏向生成狗

Usage:
    python inference_lora.py --lora_path checkpoints/lora/final --prompt "a photo of sks dog"
"""
import argparse
import os
import sys
import torch
from diffusers import StableDiffusionPipeline

# 复用模型下载工具函数
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'sd_inference'))
from utils import get_model_path, SD_V15_MODELSCOPE


def main():
    # ---- 命令行参数 ----
    parser = argparse.ArgumentParser(description='Inference with LoRA')
    parser.add_argument('--lora_path', type=str, required=True,
                        help='LoRA 权重目录（训练时保存的 checkpoints/lora/final）')
    parser.add_argument('--prompt', type=str, required=True,
                        help='正面提示词，描述你想要生成的内容')
    parser.add_argument('--negative_prompt', type=str, default='low quality, blurry',
                        help='负面提示词，描述不想要的内容（低质量、模糊等）')
    parser.add_argument('--model', type=str, default=SD_V15_MODELSCOPE,
                        help='基座模型 ID')
    parser.add_argument('--steps', type=int, default=30,
                        help='去噪步数，越多质量越好但越慢（10~50）')
    parser.add_argument('--guidance_scale', type=float, default=7.5,
                        help='引导系数（CFG），越大越贴合提示词但可能过饱和（1~20）')
    parser.add_argument('--seed', type=int, default=None,
                        help='随机种子，相同种子生成相同图片（可复现）')
    parser.add_argument('--num_images', type=int, default=4,
                        help='生成图片数量')
    parser.add_argument('--output_dir', type=str, default='outputs/lora',
                        help='输出目录')
    parser.add_argument('--hf', action='store_true',
                        help='用 HuggingFace 而不是 ModelScope 下载模型')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ============================================================
    # 步骤 1：加载基座模型
    # ============================================================
    # StableDiffusionPipeline 把所有组件打包在一起：
    #   tokenizer + text_encoder（文字编码）
    #   + VAE（潜空间转换）
    #   + UNet（去噪）
    #   + scheduler（调度去噪过程）
    model_path = get_model_path(args.model, use_modelscope=not args.hf)
    print(f"Loading model from: {model_path}")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,   # 半精度推理，节省一半显存
        safety_checker=None,          # 禁用安全检查（NSFW 过滤），避免误拦
    )

    # ============================================================
    # 步骤 2：加载 LoRA 权重
    # ============================================================
    # load_adapter 会把训练好的 LoRA A、B 矩阵加载到 UNet 的注意力层旁
    # 原权重 W 不变，推理时实际使用 W + (alpha/rank) × B × A
    print(f"Loading LoRA: {args.lora_path}")
    pipe.unet.load_adapter(args.lora_path)

    # 把整个 pipeline 搬到 GPU
    pipe = pipe.to('cuda')

    # enable_attention_slicing：分片计算注意力，减少显存峰值
    # 代价是推理略慢，但能在显存较小的 GPU 上跑
    pipe.enable_attention_slicing()

    # ============================================================
    # 步骤 3：设置随机种子
    # ============================================================
    # 种子控制初始噪声的随机状态
    # 相同种子 + 相同参数 = 相同图片（可复现）
    # 不设种子 = 每次随机
    generator = None
    if args.seed is not None:
        generator = torch.Generator('cuda').manual_seed(args.seed)

    # ============================================================
    # 步骤 4：生成图片
    # ============================================================
    # pipeline() 内部做的事：
    #   1. 文字编码：prompt → text_encoder → 77×768 语义向量
    #   2. 初始化：生成随机噪声 x_T (64×64×4)
    #   3. 循环 steps 次（从 T 到 0）：
    #      a. UNet 预测噪声：noise_pred = UNet(x_t, t, text_emb)
    #      b. 调度器去噪：x_{t-1} = scheduler.step(noise_pred, t, x_t)
    #   4. VAE 解码：64×64×4 潜变量 → 512×512×3 图片
    print(f"Prompt: {args.prompt}")
    images = pipe(
        prompt=args.prompt,                    # 正面提示词
        negative_prompt=args.negative_prompt,  # 负面提示词（不想看到的内容）
        num_inference_steps=args.steps,        # 去噪步数（默认 30 步）
        guidance_scale=args.guidance_scale,    # CFG 引导系数
        # CFG 原理：同时用有条件和无条件预测，加权组合
        #   noise = noise_uncond + guidance_scale × (noise_cond - noise_uncond)
        #   guidance_scale 越大，生成越贴合 prompt，但过大（>15）会过饱和
        num_images_per_prompt=args.num_images, # 每个 prompt 生成几张图
        generator=generator,                   # 随机种子
    ).images

    # ============================================================
    # 步骤 5：保存图片
    # ============================================================
    for i, img in enumerate(images):
        path = os.path.join(args.output_dir, f'lora_{i:03d}.png')
        img.save(path)
        print(f"Saved: {path}")

    print("Done!")


if __name__ == '__main__':
    main()
