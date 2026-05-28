"""
文生图（Text-to-Image）— 纯文本生成图片

原理：
    1. CLIP tokenizer 把文字切成 token
    2. CLIP Text Encoder 把 token 编码成语义向量（768 维）
    3. 生成随机噪声（64×64×4 潜变量）
    4. UNet 逐步去噪 30 步，每步都通过 cross-attention 注入文字语义
    5. VAE 解码潜变量为 512×512 图片

cross-attention 注入文字的原理：
    UNet 每层都有 attention 层，其中：
      Q（Query）= 图像特征（"我在关注什么"）
      K（Key）= 文字语义（"文字提供了什么信息"）
      V（Value）= 文字语义（"文字的具体内容"）
    注意力权重 = softmax(Q·K^T / √d)，图像特征会"关注"最相关的文字

Usage:
    python txt2img.py --prompt "a cat sitting on a chair"
    python txt2img.py --prompt "cyberpunk city" --negative_prompt "blurry" --steps 30 --num_images 4
"""
import argparse
import os
import torch
from diffusers import StableDiffusionPipeline
from utils import get_model_path, SD_V15_MODELSCOPE


def main():
    # ---- 命令行参数 ----
    parser = argparse.ArgumentParser(description='Text-to-Image with Stable Diffusion')
    parser.add_argument('--prompt', type=str, required=True,
                        help='正面提示词，描述你想要的图片内容')
    parser.add_argument('--negative_prompt', type=str, default='low quality, blurry, distorted',
                        help='负面提示词，描述不想要的内容')
    parser.add_argument('--model', type=str, default=SD_V15_MODELSCOPE,
                        help='模型 ID（ModelScope 格式）')
    parser.add_argument('--steps', type=int, default=30,
                        help='去噪步数（10~50），越多质量越好但越慢')
    parser.add_argument('--guidance_scale', type=float, default=7.5,
                        help='CFG 引导系数（1~20），越大越贴合 prompt')
    parser.add_argument('--seed', type=int, default=None,
                        help='随机种子，相同种子 = 相同图片')
    parser.add_argument('--num_images', type=int, default=1,
                        help='生成图片数量')
    parser.add_argument('--height', type=int, default=512,
                        help='输出图片高度（像素），必须是 8 的倍数')
    parser.add_argument('--width', type=int, default=512,
                        help='输出图片宽度（像素），必须是 8 的倍数')
    parser.add_argument('--output_dir', type=str, default='outputs/txt2img',
                        help='输出目录')
    parser.add_argument('--hf', action='store_true',
                        help='用 HuggingFace 而不是 ModelScope')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ============================================================
    # 步骤 1：加载模型
    # ============================================================
    # StableDiffusionPipeline 是 diffusers 库的核心类
    # 它把 SD 的所有组件打包成一个统一的推理接口
    # 内部包含：tokenizer + text_encoder + vae + unet + scheduler
    model_path = get_model_path(args.model, use_modelscope=not args.hf)
    print(f"Loading pipeline from: {model_path}")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,   # 半精度推理，显存减半
        safety_checker=None,          # 禁用 NSFW 检测
    )

    # 搬到 GPU
    pipe = pipe.to('cuda')

    # 注意力分片：把大的 attention 矩阵拆成小块计算
    # 好处：显存峰值降低，可以在小显存 GPU 上跑
    # 代价：推理速度略慢（约 5-10%）
    pipe.enable_attention_slicing()

    # ============================================================
    # 步骤 2：设置随机种子
    # ============================================================
    # torch.Generator 控制随机数生成器的状态
    # 同一个种子 → 同一个初始噪声 → 同一张图片
    # 不设种子 → 每次随机 → 每张图都不一样
    generator = None
    if args.seed is not None:
        generator = torch.Generator('cuda').manual_seed(args.seed)

    # ============================================================
    # 步骤 3：生成图片
    # ============================================================
    # pipe() 内部完整流程：
    #
    # 3a. 文字编码：
    #     "a cat" → [49406, 320, 6827, 49407, 0, ..., 0]  (77 个 token)
    #     → text_encoder → [1, 77, 768] 语义向量
    #
    # 3b. 初始化噪声：
    #     生成 [1, 4, 64, 64] 的随机高斯噪声（对应 512×512 图片的潜空间）
    #     为什么是 64×64？因为 VAE 把图片压缩了 8 倍：512/8 = 64
    #     为什么是 4 通道？因为 SD 的 VAE 输出 4 通道潜变量
    #
    # 3c. 逐步去噪（循环 steps 次）：
    #     for t in [999, 998, ..., 0]:  （从纯噪声到干净图）
    #         noise_pred = UNet(x_t, t, text_emb)  # 预测当前噪声
    #         x_{t-1} = scheduler.step(noise_pred, x_t, t)  # 去一步噪
    #
    #     去噪调度器（DDPM scheduler）决定每步去多少噪声：
    #       - 前期（t 大）：去掉大量噪声，确定大体结构
    #       - 后期（t 小）：去掉少量噪声，精修细节
    #
    # 3d. VAE 解码：
    #     64×64×4 潜变量 → VAE.decoder → 512×512×3 RGB 图片
    #     * (1/0.18215) 还原缩放（训练时乘的因子）
    print(f"Generating {args.num_images} image(s)...")
    print(f"Prompt: {args.prompt}")
    images = pipe(
        prompt=args.prompt,                    # 正面提示词
        negative_prompt=args.negative_prompt,  # 负面提示词
        num_inference_steps=args.steps,        # 去噪步数
        guidance_scale=args.guidance_scale,    # CFG 引导系数
        # CFG 工作原理：
        #   同时做两次 UNet 前向传播：
        #   1. 无条件预测：noise_uncond = UNet(x_t, t, ∅)  （空 prompt）
        #   2. 有条件预测：noise_cond = UNet(x_t, t, text_emb)
        #   最终噪声 = noise_uncond + guidance_scale × (noise_cond - noise_uncond)
        #   guidance_scale=1 时等同于无引导，越大越"听话"但可能过饱和
        num_images_per_prompt=args.num_images, # 每个 prompt 生成几张
        generator=generator,                   # 随机种子
        height=args.height,                    # 输出高度（像素）
        width=args.width,                      # 输出宽度（像素）
        # height/width 必须是 8 的倍数（因为 VAE 压缩 8 倍）
        # 常见值：512, 768, 1024
    ).images

    # ============================================================
    # 步骤 4：保存图片
    # ============================================================
    for i, img in enumerate(images):
        path = os.path.join(args.output_dir, f'txt2img_{i:03d}.png')
        img.save(path)
        print(f"Saved: {path}")

    print("Done!")


if __name__ == '__main__':
    main()
