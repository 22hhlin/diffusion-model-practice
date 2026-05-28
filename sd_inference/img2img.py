"""
图生图（Image-to-Image）— 基于原图 + 文字生成新图

和文生图的区别：
    文生图：从纯随机噪声开始去噪 → 完全由文字控制
    图生图：从原图加噪后开始去噪 → 保留原图结构 + 文字控制风格

strength 参数控制变化程度：
    strength=0.1：几乎不变，只做微调
    strength=0.5：保留原图结构，改变风格
    strength=0.75：较大变化，但能看出原图影子
    strength=1.0：等同于文生图，完全忽略原图

内部原理：
    1. 原图 → VAE 编码 → 潜变量 x_0
    2. 根据 strength 计算起始时间步 t_start = strength × 1000
    3. 对 x_0 加噪到 t_start 步：x_t = √(ᾱ_t)·x_0 + √(1-ᾱ_t)·ε
    4. 从 t_start 开始去噪到 0（而不是从 999 开始）
    5. t_start 越大（strength 越大），加的噪声越多，原图信息丢失越多

适用场景：
    - 风格迁移：照片 → 油画风格
    - 细节修改：改变颜色、材质
    - 局部重绘：配合 mask 使用

Usage:
    python img2img.py --image input.png --prompt "oil painting style"
    python img2img.py --image photo.jpg --prompt "anime style" --strength 0.75
"""
import argparse
import os
import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
from utils import get_model_path, SD_V15_MODELSCOPE


def main():
    # ---- 命令行参数 ----
    parser = argparse.ArgumentParser(description='Image-to-Image with Stable Diffusion')
    parser.add_argument('--image', type=str, required=True,
                        help='输入原图路径')
    parser.add_argument('--prompt', type=str, required=True,
                        help='正面提示词，描述目标风格/效果')
    parser.add_argument('--negative_prompt', type=str, default='low quality, blurry, distorted',
                        help='负面提示词')
    parser.add_argument('--model', type=str, default=SD_V15_MODELSCOPE,
                        help='模型 ID')
    parser.add_argument('--strength', type=float, default=0.75,
                        help='变化强度（0.0~1.0）：0=不变，1=完全重生成')
    parser.add_argument('--steps', type=int, default=30,
                        help='去噪步数')
    parser.add_argument('--guidance_scale', type=float, default=7.5,
                        help='CFG 引导系数')
    parser.add_argument('--seed', type=int, default=None,
                        help='随机种子')
    parser.add_argument('--output_dir', type=str, default='outputs/img2img',
                        help='输出目录')
    parser.add_argument('--hf', action='store_true',
                        help='用 HuggingFace 而不是 ModelScope')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ============================================================
    # 步骤 1：加载模型
    # ============================================================
    # Img2ImgPipeline 和 txt2img 用的是同一个模型
    # 区别在于：img2img 不从纯噪声开始，而是从原图加噪后开始去噪
    model_path = get_model_path(args.model, use_modelscope=not args.hf)
    print(f"Loading pipeline from: {model_path}")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipe = pipe.to('cuda')
    pipe.enable_attention_slicing()

    # ============================================================
    # 步骤 2：加载并预处理原图
    # ============================================================
    # VAE 要求输入 512×512 的 RGB 图片
    # 如果原图尺寸不同，需要 resize
    init_image = Image.open(args.image).convert('RGB')
    init_image = init_image.resize((512, 512))
    print(f"Input image: {args.image}")

    # ============================================================
    # 步骤 3：设置随机种子
    # ============================================================
    generator = None
    if args.seed is not None:
        generator = torch.Generator('cuda').manual_seed(args.seed)

    # ============================================================
    # 步骤 4：生成图片
    # ============================================================
    # pipe() 内部流程（和 txt2img 的区别用 ★ 标注）：
    #
    # 4a. 文字编码：prompt → text_encoder → 语义向量
    #
    # 4b. ★ 原图编码（txt2img 没有这步）：
    #     init_image → VAE.encode → 潜变量 x_0 (64×64×4)
    #
    # 4c. ★ 计算起始时间步（txt2img 从 999 开始）：
    #     t_start = int(strength × num_train_timesteps)
    #     例如 strength=0.75 → t_start = 750（从第 750 步开始去噪）
    #
    # 4d. ★ 对原图加噪到 t_start（txt2img 直接用纯噪声）：
    #     x_{t_start} = √(ᾱ_{t_start})·x_0 + √(1-ᾱ_{t_start})·ε
    #     strength 越大，加的噪声越多，原图信息丢失越多
    #
    # 4e. 从 t_start 去噪到 0：
    #     for t in [t_start, t_start-1, ..., 0]:
    #         noise_pred = UNet(x_t, t, text_emb)
    #         x_{t-1} = scheduler.step(noise_pred, x_t, t)
    #
    # 4f. VAE 解码：潜变量 → RGB 图片
    #
    print(f"Prompt: {args.prompt}")
    print(f"Strength: {args.strength}")
    images = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        image=init_image,                    # ★ 原图（文生图没有这个参数）
        strength=args.strength,              # ★ 变化强度（文生图没有这个参数）
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    ).images

    # ============================================================
    # 步骤 5：保存图片
    # ============================================================
    for i, img in enumerate(images):
        path = os.path.join(args.output_dir, f'img2img_{i:03d}.png')
        img.save(path)
        print(f"Saved: {path}")

    print("Done!")


if __name__ == '__main__':
    main()
