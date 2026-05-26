"""
Text-to-Image generation with Stable Diffusion.
Usage:
    python txt2img.py --prompt "a cat sitting on a chair"
    python txt2img.py --prompt "cyberpunk city" --negative_prompt "blurry" --steps 30 --num_images 4
"""
import argparse
import os
import torch
from diffusers import StableDiffusionPipeline


def main():
    parser = argparse.ArgumentParser(description='Text-to-Image with Stable Diffusion')
    parser.add_argument('--prompt', type=str, required=True, help='Text prompt')
    parser.add_argument('--negative_prompt', type=str, default='low quality, blurry, distorted')
    parser.add_argument('--model', type=str, default='runwayml/stable-diffusion-v1-5')
    parser.add_argument('--steps', type=int, default=30, help='Inference steps')
    parser.add_argument('--guidance_scale', type=float, default=7.5, help='CFG scale')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--num_images', type=int, default=1, help='Number of images')
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--output_dir', type=str, default='outputs/txt2img')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load pipeline
    print(f"Loading model: {args.model}")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipe = pipe.to('cuda')
    pipe.enable_attention_slicing()  # Save VRAM

    # Set seed
    generator = None
    if args.seed is not None:
        generator = torch.Generator('cuda').manual_seed(args.seed)

    # Generate
    print(f"Generating {args.num_images} image(s)...")
    print(f"Prompt: {args.prompt}")
    images = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        num_images_per_prompt=args.num_images,
        generator=generator,
        height=args.height,
        width=args.width,
    ).images

    # Save
    for i, img in enumerate(images):
        path = os.path.join(args.output_dir, f'txt2img_{i:03d}.png')
        img.save(path)
        print(f"Saved: {path}")

    print("Done!")


if __name__ == '__main__':
    main()
