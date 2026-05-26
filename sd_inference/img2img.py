"""
Image-to-Image generation with Stable Diffusion.
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
    parser = argparse.ArgumentParser(description='Image-to-Image with Stable Diffusion')
    parser.add_argument('--image', type=str, required=True, help='Input image path')
    parser.add_argument('--prompt', type=str, required=True, help='Text prompt')
    parser.add_argument('--negative_prompt', type=str, default='low quality, blurry, distorted')
    parser.add_argument('--model', type=str, default=SD_V15_MODELSCOPE)
    parser.add_argument('--strength', type=float, default=0.75,
                        help='Denoising strength (0.0=keep original, 1.0=fully regenerate)')
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--guidance_scale', type=float, default=7.5)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--output_dir', type=str, default='outputs/img2img')
    parser.add_argument('--hf', action='store_true', help='Use HuggingFace instead of ModelScope')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load pipeline
    model_path = get_model_path(args.model, use_modelscope=not args.hf)
    print(f"Loading pipeline from: {model_path}")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipe = pipe.to('cuda')
    pipe.enable_attention_slicing()

    # Load input image
    init_image = Image.open(args.image).convert('RGB')
    init_image = init_image.resize((512, 512))
    print(f"Input image: {args.image}")

    # Set seed
    generator = None
    if args.seed is not None:
        generator = torch.Generator('cuda').manual_seed(args.seed)

    # Generate
    print(f"Prompt: {args.prompt}")
    print(f"Strength: {args.strength}")
    images = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        image=init_image,
        strength=args.strength,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    ).images

    # Save
    for i, img in enumerate(images):
        path = os.path.join(args.output_dir, f'img2img_{i:03d}.png')
        img.save(path)
        print(f"Saved: {path}")

    print("Done!")


if __name__ == '__main__':
    main()
