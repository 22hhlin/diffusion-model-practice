"""
Inpainting with Stable Diffusion.
Usage:
    python inpainting.py --image input.png --mask mask.png --prompt "a red car"
    White areas in mask = regions to repaint.
"""
import argparse
import os
import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
from utils import get_model_path, SD_INPAINT_MODELSCOPE


def main():
    parser = argparse.ArgumentParser(description='Inpainting with Stable Diffusion')
    parser.add_argument('--image', type=str, required=True, help='Input image path')
    parser.add_argument('--mask', type=str, required=True, help='Mask image path (white=inpaint)')
    parser.add_argument('--prompt', type=str, required=True, help='Text prompt for inpainting')
    parser.add_argument('--negative_prompt', type=str, default='low quality, blurry, distorted')
    parser.add_argument('--model', type=str, default=SD_INPAINT_MODELSCOPE)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--guidance_scale', type=float, default=7.5)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--output_dir', type=str, default='outputs/inpainting')
    parser.add_argument('--hf', action='store_true', help='Use HuggingFace instead of ModelScope')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load pipeline
    model_path = get_model_path(args.model, use_modelscope=not args.hf)
    print(f"Loading pipeline from: {model_path}")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipe = pipe.to('cuda')
    pipe.enable_attention_slicing()

    # Load images
    image = Image.open(args.image).convert('RGB').resize((512, 512))
    mask = Image.open(args.mask).convert('RGB').resize((512, 512))
    print(f"Input image: {args.image}")
    print(f"Mask: {args.mask}")

    # Set seed
    generator = None
    if args.seed is not None:
        generator = torch.Generator('cuda').manual_seed(args.seed)

    # Generate
    print(f"Prompt: {args.prompt}")
    images = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        image=image,
        mask_image=mask,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    ).images

    # Save
    for i, img in enumerate(images):
        path = os.path.join(args.output_dir, f'inpaint_{i:03d}.png')
        img.save(path)
        print(f"Saved: {path}")

    print("Done!")


if __name__ == '__main__':
    main()
