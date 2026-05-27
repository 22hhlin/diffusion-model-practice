"""
Inference with LoRA fine-tuned Stable Diffusion.
Usage:
    python inference_lora.py --lora_path checkpoints/lora/final --prompt "a photo of sks dog"
"""
import argparse
import os
import sys
import torch
from diffusers import StableDiffusionPipeline

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'sd_inference'))
from utils import get_model_path, SD_V15_MODELSCOPE


def main():
    parser = argparse.ArgumentParser(description='Inference with LoRA')
    parser.add_argument('--lora_path', type=str, required=True, help='LoRA checkpoint path')
    parser.add_argument('--prompt', type=str, required=True, help='Text prompt')
    parser.add_argument('--negative_prompt', type=str, default='low quality, blurry')
    parser.add_argument('--model', type=str, default=SD_V15_MODELSCOPE)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--guidance_scale', type=float, default=7.5)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--num_images', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default='outputs/lora')
    parser.add_argument('--hf', action='store_true', help='Use HuggingFace instead of ModelScope')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load pipeline
    model_path = get_model_path(args.model, use_modelscope=not args.hf)
    print(f"Loading model from: {model_path}")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path, torch_dtype=torch.float16, safety_checker=None
    )

    # Load LoRA
    print(f"Loading LoRA: {args.lora_path}")
    pipe.unet.load_adapter(args.lora_path)
    pipe = pipe.to('cuda')
    pipe.enable_attention_slicing()

    # Set seed
    generator = None
    if args.seed is not None:
        generator = torch.Generator('cuda').manual_seed(args.seed)

    # Generate
    print(f"Prompt: {args.prompt}")
    images = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        num_images_per_prompt=args.num_images,
        generator=generator,
    ).images

    # Save
    for i, img in enumerate(images):
        path = os.path.join(args.output_dir, f'lora_{i:03d}.png')
        img.save(path)
        print(f"Saved: {path}")

    print("Done!")


if __name__ == '__main__':
    main()
