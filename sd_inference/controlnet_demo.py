"""
ControlNet with Canny edge detection.
Usage:
    python controlnet_demo.py --image input.png --prompt "a beautiful house"
    Uses Canny edges to control the structure of generated images.
"""
import argparse
import os
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from controlnet_aux import CannyDetector
from utils import get_model_path, SD_V15_MODELSCOPE, CONTROLNET_CANNY_MODELSCOPE


def main():
    parser = argparse.ArgumentParser(description='ControlNet Canny Demo')
    parser.add_argument('--image', type=str, required=True, help='Input image path')
    parser.add_argument('--prompt', type=str, required=True, help='Text prompt')
    parser.add_argument('--negative_prompt', type=str, default='low quality, blurry')
    parser.add_argument('--model', type=str, default=SD_V15_MODELSCOPE)
    parser.add_argument('--controlnet_model', type=str,
                        default=CONTROLNET_CANNY_MODELSCOPE)
    parser.add_argument('--low_threshold', type=int, default=100)
    parser.add_argument('--high_threshold', type=int, default=200)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--guidance_scale', type=float, default=7.5)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--output_dir', type=str, default='outputs/controlnet')
    parser.add_argument('--hf', action='store_true', help='Use HuggingFace instead of ModelScope')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Extract Canny edges
    print("Extracting Canny edges...")
    input_image = Image.open(args.image).convert('RGB').resize((512, 512))
    canny = CannyDetector()
    canny_image = canny(input_image, low_threshold=args.low_threshold,
                        high_threshold=args.high_threshold)

    # Save edge image
    canny_path = os.path.join(args.output_dir, 'canny_edges.png')
    canny_image.save(canny_path)
    print(f"Canny edges saved: {canny_path}")

    # Load pipeline
    cn_path = get_model_path(args.controlnet_model, use_modelscope=not args.hf)
    model_path = get_model_path(args.model, use_modelscope=not args.hf)
    print(f"Loading ControlNet: {cn_path}")
    controlnet = ControlNetModel.from_pretrained(cn_path, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        model_path,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
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
        image=canny_image,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    ).images

    # Save
    for i, img in enumerate(images):
        path = os.path.join(args.output_dir, f'controlnet_{i:03d}.png')
        img.save(path)
        print(f"Saved: {path}")

    print("Done!")


if __name__ == '__main__':
    main()
