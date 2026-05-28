"""
EmoSet LoRA fine-tuning pipeline: collect images + caption + train.

This script:
1. Collects images from nested folders into a flat directory with sequential numbering
2. Auto-generates captions with BLIP2
3. Runs LoRA fine-tuning

Usage:
    # Full pipeline (collect + caption + train)
    python run_emoset_pipeline.py --image_dir /mnt/workspace/data/EmoEdit_origin_image/origin_image

    # Step by step
    python run_emoset_pipeline.py --image_dir /mnt/workspace/data/EmoEdit_origin_image/origin_image --step collect
    python run_emoset_pipeline.py --image_dir /mnt/workspace/data/EmoEdit_origin_image/origin_image --step caption
    python run_emoset_pipeline.py --image_dir /mnt/workspace/data/EmoEdit_origin_image/origin_image --step train
"""
import argparse
import os
import json
import shutil
import torch
from PIL import Image
from tqdm import tqdm


VALID_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def collect_images(image_dir, output_dir):
    """Collect all images from nested folders into a flat directory with sequential numbering."""
    if not os.path.isdir(image_dir):
        print(f"Error: {image_dir} not found")
        return 0

    os.makedirs(output_dir, exist_ok=True)

    # Check if already collected
    existing = [f for f in os.listdir(output_dir) if os.path.splitext(f)[1].lower() in VALID_EXT]
    if existing:
        print(f"Already collected {len(existing)} images in {output_dir}, skipping...")
        return len(existing)

    # Scan all subfolders recursively
    all_images = []
    for root, dirs, files in os.walk(image_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in VALID_EXT:
                all_images.append(os.path.join(root, f))

    all_images.sort()
    print(f"Found {len(all_images)} images in {image_dir}")

    # Copy and rename with sequential numbering
    for i, src_path in enumerate(tqdm(all_images, desc="Collecting")):
        ext = os.path.splitext(src_path)[1].lower()
        dst_path = os.path.join(output_dir, f'{i:06d}{ext}')
        shutil.copy2(src_path, dst_path)

    print(f"Collected {len(all_images)} images to {output_dir}")
    return len(all_images)


def generate_captions(image_dir, output_meta, max_images=None):
    """Generate captions with BLIP2."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'sd_inference'))
    from utils import get_model_path
    from transformers import Blip2Processor, Blip2ForConditionalGeneration

    # Scan images
    all_files = os.listdir(image_dir)
    print(f"Total files in {image_dir}: {len(all_files)}")
    images = sorted([f for f in all_files
                     if os.path.splitext(f)[1].lower() in VALID_EXT])
    print(f"Valid image files: {len(images)}")

    if not images:
        print(f"Error: No images found in {image_dir}")
        return

    if max_images:
        images = images[:max_images]
        print(f"Limited to {max_images} images")

    print(f"Captioning {len(images)} images...")

    # Load BLIP2 captioning model
    local_blip2 = '/mnt/workspace/models/modelscope/models/Salesforce/blip2-opt-2.7b'
    if os.path.isdir(local_blip2):
        blip2_path = local_blip2
        print(f"Using local BLIP2: {blip2_path}")
    else:
        blip2_path = get_model_path('Salesforce/blip2-opt-2.7b')
    print(f"Loading BLIP2 from: {blip2_path}")

    processor = Blip2Processor.from_pretrained(blip2_path)
    model = Blip2ForConditionalGeneration.from_pretrained(
        blip2_path, torch_dtype=torch.float16
    ).to('cuda')
    print("BLIP2 loaded successfully!")

    metadata = []
    batch_size = 4

    print("Generating captions with BLIP2...")
    for i in tqdm(range(0, len(images), batch_size), desc="BLIP2"):
        batch_files = images[i:i+batch_size]
        batch_imgs = []
        for f in batch_files:
            img = Image.open(os.path.join(image_dir, f)).convert('RGB')
            batch_imgs.append(img.resize((512, 512)))

        inputs = processor(images=batch_imgs, return_tensors="pt").to('cuda')
        ids = model.generate(**inputs, max_new_tokens=50)
        captions = processor.batch_decode(ids, skip_special_tokens=True)

        for f, cap in zip(batch_files, captions):
            metadata.append({
                'file_name': os.path.abspath(os.path.join(image_dir, f)),
                'text': cap.strip(),
            })

        del inputs
        torch.cuda.empty_cache()

    # Save metadata
    with open(output_meta, 'w') as f:
        for m in metadata:
            f.write(json.dumps(m) + '\n')

    print(f"\nSaved {len(metadata)} captions to {output_meta}")

    # Show samples
    print("\nSample captions:")
    for m in metadata[:5]:
        print(f"  {os.path.basename(m['file_name'])}: {m['text']}")

    return metadata


def train_lora(data_root, epochs=20, rank=16, lr=1e-4, batch_size=4, resolution=256):
    """Run LoRA fine-tuning."""
    import sys
    sys.path.insert(0, os.path.dirname(__file__))

    meta_path = os.path.join(data_root, 'metadata.jsonl')
    if not os.path.exists(meta_path):
        print(f"Error: {meta_path} not found. Run caption step first.")
        return

    print(f"\nStarting LoRA training...")
    print(f"  Data: {meta_path}")
    print(f"  Epochs: {epochs}, Rank: {rank}, LR: {lr}, Batch: {batch_size}, Resolution: {resolution}")

    from train_lora import main as train_main

    sys.argv = [
        'train_lora.py',
        '--data_dir', data_root,
        '--epochs', str(epochs),
        '--rank', str(rank),
        '--lr', str(lr),
        '--batch_size', str(batch_size),
        '--resolution', str(resolution),
    ]
    train_main()


def main():
    parser = argparse.ArgumentParser(description='EmoSet LoRA Pipeline')
    parser.add_argument('--image_dir', type=str,
                        default='/mnt/workspace/data/EmoEdit_origin_image/origin_image',
                        help='Source image directory (with nested folders)')
    parser.add_argument('--data_root', type=str, default=None,
                        help='Working directory for processed data (default: same as image_dir parent)')
    parser.add_argument('--step', type=str, default='all',
                        choices=['collect', 'caption', 'train', 'all'])
    parser.add_argument('--max_images', type=int, default=None,
                        help='Max images for captioning (None=all)')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--rank', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--resolution', type=int, default=256)
    args = parser.parse_args()

    # Default data_root to parent of image_dir
    if args.data_root is None:
        args.data_root = os.path.dirname(args.image_dir.rstrip('/'))

    # Flat directory for collected images
    flat_dir = os.path.join(args.data_root, 'images')

    if args.step in ('collect', 'all'):
        collect_images(args.image_dir, flat_dir)

    if args.step in ('caption', 'all'):
        meta_path = os.path.join(args.data_root, 'metadata.jsonl')
        generate_captions(flat_dir, meta_path, args.max_images)

    if args.step in ('train', 'all'):
        train_lora(args.data_root, args.epochs, args.rank, args.lr, args.batch_size, args.resolution)

    print("\n=== Pipeline Complete ===")
    print(f"Generate images: python inference_lora.py --lora_path checkpoints/lora/final --prompt 'a photo'")


if __name__ == '__main__':
    main()
