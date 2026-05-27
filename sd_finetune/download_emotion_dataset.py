"""
Download an open-source emotion image dataset from HuggingFace
and auto-generate captions with BLIP2.

Supported datasets:
1. "emotion" - text emotion (not images, skip)
2. "jamescalam/youtube-captions-emotions" - text
3. Best option: Use LAION-Aesthetics subset or similar with emotion filtering

Actually, the most practical approach:
Download a small emotion dataset and use BLIP2 to caption it.

Usage:
    python download_emotion_dataset.py --output_dir ./emotion_data --num_images 200
"""
import argparse
import os
import json
import torch
from PIL import Image
from tqdm import tqdm


def download_from_hf(dataset_id, split, output_dir, num_images):
    """Download images from a HuggingFace dataset."""
    from datasets import load_dataset

    print(f"Loading dataset: {dataset_id}")
    ds = load_dataset(dataset_id, split=split, streaming=True)

    os.makedirs(output_dir, exist_ok=True)
    count = 0

    for i, item in enumerate(tqdm(ds, total=num_images, desc="Downloading")):
        if count >= num_images:
            break

        # Try to get image from common field names
        image = item.get('image') or item.get('img') or item.get('pixel_values')
        if image is None:
            continue

        if isinstance(image, dict):  # {'bytes': ...}
            from io import BytesIO
            image = Image.open(BytesIO(image['bytes']))

        if not isinstance(image, Image.Image):
            continue

        # Get label if available
        label = item.get('label') or item.get('emotion') or item.get('text', '')

        # Save
        img_path = os.path.join(output_dir, f'{count:05d}.png')
        image.convert('RGB').save(img_path)
        count += 1

    print(f"Downloaded {count} images to {output_dir}")
    return count


def generate_captions_with_blip2(image_dir, output_meta):
    """Generate captions for all images using BLIP2."""
    from transformers import Blip2Processor, Blip2ForConditionalGeneration

    print("Loading BLIP2 model...")
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        torch_dtype=torch.float16,
    ).to('cuda')

    # Scan images
    valid_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    images = sorted([
        f for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in valid_ext
    ])

    print(f"Generating captions for {len(images)} images...")
    metadata = []

    for i in tqdm(range(0, len(images), 4), desc="Captioning"):
        batch_files = images[i:i+4]
        batch_images = []
        for f in batch_files:
            img = Image.open(os.path.join(image_dir, f)).convert('RGB')
            batch_images.append(img.resize((512, 512)))

        inputs = processor(images=batch_images, return_tensors="pt").to('cuda', torch.float16)
        generated_ids = model.generate(**inputs, max_new_tokens=50)
        captions = processor.batch_decode(generated_ids, skip_special_tokens=True)

        for f, cap in zip(batch_files, captions):
            metadata.append({
                'file_name': os.path.join(image_dir, f),
                'text': cap.strip(),
            })

        del inputs
        torch.cuda.empty_cache()

    # Save metadata
    with open(output_meta, 'w') as f:
        for m in metadata:
            f.write(json.dumps(m) + '\n')

    print(f"Saved {len(metadata)} captions to {output_meta}")
    return metadata


def main():
    parser = argparse.ArgumentParser(description='Download emotion dataset and generate captions')
    parser.add_argument('--dataset', type=str,
                        default='lambdalabs/pokemon-blip-captions',
                        help='HuggingFace dataset ID')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--output_dir', type=str, default='./emotion_data')
    parser.add_argument('--num_images', type=int, default=200,
                        help='Number of images to download')
    parser.add_argument('--skip_download', action='store_true',
                        help='Skip download, only generate captions')
    parser.add_argument('--resolution', type=int, default=512)
    args = parser.parse_args()

    if not args.skip_download:
        download_from_hf(args.dataset, args.split, args.output_dir, args.num_images)

    # Generate captions
    meta_path = os.path.join(os.path.dirname(args.output_dir), 'metadata.jsonl')
    generate_captions_with_blip2(args.output_dir, meta_path)

    print(f"\nDone! Use for LoRA training:")
    print(f"  python train_lora.py --data_dir {os.path.dirname(args.output_dir)}")


if __name__ == '__main__':
    main()
