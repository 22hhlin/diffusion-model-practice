"""
Download images from HuggingFace and auto-generate captions with BLIP2.

Usage:
    # Download 200 images from a dataset and generate captions
    python download_and_caption.py --num_images 200

    # Use a specific dataset
    python download_and_caption.py --dataset lambdalabs/pokemon-blip-captions --num_images 100

    # Only caption existing images
    python download_and_caption.py --image_dir ./my_images --skip_download
"""
import argparse
import os
import json
import torch
from PIL import Image
from tqdm import tqdm


def download_images(dataset_id, split, output_dir, num_images, hf_token=None):
    """Download images from HuggingFace dataset."""
    from datasets import load_dataset

    print(f"Loading: {dataset_id}")
    ds = load_dataset(dataset_id, split=split, streaming=True, token=hf_token)

    os.makedirs(output_dir, exist_ok=True)
    count = 0

    for item in tqdm(ds, desc="Downloading"):
        if count >= num_images:
            break

        # Get image
        image = None
        for key in ['image', 'img', 'pixel_values', 'bytes']:
            if key in item:
                image = item[key]
                break

        if image is None:
            continue
        if isinstance(image, dict) and 'bytes' in image:
            from io import BytesIO
            image = Image.open(BytesIO(image['bytes']))
        if not isinstance(image, Image.Image):
            continue

        img_path = os.path.join(output_dir, f'{count:05d}.png')
        image.convert('RGB').resize((512, 512)).save(img_path)
        count += 1

    print(f"Downloaded {count} images to {output_dir}")
    return count


def generate_captions(image_dir, output_meta, batch_size=4):
    """Generate captions with BLIP2."""
    from transformers import Blip2Processor, Blip2ForConditionalGeneration

    print("Loading BLIP2...")
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
    ).to('cuda')

    valid_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    images = sorted([f for f in os.listdir(image_dir)
                     if os.path.splitext(f)[1].lower() in valid_ext])

    print(f"Captioning {len(images)} images...")
    metadata = []

    for i in tqdm(range(0, len(images), batch_size), desc="BLIP2"):
        batch_files = images[i:i+batch_size]
        batch_imgs = [Image.open(os.path.join(image_dir, f)).convert('RGB')
                      for f in batch_files]

        inputs = processor(images=batch_imgs, return_tensors="pt").to('cuda', torch.float16)
        ids = model.generate(**inputs, max_new_tokens=50)
        captions = processor.batch_decode(ids, skip_special_tokens=True)

        for f, cap in zip(batch_files, captions):
            metadata.append({
                'file_name': os.path.abspath(os.path.join(image_dir, f)),
                'text': cap.strip(),
            })
        del inputs
        torch.cuda.empty_cache()

    with open(output_meta, 'w') as f:
        for m in metadata:
            f.write(json.dumps(m) + '\n')

    print(f"Saved {len(metadata)} captions to {output_meta}")
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='lambdalabs/pokemon-blip-captions')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--image_dir', type=str, default='./downloaded_images')
    parser.add_argument('--num_images', type=int, default=200)
    parser.add_argument('--skip_download', action='store_true')
    parser.add_argument('--metadata', type=str, default='metadata.jsonl')
    args = parser.parse_args()

    if not args.skip_download:
        download_images(args.dataset, args.split, args.image_dir, args.num_images)

    meta_path = os.path.join(os.path.dirname(os.path.abspath(args.image_dir)), args.metadata)
    generate_captions(args.image_dir, meta_path)

    print(f"\nDone! Train with:")
    print(f"  python train_lora.py --data_dir {os.path.dirname(os.path.abspath(args.image_dir))}")


if __name__ == '__main__':
    main()
