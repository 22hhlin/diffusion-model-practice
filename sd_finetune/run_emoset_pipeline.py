"""
EmoSet LoRA fine-tuning pipeline: download + caption + train.

EmoSet has 8 emotion categories, 118K images, no text captions.
This script:
1. Downloads EmoSet from ModelScope (weisir001/EmoSet)
2. Auto-generates captions with BLIP2
3. Runs LoRA fine-tuning

Usage:
    # Full pipeline
    python run_emoset_pipeline.py --data_root /mnt/workspace/EmoSet

    # Step by step
    python run_emoset_pipeline.py --data_root /mnt/workspace/EmoSet --step download
    python run_emoset_pipeline.py --data_root /mnt/workspace/EmoSet --step caption
    python run_emoset_pipeline.py --data_root /mnt/workspace/EmoSet --step train
"""
import argparse
import os
import json
import torch
from PIL import Image
from tqdm import tqdm


# EmoSet emotion labels (from the official repo)
EMOTIONS = ['amusement', 'awe', 'contentment', 'excitement', 'anger', 'disgust', 'fear', 'sadness']

# Emotion descriptions for better captions
EMOTION_DESC = {
    'amusement': 'a funny and amusing scene',
    'awe': 'an awe-inspiring and magnificent scene',
    'contentment': 'a peaceful and content scene',
    'excitement': 'an exciting and thrilling scene',
    'anger': 'an angry and intense scene',
    'disgust': 'a disgusting and repulsive scene',
    'fear': 'a scary and fearful scene',
    'sadness': 'a sad and melancholic scene',
}


def download_emoset(data_root):
    """Download EmoSet from ModelScope."""
    from modelscope.msdatasets import MsDataset

    os.makedirs(data_root, exist_ok=True)

    # Check if already downloaded
    image_dir = os.path.join(data_root, 'image')
    if os.path.isdir(image_dir):
        total = sum(len(os.listdir(os.path.join(image_dir, d)))
                    for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d)))
        if total > 0:
            print(f"EmoSet already exists ({total} images), skipping download...")
            return

    print("Downloading EmoSet from ModelScope (weisir001/EmoSet)...")
    print("This may take a while on first run...")

    # Download from ModelScope
    ds = MsDataset.load('weisir001/EmoSet', split='train')

    # Organize into folder structure by emotion
    os.makedirs(image_dir, exist_ok=True)

    print(f"Processing {len(ds)} images...")
    for i, item in enumerate(tqdm(ds, desc="Saving images")):
        # Get image and emotion label
        image = item.get('image') or item.get('img')
        emotion = item.get('emotion') or item.get('label', 'unknown')

        if image is None:
            continue
        if isinstance(image, dict) and 'bytes' in image:
            from io import BytesIO
            image = Image.open(BytesIO(image['bytes']))
        if not isinstance(image, Image.Image):
            continue

        # Create emotion folder
        emotion = str(emotion).lower().replace(' ', '_')
        emotion_dir = os.path.join(image_dir, emotion)
        os.makedirs(emotion_dir, exist_ok=True)

        # Save image
        img_path = os.path.join(emotion_dir, f'{emotion}_{i:06d}.jpg')
        image.convert('RGB').save(img_path)

    # Count results
    total = sum(len(os.listdir(os.path.join(image_dir, d)))
                for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d)))
    print(f"Download complete! {total} images saved to {image_dir}")


def scan_emoset(data_root):
    """Scan EmoSet and return list of (image_path, emotion)."""
    items = []
    image_dir = os.path.join(data_root, 'image')

    if not os.path.isdir(image_dir):
        print(f"Error: {image_dir} not found")
        return items

    for emotion in EMOTIONS:
        emotion_dir = os.path.join(image_dir, emotion)
        if not os.path.isdir(emotion_dir):
            continue
        for img_name in os.listdir(emotion_dir):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                items.append({
                    'file_name': os.path.join(emotion_dir, img_name),
                    'emotion': emotion,
                })

    print(f"Found {len(items)} images across {len(EMOTIONS)} emotions")
    return items


def generate_captions(data_root, max_images=None):
    """Generate captions with BLIP2 + emotion labels."""
    from transformers import Blip2Processor, Blip2ForConditionalGeneration

    items = scan_emoset(data_root)
    if not items:
        return

    if max_images:
        items = items[:max_images]
        print(f"Limited to {max_images} images")

    print("Loading BLIP2...")
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
    ).to('cuda')

    metadata = []
    batch_size = 4

    print("Generating captions...")
    for i in tqdm(range(0, len(items), batch_size), desc="BLIP2"):
        batch = items[i:i+batch_size]
        images = []
        for item in batch:
            img = Image.open(item['file_name']).convert('RGB')
            images.append(img.resize((512, 512)))

        inputs = processor(images=images, return_tensors="pt").to('cuda', torch.float16)
        ids = model.generate(**inputs, max_new_tokens=50)
        captions = processor.batch_decode(ids, skip_special_tokens=True)

        for item, cap in zip(batch, captions):
            # Combine BLIP2 caption with emotion context
            blip_cap = cap.strip()
            emotion = item['emotion']
            # Final caption: "blip2 description, a scene expressing {emotion}"
            final_caption = f"{blip_cap}, expressing {emotion}"

            metadata.append({
                'file_name': os.path.abspath(item['file_name']),
                'text': final_caption,
                'emotion': emotion,
            })

        del inputs
        torch.cuda.empty_cache()

    # Save metadata
    meta_path = os.path.join(data_root, 'metadata.jsonl')
    with open(meta_path, 'w') as f:
        for m in metadata:
            f.write(json.dumps(m) + '\n')

    print(f"\nSaved {len(metadata)} captions to {meta_path}")

    # Show samples
    print("\nSample captions:")
    seen = set()
    for m in metadata:
        if m['emotion'] not in seen:
            seen.add(m['emotion'])
            print(f"  [{m['emotion']}] {m['text']}")

    return metadata


def train_lora(data_root, epochs=50, rank=4, lr=1e-4):
    """Run LoRA fine-tuning."""
    import sys
    sys.path.insert(0, os.path.dirname(__file__))

    meta_path = os.path.join(data_root, 'metadata.jsonl')
    if not os.path.exists(meta_path):
        print(f"Error: {meta_path} not found. Run caption step first.")
        return

    print(f"\nStarting LoRA training...")
    print(f"  Data: {meta_path}")
    print(f"  Epochs: {epochs}, Rank: {rank}, LR: {lr}")

    # Import and run training
    from train_lora import ImageDataset, main as train_main
    import sys

    # Set up args
    sys.argv = [
        'train_lora.py',
        '--data_dir', data_root,
        '--epochs', str(epochs),
        '--rank', str(rank),
        '--lr', str(lr),
    ]
    train_main()


def main():
    parser = argparse.ArgumentParser(description='EmoSet LoRA Pipeline')
    parser.add_argument('--data_root', type=str, default='/mnt/workspace/EmoSet')
    parser.add_argument('--step', type=str, default='all',
                        choices=['download', 'caption', 'train', 'all'])
    parser.add_argument('--max_images', type=int, default=None,
                        help='Max images for captioning (None=all)')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--rank', type=int, default=4)
    args = parser.parse_args()

    if args.step in ('download', 'all'):
        download_emoset(args.data_root)

    if args.step in ('caption', 'all'):
        generate_captions(args.data_root, args.max_images)

    if args.step in ('train', 'all'):
        train_lora(args.data_root, args.epochs, args.rank)

    print("\n=== Pipeline Complete ===")
    print(f"Generate images: python inference_lora.py --lora_path checkpoints/lora/final --prompt 'a scene expressing amusement'")


if __name__ == '__main__':
    main()
