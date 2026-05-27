"""
Prepare EmoSet dataset for LoRA fine-tuning.
Supports two caption strategies:
1. Label-based: "a photo expressing [emotion]"
2. BLIP2-based: auto-generate detailed captions with BLIP2

Usage:
    # Label-based captions
    python prepare_emotion_data.py --data_dir /path/to/EmoSet --mode label

    # BLIP2 auto-captions (slower but better quality)
    python prepare_emotion_data.py --data_dir /path/to/EmoSet --mode blip2
"""
import argparse
import os
import json
import torch
from PIL import Image
from tqdm import tqdm


# EmoSet emotion categories (common set, adjust if your version differs)
EMOTION_LABELS = {
    0: 'amusement',
    1: 'anger',
    2: 'awe',
    3: 'contentment',
    4: 'disgust',
    5: 'excitement',
    6: 'fear',
    7: 'sadness',
}

# Extended emotion descriptions for better prompts
EMOTION_PROMPTS = {
    'amusement': 'a funny and amusing scene that makes people laugh',
    'anger': 'an intense and angry scene expressing frustration',
    'awe': 'an awe-inspiring and magnificent scene',
    'contentment': 'a peaceful and content scene conveying satisfaction',
    'disgust': 'a disgusting and repulsive scene',
    'excitement': 'an exciting and thrilling scene full of energy',
    'fear': 'a scary and fearful scene creating terror',
    'sadness': 'a sad and melancholic scene expressing sorrow',
}


def scan_emoset(data_dir):
    """Scan EmoSet directory and return list of (image_path, emotion_label).

    Supports two common EmoSet structures:
    1. Folder-based: data_dir/{emotion_name}/images...
    2. Annotation file: data_dir/images/ + annotation.json
    """
    items = []

    # Try folder-based structure
    for emotion_name in os.listdir(data_dir):
        emotion_dir = os.path.join(data_dir, emotion_name)
        if not os.path.isdir(emotion_dir):
            continue
        # Check if this looks like an emotion folder
        valid_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        images = [f for f in os.listdir(emotion_dir)
                  if os.path.splitext(f)[1].lower() in valid_ext]
        if images:
            for img_name in images:
                items.append({
                    'file_name': os.path.join(emotion_dir, img_name),
                    'emotion': emotion_name,
                })

    # Try annotation file
    if not items:
        for ann_file in ['annotation.json', 'annotations.json', 'labels.json']:
            ann_path = os.path.join(data_dir, ann_file)
            if os.path.exists(ann_path):
                with open(ann_path) as f:
                    annotations = json.load(f)
                img_dir = os.path.join(data_dir, 'images')
                if not os.path.isdir(img_dir):
                    img_dir = data_dir
                for img_name, label in annotations.items():
                    img_path = os.path.join(img_dir, img_name)
                    if os.path.exists(img_path):
                        if isinstance(label, int):
                            emotion = EMOTION_LABELS.get(label, f'emotion_{label}')
                        else:
                            emotion = str(label)
                        items.append({
                            'file_name': img_path,
                            'emotion': emotion,
                        })
                break

    return items


def generate_label_caption(emotion, use_extended=True):
    """Generate caption from emotion label."""
    if use_extended and emotion in EMOTION_PROMPTS:
        return EMOTION_PROMPTS[emotion]
    return f'a photo expressing {emotion}'


def generate_blip2_captions(items, batch_size=4):
    """Generate captions using BLIP2."""
    from transformers import Blip2Processor, Blip2ForConditionalGeneration

    print("Loading BLIP2 model...")
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        torch_dtype=torch.float16,
    ).to('cuda')

    captions = []
    for i in tqdm(range(0, len(items), batch_size), desc="Generating captions"):
        batch = items[i:i+batch_size]
        images = []
        for item in batch:
            img = Image.open(item['file_name']).convert('RGB')
            images.append(img)

        inputs = processor(images=images, return_tensors="pt").to('cuda', torch.float16)
        generated_ids = model.generate(**inputs, max_new_tokens=50)
        batch_captions = processor.batch_decode(generated_ids, skip_special_tokens=True)

        for cap in batch_captions:
            captions.append(cap.strip())

        # Free memory
        del inputs
        torch.cuda.empty_cache()

    return captions


def main():
    parser = argparse.ArgumentParser(description='Prepare EmoSet for LoRA fine-tuning')
    parser.add_argument('--data_dir', type=str, required=True, help='EmoSet dataset path')
    parser.add_argument('--mode', type=str, default='label', choices=['label', 'blip2'],
                        help='Caption generation mode')
    parser.add_argument('--output', type=str, default='metadata.jsonl',
                        help='Output metadata filename')
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--max_images', type=int, default=None,
                        help='Max number of images to process (for testing)')
    args = parser.parse_args()

    # Scan dataset
    print(f"Scanning EmoSet at: {args.data_dir}")
    items = scan_emoset(args.data_dir)

    if not items:
        print("No images found! Check the dataset structure.")
        print("Expected: data_dir/{emotion_name}/image.jpg")
        print("   or:   data_dir/images/ + annotation.json")
        return

    print(f"Found {len(items)} images")

    # Limit if specified
    if args.max_images:
        items = items[:args.max_images]
        print(f"Limited to {len(items)} images")

    # Generate captions
    if args.mode == 'label':
        print("Generating captions from emotion labels...")
        for item in items:
            item['caption'] = generate_label_caption(item['emotion'])
    elif args.mode == 'blip2':
        print("Generating captions with BLIP2...")
        captions = generate_blip2_captions(items)
        for item, cap in zip(items, captions):
            # Combine BLIP2 caption with emotion
            item['caption'] = f"{cap}, expressing {item['emotion']}"

    # Process images and save
    output_dir = os.path.join(os.path.dirname(args.data_dir), 'emotoset_processed')
    os.makedirs(output_dir, exist_ok=True)
    metadata = []

    print("Processing images...")
    for item in tqdm(items):
        try:
            img = Image.open(item['file_name']).convert('RGB')
            # Center crop to square
            w, h = img.size
            size = min(w, h)
            left, top = (w - size) // 2, (h - size) // 2
            img = img.crop((left, top, left + size, top + size))
            img = img.resize((args.resolution, args.resolution), Image.LANCZOS)

            # Save
            img_name = os.path.basename(item['file_name'])
            out_name = os.path.splitext(img_name)[0] + '.png'
            out_path = os.path.join(output_dir, out_name)
            img.save(out_path)

            metadata.append({
                'file_name': out_path,
                'text': item['caption'],
                'emotion': item['emotion'],
            })
        except Exception as e:
            print(f"  Error: {item['file_name']}: {e}")

    # Save metadata
    meta_path = os.path.join(os.path.dirname(args.data_dir), args.output)
    with open(meta_path, 'w') as f:
        for m in metadata:
            f.write(json.dumps(m) + '\n')

    print(f"\nDone! Processed {len(metadata)} images")
    print(f"Metadata: {meta_path}")
    print(f"Images: {output_dir}")

    # Show some examples
    print("\nSample captions:")
    emotions_seen = set()
    for m in metadata:
        if m['emotion'] not in emotions_seen:
            emotions_seen.add(m['emotion'])
            print(f"  [{m['emotion']}] {m['caption']}")

    print(f"\nNext: python train_lora.py --data_dir {os.path.dirname(args.data_dir)}")


if __name__ == '__main__':
    main()
