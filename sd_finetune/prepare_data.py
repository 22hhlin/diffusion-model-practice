"""
Prepare training data for LoRA fine-tuning.
Scans a folder of images and creates a metadata file.

Usage:
    python prepare_data.py --data_dir ./my_images --prompt "a photo of sks dog"
    python prepare_data.py --data_dir ./my_images --auto_caption
"""
import argparse
import os
import json
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description='Prepare data for LoRA training')
    parser.add_argument('--data_dir', type=str, required=True, help='Folder with training images')
    parser.add_argument('--prompt', type=str, default=None,
                        help='Fixed prompt for all images (e.g., "a photo of sks dog")')
    parser.add_argument('--auto_caption', action='store_true',
                        help='Use filename as caption (no extension)')
    parser.add_argument('--output', type=str, default='metadata.jsonl')
    parser.add_argument('--resolution', type=int, default=512)
    args = parser.parse_args()

    if not args.prompt and not args.auto_caption:
        print("Error: specify --prompt or --auto_caption")
        return

    # Scan images
    valid_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    images = sorted([
        f for f in os.listdir(args.data_dir)
        if os.path.splitext(f)[1].lower() in valid_ext
    ])

    if not images:
        print(f"No images found in {args.data_dir}")
        return

    print(f"Found {len(images)} images")

    # Resize and create metadata
    output_dir = os.path.join(args.data_dir, 'processed')
    os.makedirs(output_dir, exist_ok=True)
    metadata = []

    for img_name in images:
        img_path = os.path.join(args.data_dir, img_name)
        try:
            img = Image.open(img_path).convert('RGB')
            # Resize to target resolution (center crop to square)
            w, h = img.size
            size = min(w, h)
            left = (w - size) // 2
            top = (h - size) // 2
            img = img.crop((left, top, left + size, top + size))
            img = img.resize((args.resolution, args.resolution), Image.LANCZOS)

            # Save processed image
            out_name = os.path.splitext(img_name)[0] + '.png'
            out_path = os.path.join(output_dir, out_name)
            img.save(out_path)

            # Caption
            if args.prompt:
                caption = args.prompt
            else:
                caption = os.path.splitext(img_name)[0].replace('_', ' ')

            metadata.append({
                'file_name': out_path,
                'text': caption,
            })
            print(f"  Processed: {img_name} -> {out_name}")

        except Exception as e:
            print(f"  Error processing {img_name}: {e}")

    # Save metadata
    meta_path = os.path.join(args.data_dir, args.output)
    with open(meta_path, 'w') as f:
        for item in metadata:
            f.write(json.dumps(item) + '\n')

    print(f"\nMetadata saved: {meta_path}")
    print(f"Processed images: {output_dir}")
    print(f"Total: {len(metadata)} images")
    print(f"\nNext step: python train_lora.py --data_dir {args.data_dir}")


if __name__ == '__main__':
    main()
