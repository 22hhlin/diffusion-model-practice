"""
LoRA fine-tuning for Stable Diffusion.
Usage:
    python train_lora.py --data_dir ./my_images --prompt "a photo of sks dog" --epochs 100
"""
import argparse
import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from diffusers import StableDiffusionPipeline, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model
import json
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'sd_inference'))
from utils import get_model_path, SD_V15_MODELSCOPE


class ImageDataset(Dataset):
    def __init__(self, data_dir, tokenizer, resolution=512):
        self.tokenizer = tokenizer
        self.resolution = resolution

        # Load metadata
        meta_path = os.path.join(data_dir, 'metadata.jsonl')
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                self.items = [json.loads(line) for line in f]
        else:
            # Fallback: use all images with fixed prompt
            valid_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
            self.items = []
            for f in sorted(os.listdir(data_dir)):
                if os.path.splitext(f)[1].lower() in valid_ext:
                    self.items.append({
                        'file_name': os.path.join(data_dir, f),
                        'text': 'a photo'
                    })

        print(f"Dataset: {len(self.items)} images")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]

        # Load image
        image = Image.open(item['file_name']).convert('RGB')
        image = image.resize((self.resolution, self.resolution), Image.LANCZOS)
        image = torch.tensor(list(image.getdata()), dtype=torch.float32)
        image = image.view(self.resolution, self.resolution, 3)
        image = image.permute(2, 0, 1) / 127.5 - 1.0  # [-1, 1]

        # Tokenize text
        tokens = self.tokenizer(
            item['text'],
            padding='max_length',
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors='pt',
        )

        return {
            'pixel_values': image,
            'input_ids': tokens.input_ids.squeeze(0),
        }


def main():
    parser = argparse.ArgumentParser(description='LoRA fine-tuning for SD')
    parser.add_argument('--data_dir', type=str, required=True, help='Training data directory')
    parser.add_argument('--model', type=str, default=SD_V15_MODELSCOPE)
    parser.add_argument('--hf', action='store_true', help='Use HuggingFace instead of ModelScope')
    parser.add_argument('--prompt', type=str, default=None,
                        help='Override prompt for all images')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--rank', type=int, default=16, help='LoRA rank')
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--save_dir', type=str, default='checkpoints/lora')
    parser.add_argument('--save_every', type=int, default=10, help='Save checkpoint every N epochs')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = 'cuda'

    # Load model components
    model_path = get_model_path(args.model, use_modelscope=not args.hf)
    print(f"Loading model from: {model_path}")
    pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder.to(device)
    vae = pipe.vae.to(device)
    unet = pipe.unet.to(device)

    # Freeze base model
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    unet.enable_gradient_checkpointing()

    # Apply LoRA to UNet
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank * 2,
        target_modules=['to_k', 'to_q', 'to_v', 'to_out.0'],
        lora_dropout=0.05,
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    # Dataset
    dataset = ImageDataset(args.data_dir, tokenizer, args.resolution)

    # Override prompt if specified
    if args.prompt:
        for item in dataset.items:
            item['text'] = args.prompt

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # Optimizer (8-bit saves ~30% memory)
    try:
        from bitsandbytes import AdamW8bit
        optimizer = AdamW8bit(unet.parameters(), lr=args.lr)
        print("Using AdamW8bit optimizer")
    except ImportError:
        optimizer = torch.optim.AdamW(unet.parameters(), lr=args.lr)
        print("bitsandbytes not found, using standard AdamW")
    noise_scheduler = DDPMScheduler.from_pretrained(model_path, subfolder='scheduler')

    # Training loop
    print(f"Training for {args.epochs} epochs, {len(dataloader)} steps/epoch...")
    for epoch in range(args.epochs):
        unet.train()
        total_loss = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            pixel_values = batch['pixel_values'].to(device, dtype=torch.float16)
            input_ids = batch['input_ids'].to(device)

            # Encode image to latent
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215
                encoder_hidden_states = text_encoder(input_ids)[0]

            # Sample noise and timestep
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                      (latents.shape[0],), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Predict noise
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            # Loss
            loss = torch.nn.functional.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{args.epochs}: Loss = {avg_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            ckpt_dir = os.path.join(args.save_dir, f'epoch_{epoch+1}')
            unet.save_pretrained(ckpt_dir)
            print(f"Saved checkpoint: {ckpt_dir}")

    # Save final
    final_dir = os.path.join(args.save_dir, 'final')
    unet.save_pretrained(final_dir)
    print(f"\nTraining complete! LoRA saved to: {final_dir}")
    print(f"\nUse: python inference_lora.py --lora_path {final_dir} --prompt 'your prompt'")


if __name__ == '__main__':
    main()
