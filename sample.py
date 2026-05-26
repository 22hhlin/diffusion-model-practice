"""
Sampling script for Latent Diffusion Model on MNIST.
Usage:
    python sample.py                        # Generate 16 samples
    python sample.py --num_samples 64       # Generate 64 samples
    python sample.py --show_process         # Show denoising process
"""
import argparse
import os
import torch
import matplotlib.pyplot as plt
import numpy as np

from models import VAE, UNet, GaussianDiffusion


def load_models(checkpoint_dir='checkpoints', device='cuda', timesteps=1000):
    """Load trained VAE and diffusion model."""
    vae = VAE(latent_dim=1024).to(device)
    vae.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'vae.pt'), map_location=device))
    vae.eval()

    unet = UNet(in_channels=64, base_ch=128).to(device)
    unet.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'unet.pt'), map_location=device))
    unet.eval()

    diffusion = GaussianDiffusion(unet, timesteps=timesteps).to(device)
    return vae, diffusion


@torch.no_grad()
def generate_samples(vae, diffusion, num_samples=16, device='cuda'):
    """Generate images from noise."""
    z = diffusion.sample((num_samples, 64, 4, 4), device)
    z = z.view(-1, 1024)  # Reshape to flat vector for VAE decoder
    images = vae.decode(z)
    return images.cpu()


@torch.no_grad()
def generate_with_process(vae, diffusion, num_samples=4, device='cuda', every_n=200):
    """Generate images and save intermediate steps."""
    z, intermediates = diffusion.sample_with_progress(
        (num_samples, 64, 4, 4), device, every_n=every_n
    )
    # Decode all intermediates
    decoded = []
    for inter in intermediates:
        inter = inter.to(device)
        inter = inter.view(-1, 1024)  # Reshape to flat vector for VAE decoder
        decoded.append(vae.decode(inter).cpu())
    return decoded


def save_image_grid(images, path, nrow=4):
    """Save a grid of images."""
    images = (images + 1) / 2  # [-1,1] -> [0,1]
    images = images.clamp(0, 1)

    n = images.shape[0]
    ncol = min(nrow, n)
    nrows = (n + ncol - 1) // ncol

    fig, axes = plt.subplots(nrows, ncol, figsize=(ncol * 2, nrows * 2))
    if nrows == 1:
        axes = [axes]
    axes = np.array(axes).reshape(nrows, ncol)

    for i in range(nrows):
        for j in range(ncol):
            idx = i * ncol + j
            axes[i, j].imshow(images[idx, 0], cmap='gray')
            axes[i, j].axis('off')

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved to {path}")


def save_process_visualization(decoded_steps, path, num_samples=4):
    """Visualize the denoising process."""
    steps = len(decoded_steps)
    fig, axes = plt.subplots(num_samples, steps, figsize=(steps * 2, num_samples * 2))

    for t in range(steps):
        imgs = (decoded_steps[t] + 1) / 2
        imgs = imgs.clamp(0, 1)
        for s in range(num_samples):
            axes[s, t].imshow(imgs[s, 0], cmap='gray')
            axes[s, t].axis('off')
            if s == 0:
                axes[s, t].set_title(f't={t}', fontsize=10)

    plt.suptitle('Denoising Process', fontsize=14)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved to {path}")


def main():
    parser = argparse.ArgumentParser(description='Sample from Latent Diffusion Model')
    parser.add_argument('--num_samples', type=int, default=16)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--show_process', action='store_true', help='Show denoising process')
    parser.add_argument('--nrow', type=int, default=4)
    args = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading models from {args.checkpoint_dir}...")
    vae, diffusion = load_models(args.checkpoint_dir, device, args.timesteps)

    if args.show_process:
        print("Generating samples with denoising process...")
        decoded = generate_with_process(vae, diffusion, num_samples=4, device=device, every_n=200)
        save_process_visualization(decoded, os.path.join(args.output_dir, 'denoising_process.png'))
    else:
        print(f"Generating {args.num_samples} samples...")
        images = generate_samples(vae, diffusion, args.num_samples, device)
        save_image_grid(images, os.path.join(args.output_dir, 'samples.png'), nrow=args.nrow)

    print("Done!")


if __name__ == '__main__':
    main()
