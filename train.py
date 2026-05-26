"""
Training script for Latent Diffusion Model on MNIST.
Usage:
    python train.py --stage vae         # Train VAE first
    python train.py --stage diffusion   # Then train diffusion model
    python train.py --stage all         # Train both sequentially
"""
import argparse
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from models import VAE, UNet, GaussianDiffusion


def get_mnist_loader(batch_size=128, num_workers=2):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # Scale to [-1, 1]
    ])
    dataset = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, pin_memory=True)


def train_vae(epochs=30, lr=1e-3, batch_size=128, device='cuda', save_dir='checkpoints'):
    """Train VAE on MNIST."""
    os.makedirs(save_dir, exist_ok=True)
    loader = get_mnist_loader(batch_size)
    vae = VAE(latent_dim=1024).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=lr)

    print(f"Training VAE on {device} for {epochs} epochs...")
    for epoch in range(epochs):
        vae.train()
        total_loss, total_recon, total_kl = 0, 0, 0
        pbar = tqdm(loader, desc=f"VAE Epoch {epoch+1}/{epochs}")
        for x, _ in pbar:
            x = x.to(device)
            recon, mu, logvar = vae(x)
            loss, recon_loss, kl_loss = vae.loss(x, recon, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            pbar.set_postfix(loss=loss.item() / x.size(0))

        n = len(loader.dataset)
        print(f"  Loss: {total_loss/n:.4f}, Recon: {total_recon/n:.4f}, KL: {total_kl/n:.4f}")

    torch.save(vae.state_dict(), os.path.join(save_dir, 'vae.pt'))
    print(f"VAE saved to {save_dir}/vae.pt")
    return vae


def train_diffusion(vae, epochs=100, lr=1e-3, timesteps=1000, batch_size=128,
                    device='cuda', save_dir='checkpoints'):
    """Train diffusion model in VAE latent space."""
    os.makedirs(save_dir, exist_ok=True)
    loader = get_mnist_loader(batch_size)

    unet = UNet(in_channels=64, base_ch=128).to(device)
    diffusion = GaussianDiffusion(unet, timesteps=timesteps).to(device)
    optimizer = optim.Adam(unet.parameters(), lr=lr)

    print(f"Training Diffusion on {device} for {epochs} epochs...")
    for epoch in range(epochs):
        diffusion.train()
        total_loss = 0
        pbar = tqdm(loader, desc=f"Diffusion Epoch {epoch+1}/{epochs}")
        for x, _ in pbar:
            x = x.to(device)
            with torch.no_grad():
                z, _, _ = vae.encode(x)
                # Normalize latent to roughly unit variance
                z = z / z.std()
                # Reshape flat vector to 2D feature map for UNet
                z = z.view(-1, 64, 4, 4)

            loss = diffusion.training_loss(z)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(loader)
        print(f"  Avg Loss: {avg_loss:.4f}")

        # Save checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            torch.save(unet.state_dict(), os.path.join(save_dir, f'unet_epoch{epoch+1}.pt'))

    torch.save(unet.state_dict(), os.path.join(save_dir, 'unet.pt'))
    print(f"Diffusion model saved to {save_dir}/unet.pt")
    return diffusion


def main():
    parser = argparse.ArgumentParser(description='Train Latent Diffusion Model on MNIST')
    parser.add_argument('--stage', type=str, default='all', choices=['vae', 'diffusion', 'all'],
                        help='Which stage to train')
    parser.add_argument('--epochs-vae', type=int, default=30)
    parser.add_argument('--epochs-diff', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--save-dir', type=str, default='checkpoints')
    args = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    vae = None
    if args.stage in ('vae', 'all'):
        vae = train_vae(args.epochs_vae, args.lr, args.batch_size, device, args.save_dir)

    if args.stage in ('diffusion', 'all'):
        if vae is None:
            vae = VAE(latent_dim=1024).to(device)
            ckpt = os.path.join(args.save_dir, 'vae.pt')
            vae.load_state_dict(torch.load(ckpt, map_location=device))
            print(f"Loaded VAE from {ckpt}")
        vae.eval()
        train_diffusion(vae, args.epochs_diff, args.lr, args.timesteps,
                        args.batch_size, device, args.save_dir)


if __name__ == '__main__':
    main()
