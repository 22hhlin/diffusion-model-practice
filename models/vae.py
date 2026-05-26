"""
Simplified VAE for MNIST.
Compresses 28x28 images to 4x4 latent space.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, in_channels=1, latent_dim=64):
        super().__init__()
        # 28x28 -> 14x14 -> 7x7 -> 4x4
        self.conv1 = nn.Conv2d(in_channels, 32, 3, stride=2, padding=1)   # 28 -> 14
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)            # 14 -> 7
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)           # 7 -> 4
        self.norm1 = nn.GroupNorm(8, 32)
        self.norm2 = nn.GroupNorm(8, 64)
        self.norm3 = nn.GroupNorm(8, 128)

        # 4x4x128 = 2048
        self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 4, latent_dim)

    def forward(self, x):
        h = F.silu(self.norm1(self.conv1(x)))
        h = F.silu(self.norm2(self.conv2(h)))
        h = F.silu(self.norm3(self.conv3(h)))
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim=64, out_channels=1):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128 * 4 * 4)

        # 4x4 -> 7x7 -> 14x14 -> 28x28
        self.deconv1 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=0)  # 4 -> 7
        self.deconv2 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)   # 7 -> 14
        self.deconv3 = nn.ConvTranspose2d(32, out_channels, 3, stride=2, padding=1, output_padding=1)  # 14 -> 28
        self.norm1 = nn.GroupNorm(8, 64)
        self.norm2 = nn.GroupNorm(8, 32)

    def forward(self, z):
        h = self.fc(z).view(-1, 128, 4, 4)
        h = F.silu(self.norm1(self.deconv1(h)))
        h = F.silu(self.norm2(self.deconv2(h)))
        return torch.tanh(self.deconv3(h))


class VAE(nn.Module):
    def __init__(self, in_channels=1, latent_dim=64):
        super().__init__()
        self.encoder = Encoder(in_channels, latent_dim)
        self.decoder = Decoder(latent_dim, in_channels)
        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        recon = self.decode(z)
        return recon, mu, logvar

    def loss(self, x, recon, mu, logvar):
        recon_loss = F.mse_loss(recon, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_loss, recon_loss, kl_loss
