"""
Simplified U-Net for latent diffusion on MNIST.
Operates on 4x4 latent space.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(4, out_ch)
        self.norm2 = nn.GroupNorm(4, out_ch)
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.residual = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = F.silu(self.norm1(self.conv1(x)))
        h = h + self.time_mlp(F.silu(t_emb))[:, :, None, None]
        h = F.silu(self.norm2(self.conv2(h)))
        return h + self.residual(x)


class UNet(nn.Module):
    """
    3-level U-Net for 4x4 latent diffusion.
    Input: (B, latent_dim, 4, 4) + timestep -> (B, latent_dim, 4, 4)
    """
    def __init__(self, in_channels=64, base_ch=128, time_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Encoder
        self.enc1 = ResBlock(in_channels, base_ch, time_dim)       # 4x4
        self.enc2 = ResBlock(base_ch, base_ch * 2, time_dim)       # 2x2
        self.down1 = nn.Conv2d(base_ch, base_ch, 3, stride=2, padding=1)  # 4->2
        self.down2 = nn.Conv2d(base_ch * 2, base_ch * 2, 3, stride=2, padding=1)  # 2->1

        # Bottleneck
        self.mid = ResBlock(base_ch * 2, base_ch * 2, time_dim)    # 1x1

        # Decoder
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch * 2, 2, stride=2)  # 1->2
        self.up2 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)      # 2->4
        self.dec1 = ResBlock(base_ch * 4, base_ch * 2, time_dim)   # skip connection
        self.dec2 = ResBlock(base_ch * 2, base_ch, time_dim)       # skip connection

        # Output
        self.out = nn.Sequential(
            nn.GroupNorm(4, base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, in_channels, 1),
        )

    def forward(self, x, t):
        t_emb = self.time_mlp(t.float())

        # Encoder
        h1 = self.enc1(x, t_emb)           # (B, 128, 4, 4)
        h2 = self.enc2(self.down1(h1), t_emb)  # (B, 256, 2, 2)

        # Bottleneck
        h = self.mid(self.down2(h2), t_emb)   # (B, 256, 1, 1)

        # Decoder with skip connections
        h = self.up1(h)                        # (B, 256, 2, 2)
        h = self.dec1(torch.cat([h, h2], dim=1), t_emb)  # (B, 256, 2, 2)  cat: 256+256=512
        h = self.up2(h)                        # (B, 128, 4, 4)
        h = self.dec2(torch.cat([h, h1], dim=1), t_emb)  # (B, 128, 4, 4)  cat: 128+128=256

        return self.out(h)
