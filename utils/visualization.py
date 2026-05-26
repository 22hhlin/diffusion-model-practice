"""
Visualization utilities for diffusion model training and sampling.
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os


def plot_images(images, title=None, nrow=4, figsize=None, save_path=None):
    """Plot a grid of images.

    Args:
        images: Tensor of shape (N, C, H, W) in [-1, 1]
        title: Optional title
        nrow: Number of images per row
        figsize: Optional figure size
        save_path: Optional path to save the figure
    """
    images = (images + 1) / 2  # [-1,1] -> [0,1]
    images = images.clamp(0, 1).numpy()

    n = images.shape[0]
    ncol = min(nrow, n)
    nrows = (n + ncol - 1) // ncol

    if figsize is None:
        figsize = (ncol * 2, nrows * 2)

    fig, axes = plt.subplots(nrows, ncol, figsize=figsize)
    if nrows == 1 and ncol == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncol == 1:
        axes = axes[:, np.newaxis]

    for i in range(nrows):
        for j in range(ncol):
            idx = i * ncol + j
            if idx < n:
                if images.shape[1] == 1:
                    axes[i, j].imshow(images[idx, 0], cmap='gray')
                else:
                    axes[i, j].imshow(images[idx].transpose(1, 2, 0))
            axes[i, j].axis('off')

    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_training_curves(losses, title='Training Loss', save_path=None):
    """Plot training loss curves.

    Args:
        losses: List or dict of loss values
        title: Plot title
        save_path: Optional path to save
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    if isinstance(losses, dict):
        for name, values in losses.items():
            ax.plot(values, label=name)
        ax.legend()
    else:
        ax.plot(losses)

    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_sampling_gif(image_list, save_path, duration=100):
    """Create a GIF showing the sampling process.

    Args:
        image_list: List of tensors, each (N, C, H, W)
        save_path: Path to save GIF
        duration: Duration per frame in ms
    """
    frames = []
    for images in image_list:
        # Take first image, convert to numpy
        img = images[0]
        img = (img + 1) / 2
        img = img.clamp(0, 1)

        if img.shape[0] == 1:
            img = img[0]  # Grayscale
            img = (img.numpy() * 255).astype(np.uint8)
        else:
            img = img.numpy().transpose(1, 2, 0)
            img = (img * 255).astype(np.uint8)

        # Scale up for visibility
        img = Image.fromarray(img)
        img = img.resize((128, 128), Image.NEAREST)
        frames.append(img)

    frames[0].save(
        save_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0
    )
    print(f"GIF saved to {save_path}")


def interpolate_latent(z1, z2, n_steps=10):
    """Interpolate between two latent vectors.

    Args:
        z1, z2: Latent vectors of shape (C, H, W)
        n_steps: Number of interpolation steps

    Returns:
        Tensor of shape (n_steps, C, H, W)
    """
    alphas = torch.linspace(0, 1, n_steps)
    return torch.stack([z1 * (1 - a) + z2 * a for a in alphas])
