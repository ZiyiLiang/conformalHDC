"""Position hypervector encoding of binary images (MNIST)."""
import numpy as np
import torch

from .level import bipolar_sign


def make_position_hvs(n_positions, dim, device):
    return torch.randint(0, 2, (n_positions, dim), device=device, dtype=torch.float32) * 2 - 1


@torch.no_grad()
def encode_binary_images(pixels, pos_hvs, batch_size=512):
    """Bipolar HVs bundling the position HVs of the set pixels of (n, n_positions) images."""
    return np.concatenate([
        bipolar_sign(pixels[i:i + batch_size].to(pos_hvs.device, torch.float32) @ pos_hvs).cpu().numpy()
        for i in range(0, len(pixels), batch_size)
    ])
