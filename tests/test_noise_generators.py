import sys
import os
import torch
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from nodes_sampler import _perlin_noise, _spectral_noise

shape = (1, 4, 128, 128)
device = torch.device('cpu')

perlin = _perlin_noise(shape, device)
spectral = _spectral_noise(shape, device)

print(f"Perlin: mean={perlin.mean().item():.4f}, std={perlin.std().item():.4f}, max={perlin.max().item():.4f}, min={perlin.min().item():.4f}")
print(f"Spectral: mean={spectral.mean().item():.4f}, std={spectral.std().item():.4f}, max={spectral.max().item():.4f}, min={spectral.min().item():.4f}")
