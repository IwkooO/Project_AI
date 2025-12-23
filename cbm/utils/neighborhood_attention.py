"""
Neighborhood Attention (NA) utilities.

This module provides a pure-PyTorch implementation of Neighborhood Attention via
an additive attention bias/mask, without custom CUDA kernels.

Key idea:
  - Allow attention only within a local KxK neighborhood (Chebyshev distance),
    and block all other pairs by adding a large negative bias (≈ -inf).
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Final

import torch


def _validate_kernel_size(kernel_size: int) -> int:
    k = int(kernel_size)
    if k <= 0:
        raise ValueError(f"kernel_size must be > 0, got {kernel_size}")
    if k % 2 == 0:
        raise ValueError(f"kernel_size must be odd (e.g. 3,5,7), got {kernel_size}")
    return k


def _validate_num_patches_square(num_patches: int) -> int:
    p = int(num_patches)
    if p <= 0:
        raise ValueError(f"num_patches must be > 0, got {num_patches}")
    side = int(math.isqrt(p))
    if side * side != p:
        raise ValueError(f"num_patches ({p}) must be a perfect square grid.")
    return side


@lru_cache(maxsize=128)
def _neighbor_bool_mask_cpu(num_patches: int, kernel_size: int) -> torch.Tensor:
    """
    Cached boolean neighborhood mask on CPU.

    Returns:
        torch.BoolTensor of shape [P, P] where True means "allowed attention".
    """
    k = _validate_kernel_size(kernel_size)
    side = _validate_num_patches_square(num_patches)
    radius: Final[int] = k // 2

    coords = torch.arange(side, device="cpu")
    grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
    patches_coords = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=1)  # [P, 2]

    diff = patches_coords.unsqueeze(1) - patches_coords.unsqueeze(0)  # [P, P, 2]
    abs_diff = diff.abs()
    is_neighbor = (abs_diff[:, :, 0] <= radius) & (abs_diff[:, :, 1] <= radius)  # Chebyshev
    return is_neighbor.contiguous()


def create_local_mask(num_patches: int, kernel_size: int = 5) -> torch.Tensor:
    """
    Generates a 2D additive attention mask/bias for Neighborhood Attention.

    Convention:
      - 0.0: allowed (j is within i's neighborhood)
      - -inf: blocked (j is outside the neighborhood)

    Note:
      For dtype safety (fp16/bf16), consider using `get_local_attention_bias(...)`
      which uses `torch.finfo(dtype).min` instead of `-inf`.

    Returns:
        torch.FloatTensor (CPU) of shape [P, P].
    """
    is_neighbor = _neighbor_bool_mask_cpu(int(num_patches), int(kernel_size))
    p = int(num_patches)
    mask = torch.full((p, p), float("-inf"), dtype=torch.float32, device="cpu")
    mask.masked_fill_(is_neighbor, 0.0)
    return mask


def get_local_attention_bias(
    num_patches: int,
    kernel_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Returns an additive attention bias suitable for MultiheadAttention / TransformerEncoder.

    This is a device/dtype-aware version of `create_local_mask`, using finfo.min
    as the blocking value (more robust than -inf for some reduced-precision paths).

    Shape:
      - [P, P]
    """
    is_neighbor = _neighbor_bool_mask_cpu(int(num_patches), int(kernel_size))
    p = int(num_patches)

    if dtype.is_floating_point:
        neg = torch.finfo(dtype).min
    else:
        raise TypeError(f"dtype must be floating point for attention bias, got {dtype}")

    bias = torch.full((p, p), neg, device=device, dtype=dtype)
    bias.masked_fill_(is_neighbor.to(device=device), 0.0)
    return bias


