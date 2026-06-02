"""Oversized spatial teacher for V2 HDR hierarchy feasibility experiments."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_CHANNELS = 4
BITS_PER_CHANNEL = 32
PLANE_CHANNELS = MAX_CHANNELS * BITS_PER_CHANNEL
INPUT_CHANNELS = PLANE_CHANNELS + 1


def bits_to_planes(bits: np.ndarray) -> np.ndarray:
    """Convert HxWxC uint32 values to padded (4*32)xHxW uint8 planes."""
    if bits.dtype != np.uint32 or bits.ndim != 3:
        raise ValueError("expected HxWxC uint32 values")
    height, width, channels = bits.shape
    if channels > MAX_CHANNELS:
        raise ValueError(f"too many channels: {channels}")
    shifts = np.arange(31, -1, -1, dtype=np.uint32)
    unpacked = ((bits[..., None] >> shifts) & 1).astype(np.uint8)
    unpacked = unpacked.transpose(2, 3, 0, 1).reshape(
        channels * BITS_PER_CHANNEL, height, width
    )
    planes = np.zeros((PLANE_CHANNELS, height, width), dtype=np.uint8)
    planes[:channels * BITS_PER_CHANNEL] = unpacked
    return planes


def make_teacher_sample(
    bits: np.ndarray,
    residuals: np.ndarray,
    pass_map: np.ndarray,
    pass_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create context, target planes, and selected loss mask for one pass."""
    if bits.shape != residuals.shape:
        raise ValueError("bits and residuals shape mismatch")
    if pass_map.shape != bits.shape[:2]:
        raise ValueError("pass map shape mismatch")
    return make_teacher_sample_from_planes(
        bits_to_planes(bits),
        bits_to_planes(residuals),
        pass_map,
        pass_index,
        bits.shape[2],
    )


def make_teacher_sample_from_planes(
    bit_planes: np.ndarray,
    residual_planes: np.ndarray,
    pass_map: np.ndarray,
    pass_index: int,
    channels: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create one pass sample from cached uint8 planes."""
    known = pass_map < pass_index
    selected = pass_map == pass_index
    context = np.zeros((INPUT_CHANNELS, *pass_map.shape), dtype=np.uint8)
    context[:PLANE_CHANNELS] = bit_planes * known[None, ...]
    context[PLANE_CHANNELS] = known
    loss_mask = np.zeros_like(residual_planes, dtype=bool)
    loss_mask[:channels * BITS_PER_CHANNEL] = selected[None, ...]
    return context, residual_planes, loss_mask


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(F.gelu(self.conv1(x)))


class HierarchyTeacher(nn.Module):
    def __init__(self, hidden_channels: int = 96, blocks: int = 6):
        super().__init__()
        self.stem = nn.Conv2d(INPUT_CHANNELS, hidden_channels, 5, padding=2)
        self.blocks = nn.Sequential(
            *[ResBlock(hidden_channels) for _ in range(blocks)]
        )
        self.head = nn.Conv2d(hidden_channels, PLANE_CHANNELS, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.stem(x))
        return self.head(self.blocks(hidden))
