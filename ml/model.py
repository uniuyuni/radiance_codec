"""Shared causal byte language model for ML entropy coding."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

EMBED_DIM = 64
CHANNELS = 64
DILATIONS = [1, 2, 4, 8, 16, 32, 64]
RF = 1 + sum((3 - 1) * d for d in DILATIONS)


class ChannelLayerNorm(nn.Module):
    """Normalize channels independently at each byte position."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class CausalResBlock(nn.Module):
    def __init__(self, ch: int, dilation: int):
        super().__init__()
        self.pad = (3 - 1) * dilation
        self.conv1 = nn.Conv1d(ch, 2 * ch, 3, dilation=dilation, padding=0)
        self.conv2 = nn.Conv1d(ch, ch, 1)
        self.ln = ChannelLayerNorm(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.pad(x, (self.pad, 0))
        h = self.conv1(h)
        a, b = h.chunk(2, dim=1)
        return self.ln(x + self.conv2(torch.tanh(a) * torch.sigmoid(b)))


class ByteLM(nn.Module):
    def __init__(self, ch: int = CHANNELS, dilations=DILATIONS):
        super().__init__()
        self.embed = nn.Embedding(256, EMBED_DIM)
        self.proj_in = nn.Conv1d(EMBED_DIM, ch, 1)
        self.blocks = nn.ModuleList([CausalResBlock(ch, d) for d in dilations])
        self.head = nn.Conv1d(ch, 256, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj_in(self.embed(x).transpose(1, 2))
        for blk in self.blocks:
            h = blk(h)
        return self.head(h)
