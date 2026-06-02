"""Train the oversized V2 spatial teacher and estimate conditional bit cost."""
from __future__ import annotations

import json
import math
import os
import random
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from hierarchy_teacher import (
    HierarchyTeacher,
    bits_to_planes,
    make_teacher_sample_from_planes,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "ml" / "hierarchy_data"
CHECKPOINT_DIR = ROOT / "ml" / "checkpoints_hierarchy"
CHECKPOINT_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 8
STEPS_PER_EPOCH = 100
EPOCHS = 8
EVAL_SAMPLES = 64
SEED = 0


def pick_device() -> torch.device:
    if os.environ.get("RADIANCE_CODEC_FORCE_CPU", "0") == "1":
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@lru_cache(maxsize=32)
def load_tile(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    with np.load(path) as tile:
        bits = tile["bits"]
        return (
            bits_to_planes(bits),
            bits_to_planes(tile["residuals"]),
            tile["pass_map"],
            bits.shape[2],
        )


class HierarchyDataset(Dataset):
    def __init__(self, split: str, virtual_len: int):
        self.root = DATA_ROOT / split
        manifest = json.loads((self.root / "manifest.json").read_text())
        tile_size = int(manifest["tile_size"])
        self.tiles = [
            str(self.root / item["path"])
            for item in manifest["tiles"]
            if item["shape"][0] == tile_size and item["shape"][1] == tile_size
        ]
        if not self.tiles:
            raise RuntimeError(f"no hierarchy tiles for {split}")
        self.virtual_len = virtual_len

    def __len__(self) -> int:
        return self.virtual_len

    def __getitem__(self, _index: int):
        bit_planes, residual_planes, pass_map, channels = load_tile(
            random.choice(self.tiles)
        )
        pass_index = random.randint(1, int(pass_map.max()))
        context, target, loss_mask = make_teacher_sample_from_planes(
            bit_planes, residual_planes, pass_map, pass_index, channels
        )
        return (
            torch.from_numpy(context),
            torch.from_numpy(target),
            torch.from_numpy(loss_mask),
        )


def masked_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    losses = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return losses.masked_select(loss_mask).mean()


def evaluate(
    model: HierarchyTeacher,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for context, target, loss_mask in loader:
            logits = model(context.to(device, dtype=torch.float32))
            loss = masked_loss(
                logits,
                target.to(device, dtype=torch.float32),
                loss_mask.to(device),
            )
            losses.append(float(loss.detach()))
    return float(np.mean(losses)) / math.log(2) * 32


def main() -> int:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = pick_device()
    print(f"device: {device}")

    train_data = HierarchyDataset("train", STEPS_PER_EPOCH * BATCH_SIZE)
    eval_data = HierarchyDataset("test", EVAL_SAMPLES)
    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        drop_last=True,
    )
    eval_loader = DataLoader(
        eval_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    print(f"tiles: train={len(train_data.tiles)}, test={len(eval_data.tiles)}")

    model = HierarchyTeacher().to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"teacher: {parameters / 1e6:.2f}M parameters")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    history = []
    best_eval = float("inf")

    print(f"{'epoch':>5s} {'train bits/value':>17s} {'eval bits/value':>16s} {'time':>8s}")
    print("-" * 54)
    for epoch in range(EPOCHS):
        model.train()
        t0 = time.perf_counter()
        losses = []
        for context, target, loss_mask in train_loader:
            logits = model(context.to(device, dtype=torch.float32))
            loss = masked_loss(
                logits,
                target.to(device, dtype=torch.float32),
                loss_mask.to(device),
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        train_bits = float(np.mean(losses)) / math.log(2) * 32
        eval_bits = evaluate(model, eval_loader, device)
        elapsed = time.perf_counter() - t0
        marker = ""
        if eval_bits < best_eval:
            best_eval = eval_bits
            torch.save(model.state_dict(), CHECKPOINT_DIR / "model_best.pt")
            marker = " *"
        history.append(
            {
                "epoch": epoch,
                "train_bits_per_value": train_bits,
                "eval_bits_per_value": eval_bits,
                "time_sec": elapsed,
            }
        )
        (CHECKPOINT_DIR / "metrics.json").write_text(
            json.dumps(
                {
                    "device": str(device),
                    "parameters": parameters,
                    "history": history,
                    "best_eval_bits_per_value": best_eval,
                },
                indent=2,
            )
        )
        print(f"{epoch:5d} {train_bits:17.3f} {eval_bits:16.3f} {elapsed:7.1f}s{marker}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
