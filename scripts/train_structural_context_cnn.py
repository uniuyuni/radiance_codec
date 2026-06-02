"""Train a decoder-context CNN for structural residual bit probabilities.

This is a narrow feasibility probe for the V2 direction:

* The codec first decodes sign, exponent, and high mantissa bits.
* For a later field such as mantissa_mid, the decoder also knows the selected
  structural predictor mode.
* A small CNN predicts per-pixel residual bit probabilities for that field.

The model predicts probabilities only. The residual bits remain exact symbols
that an entropy coder must store, so the experiment is still lossless.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_structural_predictors import (  # noqa: E402
    DATA_DIR,
    FIELD_GROUPS,
    float_to_bits,
    load_blosc_ratios,
    predictors,
    read_exr,
)

TRAIN_DIR = ROOT / "ml" / "training_data"
CHECKPOINT_DIR = ROOT / "ml" / "checkpoints_structural_context"
CHECKPOINT_DIR.mkdir(exist_ok=True)
RESULTS_DIR = ROOT / "results"

MAX_CHANNELS = 4
KNOWN_BITS = tuple(range(31, 14, -1))
MODE_NAMES = ["zero", "west", "north", "northwest", "northeast", "float_med", "float_planar"]


@dataclass(frozen=True)
class LoadedImage:
    path: Path
    bits: np.ndarray
    predictor_bits: dict[str, np.ndarray]


class ResidualContextCNN(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, hidden: int = 96):
        super().__init__()
        self.stem = nn.Conv2d(input_channels, hidden, 5, padding=2)
        self.blocks = nn.Sequential(
            ResBlock(hidden),
            ResBlock(hidden),
            ResBlock(hidden),
            ResBlock(hidden),
        )
        self.head = nn.Conv2d(hidden, output_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(F.gelu(self.stem(x))))


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(F.gelu(self.conv1(x)))


def pick_device() -> torch.device:
    if os.environ.get("RADIANCE_CODEC_FORCE_CPU", "0") == "1":
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@lru_cache(maxsize=6)
def load_image(path_text: str) -> LoadedImage:
    path = Path(path_text)
    pixels = read_exr(path)
    bits = float_to_bits(pixels)
    return LoadedImage(path=path, bits=bits, predictor_bits=predictors(pixels))


def bit_planes(bits: np.ndarray, bit_indices: tuple[int, ...]) -> np.ndarray:
    height, width, channels = bits.shape
    shifts = np.asarray(bit_indices, dtype=np.uint32)
    unpacked = ((bits[..., None] >> shifts) & 1).astype(np.float32)
    unpacked = unpacked.transpose(2, 3, 0, 1).reshape(
        channels * len(bit_indices),
        height,
        width,
    )
    planes = np.zeros((MAX_CHANNELS * len(bit_indices), height, width), dtype=np.float32)
    planes[:channels * len(bit_indices)] = unpacked
    return planes


def mode_planes(mode_index: int, height: int, width: int) -> np.ndarray:
    planes = np.zeros((len(MODE_NAMES), height, width), dtype=np.float32)
    planes[mode_index] = 1.0
    return planes


def make_sample(
    image: LoadedImage,
    y: int,
    x: int,
    tile_size: int,
    field: str,
    mode_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mode = MODE_NAMES[mode_index]
    sl = np.s_[y:y + tile_size, x:x + tile_size]
    bits = image.bits[sl]
    residual = bits ^ image.predictor_bits[mode][sl]
    height, width, channels = bits.shape
    context = np.concatenate(
        [
            bit_planes(bits, KNOWN_BITS),
            mode_planes(mode_index, height, width),
        ],
        axis=0,
    )
    target_bits = FIELD_GROUPS[field]
    target = bit_planes(residual, target_bits)
    mask = np.zeros_like(target, dtype=bool)
    mask[:channels * len(target_bits)] = True
    return context, target, mask


def random_tile_origin(image: LoadedImage, tile_size: int) -> tuple[int, int]:
    height, width, _ = image.bits.shape
    max_y = max(0, height - tile_size)
    max_x = max(0, width - tile_size)
    y = random.randrange(0, max_y + 1, tile_size) if max_y else 0
    x = random.randrange(0, max_x + 1, tile_size) if max_x else 0
    return y, x


def kt_binary_cost(ones: int, total: int, alpha: float = 0.5) -> float:
    if total == 0:
        return 0.0
    zeros = total - ones
    log_probability = (
        math.lgamma(ones + alpha)
        + math.lgamma(zeros + alpha)
        - math.lgamma(total + 2.0 * alpha)
        - 2.0 * math.lgamma(alpha)
        + math.lgamma(2.0 * alpha)
    )
    return -log_probability / math.log(2.0)


def adaptive_cost(bits: np.ndarray, bit_indices: tuple[int, ...]) -> float:
    flat = bits.reshape(-1)
    total = flat.size
    cost = 0.0
    for bit_index in bit_indices:
        ones = int(((flat >> bit_index) & 1).sum())
        cost += kt_binary_cost(ones, total)
    return cost


def bce_bits(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    losses = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return losses.masked_select(mask).sum() / math.log(2.0)


def bce_bits_per_sample(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    losses = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    losses = losses * mask.to(dtype=losses.dtype)
    return losses.flatten(1).sum(dim=1) / math.log(2.0)


def train_epoch(
    model: ResidualContextCNN,
    optimizer: torch.optim.Optimizer,
    train_paths: list[Path],
    device: torch.device,
    args: argparse.Namespace,
) -> float:
    model.train()
    losses = []
    for _ in range(args.steps_per_epoch):
        contexts = []
        targets = []
        masks = []
        for _batch in range(args.batch_size):
            image = load_image(str(random.choice(train_paths)))
            y, x = random_tile_origin(image, args.tile_size)
            mode_index = random.randrange(len(MODE_NAMES))
            context, target, mask = make_sample(
                image, y, x, args.tile_size, args.field, mode_index
            )
            contexts.append(context)
            targets.append(target)
            masks.append(mask)
        xb = torch.from_numpy(np.stack(contexts)).to(device)
        target = torch.from_numpy(np.stack(targets)).to(device)
        mask = torch.from_numpy(np.stack(masks)).to(device)
        logits = model(xb)
        loss = bce_bits(logits, target, mask) / mask.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()) * len(FIELD_GROUPS[args.field]))
    return float(np.mean(losses))


def evaluate_image(
    model: ResidualContextCNN,
    path: Path,
    device: torch.device,
    args: argparse.Namespace,
) -> dict:
    image = load_image(str(path))
    bits = image.bits
    height, width, channels = bits.shape
    raw_bits = bits.size * 32
    target_bits = FIELD_GROUPS[args.field]
    adaptive_total = 0.0
    context_total = 0.0
    hybrid_total = 0.0
    target_adaptive_total = 0.0
    target_context_total = 0.0
    target_hybrid_total = 0.0
    mode_overhead_total = 0.0
    model.eval()

    with torch.no_grad():
        for y in range(0, height, args.tile_size):
            for x in range(0, width, args.tile_size):
                sl = np.s_[y:y + args.tile_size, x:x + args.tile_size]
                tile = bits[sl]
                tile_values = tile.size
                mode_overhead_total += args.mode_overhead_bits * len(FIELD_GROUPS)

                for field, field_bits in FIELD_GROUPS.items():
                    raw_cost = tile_values * len(field_bits)
                    candidates = {"raw": raw_cost}
                    for mode in MODE_NAMES:
                        residual = tile ^ image.predictor_bits[mode][sl]
                        candidates[mode] = adaptive_cost(residual, field_bits)
                    best_adaptive = min(candidates.values()) + args.mode_overhead_bits
                    adaptive_total += best_adaptive
                    if field != args.field:
                        context_total += best_adaptive
                        hybrid_total += best_adaptive
                        continue

                    target_adaptive_total += best_adaptive
                    model_costs = [raw_cost + args.mode_overhead_bits]
                    contexts = []
                    targets = []
                    masks = []
                    for mode_index, mode in enumerate(MODE_NAMES):
                        context, target, mask = make_sample(
                            image, y, x, args.tile_size, field, mode_index
                        )
                        contexts.append(context)
                        targets.append(target)
                        masks.append(mask)
                    logits = model(torch.from_numpy(np.stack(contexts)).to(device))
                    costs = bce_bits_per_sample(
                        logits,
                        torch.from_numpy(np.stack(targets)).to(device),
                        torch.from_numpy(np.stack(masks)).to(device),
                    )
                    model_costs.extend(
                        float(cost.detach().cpu()) + args.mode_overhead_bits
                        for cost in costs
                    )
                    best_context = min(model_costs)
                    target_context_total += best_context
                    context_total += best_context
                    best_hybrid = min(best_adaptive, best_context) + args.ml_switch_overhead_bits
                    target_hybrid_total += best_hybrid
                    hybrid_total += best_hybrid

    return {
        "image": path.name,
        "shape": [height, width, channels],
        "field": args.field,
        "tile_size": args.tile_size,
        "raw_bits": raw_bits,
        "adaptive_bits": adaptive_total,
        "context_bits": context_total,
        "hybrid_bits": hybrid_total,
        "adaptive_ratio": raw_bits / adaptive_total,
        "context_ratio": raw_bits / context_total,
        "hybrid_ratio": raw_bits / hybrid_total,
        "target_adaptive_bits": target_adaptive_total,
        "target_context_bits": target_context_total,
        "target_hybrid_bits": target_hybrid_total,
        "target_gain": target_adaptive_total / target_context_total,
        "target_hybrid_gain": target_adaptive_total / target_hybrid_total,
        "mode_overhead_bits": mode_overhead_total,
    }


def geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def collect_readable(paths: list[Path], limit: int | None) -> list[Path]:
    readable = []
    for path in paths:
        try:
            load_image(str(path))
        except Exception as error:
            print(f"SKIP {path.name}: {error}")
            continue
        readable.append(path)
        if limit is not None and len(readable) >= limit:
            break
    return readable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--field", choices=sorted(FIELD_GROUPS), default="mantissa_mid")
    parser.add_argument("--tile-size", type=int, default=64)
    parser.add_argument("--train-limit", type=int, default=12)
    parser.add_argument("--eval-limit", type=int)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--steps-per-epoch", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mode-overhead-bits", type=int, default=4)
    parser.add_argument("--ml-switch-overhead-bits", type=int, default=1)
    parser.add_argument("--checkpoint")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device()
    print(f"device: {device}")

    train_paths = collect_readable(sorted(TRAIN_DIR.glob("*.exr")), args.train_limit)
    test_paths = collect_readable(sorted(DATA_DIR.glob("*.exr")), args.eval_limit)
    if args.eval_limit is not None:
        test_paths = test_paths[:args.eval_limit]
    if not train_paths or not test_paths:
        raise RuntimeError("missing train or test EXR files")

    input_channels = MAX_CHANNELS * len(KNOWN_BITS) + len(MODE_NAMES)
    output_channels = MAX_CHANNELS * len(FIELD_GROUPS[args.field])
    model = ResidualContextCNN(input_channels, output_channels).to(device)
    checkpoint = (
        Path(args.checkpoint)
        if args.checkpoint
        else CHECKPOINT_DIR / f"{args.field}_tile{args.tile_size}_latest.pt"
    )
    if checkpoint.exists():
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        print(f"loaded checkpoint: {checkpoint}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"field={args.field} tile={args.tile_size} "
        f"train_images={len(train_paths)} test_images={len(test_paths)} "
        f"params={parameters / 1e6:.2f}M"
    )

    history = []
    best_context = 0.0
    for epoch in range(args.epochs):
        start = time.perf_counter()
        train_bits = train_epoch(model, optimizer, train_paths, device, args)
        elapsed = time.perf_counter() - start
        torch.save(model.state_dict(), checkpoint)
        history.append(
            {
                "epoch": epoch,
                "train_target_bits_per_value": train_bits,
                "time_sec": elapsed,
            }
        )
        print(
            f"epoch={epoch:3d} train_target={train_bits:.3f} "
            f"bits/value time={elapsed:.1f}s"
        )

    results = [evaluate_image(model, path, device, args) for path in test_paths]
    blosc = load_blosc_ratios()
    print(
        f"{'image':43s} {'adapt':>9s} {'context':>9s} "
        f"{'hybrid':>9s} {'target':>8s} {'Blosc':>8s}"
    )
    print("-" * 98)
    for result in results:
        blosc_ratio = blosc.get(Path(result["image"]).stem)
        result["blosc_ratio"] = blosc_ratio
        if result["context_ratio"] > best_context:
            best_context = result["context_ratio"]
        blosc_text = f"{blosc_ratio:7.2f}x" if blosc_ratio is not None else "      -"
        print(
            f"{Path(result['image']).stem:43s} "
            f"{result['adaptive_ratio']:8.2f}x "
            f"{result['context_ratio']:8.2f}x "
            f"{result['hybrid_ratio']:8.2f}x "
            f"{result['target_hybrid_gain']:7.3f}x "
            f"{blosc_text}"
        )
    print("-" * 98)
    for key in ["adaptive_ratio", "context_ratio", "hybrid_ratio", "blosc_ratio"]:
        values = [row[key] for row in results if row.get(key) is not None]
        print(f"geomean {key:15s}: {geomean(values):.3f}x")

    output = RESULTS_DIR / f"structural_context_{args.field}_tile{args.tile_size}.json"
    output.write_text(
        json.dumps(
            {
                "args": vars(args),
                "history": history,
                "results": results,
            },
            indent=2,
        )
    )
    print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
