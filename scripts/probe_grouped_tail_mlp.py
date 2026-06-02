"""Probe learned probabilities for grouped-delta low mantissa payload bits.

The goal is not to predict replacement values.  The model predicts
P(payload_bit=1), and an entropy coder would still store the exact bit.  This
lets us test whether low mantissa residual bits contain structure that the
hand-built GDX2 context families do not capture.

Two feature sets are useful:

* strict: only payload bits and metadata available in the current GDX2 payload
  decode schedule.
* oracle: strict plus actual ordered-float high body bits.  This is not
  available in the current payload-only schedule, but it tests whether an
  exponent/high-mantissa-aware progressive decoder would be worth designing.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_structural_predictors import DATA_DIR, float_to_bits, read_exr  # noqa: E402
from evaluate_structural_delta_context import bits_to_ordered  # noqa: E402
from structural_delta_grouped_roundtrip import choose_group_payloads  # noqa: E402

RESULTS_DIR = ROOT / "results"
MAX_CHANNELS = 4
CONTEXT_FAMILIES = (
    "base",
    "prev_only",
    "wn_prev",
    "spatial_prev_pc",
    "spatial_prev_channel",
    "spatial_hi_neighbors",
    "spatial_hi_channel",
    "spatial_hi_pc",
    "prev_xy_channel",
)


@dataclass
class PreparedImage:
    path: Path
    bits: np.ndarray
    ordered: np.ndarray
    payload: np.ndarray
    modes: dict[tuple[int, int, str], str]
    tile_size: int


class TailMLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def pick_device() -> torch.device:
    if os.environ.get("RADIANCE_CODEC_FORCE_CPU", "0") == "1":
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def parse_bits(text: str) -> tuple[int, ...]:
    if ":" in text:
        start_text, end_text = text.split(":", 1)
        return tuple(range(int(start_text), int(end_text)))
    return tuple(int(part) for part in text.split(",") if part)


def one_hot(index: int, size: int) -> list[float]:
    values = [0.0] * size
    if 0 <= index < size:
        values[index] = 1.0
    return values


def prepare_image(
    path: Path,
    crop_size: int,
    tile_size: int,
    previous_bits: int,
) -> PreparedImage:
    pixels = read_exr(path)
    if crop_size:
        pixels = pixels[:crop_size, :crop_size]
    pixels = np.ascontiguousarray(pixels, dtype=np.float32)
    bits = float_to_bits(pixels).copy()
    ordered = bits_to_ordered(bits)
    modes, payloads, _carries = choose_group_payloads(
        bits,
        pixels,
        tile_size,
        previous_bits,
        "body",
    )
    return PreparedImage(
        path=path,
        bits=bits,
        ordered=ordered,
        payload=payloads["body"],
        modes=modes,
        tile_size=tile_size,
    )


def collect_images(
    glob_text: str,
    limit: int,
    crop_size: int,
    tile_size: int,
    previous_bits: int,
) -> list[PreparedImage]:
    prepared: list[PreparedImage] = []
    for path in sorted(DATA_DIR.glob(glob_text)):
        try:
            prepared.append(prepare_image(path, crop_size, tile_size, previous_bits))
        except Exception as error:
            print(f"SKIP {path.name}: {error}")
            continue
        print(f"loaded {path.name}: shape={prepared[-1].payload.shape}")
        if limit and len(prepared) >= limit:
            break
    if not prepared:
        raise RuntimeError(f"no usable images matched {DATA_DIR / glob_text}")
    return prepared


def mode_vocab(images: list[PreparedImage]) -> dict[str, int]:
    names = sorted({mode for image in images for mode in image.modes.values()})
    return {name: index for index, name in enumerate(names)}


def payload_bit(value: int, bit: int) -> int:
    return (value >> bit) & 1


def feature_row(
    image: PreparedImage,
    y: int,
    x: int,
    c: int,
    bit: int,
    bits: tuple[int, ...],
    mode_to_id: dict[str, int],
    feature_set: str,
) -> tuple[list[float], int, int, int]:
    height, width, channels = image.payload.shape
    tile_y = (y // image.tile_size) * image.tile_size
    tile_x = (x // image.tile_size) * image.tile_size
    local_y = y - tile_y
    local_x = x - tile_x
    mode = image.modes[(tile_y, tile_x, "body")]
    mode_id = mode_to_id[mode]

    value = int(image.payload[y, x, c])
    west = int(image.payload[y, x - 1, c]) if local_x > 0 else 0
    north = int(image.payload[y - 1, x, c]) if local_y > 0 else 0
    northwest = int(image.payload[y - 1, x - 1, c]) if local_y > 0 and local_x > 0 else 0
    northeast = (
        int(image.payload[y - 1, x + 1, c])
        if local_y > 0 and local_x + 1 < min(image.tile_size, width - tile_x)
        else 0
    )
    previous_channel = int(image.payload[y, x, c - 1]) if c > 0 else 0

    target = payload_bit(value, bit)
    features: list[float] = []
    features.extend(one_hot(bits.index(bit), len(bits)))
    features.extend(one_hot(c, MAX_CHANNELS))
    features.extend(one_hot(mode_id, len(mode_to_id)))
    features.append(float(channels) / MAX_CHANNELS)
    features.append((local_x & 1) * 1.0)
    features.append((local_y & 1) * 1.0)
    features.append((local_x & 3) / 3.0)
    features.append((local_y & 3) / 3.0)

    # Current residual higher bits.  These are already decoded when coding
    # lower mantissa planes.
    for offset in range(1, 9):
        previous_bit = bit + offset
        features.append(float(payload_bit(value, previous_bit)) if previous_bit <= 30 else 0.0)

    # Existing GDX-like spatial evidence for the current bit and nearby higher
    # bits.  North-east is decoder-safe because the previous row is done.
    neighbors = (west, north, northwest, northeast, previous_channel)
    for neighbor in neighbors:
        features.append(float(payload_bit(neighbor, bit)))
    for offset in (1, 2, 3, 4):
        previous_bit = bit + offset
        for neighbor in (west, north, previous_channel):
            features.append(
                float(payload_bit(neighbor, previous_bit))
                if previous_bit <= 30
                else 0.0
            )

    # Compact numeric summaries help a small MLP see monotonic relationships
    # without requiring it to reconstruct integers from only binary features.
    high_tail = (value >> max(bit + 1, 0)) & ((1 << min(12, max(0, 30 - bit))) - 1)
    features.append(high_tail / float(max(1, (1 << min(12, max(0, 30 - bit))) - 1)))
    features.append(((west ^ value) & 0x7fff) / 32767.0)
    features.append(((north ^ value) & 0x7fff) / 32767.0)

    oracle_key = 0
    if feature_set == "oracle":
        ordered = int(image.ordered[y, x, c])
        west_ordered = int(image.ordered[y, x - 1, c]) if local_x > 0 else 0
        north_ordered = int(image.ordered[y - 1, x, c]) if local_y > 0 else 0
        for high_bit in range(30, 14, -1):
            features.append(float(payload_bit(ordered, high_bit)))
        for high_bit in range(30, 22, -1):
            features.append(float(payload_bit(west_ordered, high_bit)))
            features.append(float(payload_bit(north_ordered, high_bit)))
        exponent = (int(image.bits[y, x, c]) >> 23) & 0xff
        features.append(exponent / 255.0)
        features.append(((ordered >> 15) & 0xffff) / 65535.0)
        oracle_key = ((ordered >> 15) & 0xffff) | (exponent << 16)

    tabular_key = 0
    tabular_key |= bits.index(bit)
    tabular_key |= c << 5
    tabular_key |= mode_id << 8
    tabular_key |= (value >> (bit + 1) & 0x0f) << 14
    tabular_key |= payload_bit(west, bit) << 18
    tabular_key |= payload_bit(north, bit) << 19
    tabular_key |= payload_bit(northwest, bit) << 20
    tabular_key |= payload_bit(northeast, bit) << 21
    tabular_key |= payload_bit(previous_channel, bit) << 22
    return features, target, tabular_key, oracle_key


def sample_dataset(
    images: list[PreparedImage],
    sample_count: int,
    bits: tuple[int, ...],
    mode_to_id: dict[str, int],
    feature_set: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    features: list[list[float]] = []
    labels: list[int] = []
    tabular_keys: list[int] = []
    oracle_keys: list[int] = []
    for _ in range(sample_count):
        image = rng.choice(images)
        height, width, channels = image.payload.shape
        y = rng.randrange(height)
        x = rng.randrange(width)
        c = rng.randrange(channels)
        bit = rng.choice(bits)
        row, label, tabular_key, oracle_key = feature_row(
            image,
            y,
            x,
            c,
            bit,
            bits,
            mode_to_id,
            feature_set,
        )
        features.append(row)
        labels.append(label)
        tabular_keys.append(tabular_key)
        oracle_keys.append(oracle_key)
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels, dtype=np.float32),
        np.asarray(tabular_keys, dtype=np.uint64),
        np.asarray(oracle_keys, dtype=np.uint64),
    )


def prior_bits(train_labels: np.ndarray, eval_labels: np.ndarray) -> float:
    p = (float(train_labels.sum()) + 0.5) / (float(train_labels.size) + 1.0)
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return float(
        -(
            eval_labels * math.log2(p)
            + (1.0 - eval_labels) * math.log2(1.0 - p)
        ).mean()
    )


def tabular_bits(
    train_keys: np.ndarray,
    train_labels: np.ndarray,
    eval_keys: np.ndarray,
    eval_labels: np.ndarray,
) -> float:
    counts: dict[int, list[int]] = {}
    for key, label in zip(train_keys, train_labels, strict=True):
        total, ones = counts.setdefault(int(key), [0, 0])
        total += 1
        ones += int(label)
        counts[int(key)] = [total, ones]
    total_bits = 0.0
    for key, label in zip(eval_keys, eval_labels, strict=True):
        total, ones = counts.get(int(key), [0, 0])
        p = (ones + 0.5) / (total + 1.0)
        total_bits += -math.log2(p if label else 1.0 - p)
    return total_bits / float(eval_labels.size)


def train_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    eval_x: np.ndarray,
    eval_y: np.ndarray,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[TailMLP, float, list[dict[str, float]]]:
    model = TailMLP(train_x.shape[1], args.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    x_train = torch.from_numpy(train_x).to(device)
    y_train = torch.from_numpy(train_y).to(device)
    x_eval = torch.from_numpy(eval_x).to(device)
    y_eval = torch.from_numpy(eval_y).to(device)
    history: list[dict[str, float]] = []
    rng = np.random.default_rng(args.seed)
    best_eval = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(train_y.size)
        losses = []
        for start in range(0, train_y.size, args.batch_size):
            indices = torch.from_numpy(order[start:start + args.batch_size]).to(device)
            logits = model(x_train.index_select(0, indices))
            target = y_train.index_select(0, indices)
            loss = F.binary_cross_entropy_with_logits(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()) / math.log(2.0))

        model.eval()
        with torch.no_grad():
            eval_logits = model(x_eval)
            eval_loss = F.binary_cross_entropy_with_logits(eval_logits, y_eval)
        row = {
            "epoch": float(epoch),
            "train_bits_per_sample": float(np.mean(losses)),
            "eval_bits_per_sample": float(eval_loss.detach().cpu()) / math.log(2.0),
        }
        history.append(row)
        if row["eval_bits_per_sample"] < best_eval:
            best_eval = row["eval_bits_per_sample"]
            best_state = copy.deepcopy(model.state_dict())
        print(
            f"  epoch={epoch:02d} train={row['train_bits_per_sample']:.4f} "
            f"eval={row['eval_bits_per_sample']:.4f} bits/sample"
        )
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_eval, history


def kt_cost(ones: int, total: int) -> float:
    if total == 0:
        return 0.0
    zeros = total - ones
    log_probability = (
        math.lgamma(ones + 0.5)
        + math.lgamma(zeros + 0.5)
        - math.lgamma(total + 1.0)
        - 2.0 * math.lgamma(0.5)
    )
    return -log_probability / math.log(2.0)


def shift_plane(plane: np.ndarray, dy: int, dx: int) -> np.ndarray:
    out = np.zeros_like(plane)
    height, width, _channels = plane.shape
    dst_y0 = max(0, -dy)
    dst_y1 = min(height, height - dy)
    dst_x0 = max(0, -dx)
    dst_x1 = min(width, width - dx)
    src_y0 = dst_y0 + dy
    src_y1 = dst_y1 + dy
    src_x0 = dst_x0 + dx
    src_x1 = dst_x1 + dx
    out[dst_y0:dst_y1, dst_x0:dst_x1] = plane[src_y0:src_y1, src_x0:src_x1]
    return out


def previous_plane(payload: np.ndarray, bit: int, offset: int) -> np.ndarray:
    previous_bit = bit + offset
    if previous_bit > 30:
        return np.zeros(payload.shape, dtype=np.uint16)
    return ((payload >> np.uint32(previous_bit)) & np.uint32(1)).astype(np.uint16)


def family_context(payload: np.ndarray, bit: int, family: str) -> tuple[np.ndarray, int]:
    plane = ((payload >> np.uint32(bit)) & np.uint32(1)).astype(np.uint16)
    w = shift_plane(plane, 0, -1)
    n = shift_plane(plane, -1, 0)
    nw = shift_plane(plane, -1, -1)
    ne = shift_plane(plane, -1, 1)
    p1 = previous_plane(payload, bit, 1)
    p2 = previous_plane(payload, bit, 2)
    p3 = previous_plane(payload, bit, 3)
    p4 = previous_plane(payload, bit, 4)
    context = np.zeros(payload.shape, dtype=np.uint16)

    if family == "base":
        context |= w | (n << 1) | (nw << 2) | (ne << 3)
        context |= p1 << 4
        context |= p2 << 5
        context |= p3 << 6
        context |= p4 << 7
        return context, 16 << 4
    if family == "prev_only":
        context |= p1 | (p2 << 1) | (p3 << 2) | (p4 << 3)
        return context, 1 << 4
    if family == "wn_prev":
        context |= w | (n << 1) | (p1 << 2) | (p2 << 3) | (p3 << 4) | (p4 << 5)
        return context, 1 << 6

    pc = np.zeros_like(plane)
    pc[..., 1:] = plane[..., :-1]
    if family == "spatial_prev_pc":
        context |= w | (n << 1) | (nw << 2) | (ne << 3) | (pc << 4)
        context |= p1 << 5
        context |= p2 << 6
        context |= p3 << 7
        context |= p4 << 8
        return context, 1 << 9

    channels = payload.shape[2]
    c = np.arange(channels, dtype=np.uint16)[None, None, :]
    c0 = np.broadcast_to((c == 0).astype(np.uint16), payload.shape)
    c1 = np.broadcast_to((c == 1).astype(np.uint16), payload.shape)
    c2 = np.broadcast_to((c == 2).astype(np.uint16), payload.shape)
    if family == "spatial_prev_channel":
        context |= w | (n << 1) | (nw << 2) | (ne << 3)
        context |= c0 << 4
        context |= c1 << 5
        context |= c2 << 6
        context |= p1 << 7
        context |= p2 << 8
        context |= p3 << 9
        context |= p4 << 10
        return context, 1 << 11

    wp1 = shift_plane(p1, 0, -1)
    np1 = shift_plane(p1, -1, 0)
    wp2 = shift_plane(p2, 0, -1)
    np2 = shift_plane(p2, -1, 0)
    if family == "spatial_hi_neighbors":
        context |= w | (n << 1) | (nw << 2) | (ne << 3)
        context |= p1 << 4
        context |= wp1 << 5
        context |= np1 << 6
        context |= p2 << 7
        context |= wp2 << 8
        context |= np2 << 9
        return context, 1 << 10
    if family == "spatial_hi_channel":
        context |= w | (n << 1)
        context |= c0 << 2
        context |= c1 << 3
        context |= c2 << 4
        context |= p1 << 5
        context |= wp1 << 6
        context |= np1 << 7
        context |= p2 << 8
        context |= wp2 << 9
        context |= np2 << 10
        return context, 1 << 11
    if family == "spatial_hi_pc":
        context |= w | (n << 1) | (nw << 2) | (ne << 3) | (pc << 4)
        context |= p1 << 5
        context |= wp1 << 6
        context |= np1 << 7
        context |= p2 << 8
        context |= wp2 << 9
        context |= np2 << 10
        return context, 1 << 11
    if family == "prev_xy_channel":
        height, width, channels = payload.shape
        x_lsb = np.broadcast_to(
            (np.arange(width, dtype=np.uint16)[None, :, None] & 1),
            payload.shape,
        )
        y_lsb = np.broadcast_to(
            (np.arange(height, dtype=np.uint16)[:, None, None] & 1),
            payload.shape,
        )
        c = np.arange(channels, dtype=np.uint16)[None, None, :]
        c0 = np.broadcast_to((c == 0).astype(np.uint16), payload.shape)
        c1 = np.broadcast_to((c == 1).astype(np.uint16), payload.shape)
        c2 = np.broadcast_to((c == 2).astype(np.uint16), payload.shape)
        context |= p1
        context |= p2 << 1
        context |= x_lsb << 2
        context |= y_lsb << 3
        context |= c0 << 4
        context |= c1 << 5
        context |= c2 << 6
        return context, 1 << 7
    raise ValueError(f"unknown family: {family}")


def context_family_cost(payload: np.ndarray, bit: int, family: str) -> float:
    context, context_count = family_context(payload, bit, family)
    symbols = ((payload >> np.uint32(bit)) & np.uint32(1)).reshape(-1)
    return context_cost_from_flat(context.reshape(-1), symbols, context_count)


def context_cost_from_flat(
    context_flat: np.ndarray,
    symbols: np.ndarray,
    context_count: int,
) -> float:
    totals = np.bincount(context_flat, minlength=context_count)
    ones = np.bincount(context_flat, weights=symbols, minlength=context_count)
    cost = 0.0
    for context_id in range(context_count):
        total = int(totals[context_id])
        if total:
            cost += kt_cost(int(ones[context_id]), total)
    return cost


def best_family_bits_per_sample(images: list[PreparedImage], bits: tuple[int, ...]) -> float:
    total_cost = 0.0
    total_symbols = 0
    for image in images:
        height, width, _channels = image.payload.shape
        for y in range(0, height, image.tile_size):
            for x in range(0, width, image.tile_size):
                tile = image.payload[y:y + image.tile_size, x:x + image.tile_size]
                for bit in bits:
                    best = min(
                        context_family_cost(tile, bit, family)
                        for family in CONTEXT_FAMILIES
                    )
                    total_cost += best
                    total_symbols += tile.size
    return total_cost / float(total_symbols)


def mlp_full_bits_per_sample(
    model: TailMLP,
    images: list[PreparedImage],
    bits: tuple[int, ...],
    mode_to_id: dict[str, int],
    feature_set: str,
    device: torch.device,
    batch_size: int,
) -> float:
    model.eval()
    total_bits = 0.0
    total_symbols = 0
    rows: list[list[float]] = []
    labels: list[float] = []

    def flush() -> None:
        nonlocal total_bits, total_symbols, rows, labels
        if not rows:
            return
        xb = torch.from_numpy(np.asarray(rows, dtype=np.float32)).to(device)
        yb = torch.from_numpy(np.asarray(labels, dtype=np.float32)).to(device)
        with torch.no_grad():
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb, reduction="sum")
        total_bits += float(loss.detach().cpu()) / math.log(2.0)
        total_symbols += len(labels)
        rows = []
        labels = []

    for image in images:
        height, width, channels = image.payload.shape
        for y in range(height):
            for x in range(width):
                for c in range(channels):
                    for bit in bits:
                        row, label, _tabular_key, _oracle_key = feature_row(
                            image,
                            y,
                            x,
                            c,
                            bit,
                            bits,
                            mode_to_id,
                            feature_set,
                        )
                        rows.append(row)
                        labels.append(float(label))
                        if len(rows) >= batch_size:
                            flush()
    flush()
    return total_bits / float(total_symbols)


def predict_logits(
    model: TailMLP,
    rows: list[list[float]],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    logits: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            xb = torch.from_numpy(
                np.asarray(rows[start:start + batch_size], dtype=np.float32)
            ).to(device)
            logits.append(model(xb).detach().cpu().numpy())
    return np.concatenate(logits) if logits else np.empty((0,), dtype=np.float32)


def mlp_binned_context_bits_per_sample(
    model: TailMLP,
    images: list[PreparedImage],
    bits: tuple[int, ...],
    mode_to_id: dict[str, int],
    feature_set: str,
    device: torch.device,
    batch_size: int,
    bins: int,
) -> tuple[float, float, float, float, float]:
    total_family_cost = 0.0
    total_mlp_context_cost = 0.0
    total_hybrid_cost = 0.0
    total_augmented_cost = 0.0
    total_augmented_hybrid_cost = 0.0
    total_symbols = 0
    for image in images:
        height, width, channels = image.payload.shape
        for tile_y in range(0, height, image.tile_size):
            for tile_x in range(0, width, image.tile_size):
                tile = image.payload[
                    tile_y:tile_y + image.tile_size,
                    tile_x:tile_x + image.tile_size,
                ]
                tile_h, tile_w, _ = tile.shape
                for bit in bits:
                    family_cost = min(
                        context_family_cost(tile, bit, family)
                        for family in CONTEXT_FAMILIES
                    )
                    rows: list[list[float]] = []
                    labels: list[int] = []
                    for dy in range(tile_h):
                        y = tile_y + dy
                        for dx in range(tile_w):
                            x = tile_x + dx
                            for c in range(channels):
                                row, label, _tabular_key, _oracle_key = feature_row(
                                    image,
                                    y,
                                    x,
                                    c,
                                    bit,
                                    bits,
                                    mode_to_id,
                                    feature_set,
                                )
                                rows.append(row)
                                labels.append(label)
                    logits = predict_logits(model, rows, device, batch_size)
                    probs = 1.0 / (1.0 + np.exp(-logits))
                    context = np.clip((probs * bins).astype(np.int64), 0, bins - 1)
                    label_array = np.asarray(labels, dtype=np.uint8)
                    totals = np.bincount(context, minlength=bins)
                    ones = np.bincount(context, weights=label_array, minlength=bins)
                    mlp_context_cost = 0.0
                    for context_id in range(bins):
                        total = int(totals[context_id])
                        if total:
                            mlp_context_cost += kt_cost(int(ones[context_id]), total)
                    best_augmented = float("inf")
                    for family in CONTEXT_FAMILIES:
                        family_context_ids, family_context_count = family_context(
                            tile,
                            bit,
                            family,
                        )
                        combined_context = (
                            family_context_ids.reshape(-1).astype(np.int64) * bins
                            + context
                        )
                        augmented_cost = context_cost_from_flat(
                            combined_context,
                            label_array,
                            family_context_count * bins,
                        )
                        best_augmented = min(best_augmented, augmented_cost)
                    total_family_cost += family_cost
                    total_mlp_context_cost += mlp_context_cost
                    total_hybrid_cost += min(family_cost, mlp_context_cost)
                    total_augmented_cost += best_augmented
                    total_augmented_hybrid_cost += min(family_cost, best_augmented)
                    total_symbols += len(labels)
    return (
        total_family_cost / float(total_symbols),
        total_mlp_context_cost / float(total_symbols),
        total_hybrid_cost / float(total_symbols),
        total_augmented_cost / float(total_symbols),
        total_augmented_hybrid_cost / float(total_symbols),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-glob", default="*.exr")
    parser.add_argument("--eval-glob", default="ph_studio_small_03_1k.exr")
    parser.add_argument("--train-limit", type=int, default=4)
    parser.add_argument("--eval-limit", type=int, default=1)
    parser.add_argument("--allow-train-eval-overlap", action="store_true")
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--previous-bits", type=int, default=4)
    parser.add_argument("--bits", default="0:15")
    parser.add_argument("--feature-set", choices=["strict", "oracle", "both"], default="both")
    parser.add_argument("--train-samples", type=int, default=50000)
    parser.add_argument("--eval-samples", type=int, default=20000)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--full-eval", action="store_true")
    parser.add_argument("--full-eval-batch-size", type=int, default=8192)
    parser.add_argument("--mlp-context-bins", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=11)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    bits = parse_bits(args.bits)
    device = pick_device()
    print(f"device: {device}")
    print(f"target bits: {bits}")

    train_images = collect_images(
        args.train_glob,
        args.train_limit,
        args.crop_size,
        args.tile_size,
        args.previous_bits,
    )
    eval_images = collect_images(
        args.eval_glob,
        args.eval_limit,
        args.crop_size,
        args.tile_size,
        args.previous_bits,
    )
    if not args.allow_train_eval_overlap:
        eval_names = {image.path.name for image in eval_images}
        train_images = [image for image in train_images if image.path.name not in eval_names]
        if not train_images:
            raise RuntimeError(
                "train/eval split is empty after removing overlap; "
                "use --allow-train-eval-overlap for a self-fit smoke test"
            )
    mode_to_id = mode_vocab(train_images + eval_images)
    print(f"mode vocab: {mode_to_id}")

    feature_sets = ["strict", "oracle"] if args.feature_set == "both" else [args.feature_set]
    output: dict[str, object] = {
        "args": vars(args),
        "mode_to_id": mode_to_id,
        "results": {},
    }
    for feature_set in feature_sets:
        print(f"\nfeature_set={feature_set}")
        train_x, train_y, train_keys, train_oracle_keys = sample_dataset(
            train_images,
            args.train_samples,
            bits,
            mode_to_id,
            feature_set,
            args.seed,
        )
        eval_x, eval_y, eval_keys, eval_oracle_keys = sample_dataset(
            eval_images,
            args.eval_samples,
            bits,
            mode_to_id,
            feature_set,
            args.seed + 1009,
        )
        bit_prior = prior_bits(train_y, eval_y)
        tabular = tabular_bits(train_keys, train_y, eval_keys, eval_y)
        oracle_tabular = None
        if feature_set == "oracle":
            oracle_tabular = tabular_bits(
                train_oracle_keys,
                train_y,
                eval_oracle_keys,
                eval_y,
            )
        model, mlp, history = train_model(
            train_x,
            train_y,
            eval_x,
            eval_y,
            device,
            args,
        )
        family_full = None
        mlp_full = None
        mlp_context_full = None
        hybrid_context_full = None
        augmented_context_full = None
        augmented_hybrid_full = None
        if args.full_eval:
            print("  running full-crop family/MLP comparison...")
            mlp_full = mlp_full_bits_per_sample(
                model,
                eval_images,
                bits,
                mode_to_id,
                feature_set,
                device,
                args.full_eval_batch_size,
            )
            (
                family_full,
                mlp_context_full,
                hybrid_context_full,
                augmented_context_full,
                augmented_hybrid_full,
            ) = mlp_binned_context_bits_per_sample(
                model,
                eval_images,
                bits,
                mode_to_id,
                feature_set,
                device,
                args.full_eval_batch_size,
                args.mlp_context_bins,
            )
        result = {
            "samples_train": int(train_y.size),
            "samples_eval": int(eval_y.size),
            "positive_rate_train": float(train_y.mean()),
            "positive_rate_eval": float(eval_y.mean()),
            "bit_prior_bits_per_sample": bit_prior,
            "tabular_strict_bits_per_sample": tabular,
            "tabular_oracle_bits_per_sample": oracle_tabular,
            "mlp_bits_per_sample": mlp,
            "full_family_bits_per_sample": family_full,
            "full_mlp_bits_per_sample": mlp_full,
            "full_mlp_context_bits_per_sample": mlp_context_full,
            "full_hybrid_family_mlp_context_bits_per_sample": hybrid_context_full,
            "full_augmented_family_mlp_context_bits_per_sample": augmented_context_full,
            "full_augmented_hybrid_bits_per_sample": augmented_hybrid_full,
            "full_gain_vs_family": (
                None if family_full is None or mlp_full is None else family_full / mlp_full
            ),
            "full_context_gain_vs_family": (
                None
                if family_full is None or mlp_context_full is None
                else family_full / mlp_context_full
            ),
            "full_hybrid_gain_vs_family": (
                None
                if family_full is None or hybrid_context_full is None
                else family_full / hybrid_context_full
            ),
            "full_augmented_gain_vs_family": (
                None
                if family_full is None or augmented_context_full is None
                else family_full / augmented_context_full
            ),
            "full_augmented_hybrid_gain_vs_family": (
                None
                if family_full is None or augmented_hybrid_full is None
                else family_full / augmented_hybrid_full
            ),
            "gain_vs_bit_prior": bit_prior / mlp,
            "gain_vs_tabular_strict": tabular / mlp,
            "history": history,
        }
        output["results"][feature_set] = result
        print(
            f"summary {feature_set}: "
            f"pos_train={train_y.mean():.4f} pos_eval={eval_y.mean():.4f} "
            f"prior={bit_prior:.4f} "
            f"tabular={tabular:.4f} mlp={mlp:.4f} "
            f"gain_vs_tabular={tabular / mlp:.3f}x"
        )
        if oracle_tabular is not None:
            print(f"  oracle_tabular={oracle_tabular:.4f} bits/sample")
        if family_full is not None and mlp_full is not None:
            print(
                f"  full-crop family={family_full:.4f} "
                f"mlp={mlp_full:.4f} gain={family_full / mlp_full:.3f}x"
            )
        if (
            family_full is not None
            and mlp_context_full is not None
            and hybrid_context_full is not None
        ):
            print(
                f"  mlp-context bins={args.mlp_context_bins}: "
                f"mlp_ctx={mlp_context_full:.4f} "
                f"hybrid={hybrid_context_full:.4f} "
                f"hybrid_gain={family_full / hybrid_context_full:.3f}x"
            )
        if (
            family_full is not None
            and augmented_context_full is not None
            and augmented_hybrid_full is not None
        ):
            print(
                f"  family x mlp-bin: augmented={augmented_context_full:.4f} "
                f"hybrid={augmented_hybrid_full:.4f} "
                f"hybrid_gain={family_full / augmented_hybrid_full:.3f}x"
            )

    safe_eval = (
        args.eval_glob
        .replace("*", "star")
        .replace("/", "_")
        .replace(":", "-")
        .replace(".", "_")
    )
    output_path = RESULTS_DIR / (
        f"grouped_tail_mlp_{args.feature_set}_{safe_eval}_crop{args.crop_size}_"
        f"bits{args.bits.replace(':', '-')}.json"
    )
    output_path.write_text(json.dumps(output, indent=2))
    print(f"\nSaved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
