"""Train/test tile-split MLP context mixer for hard float tails.

This is a route-E probe: it asks whether a learned probability model can beat
the current fixed GDX context families on held-out tiles from a hard float32
sample.  The model predicts exact payload bits; an entropy coder would still
store every bit losslessly.

Reported costs:

* direct MLP: cross entropy from fixed model probabilities;
* MLP-bin KT: model probability bin as an adaptive universal context;
* augmented KT: current GDX family contexts crossed with MLP probability bins.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_structural_predictors import DATA_DIR, float_to_bits, read_exr  # noqa: E402
from evaluate_structural_delta_context import bits_to_ordered  # noqa: E402
from probe_grouped_tail_mlp import (  # noqa: E402
    CONTEXT_FAMILIES,
    PreparedImage,
    TailMLP,
    context_cost_from_flat,
    context_family_cost,
    family_context,
    feature_row,
    mode_vocab,
    parse_bits,
)
from structural_delta_grouped_roundtrip import choose_group_payloads  # noqa: E402

RESULTS_DIR = ROOT / "results"


@dataclass(frozen=True)
class TileOrigin:
    y: int
    x: int


def pick_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
    return torch.device("cpu")


def prepare_image(path: Path, crop_size: int, tile_size: int, previous_bits: int) -> PreparedImage:
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


def tile_origins(image: PreparedImage) -> list[TileOrigin]:
    height, width, _channels = image.payload.shape
    return [
        TileOrigin(y, x)
        for y in range(0, height, image.tile_size)
        for x in range(0, width, image.tile_size)
    ]


def split_tiles(image: PreparedImage, pattern: str) -> tuple[list[TileOrigin], list[TileOrigin]]:
    origins = tile_origins(image)
    if len(origins) < 2:
        raise ValueError("tile split needs at least two tiles")
    if pattern == "checker":
        train = []
        eval_tiles = []
        for origin in origins:
            ty = origin.y // image.tile_size
            tx = origin.x // image.tile_size
            if (ty + tx) & 1:
                eval_tiles.append(origin)
            else:
                train.append(origin)
    elif pattern == "bottom":
        max_y = max(origin.y for origin in origins)
        train = [origin for origin in origins if origin.y != max_y]
        eval_tiles = [origin for origin in origins if origin.y == max_y]
    elif pattern == "right":
        max_x = max(origin.x for origin in origins)
        train = [origin for origin in origins if origin.x != max_x]
        eval_tiles = [origin for origin in origins if origin.x == max_x]
    elif pattern == "last":
        train = origins[:-1]
        eval_tiles = origins[-1:]
    else:
        raise ValueError(f"unknown eval pattern: {pattern}")
    if not train or not eval_tiles:
        raise ValueError(f"pattern {pattern!r} produced empty split")
    return train, eval_tiles


def sample_rows(
    image: PreparedImage,
    tiles: list[TileOrigin],
    sample_count: int,
    bits: tuple[int, ...],
    mode_to_id: dict[str, int],
    feature_set: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    features: list[list[float]] = []
    labels: list[int] = []
    height, width, channels = image.payload.shape
    for _ in range(sample_count):
        origin = rng.choice(tiles)
        tile_h = min(image.tile_size, height - origin.y)
        tile_w = min(image.tile_size, width - origin.x)
        y = origin.y + rng.randrange(tile_h)
        x = origin.x + rng.randrange(tile_w)
        c = rng.randrange(channels)
        bit = rng.choice(bits)
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
        features.append(row)
        labels.append(label)
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.float32)


def train_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    eval_x: np.ndarray,
    eval_y: np.ndarray,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[TailMLP, list[dict[str, float]]]:
    model = TailMLP(train_x.shape[1], args.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    x_train = torch.from_numpy(train_x).to(device)
    y_train = torch.from_numpy(train_y).to(device)
    x_eval = torch.from_numpy(eval_x).to(device)
    y_eval = torch.from_numpy(eval_y).to(device)
    rng = np.random.default_rng(args.seed)
    best_eval = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    history = []
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
        eval_bits = float(eval_loss.detach().cpu()) / math.log(2.0)
        row = {
            "epoch": float(epoch),
            "train_bits": float(np.mean(losses)),
            "eval_bits": eval_bits,
        }
        history.append(row)
        if eval_bits < best_eval:
            best_eval = eval_bits
            best_state = copy.deepcopy(model.state_dict())
        print(
            f"  epoch={epoch:02d} train={row['train_bits']:.4f} "
            f"eval={eval_bits:.4f} bits/sample",
        )
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def predict_logits(model: TailMLP, rows: list[list[float]], device: torch.device, batch_size: int) -> np.ndarray:
    out = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            xb = torch.from_numpy(
                np.asarray(rows[start:start + batch_size], dtype=np.float32),
            ).to(device)
            out.append(model(xb).detach().cpu().numpy())
    return np.concatenate(out) if out else np.empty((0,), dtype=np.float32)


def bce_bits_from_logits(logits: np.ndarray, labels: np.ndarray) -> float:
    logits_t = torch.from_numpy(logits.astype(np.float32))
    labels_t = torch.from_numpy(labels.astype(np.float32))
    loss = F.binary_cross_entropy_with_logits(logits_t, labels_t, reduction="sum")
    return float(loss) / math.log(2.0)


def kt_context_cost(context: np.ndarray, labels: np.ndarray, context_count: int) -> float:
    return context_cost_from_flat(
        context.astype(np.int64).reshape(-1),
        labels.astype(np.uint8).reshape(-1),
        context_count,
    )


def evaluate_full(
    image: PreparedImage,
    eval_tiles: list[TileOrigin],
    model: TailMLP,
    bits: tuple[int, ...],
    mode_to_id: dict[str, int],
    feature_set: str,
    device: torch.device,
    args: argparse.Namespace,
) -> dict:
    totals = {
        "gdx_bits": 0.0,
        "direct_mlp_bits": 0.0,
        "mlp_bin_kt_bits": 0.0,
        "augmented_kt_bits": 0.0,
        "hybrid_augmented_bits": 0.0,
        "symbols": 0,
    }
    picks = {"augmented": 0, "gdx": 0}
    channels = image.payload.shape[2]
    for origin in eval_tiles:
        tile = image.payload[
            origin.y:origin.y + image.tile_size,
            origin.x:origin.x + image.tile_size,
        ]
        tile_h, tile_w, _ = tile.shape
        for bit in bits:
            gdx = min(
                context_family_cost(tile, bit, family)
                for family in CONTEXT_FAMILIES
            )
            rows: list[list[float]] = []
            labels: list[int] = []
            for dy in range(tile_h):
                y = origin.y + dy
                for dx in range(tile_w):
                    x = origin.x + dx
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
            label_array = np.asarray(labels, dtype=np.uint8)
            logits = predict_logits(model, rows, device, args.full_eval_batch_size)
            direct = bce_bits_from_logits(logits, label_array.astype(np.float32))
            probs = 1.0 / (1.0 + np.exp(-logits))
            mlp_context = np.clip(
                (probs * args.mlp_context_bins).astype(np.int64),
                0,
                args.mlp_context_bins - 1,
            )
            mlp_bin = kt_context_cost(
                mlp_context,
                label_array,
                args.mlp_context_bins,
            )
            best_augmented = float("inf")
            for family in CONTEXT_FAMILIES:
                family_ids, family_count = family_context(tile, bit, family)
                combined = (
                    family_ids.reshape(-1).astype(np.int64) * args.mlp_context_bins
                    + mlp_context
                )
                cost = kt_context_cost(
                    combined,
                    label_array,
                    family_count * args.mlp_context_bins,
                )
                best_augmented = min(best_augmented, cost)
            totals["gdx_bits"] += gdx
            totals["direct_mlp_bits"] += direct
            totals["mlp_bin_kt_bits"] += mlp_bin
            totals["augmented_kt_bits"] += best_augmented
            totals["hybrid_augmented_bits"] += min(gdx, best_augmented)
            totals["symbols"] += len(labels)
            if best_augmented < gdx:
                picks["augmented"] += 1
            else:
                picks["gdx"] += 1
    symbols = float(totals["symbols"])
    return {
        **totals,
        "gdx_bps": totals["gdx_bits"] / symbols,
        "direct_mlp_bps": totals["direct_mlp_bits"] / symbols,
        "mlp_bin_kt_bps": totals["mlp_bin_kt_bits"] / symbols,
        "augmented_kt_bps": totals["augmented_kt_bits"] / symbols,
        "hybrid_augmented_bps": totals["hybrid_augmented_bits"] / symbols,
        "direct_gain": totals["gdx_bits"] / totals["direct_mlp_bits"],
        "mlp_bin_gain": totals["gdx_bits"] / totals["mlp_bin_kt_bits"],
        "augmented_gain": totals["gdx_bits"] / totals["augmented_kt_bits"],
        "hybrid_augmented_gain": totals["gdx_bits"] / totals["hybrid_augmented_bits"],
        "picks": picks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="sample_hilberts-mill-conference-room_2K.exr")
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--previous-bits", type=int, default=4)
    parser.add_argument("--bits", default="0:15")
    parser.add_argument("--feature-set", choices=("strict", "oracle"), default="strict")
    parser.add_argument("--eval-pattern", choices=("checker", "bottom", "right", "last"), default="checker")
    parser.add_argument("--train-samples", type=int, default=80000)
    parser.add_argument("--eval-samples", type=int, default=30000)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--full-eval-batch-size", type=int, default=8192)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--mlp-context-bins", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda", "auto"), default="cpu")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bits = parse_bits(args.bits)
    paths = sorted(DATA_DIR.glob(args.glob))
    if not paths:
        raise RuntimeError(f"no images matched {DATA_DIR / args.glob}")
    if len(paths) > 1:
        raise RuntimeError("this tile-split probe expects exactly one image")
    image = prepare_image(paths[0], args.crop_size, args.tile_size, args.previous_bits)
    train_tiles, eval_tiles = split_tiles(image, args.eval_pattern)
    mode_to_id = mode_vocab([image])
    device = pick_device(args.device)
    print(
        f"image={image.path.name} shape={image.payload.shape} "
        f"feature_set={args.feature_set} device={device}",
    )
    print(
        f"tiles train={len(train_tiles)} eval={len(eval_tiles)} "
        f"bits={bits[0]}..{bits[-1]}",
    )
    train_x, train_y = sample_rows(
        image,
        train_tiles,
        args.train_samples,
        bits,
        mode_to_id,
        args.feature_set,
        args.seed,
    )
    eval_x, eval_y = sample_rows(
        image,
        eval_tiles,
        args.eval_samples,
        bits,
        mode_to_id,
        args.feature_set,
        args.seed + 1,
    )
    model, history = train_model(train_x, train_y, eval_x, eval_y, device, args)
    full = evaluate_full(
        image,
        eval_tiles,
        model,
        bits,
        mode_to_id,
        args.feature_set,
        device,
        args,
    )
    print(
        "full eval: "
        f"gdx={full['gdx_bps']:.4f} "
        f"direct={full['direct_mlp_bps']:.4f} gain={full['direct_gain']:.4f} "
        f"mlp_bin={full['mlp_bin_kt_bps']:.4f} gain={full['mlp_bin_gain']:.4f} "
        f"aug={full['augmented_kt_bps']:.4f} gain={full['augmented_gain']:.4f} "
        f"hybrid={full['hybrid_augmented_bps']:.4f} "
        f"hybrid_gain={full['hybrid_augmented_gain']:.4f} picks={full['picks']}",
    )
    row = {
        "image": image.path.name,
        "shape": list(image.payload.shape),
        "crop_size": args.crop_size,
        "tile_size": args.tile_size,
        "bits": list(bits),
        "feature_set": args.feature_set,
        "eval_pattern": args.eval_pattern,
        "train_tiles": [origin.__dict__ for origin in train_tiles],
        "eval_tiles": [origin.__dict__ for origin in eval_tiles],
        "train_samples": args.train_samples,
        "eval_samples": args.eval_samples,
        "history": history,
        "full_eval": full,
    }
    if not args.no_save:
        output = RESULTS_DIR / (
            f"tail_mlp_tile_split_{image.path.stem}_crop{args.crop_size}"
            f"_{args.feature_set}_{args.eval_pattern}.json"
        )
        output.write_text(json.dumps(row, indent=2))
        print(f"Saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
