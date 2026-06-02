"""Estimate structural coding with spatial bit-plane context models.

This is a PIC-like practical probe. After choosing a reversible predictor, each
residual bit-plane is coded with decoder-reproducible adaptive contexts derived
from already decoded neighboring bits:

    west, north, north-west, north-east

The model has no probability side stream. It is closer to classic image context
coding than a neural net, but it directly tests whether contour/region structure
inside residual bit-planes is the missing piece.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

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

RESULTS_DIR = ROOT / "results"
DEFAULT_MODES = [
    "zero",
    "west",
    "north",
    "northwest",
    "northeast",
    "channel_green",
    "channel_previous",
    "bit_xor_planar",
    "bit_majority",
    "float_med",
    "float_planar",
]


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


def spatial_contexts(plane: np.ndarray) -> np.ndarray:
    """Return 4-neighbor context ids for an HxWxC binary plane."""
    west = np.zeros_like(plane)
    north = np.zeros_like(plane)
    northwest = np.zeros_like(plane)
    northeast = np.zeros_like(plane)
    west[:, 1:, :] = plane[:, :-1, :]
    north[1:, :, :] = plane[:-1, :, :]
    northwest[1:, 1:, :] = plane[:-1, :-1, :]
    northeast[1:, :-1, :] = plane[:-1, 1:, :]
    return west | (north << 1) | (northwest << 2) | (northeast << 3)


def context_bitplane_cost(
    bits: np.ndarray,
    bit_indices: tuple[int, ...],
    previous_bits: int,
) -> float:
    cost = 0.0
    for bit_index in bit_indices:
        plane = ((bits >> bit_index) & 1).astype(np.uint8, copy=False)
        context = spatial_contexts(plane)
        for offset in range(1, previous_bits + 1):
            previous_bit = bit_index + offset
            if previous_bit > 31:
                continue
            previous_plane = ((bits >> previous_bit) & 1).astype(np.uint8, copy=False)
            context |= previous_plane << (3 + offset)
        context_flat = context.reshape(-1)
        plane_flat = plane.reshape(-1)
        context_count = 16 << previous_bits
        totals = np.bincount(context_flat, minlength=context_count)
        ones = np.bincount(context_flat, weights=plane_flat, minlength=context_count)
        for context_id in range(context_count):
            cost += kt_binary_cost(int(ones[context_id]), int(totals[context_id]))
    return cost


def estimate_image(
    path: Path,
    tile_size: int,
    mode_overhead_bits: int,
    previous_bits: int,
    modes: list[str],
) -> dict:
    pixels = read_exr(path)
    bits = float_to_bits(pixels)
    predictor_bits = {
        mode: pred
        for mode, pred in predictors(pixels).items()
        if mode in modes
    }
    height, width, channels = bits.shape
    raw_bits = bits.size * 32
    total_bits = 0.0
    field_bits_total = Counter()
    mode_histogram = Counter()
    field_mode_histogram = {field: Counter() for field in FIELD_GROUPS}
    tile_count = 0

    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            tile_count += 1
            sl = np.s_[y:y + tile_size, x:x + tile_size]
            tile = bits[sl]
            tile_values = tile.size
            for field, bit_indices in FIELD_GROUPS.items():
                candidates = {"raw": tile_values * len(bit_indices)}
                for name, pred in predictor_bits.items():
                    residual = tile ^ pred[sl]
                    candidates[name] = context_bitplane_cost(
                        residual,
                        bit_indices,
                        previous_bits,
                    )
                best_mode = min(candidates, key=candidates.__getitem__)
                best_bits = candidates[best_mode] + mode_overhead_bits
                total_bits += best_bits
                field_bits_total[field] += best_bits
                mode_histogram[best_mode] += 1
                field_mode_histogram[field][best_mode] += 1

    return {
        "image": path.name,
        "shape": [height, width, channels],
        "tile_size": tile_size,
        "previous_bits": previous_bits,
        "modes": modes,
        "tile_count": tile_count,
        "raw_bits": raw_bits,
        "estimated_bits": total_bits,
        "estimated_bytes": math.ceil(total_bits / 8),
        "ratio": raw_bits / total_bits,
        "field_bits": dict(field_bits_total),
        "field_bits_per_value": {
            field: bits_value / bits.size
            for field, bits_value in field_bits_total.items()
        },
        "mode_histogram": dict(mode_histogram),
        "field_mode_histogram": {
            field: dict(histogram)
            for field, histogram in field_mode_histogram.items()
        },
    }


def geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="*.exr")
    parser.add_argument("--tile-size", type=int, default=64)
    parser.add_argument("--mode-overhead-bits", type=int, default=4)
    parser.add_argument("--previous-bits", type=int, default=0)
    parser.add_argument("--modes", nargs="+", default=DEFAULT_MODES)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = sorted(DATA_DIR.glob(args.glob))
    if not paths:
        raise RuntimeError(f"no images matched {DATA_DIR / args.glob}")
    blosc = load_blosc_ratios()
    results = []
    print(
        f"{'image':43s} {'ctx':>9s} {'Blosc':>9s} "
        f"{'hybrid':>9s} {'pick':>8s}"
    )
    print("-" * 86)
    for path in paths:
        result = estimate_image(
            path,
            args.tile_size,
            args.mode_overhead_bits,
            args.previous_bits,
            args.modes,
        )
        blosc_ratio = blosc.get(path.stem)
        if blosc_ratio is None:
            hybrid_ratio = result["ratio"]
            pick = "ctx"
        else:
            hybrid_ratio = max(result["ratio"], blosc_ratio)
            pick = "ctx" if result["ratio"] >= blosc_ratio else "Blosc"
        result["blosc_ratio"] = blosc_ratio
        result["hybrid_ratio"] = hybrid_ratio
        result["hybrid_pick"] = pick
        results.append(result)
        blosc_text = f"{blosc_ratio:8.2f}x" if blosc_ratio is not None else "       -"
        print(
            f"{path.stem:43s} {result['ratio']:8.2f}x "
            f"{blosc_text} {hybrid_ratio:8.2f}x {pick:>8s}"
        )
    print("-" * 86)
    for key in ["ratio", "blosc_ratio", "hybrid_ratio"]:
        values = [result[key] for result in results if result[key] is not None]
        print(f"geomean {key:12s}: {geomean(values):.3f}x")

    output = RESULTS_DIR / (
        f"structural_spatial_context_tile{args.tile_size}"
        f"_prev{args.previous_bits}_{len(args.modes)}modes.json"
    )
    output.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
