"""Estimate structural predictor with quantized probability side information."""
from __future__ import annotations

import argparse
import json
import math
import sys
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


def binary_cost(ones: int, total: int, probability: float) -> float:
    probability = min(max(probability, 1e-6), 1.0 - 1e-6)
    zeros = total - ones
    return -ones * math.log2(probability) - zeros * math.log2(1.0 - probability)


def best_entropy_cost(symbols: np.ndarray) -> float:
    ones = int(symbols.sum())
    total = int(symbols.size)
    if ones == 0 or ones == total:
        return 0.0
    p = ones / total
    return binary_cost(ones, total, p)


def quantized_cost(symbols: np.ndarray, bins: np.ndarray) -> float:
    ones = int(symbols.sum())
    total = int(symbols.size)
    if total == 0:
        return 0.0
    p = ones / total
    q = float(bins[int(np.argmin(np.abs(bins - p)))])
    return binary_cost(ones, total, q)


def image_estimate(path: Path, tile_size: int, prob_bins: int) -> dict:
    pixels = read_exr(path)
    bits = float_to_bits(pixels)
    predictor_bits = predictors(pixels)
    height, width, channels = bits.shape
    bins = np.linspace(0.0, 1.0, prob_bins + 2, dtype=np.float64)[1:-1]
    prob_bin_bits = math.ceil(math.log2(prob_bins))
    mode_bits = math.ceil(math.log2(len(predictor_bits) + 1))
    raw_bits = bits.size * 32
    oracle_bits = 0.0
    binned_bits = 0.0
    raw_fallback_bits = 0.0

    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            sl = np.s_[y:y + tile_size, x:x + tile_size]
            tile_values = bits[sl].size
            for bit_indices in FIELD_GROUPS.values():
                raw_cost = tile_values * len(bit_indices)
                best_oracle = raw_cost
                best_binned = raw_cost + mode_bits
                for pred in predictor_bits.values():
                    residual = bits[sl] ^ pred[sl]
                    flat = residual.reshape(-1)
                    oracle = mode_bits
                    binned = mode_bits
                    for bit_index in bit_indices:
                        symbols = ((flat >> bit_index) & 1).astype(np.uint8)
                        oracle += best_entropy_cost(symbols)
                        binned += quantized_cost(symbols, bins) + prob_bin_bits
                    if oracle < best_oracle:
                        best_oracle = oracle
                    if binned < best_binned:
                        best_binned = binned
                oracle_bits += best_oracle
                if raw_cost < best_binned:
                    raw_fallback_bits += raw_cost + mode_bits
                    binned_bits += raw_cost + mode_bits
                else:
                    binned_bits += best_binned

    return {
        "image": path.name,
        "shape": [height, width, channels],
        "tile_size": tile_size,
        "prob_bins": prob_bins,
        "raw_bits": raw_bits,
        "oracle_bits": oracle_bits,
        "binned_bits": binned_bits,
        "oracle_ratio": raw_bits / oracle_bits,
        "binned_ratio": raw_bits / binned_bits,
        "raw_fallback_bits": raw_fallback_bits,
    }


def geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="*.exr")
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--prob-bins", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = sorted(DATA_DIR.glob(args.glob))
    blosc = load_blosc_ratios()
    rows = []
    print(f"{'image':43s} {'oracle':>9s} {'binned':>9s} {'Blosc':>9s} {'hybrid':>9s}")
    print("-" * 86)
    for path in paths:
        row = image_estimate(path, args.tile_size, args.prob_bins)
        blosc_ratio = blosc.get(path.stem)
        hybrid = max(row["binned_ratio"], blosc_ratio) if blosc_ratio else row["binned_ratio"]
        row["blosc_ratio"] = blosc_ratio
        row["hybrid_ratio"] = hybrid
        rows.append(row)
        print(
            f"{path.stem:43s} {row['oracle_ratio']:8.2f}x "
            f"{row['binned_ratio']:8.2f}x {blosc_ratio:8.2f}x {hybrid:8.2f}x"
        )
    print("-" * 86)
    for key in ["oracle_ratio", "binned_ratio", "blosc_ratio", "hybrid_ratio"]:
        print(f"geomean {key:13s}: {geomean([row[key] for row in rows]):.2f}x")
    out = ROOT / "results" / "structural_prob_bins_estimate.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nSaved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
