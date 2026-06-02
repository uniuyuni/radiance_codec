"""Audit likely source precision classes for float32 HDR images.

This checks whether tiles look like true float32, half-like/PXR24-like, or
decimal-origin floats that could be losslessly represented as scaled integers.
It is diagnostic: source classes can later feed route selection and learned
contexts.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_structural_predictors import DATA_DIR, float_to_bits, read_exr  # noqa: E402

RESULTS_DIR = ROOT / "results"


def common_low_zero_bits(values: np.ndarray, max_bits: int) -> int:
    flat = values.reshape(-1)
    best = 0
    for width in range(1, max_bits + 1):
        mask = np.uint32((1 << width) - 1)
        if bool(np.all((flat & mask) == 0)):
            best = width
        else:
            break
    return best


def decimal_exact_rate(values: np.ndarray, scale_power: int) -> float:
    finite = np.isfinite(values)
    if not bool(np.any(finite)):
        return 0.0
    selected = values[finite].astype(np.float64)
    scale = float(10**scale_power)
    rounded = np.rint(selected * scale)
    reconstructed = (rounded / scale).astype(np.float32)
    return float(np.mean(reconstructed == values[finite]))


def half_exact_rate(values: np.ndarray) -> float:
    finite = np.isfinite(values)
    if not bool(np.any(finite)):
        return 0.0
    selected = values[finite]
    with np.errstate(over="ignore", invalid="ignore"):
        reconstructed = selected.astype(np.float16).astype(np.float32)
    return float(np.mean(reconstructed == selected))


def classify_tile(pixels: np.ndarray, bits: np.ndarray) -> dict:
    mantissa = bits & np.uint32(0x7FFFFF)
    low_zero = common_low_zero_bits(mantissa, 23)
    half_rate = half_exact_rate(pixels)
    decimal_rates = {
        str(power): decimal_exact_rate(pixels, power)
        for power in range(0, 7)
    }
    best_decimal_power, best_decimal_rate = max(
        decimal_rates.items(),
        key=lambda item: item[1],
    )
    if low_zero >= 16:
        klass = "bfloat_or_coarser"
    elif low_zero >= 13 or half_rate == 1.0:
        klass = "half_like"
    elif low_zero >= 8:
        klass = "pxr24_like"
    elif best_decimal_rate == 1.0:
        klass = f"decimal_1e-{best_decimal_power}"
    elif low_zero >= 1:
        klass = "partly_quantized"
    else:
        klass = "true_or_dithered_float"
    return {
        "class": klass,
        "low_zero_bits": low_zero,
        "half_exact_rate": half_rate,
        "best_decimal_power": int(best_decimal_power),
        "best_decimal_exact_rate": float(best_decimal_rate),
        "decimal_exact_rates": decimal_rates,
    }


def summarize_image(path: Path, crop_size: int, tile_size: int) -> dict:
    pixels = read_exr(path)
    if crop_size:
        pixels = pixels[:crop_size, :crop_size]
    pixels = np.ascontiguousarray(pixels, dtype=np.float32)
    bits = float_to_bits(pixels).copy()
    height, width, _channels = bits.shape
    class_histogram = Counter()
    low_zero_histogram = Counter()
    tiles = []
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            sl = np.s_[y:y + tile_size, x:x + tile_size]
            row = classify_tile(pixels[sl], bits[sl])
            row["y"] = y
            row["x"] = x
            tiles.append(row)
            class_histogram[row["class"]] += 1
            low_zero_histogram[str(row["low_zero_bits"])] += 1
    whole = classify_tile(pixels, bits)
    return {
        "image": path.name,
        "shape": list(bits.shape),
        "crop_size": crop_size,
        "tile_size": tile_size,
        "whole_image": whole,
        "class_histogram": dict(class_histogram),
        "low_zero_histogram": dict(low_zero_histogram),
        "tiles": tiles,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="*.exr")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--crop-size", type=int, default=0)
    parser.add_argument("--tile-size", type=int, default=128)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = []
    for path in sorted(DATA_DIR.glob(args.glob)):
        row = summarize_image(path, args.crop_size, args.tile_size)
        rows.append(row)
        whole = row["whole_image"]
        print(
            f"{path.stem:43s} class={whole['class']:22s} "
            f"low0={whole['low_zero_bits']:2d} "
            f"half={whole['half_exact_rate']:.3f} "
            f"dec=1e-{whole['best_decimal_power']}:{whole['best_decimal_exact_rate']:.3f} "
            f"tiles={row['class_histogram']}"
        )
        if args.limit and len(rows) >= args.limit:
            break
    safe_glob = args.glob.replace("*", "star").replace(".", "_")
    output = RESULTS_DIR / (
        f"source_precision_{safe_glob}_tile{args.tile_size}_crop{args.crop_size}.json"
    )
    output.write_text(json.dumps(rows, indent=2))
    print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
