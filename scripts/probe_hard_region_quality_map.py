"""Audit where full-image transform-index candidates fail quality.

Several candidates pass top-left crops but fail on the full image.  This script
computes per-tile RGB quality for selected global candidates so hard regions can
be targeted instead of guessing from a single crop.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from probe_color_coefficient_search import forward_coeff, inverse_coeff
from probe_transform_index_quantization import (
    DATA_DIR,
    RESULTS_DIR,
    quality_stats,
    quantize_transform,
    read_exr,
)


def parse_candidates(text: str) -> list[tuple[str, float, float, str, int]]:
    candidates = []
    for spec in text.split(","):
        parts = spec.strip().split(":")
        if len(parts) != 5:
            raise ValueError(
                "candidate must be name:a:b:transform:bits, got "
                f"{spec!r}")
        name, a, b, transform, bits = parts
        candidates.append((name, float(a), float(b), transform, int(bits)))
    return candidates


def candidate_recon(
    pixels: np.ndarray,
    a: float,
    b: float,
    transform: str,
    bits: int,
) -> np.ndarray:
    if a == 0.0 and b == 0.0:
        _, recon = quantize_transform(pixels, bits, transform)
        return recon
    planes = forward_coeff(pixels, a, b)
    _, recon_planes = quantize_transform(planes, bits, transform)
    return inverse_coeff(recon_planes, a, b)


def summarize_candidate(
    pixels: np.ndarray,
    recon: np.ndarray,
    tile_size: int,
) -> dict:
    tiles = []
    for y in range(0, pixels.shape[0], tile_size):
        for x in range(0, pixels.shape[1], tile_size):
            tile = pixels[y:y + tile_size, x:x + tile_size, :]
            tile_recon = recon[y:y + tile.shape[0], x:x + tile.shape[1], :]
            quality = quality_stats(tile, tile_recon)
            tiles.append({
                "x": x,
                "y": y,
                "w": tile.shape[1],
                "h": tile.shape[0],
                **quality,
            })
    full_quality = quality_stats(pixels, recon)
    hard = sorted(
        tiles,
        key=lambda row: (
            row["signed_log2_rmse"],
            row["gradient_signed_log2_nrmse"],
        ),
        reverse=True,
    )
    return {
        "full_quality": full_quality,
        "tile_count": len(tiles),
        "passing_tiles": sum(
            1 for row in tiles
            if row["signed_log2_rmse"] <= 0.004
            and row["signed_log2_p99"] <= 0.018
            and row["gradient_signed_log2_nrmse"] <= 0.5
        ),
        "worst_tiles": hard[:20],
        "tiles": tiles,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="sample_DSCF0009.EXR")
    parser.add_argument("--crop-size", type=int, default=0)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument(
        "--candidates",
        default=(
            "rgb-gamma075-b7:0:0:gamma075:7,"
            "rgb-signedlog-b7:0:0:signed-log:7,"
            "coeff0500-gamma075-b7:0.5:0:gamma075:7"
        ),
    )
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    candidates = parse_candidates(args.candidates)
    results = []
    for path in sorted(DATA_DIR.glob(args.glob)):
        pixels = read_exr(path, args.crop_size)
        image_rows = []
        print(path.name)
        for name, a, b, transform, bits in candidates:
            recon = candidate_recon(pixels, a, b, transform, bits)
            row = summarize_candidate(pixels, recon, args.tile_size)
            q = row["full_quality"]
            image_rows.append({
                "name": name,
                "a": a,
                "b": b,
                "transform": transform,
                "bits": bits,
                **row,
            })
            print(
                f"  {name}: pass_tiles={row['passing_tiles']}/{row['tile_count']} "
                f"log={q['signed_log2_rmse']:.3e} "
                f"p99={q['signed_log2_p99']:.3e} "
                f"grad={q['gradient_signed_log2_nrmse']:.3e}")
            for tile in row["worst_tiles"][:5]:
                print(
                    f"    worst x={tile['x']} y={tile['y']} "
                    f"log={tile['signed_log2_rmse']:.3e} "
                    f"p99={tile['signed_log2_p99']:.3e} "
                    f"grad={tile['gradient_signed_log2_nrmse']:.3e}")
        results.append({
            "image": path.name,
            "shape": list(pixels.shape),
            "crop_size": args.crop_size,
            "tile_size": args.tile_size,
            "candidates": image_rows,
        })

    if not args.no_save:
        safe_glob = args.glob.replace("*", "star").replace(".", "_")
        output = RESULTS_DIR / (
            f"hard_region_quality_map_{safe_glob}"
            f"_crop{args.crop_size}_tile{args.tile_size}.json"
        )
        output.write_text(json.dumps(results, indent=2))
        print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
