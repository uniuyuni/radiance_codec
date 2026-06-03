"""Probe color decorrelation and tile transform selection for transform-index.

This is a decision probe, not a production bitstream.  It estimates two next
steps before implementing them in C++:

* color/channel decorrelation before transform-index quantization;
* global-range candidates with tile-local transform selection.

Quality is always measured after reconstructing RGB.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from probe_transform_index_quantization import (
    DATA_DIR,
    RESULTS_DIR,
    estimate_index_bits,
    quality_stats,
    quantize_transform,
    read_exr,
)


@dataclass(frozen=True)
class Candidate:
    color: str
    transform: str
    bits: int
    indices: np.ndarray
    recon_rgb: np.ndarray
    estimated_bits: float
    quality: dict[str, float]


def forward_color(pixels: np.ndarray, mode: str) -> np.ndarray:
    if mode == "rgb" or pixels.shape[2] < 3:
        return np.ascontiguousarray(pixels)
    out = np.array(pixels, dtype=np.float32, copy=True)
    r = pixels[:, :, 0].astype(np.float64)
    g = pixels[:, :, 1].astype(np.float64)
    b = pixels[:, :, 2].astype(np.float64)
    if mode == "g-diff":
        out[:, :, 0] = g.astype(np.float32)
        out[:, :, 1] = (r - g).astype(np.float32)
        out[:, :, 2] = (b - g).astype(np.float32)
    elif mode == "ycocg":
        y = (r + 2.0 * g + b) * 0.25
        co = r - b
        cg = g - (r + b) * 0.5
        out[:, :, 0] = y.astype(np.float32)
        out[:, :, 1] = co.astype(np.float32)
        out[:, :, 2] = cg.astype(np.float32)
    else:
        raise ValueError(f"unknown color mode: {mode}")
    return np.ascontiguousarray(out)


def inverse_color(planes: np.ndarray, mode: str) -> np.ndarray:
    if mode == "rgb" or planes.shape[2] < 3:
        return np.ascontiguousarray(planes)
    out = np.array(planes, dtype=np.float32, copy=True)
    a = planes[:, :, 0].astype(np.float64)
    b = planes[:, :, 1].astype(np.float64)
    c = planes[:, :, 2].astype(np.float64)
    if mode == "g-diff":
        g = a
        r = b + g
        blue = c + g
        out[:, :, 0] = r.astype(np.float32)
        out[:, :, 1] = g.astype(np.float32)
        out[:, :, 2] = blue.astype(np.float32)
    elif mode == "ycocg":
        y = a
        co = b
        cg = c
        r = y - cg * 0.5 + co * 0.5
        g = y + cg * 0.5
        blue = y - cg * 0.5 - co * 0.5
        out[:, :, 0] = r.astype(np.float32)
        out[:, :, 1] = g.astype(np.float32)
        out[:, :, 2] = blue.astype(np.float32)
    else:
        raise ValueError(f"unknown color mode: {mode}")
    return np.ascontiguousarray(out)


def parse_csv(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in text.split(",") if part.strip())


def parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(",") if part.strip())


def quality_passes(
    quality: dict[str, float],
    log_rmse_threshold: float,
    log_p99_threshold: float,
    gradient_threshold: float,
) -> bool:
    return (
        quality["signed_log2_rmse"] <= log_rmse_threshold
        and quality["signed_log2_p99"] <= log_p99_threshold
        and quality["gradient_signed_log2_nrmse"] <= gradient_threshold
    )


def summarize_global(
    pixels: np.ndarray,
    color_modes: tuple[str, ...],
    transforms: tuple[str, ...],
    bits_values: tuple[int, ...],
) -> list[Candidate]:
    candidates: list[Candidate] = []
    for color in color_modes:
        planes = forward_color(pixels, color)
        for bits in bits_values:
            for transform in transforms:
                indices, recon_planes = quantize_transform(planes, bits, transform)
                recon_rgb = inverse_color(recon_planes, color)
                estimated_bits = estimate_index_bits(indices, bits)
                candidates.append(Candidate(
                    color=color,
                    transform=transform,
                    bits=bits,
                    indices=indices,
                    recon_rgb=recon_rgb,
                    estimated_bits=estimated_bits,
                    quality=quality_stats(pixels, recon_rgb),
                ))
    return candidates


def summarize_tile_selector(
    pixels: np.ndarray,
    candidates: list[Candidate],
    tile_size: int,
    log_rmse_threshold: float,
    log_p99_threshold: float,
    gradient_threshold: float,
) -> dict:
    selected_recon = np.array(pixels, copy=True)
    selected_counter: Counter[str] = Counter()
    estimated_bits = 0.0
    selector_bits = math.ceil(math.log2(max(1, len(candidates))))
    rows = []

    for y in range(0, pixels.shape[0], tile_size):
        for x in range(0, pixels.shape[1], tile_size):
            tile = pixels[y:y + tile_size, x:x + tile_size, :]
            tile_candidates = []
            for candidate in candidates:
                indices = candidate.indices[y:y + tile.shape[0], x:x + tile.shape[1], :]
                recon = candidate.recon_rgb[y:y + tile.shape[0], x:x + tile.shape[1], :]
                quality = quality_stats(tile, recon)
                bits = estimate_index_bits(indices, candidate.bits) + selector_bits
                tile_candidates.append((candidate, bits, quality, recon))
            passing = [
                item for item in tile_candidates
                if quality_passes(
                    item[2],
                    log_rmse_threshold,
                    log_p99_threshold,
                    gradient_threshold,
                )
            ]
            if passing:
                chosen = min(passing, key=lambda item: item[1])
            else:
                chosen = min(
                    tile_candidates,
                    key=lambda item: (
                        max(
                            item[2]["signed_log2_rmse"] / log_rmse_threshold,
                            item[2]["signed_log2_p99"] / log_p99_threshold,
                            item[2]["gradient_signed_log2_nrmse"]
                            / gradient_threshold,
                        ),
                        item[1],
                    ),
                )
            candidate, bits, quality, recon = chosen
            selected_recon[y:y + tile.shape[0], x:x + tile.shape[1], :] = recon
            estimated_bits += bits
            key = f"{candidate.color}:{candidate.transform}:b{candidate.bits}"
            selected_counter[key] += 1
            rows.append({
                "x": x,
                "y": y,
                "color": candidate.color,
                "transform": candidate.transform,
                "bits": candidate.bits,
                "estimated_bits": bits,
                **quality,
            })

    full_quality = quality_stats(pixels, selected_recon)
    return {
        "estimated_bits": estimated_bits,
        "estimated_bytes": math.ceil(estimated_bits / 8.0),
        "estimated_ratio_vs_original": pixels.nbytes * 8.0 / estimated_bits,
        "selected_modes": dict(sorted(selected_counter.items())),
        "full_quality": full_quality,
        "tiles": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="sample_DSCF0009.EXR")
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--bits", default="7,8")
    parser.add_argument("--color-modes", default="rgb,g-diff,ycocg")
    parser.add_argument("--transforms", default="linear,signed-log,gamma075,asinh")
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--log-rmse-threshold", type=float, default=0.004)
    parser.add_argument("--log-p99-threshold", type=float, default=0.018)
    parser.add_argument("--gradient-threshold", type=float, default=0.5)
    parser.add_argument("--no-tile-selector", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    bits_values = parse_ints(args.bits)
    color_modes = parse_csv(args.color_modes)
    transforms = parse_csv(args.transforms)
    results = []
    for path in sorted(DATA_DIR.glob(args.glob)):
        pixels = read_exr(path, args.crop_size)
        candidates = summarize_global(
            pixels,
            color_modes,
            transforms,
            bits_values,
        )
        global_rows = []
        print(path.name)
        print("  global candidates:")
        for candidate in sorted(
            candidates,
            key=lambda c: (
                not quality_passes(
                    c.quality,
                    args.log_rmse_threshold,
                    args.log_p99_threshold,
                    args.gradient_threshold,
                ),
                c.estimated_bits,
            ),
        ):
            q = candidate.quality
            row = {
                "color": candidate.color,
                "transform": candidate.transform,
                "bits": candidate.bits,
                "estimated_bytes": math.ceil(candidate.estimated_bits / 8.0),
                "estimated_ratio_vs_original": (
                    pixels.nbytes * 8.0 / candidate.estimated_bits),
                **q,
            }
            global_rows.append(row)
            marker = "PASS" if quality_passes(
                q,
                args.log_rmse_threshold,
                args.log_p99_threshold,
                args.gradient_threshold,
            ) else "fail"
            print(
                f"    {marker} {candidate.color:<6} "
                f"{candidate.transform:<10} bits{candidate.bits}: "
                f"{row['estimated_ratio_vs_original']:6.2f}x "
                f"bytes={row['estimated_bytes']} "
                f"log={q['signed_log2_rmse']:.3e} "
                f"p99={q['signed_log2_p99']:.3e} "
                f"grad={q['gradient_signed_log2_nrmse']:.3e}")

        selector = None
        if not args.no_tile_selector:
            selector = summarize_tile_selector(
                pixels,
                candidates,
                args.tile_size,
                args.log_rmse_threshold,
                args.log_p99_threshold,
                args.gradient_threshold,
            )
            q = selector["full_quality"]
            print(
                f"  tile-selector: {selector['estimated_ratio_vs_original']:.2f}x "
                f"bytes={selector['estimated_bytes']} "
                f"modes={selector['selected_modes']} "
                f"log={q['signed_log2_rmse']:.3e} "
                f"p99={q['signed_log2_p99']:.3e} "
                f"grad={q['gradient_signed_log2_nrmse']:.3e}")
        results.append({
            "image": path.name,
            "shape": list(pixels.shape),
            "crop_size": args.crop_size,
            "bits": list(bits_values),
            "color_modes": list(color_modes),
            "transforms": list(transforms),
            "tile_size": args.tile_size,
            "global": global_rows,
            "tile_selector": selector,
        })

    if not args.no_save:
        safe_glob = args.glob.replace("*", "star").replace(".", "_")
        output = RESULTS_DIR / (
            f"color_tile_transform_index_{safe_glob}"
            f"_crop{args.crop_size}_bits{args.bits.replace(',', '-')}"
            f"_tile{args.tile_size}.json"
        )
        output.write_text(json.dumps(results, indent=2))
        print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
