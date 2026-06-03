"""Probe capped-range transform indices with a sparse bright-tail escape.

For the current X-Trans sample, almost all values are in a compact positive
range, while a small bright tail expands the global range.  This probe caps the
main transform-index range at a chosen value and stores only values above that
cap through a small tail side stream estimate.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import OpenImageIO as oiio

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
sys.path.insert(0, str(ROOT / "scripts"))

from audit_linear_index_payload import residual_stats  # noqa: E402
from benchmark_linear_index_codec import error_stats  # noqa: E402


def read_exr(path: Path, crop_size: int) -> np.ndarray:
    image_input = oiio.ImageInput.open(str(path))
    if image_input is None:
        raise RuntimeError(f"can't open {path}: {oiio.geterror()}")
    spec = image_input.spec()
    height = spec.height if crop_size == 0 else min(crop_size, spec.height)
    width = spec.width if crop_size == 0 else min(crop_size, spec.width)
    if crop_size:
        pixels = image_input.read_scanlines(
            0,
            height,
            0,
            0,
            spec.nchannels,
            oiio.FLOAT,
        )
    else:
        pixels = image_input.read_image(format=oiio.FLOAT)
    image_input.close()
    if pixels is None:
        raise RuntimeError(f"can't read {path}: {oiio.geterror()}")
    return np.ascontiguousarray(
        np.asarray(pixels, dtype=np.float32).reshape(
            height,
            spec.width,
            spec.nchannels,
        )[:, :width, :]
    )


def transform_power(values: np.ndarray, gamma: float) -> np.ndarray:
    return np.sign(values) * np.power(np.abs(values), gamma)


def inverse_power(values: np.ndarray, gamma: float) -> np.ndarray:
    return np.sign(values) * np.power(np.abs(values), 1.0 / gamma)


def entropy_bits(values: np.ndarray, alphabet: int | None = None) -> float:
    flat = values.reshape(-1).astype(np.int64)
    if flat.size == 0:
        return 0.0
    counts = np.bincount(flat, minlength=alphabet or 0)
    counts = counts[counts > 0].astype(np.float64)
    if counts.size == 0:
        return 0.0
    probs = counts / counts.sum()
    return float(-(probs * np.log2(probs)).sum() * flat.size)


def binary_entropy_bits(ones: int, total: int) -> float:
    if total <= 0 or ones <= 0 or ones >= total:
        return 0.0
    p = ones / total
    return float(-(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p)) * total)


def quality_passes(quality: dict[str, float]) -> bool:
    return (
        quality["signed_log2_rmse"] <= 0.004
        and quality["signed_log2_p99"] <= 0.018
        and quality["gradient_signed_log2_nrmse"] <= 0.5
    )


def quantize_tail_escape(
    pixels: np.ndarray,
    main_bits: int,
    tail_bits: int,
    gamma: float,
    cap: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    main_levels = (1 << main_bits) - 1
    tail_levels = (1 << tail_bits) - 1
    indices = np.zeros(pixels.shape, dtype=np.uint16)
    recon = np.array(pixels, copy=True)
    tail_total = 0
    saturated_total = 0
    tail_value_bits = 0.0
    tail_mask_bits = 0.0
    tail_by_channel = []

    for channel in range(pixels.shape[2]):
        values = pixels[:, :, channel].astype(np.float64)
        lo = float(np.min(values))
        hi = min(float(np.max(values)), cap)
        lo_t = float(transform_power(np.asarray(lo), gamma))
        hi_t = float(transform_power(np.asarray(hi), gamma))
        transformed = transform_power(np.minimum(values, cap), gamma)
        if hi_t > lo_t:
            q = np.floor((transformed - lo_t) / (hi_t - lo_t) * main_levels + 0.5)
            q = np.clip(q, 0, main_levels)
            indices[:, :, channel] = q.astype(np.uint16)
            rec_t = lo_t + q * (hi_t - lo_t) / main_levels
            recon[:, :, channel] = inverse_power(rec_t, gamma).astype(np.float32)
        else:
            q = np.zeros(values.shape, dtype=np.float64)
            recon[:, :, channel] = np.float32(lo)

        tail = values > cap
        saturated = q >= main_levels
        tail_count = int(tail.sum())
        saturated_count = int(saturated.sum())
        tail_total += tail_count
        saturated_total += saturated_count
        tail_by_channel.append(tail_count)
        tail_mask_bits += binary_entropy_bits(tail_count, saturated_count)

        if tail_count:
            tail_values = values[tail]
            tail_lo_t = hi_t
            tail_hi_t = float(transform_power(np.asarray(float(tail_values.max())), gamma))
            if tail_hi_t > tail_lo_t:
                tail_t = transform_power(tail_values, gamma)
                tail_q = np.floor(
                    (tail_t - tail_lo_t)
                    / (tail_hi_t - tail_lo_t)
                    * tail_levels
                    + 0.5
                )
                tail_q = np.clip(tail_q, 0, tail_levels).astype(np.uint16)
                tail_rec_t = tail_lo_t + tail_q * (tail_hi_t - tail_lo_t) / tail_levels
                recon[:, :, channel][tail] = inverse_power(tail_rec_t, gamma).astype(
                    np.float32
                )
                tail_value_bits += entropy_bits(tail_q, tail_levels + 1)
            else:
                recon[:, :, channel][tail] = np.float32(cap)

    side = {
        "tail_count": tail_total,
        "tail_rate": tail_total / pixels.size,
        "tail_by_channel": tail_by_channel,
        "saturated_count": saturated_total,
        "saturated_rate": saturated_total / pixels.size,
        "tail_mask_bits": tail_mask_bits,
        "tail_value_bits": tail_value_bits,
        "tail_side_bits": tail_mask_bits + tail_value_bits + 256.0,
    }
    return indices, np.ascontiguousarray(recon), side


def summarize_candidate(
    pixels: np.ndarray,
    main_bits: int,
    tail_bits: int,
    gamma: float,
    cap: float,
) -> dict:
    indices, recon, side = quantize_tail_escape(
        pixels,
        main_bits,
        tail_bits,
        gamma,
        cap,
    )
    residual = residual_stats(indices, main_bits, "med")
    main_bits_total = residual["residual_order0_entropy_bits_per_sample"] * pixels.size
    estimated_bits = main_bits_total + side["tail_side_bits"] + 73.0 * 8.0
    estimated_bytes = estimated_bits / 8.0
    quality = error_stats(pixels, recon)
    return {
        "main_bits": main_bits,
        "tail_bits": tail_bits,
        "gamma": gamma,
        "cap": cap,
        "estimated_bytes": estimated_bytes,
        "estimated_ratio_vs_original": pixels.nbytes / estimated_bytes,
        "estimated_bits_per_sample": estimated_bits / pixels.size,
        "main_bits_per_sample": residual["residual_order0_entropy_bits_per_sample"],
        "nonzero_rate": residual["nonzero_rate"],
        "tail": side,
        "quality_passes": quality_passes(quality),
        "quality": quality,
    }


def summarize_image(
    path: Path,
    crop_size: int,
    main_bits_values: tuple[int, ...],
    tail_bits_values: tuple[int, ...],
    gammas: tuple[float, ...],
    caps: tuple[float, ...],
) -> dict:
    pixels = read_exr(path, crop_size)
    rows = [
        summarize_candidate(pixels, main_bits, tail_bits, gamma, cap)
        for main_bits in main_bits_values
        for tail_bits in tail_bits_values
        for gamma in gammas
        for cap in caps
    ]
    return {
        "image": path.name,
        "shape": list(pixels.shape),
        "crop_size": crop_size,
        "rows": rows,
    }


def parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(",") if part.strip())


def parse_floats(text: str) -> tuple[float, ...]:
    return tuple(float(part) for part in text.split(",") if part.strip())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="sample_DSCF0009.EXR")
    parser.add_argument("--crop-size", type=int, default=1024)
    parser.add_argument("--main-bits", default="7")
    parser.add_argument("--tail-bits", default="6,7,8")
    parser.add_argument("--gammas", default="0.75,0.85,0.9")
    parser.add_argument("--caps", default="3.5,4.0,4.5")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    summaries = []
    for path in sorted(DATA_DIR.glob(args.glob)):
        summary = summarize_image(
            path,
            args.crop_size,
            parse_ints(args.main_bits),
            parse_ints(args.tail_bits),
            parse_floats(args.gammas),
            parse_floats(args.caps),
        )
        summaries.append(summary)
        print(path.name)
        rows = sorted(
            summary["rows"],
            key=lambda row: (not row["quality_passes"], row["estimated_bytes"]),
        )
        for row in rows[:16]:
            q = row["quality"]
            print(
                f"  {'PASS' if row['quality_passes'] else 'fail'} "
                f"main{row['main_bits']} tail{row['tail_bits']} "
                f"g={row['gamma']:.2f} cap={row['cap']:.2f} "
                f"{row['estimated_ratio_vs_original']:7.2f}x "
                f"bytes={row['estimated_bytes']:,.0f} "
                f"bps={row['estimated_bits_per_sample']:.3f} "
                f"tail={row['tail']['tail_rate']:.3%} "
                f"log={q['signed_log2_rmse']:.3e} "
                f"p99={q['signed_log2_p99']:.3e} "
                f"grad={q['gradient_signed_log2_nrmse']:.3e}"
            )
        if args.limit and len(summaries) >= args.limit:
            break

    if not args.no_save:
        safe_glob = args.glob.replace("*", "star").replace(".", "_")
        output = RESULTS_DIR / (
            f"tail_escape_range_index_{safe_glob}"
            f"_crop{args.crop_size}_main{args.main_bits.replace(',', '-')}.json"
        )
        output.write_text(json.dumps(summaries, indent=2))
        print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
