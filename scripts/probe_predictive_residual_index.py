"""Probe causal predictive-residual transform-index quantization.

StageLinearIndex currently quantizes the transformed value itself.  This probe
tests a DPCM-like variant:

    transformed_value ~= causal_predictor(reconstructed_neighbors) + q(residual)

The encoded image is still an index image, but the quantized quantity is the
prediction residual in transform space.  This is only a decision probe; it does
not write a production bitstream.
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
sys.path.insert(0, str(ROOT / "codec" / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_linear_index_payload import residual_stats  # noqa: E402
from benchmark_linear_index_codec import error_stats  # noqa: E402
import radiance_codec  # noqa: E402


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


def transform_values(values: np.ndarray, mode: str) -> np.ndarray:
    if mode == "linear":
        return values
    if mode == "signed-log":
        return np.sign(values) * np.log2(1.0 + np.abs(values))
    if mode == "sqrt":
        return np.sign(values) * np.sqrt(np.abs(values))
    if mode == "gamma075":
        return np.sign(values) * np.power(np.abs(values), 0.75)
    if mode == "asinh":
        return np.arcsinh(values)
    raise ValueError(f"unknown transform: {mode}")


def inverse_transform_values(values: np.ndarray, mode: str) -> np.ndarray:
    if mode == "linear":
        return values
    if mode == "signed-log":
        return np.sign(values) * (np.exp2(np.abs(values)) - 1.0)
    if mode == "sqrt":
        return np.sign(values) * np.power(np.abs(values), 2.0)
    if mode == "gamma075":
        return np.sign(values) * np.power(np.abs(values), 1.0 / 0.75)
    if mode == "asinh":
        return np.sinh(values)
    raise ValueError(f"unknown transform: {mode}")


def spatial_predict(
    recon: np.ndarray,
    y: int,
    x: int,
    predictor: str,
) -> float:
    west = recon[y, x - 1] if x > 0 else 0.0
    north = recon[y - 1, x] if y > 0 else 0.0
    if predictor == "west":
        return float(west)
    if predictor == "north":
        return float(north)
    if predictor == "avg":
        return float((west + north) * 0.5)
    northwest = recon[y - 1, x - 1] if x > 0 and y > 0 else 0.0
    if predictor == "med":
        if northwest >= max(west, north):
            return float(min(west, north))
        if northwest <= min(west, north):
            return float(max(west, north))
        return float(west + north - northwest)
    raise ValueError(f"unknown predictor: {predictor}")


def original_residual_range(channel: np.ndarray, predictor: str) -> tuple[float, float]:
    west = np.zeros_like(channel)
    north = np.zeros_like(channel)
    west[:, 1:] = channel[:, :-1]
    north[1:, :] = channel[:-1, :]
    if predictor == "west":
        pred = west
    elif predictor == "north":
        pred = north
    elif predictor == "avg":
        pred = (west + north) * 0.5
    elif predictor == "med":
        northwest = np.zeros_like(channel)
        northwest[1:, 1:] = channel[:-1, :-1]
        hi = np.maximum(west, north)
        lo = np.minimum(west, north)
        gradient = west + north - northwest
        pred = np.where(northwest >= hi, lo, np.where(northwest <= lo, hi, gradient))
    else:
        raise ValueError(f"unknown predictor: {predictor}")
    residual = channel - pred
    return float(np.min(residual)), float(np.max(residual))


def expand_range(lo: float, hi: float, scale: float) -> tuple[float, float]:
    center = (lo + hi) * 0.5
    half = (hi - lo) * 0.5 * scale
    return center - half, center + half


def quantize_predictive_residual(
    pixels: np.ndarray,
    bits: int,
    transform: str,
    predictor: str,
    range_scale: float,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    levels = (1 << bits) - 1
    transformed = transform_values(pixels.astype(np.float64), transform)
    indices = np.zeros(pixels.shape, dtype=np.uint16)
    recon_t = np.zeros(pixels.shape, dtype=np.float64)
    ranges: list[dict[str, float]] = []
    clipped = 0
    samples = pixels.size

    for channel in range(pixels.shape[2]):
        source = transformed[:, :, channel]
        lo, hi = original_residual_range(source, predictor)
        lo, hi = expand_range(lo, hi, range_scale)
        if not hi > lo:
            ranges.append({"lo": lo, "hi": hi, "clipped_rate": 0.0})
            continue
        step = (hi - lo) / levels
        target = recon_t[:, :, channel]
        q_target = indices[:, :, channel]
        channel_clipped = 0
        for y in range(pixels.shape[0]):
            for x in range(pixels.shape[1]):
                pred = spatial_predict(target, y, x, predictor)
                residual = source[y, x] - pred
                q = math.floor((residual - lo) / step + 0.5)
                if q < 0:
                    q = 0
                    channel_clipped += 1
                elif q > levels:
                    q = levels
                    channel_clipped += 1
                q_target[y, x] = q
                target[y, x] = pred + lo + q * step
        clipped += channel_clipped
        ranges.append({
            "lo": lo,
            "hi": hi,
            "step": step,
            "clipped_rate": channel_clipped / source.size,
        })

    recon = inverse_transform_values(recon_t, transform).astype(np.float32)
    return indices, np.ascontiguousarray(recon), ranges


def baseline_row(pixels: np.ndarray, bits: int, transform: str) -> dict:
    encoded = radiance_codec.encode_linear_index_near_lossless(
        pixels,
        bits=bits,
        effort=9,
        transform=transform,
    )
    decoded = radiance_codec.decode(encoded, pixels.shape)
    return {
        "kind": "baseline-stage-linear-index",
        "bits": bits,
        "transform": transform,
        "encoded_bytes": len(encoded),
        "ratio_vs_original": pixels.nbytes / len(encoded),
        "estimated_bits_per_sample": len(encoded) * 8 / pixels.size,
        "quality": error_stats(pixels, decoded),
    }


def candidate_row(
    pixels: np.ndarray,
    bits: int,
    transform: str,
    predictor: str,
    range_scale: float,
) -> dict:
    indices, recon, ranges = quantize_predictive_residual(
        pixels,
        bits,
        transform,
        predictor,
        range_scale,
    )
    stats = residual_stats(indices, bits, "med")
    estimated_bits_per_sample = stats["residual_order0_entropy_bits_per_sample"]
    estimated_bytes = estimated_bits_per_sample * pixels.size / 8.0 + 73.0
    return {
        "kind": "predictive-residual-index",
        "bits": bits,
        "transform": transform,
        "predictor": predictor,
        "range_scale": range_scale,
        "ranges": ranges,
        "estimated_bytes": estimated_bytes,
        "estimated_ratio_vs_original": pixels.nbytes / estimated_bytes,
        "estimated_bits_per_sample": estimated_bits_per_sample,
        "nonzero_rate": stats["nonzero_rate"],
        "value_entropy_bits_per_nonzero":
            stats["value_order0_entropy_bits_per_nonzero"],
        "quality": error_stats(pixels, recon),
    }


def summarize_image(
    path: Path,
    crop_size: int,
    bits_values: tuple[int, ...],
    transforms: tuple[str, ...],
    predictors: tuple[str, ...],
    range_scales: tuple[float, ...],
) -> dict:
    pixels = read_exr(path, crop_size)
    rows = []
    for transform in transforms:
        for bits in bits_values:
            rows.append(baseline_row(pixels, bits, transform))
            for predictor in predictors:
                for range_scale in range_scales:
                    rows.append(
                        candidate_row(
                            pixels,
                            bits,
                            transform,
                            predictor,
                            range_scale,
                        )
                    )
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


def parse_strings(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in text.split(",") if part.strip())


def quality_passes(row: dict) -> bool:
    quality = row["quality"]
    return (
        quality["signed_log2_rmse"] <= 0.004
        and quality["signed_log2_p99"] <= 0.018
        and quality["gradient_signed_log2_nrmse"] <= 0.5
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="sample_DSCF0009.EXR")
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--bits", default="7,8")
    parser.add_argument("--transforms", default="gamma075")
    parser.add_argument("--predictors", default="avg,med")
    parser.add_argument("--range-scales", default="1.0,1.25,1.5")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    bits_values = parse_ints(args.bits)
    transforms = parse_strings(args.transforms)
    predictors = parse_strings(args.predictors)
    range_scales = parse_floats(args.range_scales)

    summaries = []
    for path in sorted(DATA_DIR.glob(args.glob)):
        summary = summarize_image(
            path,
            args.crop_size,
            bits_values,
            transforms,
            predictors,
            range_scales,
        )
        summaries.append(summary)
        print(path.name)
        sorted_rows = sorted(
            summary["rows"],
            key=lambda item: (
                not quality_passes(item),
                item.get("estimated_bytes", item.get("encoded_bytes", float("inf"))),
            ),
        )
        for row in sorted_rows[:12]:
            q = row["quality"]
            if row["kind"] == "baseline-stage-linear-index":
                size = row["encoded_bytes"]
                ratio = row["ratio_vs_original"]
                label = f"baseline {row['transform']} b{row['bits']}"
            else:
                size = row["estimated_bytes"]
                ratio = row["estimated_ratio_vs_original"]
                label = (
                    f"predres {row['transform']} b{row['bits']} "
                    f"{row['predictor']} scale{row['range_scale']:g}"
                )
            print(
                f"  {'PASS' if quality_passes(row) else 'fail'} "
                f"{label:36s} "
                f"{ratio:7.2f}x bytes={size:,.0f} "
                f"bps={row['estimated_bits_per_sample']:.3f} "
                f"log={q['signed_log2_rmse']:.3e} "
                f"p99={q['signed_log2_p99']:.3e} "
                f"grad={q['gradient_signed_log2_nrmse']:.3e}"
            )
        if args.limit and len(summaries) >= args.limit:
            break

    if not args.no_save:
        safe_glob = args.glob.replace("*", "star").replace(".", "_")
        output = RESULTS_DIR / (
            f"predictive_residual_index_{safe_glob}"
            f"_crop{args.crop_size}_bits{args.bits.replace(',', '-')}.json"
        )
        output.write_text(json.dumps(summaries, indent=2))
        print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
