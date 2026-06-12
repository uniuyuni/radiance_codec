"""Benchmark near-lossless ratios by zeroing low raw mantissa bits.

This does not replace the exact-lossless track. It exercises the codec's
MantissaQuantize stage and measures the product-mode escape hatch suggested by
the exact-lossless audits: if true/dithered low mantissa tails are the wall, how
many low bits must be quantized before the GDX backend reaches 12x+?
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import OpenImageIO as oiio

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "codec" / "python"))

import radiance_codec  # noqa: E402

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"


def read_exr(path: Path, crop_size: int = 0) -> np.ndarray:
    image_input = oiio.ImageInput.open(str(path))
    if image_input is None:
        raise RuntimeError(f"can't open {path}: {oiio.geterror()}")
    spec = image_input.spec()
    if crop_size:
        height = min(crop_size, spec.height)
        width = min(crop_size, spec.width)
        pixels = image_input.read_scanlines(
            0,
            height,
            0,
            0,
            spec.nchannels,
            oiio.FLOAT,
        )
    else:
        height = spec.height
        width = spec.width
        pixels = image_input.read_image(format=oiio.FLOAT)
    image_input.close()
    if pixels is None:
        raise RuntimeError(f"can't read {path}: {oiio.geterror()}")
    return np.asarray(pixels, dtype=np.float32).reshape(
        height,
        spec.width,
        spec.nchannels,
    )[:, :width, :]


def quantize_mantissa(
    pixels: np.ndarray,
    low_bits: int,
    policy: radiance_codec.NearLosslessPolicy,
) -> np.ndarray:
    return radiance_codec.quantize_mantissa(pixels, low_bits, policy=policy)


def error_stats(original: np.ndarray, quantized: np.ndarray) -> dict[str, float]:
    diff = quantized.astype(np.float64) - original.astype(np.float64)
    finite = np.isfinite(original) & np.isfinite(quantized)
    if not bool(np.any(finite)):
        return {
            "max_abs": 0.0,
            "mae": 0.0,
            "rmse": 0.0,
            "snr_db": float("inf"),
            "max_rel_nonzero": 0.0,
            "signed_log2_rmse": 0.0,
            "gradient_signed_log2_nrmse": 0.0,
            "histogram_ks_256": 0.0,
            "psnr_peak_abs": float("inf"),
            "changed_rate": 0.0,
        }
    selected_diff = diff[finite]
    selected_original = original[finite].astype(np.float64)
    max_abs = float(np.max(np.abs(selected_diff)))
    mae = float(np.mean(np.abs(selected_diff)))
    rmse = float(np.sqrt(np.mean(selected_diff * selected_diff)))
    signal_rms = float(np.sqrt(np.mean(selected_original * selected_original)))
    snr = float("inf")
    if rmse > 0.0 and signal_rms > 0.0:
        snr = 20.0 * math.log10(signal_rms / rmse)
    nonzero = selected_original != 0.0
    max_rel = 0.0
    if bool(np.any(nonzero)):
        max_rel = float(
            np.max(np.abs(selected_diff[nonzero] / selected_original[nonzero]))
        )
    selected_quantized = quantized[finite].astype(np.float64)
    signed_log_original = (
        np.sign(selected_original) * np.log2(1.0 + np.abs(selected_original))
    )
    signed_log_quantized = (
        np.sign(selected_quantized) * np.log2(1.0 + np.abs(selected_quantized))
    )
    signed_log_diff = signed_log_quantized - signed_log_original
    signed_log_rmse = float(np.sqrt(np.mean(signed_log_diff * signed_log_diff)))
    peak = float(np.max(np.abs(selected_original)))
    psnr = float("inf")
    if rmse > 0.0 and peak > 0.0:
        psnr = 20.0 * math.log10(peak / rmse)
    changed_rate = float(np.mean(original.view(np.uint32) != quantized.view(np.uint32)))

    histogram_ks = 0.0
    if peak > 0.0:
        hist_range = (
            float(np.min(selected_original)),
            float(np.max(selected_original)),
        )
        if hist_range[1] > hist_range[0]:
            h0, edges = np.histogram(
                selected_original,
                bins=256,
                range=hist_range,
                density=False,
            )
            h1, _ = np.histogram(
                selected_quantized,
                bins=edges,
                density=False,
            )
            c0 = np.cumsum(h0, dtype=np.float64)
            c1 = np.cumsum(h1, dtype=np.float64)
            if c0[-1] > 0 and c1[-1] > 0:
                histogram_ks = float(np.max(np.abs(c0 / c0[-1] - c1 / c1[-1])))

    gradient_nrmse = 0.0
    signed_log_full_original = (
        np.sign(original.astype(np.float64))
        * np.log2(1.0 + np.abs(original.astype(np.float64)))
    )
    signed_log_full_quantized = (
        np.sign(quantized.astype(np.float64))
        * np.log2(1.0 + np.abs(quantized.astype(np.float64)))
    )
    grad_original = np.concatenate(
        [
            np.diff(signed_log_full_original, axis=0).ravel(),
            np.diff(signed_log_full_original, axis=1).ravel(),
        ]
    )
    grad_quantized = np.concatenate(
        [
            np.diff(signed_log_full_quantized, axis=0).ravel(),
            np.diff(signed_log_full_quantized, axis=1).ravel(),
        ]
    )
    grad_finite = np.isfinite(grad_original) & np.isfinite(grad_quantized)
    if bool(np.any(grad_finite)):
        grad_diff = grad_quantized[grad_finite] - grad_original[grad_finite]
        grad_rmse = float(np.sqrt(np.mean(grad_diff * grad_diff)))
        grad_rms = float(np.sqrt(np.mean(grad_original[grad_finite] ** 2)))
        if grad_rms > 0.0:
            gradient_nrmse = grad_rmse / grad_rms
    return {
        "max_abs": max_abs,
        "mae": mae,
        "rmse": rmse,
        "snr_db": snr,
        "max_rel_nonzero": max_rel,
        "signed_log2_rmse": signed_log_rmse,
        "gradient_signed_log2_nrmse": gradient_nrmse,
        "histogram_ks_256": histogram_ks,
        "psnr_peak_abs": psnr,
        "changed_rate": changed_rate,
    }


def summarize_image(
    path: Path,
    crop_size: int,
    effort: int,
    low_bits_values: tuple[int, ...],
    policies: tuple[radiance_codec.NearLosslessPolicy, ...],
) -> dict:
    pixels = read_exr(path, crop_size)
    pixels = np.ascontiguousarray(pixels, dtype=np.float32)
    rows = []
    for policy in policies:
        for low_bits in low_bits_values:
            quantized = quantize_mantissa(pixels, low_bits, policy)
            t0 = time.perf_counter()
            encoded = radiance_codec.encode_near_lossless(
                pixels,
                low_bits=low_bits,
                effort=effort,
                policy=policy,
            )
            t1 = time.perf_counter()
            decoded = radiance_codec.decode(encoded)
            t2 = time.perf_counter()
            if decoded.tobytes() != quantized.tobytes():
                raise RuntimeError(
                    f"{path.name}: decoded near-lossless image does not match "
                    f"the expected low{low_bits} quantized image")
            ratio = pixels.nbytes / len(encoded)
            stats = error_stats(pixels, quantized)
            rows.append(
                {
                    "policy": policy.name.lower(),
                    "low_bits_zeroed": low_bits,
                    "encoded_bytes": len(encoded),
                    "ratio_vs_original": ratio,
                    "encode_seconds": t1 - t0,
                    "decode_seconds": t2 - t1,
                    **stats,
                }
            )
    return {
        "image": path.name,
        "shape": list(pixels.shape),
        "crop_size": crop_size,
        "effort": effort,
        "rows": rows,
    }


def parse_low_bits(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(",") if part)


def parse_policies(text: str) -> tuple[radiance_codec.NearLosslessPolicy, ...]:
    aliases = {
        "fixed": radiance_codec.NearLosslessPolicy.FIXED,
        "tile": radiance_codec.NearLosslessPolicy.TILE,
        "exponent": radiance_codec.NearLosslessPolicy.EXPONENT,
        "tile_exponent": radiance_codec.NearLosslessPolicy.TILE_EXPONENT,
        "tile-exponent": radiance_codec.NearLosslessPolicy.TILE_EXPONENT,
        "hybrid": radiance_codec.NearLosslessPolicy.TILE_EXPONENT,
        "linear": radiance_codec.NearLosslessPolicy.LINEAR_RANGE,
        "linear_range": radiance_codec.NearLosslessPolicy.LINEAR_RANGE,
        "linear-range": radiance_codec.NearLosslessPolicy.LINEAR_RANGE,
        "log": radiance_codec.NearLosslessPolicy.LOG_RANGE,
        "log_range": radiance_codec.NearLosslessPolicy.LOG_RANGE,
        "log-range": radiance_codec.NearLosslessPolicy.LOG_RANGE,
    }
    out = []
    for part in text.split(","):
        key = part.strip().lower()
        if not key:
            continue
        if key not in aliases:
            raise ValueError(f"unknown policy {part!r}; choices: {sorted(aliases)}")
        out.append(aliases[key])
    return tuple(out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="*.exr")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--crop-size", type=int, default=0)
    parser.add_argument("--effort", type=int, default=9)
    parser.add_argument("--low-bits", default="0,4,8,12,15")
    parser.add_argument("--policies", default="fixed")
    return parser.parse_args()


def geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def main() -> int:
    args = parse_args()
    low_bits_values = parse_low_bits(args.low_bits)
    policies = parse_policies(args.policies)
    rows = []
    for path in sorted(DATA_DIR.glob(args.glob)):
        row = summarize_image(
            path,
            args.crop_size,
            args.effort,
            low_bits_values,
            policies,
        )
        rows.append(row)
        print(path.stem)
        for item in row["rows"]:
            print(
                f"  {item['policy']:13s} low{item['low_bits_zeroed']:02d}: "
                f"{item['ratio_vs_original']:7.2f}x "
                f"changed={item['changed_rate']:.3f} "
                f"max_rel={item['max_rel_nonzero']:.3e} "
                f"log_rmse={item['signed_log2_rmse']:.3e} "
                f"grad={item['gradient_signed_log2_nrmse']:.3e} "
                f"ks={item['histogram_ks_256']:.3e} "
                f"psnr={item['psnr_peak_abs']:.2f}dB"
            )
        if args.limit and len(rows) >= args.limit:
            break
    if rows:
        print("\ngeomean by policy/low bits:")
        for policy in policies:
            for low_bits in low_bits_values:
                values = [
                    item["ratio_vs_original"]
                    for row in rows
                    for item in row["rows"]
                    if item["policy"] == policy.name.lower()
                    and item["low_bits_zeroed"] == low_bits
                ]
                print(
                    f"  {policy.name.lower():13s} low{low_bits:02d}: "
                    f"{geomean(values):.3f}x"
                )
    safe_glob = args.glob.replace("*", "star").replace(".", "_")
    output = RESULTS_DIR / (
        f"mantissa_quantization_{safe_glob}_effort{args.effort}"
        f"_crop{args.crop_size}_low{args.low_bits.replace(',', '-')}"
        f"_policy{args.policies.replace(',', '-')}.json"
    )
    output.write_text(json.dumps(rows, indent=2))
    print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
