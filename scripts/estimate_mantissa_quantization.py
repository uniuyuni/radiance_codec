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


def read_exr(path: Path) -> np.ndarray:
    image_input = oiio.ImageInput.open(str(path))
    if image_input is None:
        raise RuntimeError(f"can't open {path}: {oiio.geterror()}")
    spec = image_input.spec()
    pixels = image_input.read_image(format=oiio.FLOAT)
    image_input.close()
    if pixels is None:
        raise RuntimeError(f"can't read {path}: {oiio.geterror()}")
    return np.asarray(pixels, dtype=np.float32).reshape(
        spec.height,
        spec.width,
        spec.nchannels,
    )


def quantize_mantissa(pixels: np.ndarray, low_bits: int) -> np.ndarray:
    return radiance_codec.quantize_mantissa(pixels, low_bits)


def error_stats(original: np.ndarray, quantized: np.ndarray) -> dict[str, float]:
    diff = quantized.astype(np.float64) - original.astype(np.float64)
    finite = np.isfinite(original) & np.isfinite(quantized)
    if not bool(np.any(finite)):
        return {
            "max_abs": 0.0,
            "rmse": 0.0,
            "max_rel_nonzero": 0.0,
            "psnr_peak_abs": float("inf"),
            "changed_rate": 0.0,
        }
    selected_diff = diff[finite]
    selected_original = original[finite].astype(np.float64)
    max_abs = float(np.max(np.abs(selected_diff)))
    rmse = float(np.sqrt(np.mean(selected_diff * selected_diff)))
    nonzero = selected_original != 0.0
    max_rel = 0.0
    if bool(np.any(nonzero)):
        max_rel = float(
            np.max(np.abs(selected_diff[nonzero] / selected_original[nonzero]))
        )
    peak = float(np.max(np.abs(selected_original)))
    psnr = float("inf")
    if rmse > 0.0 and peak > 0.0:
        psnr = 20.0 * math.log10(peak / rmse)
    changed_rate = float(np.mean(original.view(np.uint32) != quantized.view(np.uint32)))
    return {
        "max_abs": max_abs,
        "rmse": rmse,
        "max_rel_nonzero": max_rel,
        "psnr_peak_abs": psnr,
        "changed_rate": changed_rate,
    }


def summarize_image(
    path: Path,
    crop_size: int,
    effort: int,
    low_bits_values: tuple[int, ...],
) -> dict:
    pixels = read_exr(path)
    if crop_size:
        pixels = pixels[:crop_size, :crop_size]
    pixels = np.ascontiguousarray(pixels, dtype=np.float32)
    rows = []
    for low_bits in low_bits_values:
        quantized = quantize_mantissa(pixels, low_bits)
        t0 = time.perf_counter()
        encoded = radiance_codec.encode_near_lossless(
            pixels,
            low_bits=low_bits,
            effort=effort,
        )
        t1 = time.perf_counter()
        decoded = radiance_codec.decode(encoded, pixels.shape)
        t2 = time.perf_counter()
        if decoded.tobytes() != quantized.tobytes():
            raise RuntimeError(
                f"{path.name}: decoded near-lossless image does not match "
                f"the expected low{low_bits} quantized image")
        ratio = pixels.nbytes / len(encoded)
        stats = error_stats(pixels, quantized)
        rows.append(
            {
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="*.exr")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--crop-size", type=int, default=0)
    parser.add_argument("--effort", type=int, default=9)
    parser.add_argument("--low-bits", default="0,4,8,12,15")
    return parser.parse_args()


def geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def main() -> int:
    args = parse_args()
    low_bits_values = parse_low_bits(args.low_bits)
    rows = []
    for path in sorted(DATA_DIR.glob(args.glob)):
        row = summarize_image(path, args.crop_size, args.effort, low_bits_values)
        rows.append(row)
        print(path.stem)
        for item in row["rows"]:
            print(
                f"  low{item['low_bits_zeroed']:02d}: "
                f"{item['ratio_vs_original']:7.2f}x "
                f"changed={item['changed_rate']:.3f} "
                f"max_rel={item['max_rel_nonzero']:.3e} "
                f"psnr={item['psnr_peak_abs']:.2f}dB"
            )
        if args.limit and len(rows) >= args.limit:
            break
    if rows:
        print("\ngeomean by low bits:")
        for low_bits in low_bits_values:
            values = [
                item["ratio_vs_original"]
                for row in rows
                for item in row["rows"]
                if item["low_bits_zeroed"] == low_bits
            ]
            print(f"  low{low_bits:02d}: {geomean(values):.3f}x")
    safe_glob = args.glob.replace("*", "star").replace(".", "_")
    output = RESULTS_DIR / (
        f"mantissa_quantization_{safe_glob}_effort{args.effort}"
        f"_crop{args.crop_size}_low{args.low_bits.replace(',', '-')}.json"
    )
    output.write_text(json.dumps(rows, indent=2))
    print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
