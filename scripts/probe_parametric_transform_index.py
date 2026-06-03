"""Search parametric transform-index quantizers.

The implemented StageLinearIndex has a small fixed transform set.  This probe
keeps the same global per-channel index idea but searches continuous transform
families, mainly signed power curves:

    f(x) = sign(x) * abs(x) ** gamma

The goal is to find a transform whose bits8 quality remains strict while the
index residual image is cheaper than the current gamma075 baseline.
"""
from __future__ import annotations

import argparse
import json
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


def quantize_power(
    pixels: np.ndarray,
    bits: int,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray]:
    levels = (1 << bits) - 1
    indices = np.zeros(pixels.shape, dtype=np.uint16)
    recon = np.array(pixels, copy=True)
    for channel in range(pixels.shape[2]):
        values = pixels[:, :, channel].astype(np.float64)
        transformed = transform_power(values, gamma)
        range_values = transformed.astype(np.float32).astype(np.float64)
        lo = float(np.min(range_values))
        hi = float(np.max(range_values))
        if not hi > lo:
            recon[:, :, channel] = inverse_power(
                np.asarray(lo, dtype=np.float64),
                gamma,
            ).astype(np.float32)
            continue
        q = np.floor((transformed - lo) / (hi - lo) * levels + 0.5)
        q = np.clip(q, 0, levels)
        indices[:, :, channel] = q.astype(np.uint16)
        rec_t = lo + q * (hi - lo) / levels
        recon[:, :, channel] = inverse_power(rec_t, gamma).astype(np.float32)
    return indices, np.ascontiguousarray(recon)


def quality_passes(quality: dict[str, float]) -> bool:
    return (
        quality["signed_log2_rmse"] <= 0.004
        and quality["signed_log2_p99"] <= 0.018
        and quality["gradient_signed_log2_nrmse"] <= 0.5
    )


def tile_quality_summary(
    original: np.ndarray,
    recon: np.ndarray,
    tile_size: int,
) -> dict | None:
    if tile_size <= 0:
        return None
    rows = []
    pass_count = 0
    total_count = 0
    for y in range(0, original.shape[0], tile_size):
        for x in range(0, original.shape[1], tile_size):
            tile_original = original[y:y + tile_size, x:x + tile_size, :]
            tile_recon = recon[y:y + tile_size, x:x + tile_size, :]
            quality = error_stats(tile_original, tile_recon)
            passed = quality_passes(quality)
            pass_count += int(passed)
            total_count += 1
            rows.append({
                "x": x,
                "y": y,
                "w": int(tile_original.shape[1]),
                "h": int(tile_original.shape[0]),
                "passes": passed,
                "quality": quality,
            })
    rows.sort(key=lambda row: row["quality"]["signed_log2_rmse"], reverse=True)
    return {
        "tile_size": tile_size,
        "passing_tiles": pass_count,
        "total_tiles": total_count,
        "worst_tiles": rows[:12],
    }


def summarize_candidate(
    pixels: np.ndarray,
    bits: int,
    gamma: float,
    coder_predictor: str,
    tile_size: int,
) -> dict:
    indices, recon = quantize_power(pixels, bits, gamma)
    residual = residual_stats(indices, bits, coder_predictor)
    estimated_bits_per_sample = residual["residual_order0_entropy_bits_per_sample"]
    estimated_bytes = estimated_bits_per_sample * pixels.size / 8.0 + 73.0
    quality = error_stats(pixels, recon)
    return {
        "family": "signed-power",
        "gamma": gamma,
        "bits": bits,
        "coder_predictor": coder_predictor,
        "estimated_bits_per_sample": estimated_bits_per_sample,
        "estimated_bytes": estimated_bytes,
        "estimated_ratio_vs_original": pixels.nbytes / estimated_bytes,
        "nonzero_rate": residual["nonzero_rate"],
        "value_entropy_bits_per_nonzero":
            residual["value_order0_entropy_bits_per_nonzero"],
        "quality_passes": quality_passes(quality),
        "quality": quality,
        "tile_quality": tile_quality_summary(pixels, recon, tile_size),
    }


def summarize_image(
    path: Path,
    crop_size: int,
    bits_values: tuple[int, ...],
    gammas: tuple[float, ...],
    coder_predictor: str,
    tile_size: int,
) -> dict:
    pixels = read_exr(path, crop_size)
    rows = [
        summarize_candidate(pixels, bits, gamma, coder_predictor, tile_size)
        for bits in bits_values
        for gamma in gammas
    ]
    return {
        "image": path.name,
        "shape": list(pixels.shape),
        "crop_size": crop_size,
        "coder_predictor": coder_predictor,
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
    parser.add_argument("--bits", default="7,8")
    parser.add_argument(
        "--gammas",
        default="0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,1.0",
    )
    parser.add_argument("--coder-predictor", default="med")
    parser.add_argument("--tile-size", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    bits_values = parse_ints(args.bits)
    gammas = parse_floats(args.gammas)

    summaries = []
    for path in sorted(DATA_DIR.glob(args.glob)):
        summary = summarize_image(
            path,
            args.crop_size,
            bits_values,
            gammas,
            args.coder_predictor,
            args.tile_size,
        )
        summaries.append(summary)
        print(path.name)
        sorted_rows = sorted(
            summary["rows"],
            key=lambda row: (
                not row["quality_passes"],
                row["estimated_bytes"],
            ),
        )
        for row in sorted_rows[:16]:
            q = row["quality"]
            print(
                f"  {'PASS' if row['quality_passes'] else 'fail'} "
                f"gamma={row['gamma']:.3f} b{row['bits']} "
                f"{row['estimated_ratio_vs_original']:7.2f}x "
                f"bytes={row['estimated_bytes']:,.0f} "
                f"bps={row['estimated_bits_per_sample']:.3f} "
                f"nz={row['nonzero_rate']:.2%} "
                f"log={q['signed_log2_rmse']:.3e} "
                f"p99={q['signed_log2_p99']:.3e} "
                f"grad={q['gradient_signed_log2_nrmse']:.3e}"
            )
        if args.limit and len(summaries) >= args.limit:
            break

    if not args.no_save:
        safe_glob = args.glob.replace("*", "star").replace(".", "_")
        output = RESULTS_DIR / (
            f"parametric_transform_index_{safe_glob}"
            f"_crop{args.crop_size}_bits{args.bits.replace(',', '-')}.json"
        )
        output.write_text(json.dumps(summaries, indent=2))
        print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
