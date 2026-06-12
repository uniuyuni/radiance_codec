"""Benchmark the implemented dedicated linear-index near-lossless codec."""
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
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
sys.path.insert(0, str(ROOT / "codec" / "python"))

import radiance_codec  # noqa: E402


def display_luma(pixels: np.ndarray, white: float = 4.0, gamma: float = 2.2) -> np.ndarray:
    rgb = pixels[:, :, :3].astype(np.float64)
    normalized = np.clip(rgb / float(white), 0.0, 1.0)
    display_rgb = np.power(normalized, 1.0 / float(gamma))
    return (
        display_rgb[:, :, 0] * 0.2126
        + display_rgb[:, :, 1] * 0.7152
        + display_rgb[:, :, 2] * 0.0722
    )


def display_region_masks(pixels: np.ndarray) -> dict[str, np.ndarray]:
    value = np.max(pixels[:, :, :3].astype(np.float64), axis=2)
    return {
        "dark_0_0.25": value <= 0.25,
        "mid_0.25_1": (value > 0.25) & (value <= 1.0),
        "highlight_1_4": (value > 1.0) & (value <= 4.0),
        "extreme_gt4": value > 4.0,
        "all": np.ones(value.shape, dtype=bool),
    }


def detail_loss_stats(original_luma: np.ndarray, decoded_luma: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    orig_sum_sq = 0.0
    dec_sum_sq = 0.0
    diff_sum_sq = 0.0
    dot = 0.0
    sample_count = 0
    lost_count = 0
    visible_count = 0
    for orig_grad, dec_grad, grad_mask in (
        (
            original_luma[:, 1:] - original_luma[:, :-1],
            decoded_luma[:, 1:] - decoded_luma[:, :-1],
            mask[:, 1:] & mask[:, :-1],
        ),
        (
            original_luma[1:, :] - original_luma[:-1, :],
            decoded_luma[1:, :] - decoded_luma[:-1, :],
            mask[1:, :] & mask[:-1, :],
        ),
    ):
        if not bool(np.any(grad_mask)):
            continue
        orig = orig_grad[grad_mask]
        dec = dec_grad[grad_mask]
        diff = dec - orig
        sample_count += int(orig.size)
        orig_sum_sq += float(np.sum(orig * orig))
        dec_sum_sq += float(np.sum(dec * dec))
        diff_sum_sq += float(np.sum(diff * diff))
        dot += float(np.sum(orig * dec))
        visible = np.abs(orig) > (1.0 / 4096.0)
        visible_count += int(np.count_nonzero(visible))
        lost_count += int(np.count_nonzero(visible & (np.abs(dec) < np.abs(orig) * 0.5)))

    if sample_count == 0 or orig_sum_sq == 0.0:
        return {
            "grad_nrmse": 0.0,
            "grad_energy_ratio": 1.0,
            "grad_correlation": 1.0,
            "lost_detail_rate": 0.0,
            "visible_detail_rate": 0.0,
        }
    denom = math.sqrt(orig_sum_sq * dec_sum_sq)
    return {
        "grad_nrmse": math.sqrt(diff_sum_sq / sample_count) / math.sqrt(orig_sum_sq / sample_count),
        "grad_energy_ratio": math.sqrt(dec_sum_sq / orig_sum_sq),
        "grad_correlation": 1.0 if denom == 0.0 else dot / denom,
        "lost_detail_rate": lost_count / sample_count,
        "visible_detail_rate": visible_count / sample_count,
    }


def display_detail_stats(original: np.ndarray, decoded: np.ndarray) -> dict[str, float]:
    original_luma = display_luma(original)
    decoded_luma = display_luma(decoded)
    regions = display_region_masks(original)
    stats = {}
    for label, mask in regions.items():
        detail = detail_loss_stats(original_luma, decoded_luma, mask)
        prefix = f"display_w4_g22_{label}"
        stats[f"{prefix}_grad_nrmse"] = detail["grad_nrmse"]
        stats[f"{prefix}_grad_energy_ratio"] = detail["grad_energy_ratio"]
        stats[f"{prefix}_grad_correlation"] = detail["grad_correlation"]
        stats[f"{prefix}_lost_detail_rate"] = detail["lost_detail_rate"]
        stats[f"{prefix}_visible_detail_rate"] = detail["visible_detail_rate"]
    return stats


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


def error_stats(original: np.ndarray, decoded: np.ndarray) -> dict[str, float]:
    diff = decoded.astype(np.float64) - original.astype(np.float64)
    rmse = float(np.sqrt(np.mean(diff * diff)))
    mae = float(np.mean(np.abs(diff)))
    peak = float(np.max(np.abs(original)))
    psnr = math.inf
    if rmse > 0.0 and peak > 0.0:
        psnr = 20.0 * math.log10(peak / rmse)
    signed_log_original = (
        np.sign(original.astype(np.float64))
        * np.log2(1.0 + np.abs(original.astype(np.float64)))
    )
    signed_log_decoded = (
        np.sign(decoded.astype(np.float64))
        * np.log2(1.0 + np.abs(decoded.astype(np.float64)))
    )
    signed_log_diff = signed_log_decoded - signed_log_original
    grad_original = np.concatenate(
        [
            np.diff(signed_log_original, axis=0).reshape(-1),
            np.diff(signed_log_original, axis=1).reshape(-1),
        ]
    )
    grad_decoded = np.concatenate(
        [
            np.diff(signed_log_decoded, axis=0).reshape(-1),
            np.diff(signed_log_decoded, axis=1).reshape(-1),
        ]
    )
    grad_diff = grad_decoded - grad_original
    grad_signal = float(np.sqrt(np.mean(grad_original * grad_original)))
    grad_nrmse = 0.0
    if grad_signal > 0.0:
        grad_nrmse = float(np.sqrt(np.mean(grad_diff * grad_diff)) / grad_signal)
    stats = {
        "mae": mae,
        "rmse": rmse,
        "psnr_peak_abs": psnr,
        "signed_log2_rmse": float(np.sqrt(np.mean(signed_log_diff ** 2))),
        "signed_log2_p99": float(np.percentile(np.abs(signed_log_diff), 99.0)),
        "gradient_signed_log2_nrmse": grad_nrmse,
    }
    stats.update(display_detail_stats(original, decoded))
    return stats


def quality_passes(
    row: dict[str, float],
    log_rmse_threshold: float,
    log_p99_threshold: float,
    gradient_threshold: float,
) -> bool:
    return (
        row["signed_log2_rmse"] <= log_rmse_threshold
        and row["signed_log2_p99"] <= log_p99_threshold
        and row["gradient_signed_log2_nrmse"] <= gradient_threshold
    )


def summarize_image(
    path: Path,
    crop_size: int,
    bits_values: tuple[int, ...],
    transform: str,
    log_rmse_threshold: float,
    log_p99_threshold: float,
    gradient_threshold: float,
) -> dict:
    pixels = read_exr(path, crop_size)
    rows = []
    for bits in bits_values:
        t0 = time.perf_counter()
        encoded = radiance_codec.encode_linear_index_near_lossless(
            pixels,
            bits=bits,
            effort=9,
            transform=transform,
        )
        t1 = time.perf_counter()
        decoded = radiance_codec.decode(encoded)
        t2 = time.perf_counter()
        expected = radiance_codec.quantize_linear_index(
            pixels,
            bits=bits,
            transform=transform,
        )
        if decoded.tobytes() != expected.tobytes():
            raise RuntimeError(f"{path.name} bits{bits}: decoded quantization mismatch")
        rows.append({
            "bits": bits,
            "encoded_bytes": len(encoded),
            "ratio_vs_original": pixels.nbytes / len(encoded),
            "encode_seconds": t1 - t0,
            "decode_seconds": t2 - t1,
            **error_stats(pixels, decoded),
        })
    selected = None
    for row in rows:
        if quality_passes(
            row,
            log_rmse_threshold,
            log_p99_threshold,
            gradient_threshold,
        ):
            selected = row
            break
    return {
        "image": path.name,
        "shape": list(pixels.shape),
        "crop_size": crop_size,
        "transform": transform,
        "quality_thresholds": {
            "signed_log2_rmse": log_rmse_threshold,
            "signed_log2_p99": log_p99_threshold,
            "gradient_signed_log2_nrmse": gradient_threshold,
        },
        "selected_bits": None if selected is None else selected["bits"],
        "selected_ratio_vs_original": None if selected is None else selected["ratio_vs_original"],
        "rows": rows,
    }


def parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(",") if part.strip())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="sample_DSCF0009.EXR")
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--bits", default="6,7")
    parser.add_argument("--transform", default="linear")
    parser.add_argument("--log-rmse-threshold", type=float, default=0.004)
    parser.add_argument("--log-p99-threshold", type=float, default=0.018)
    parser.add_argument("--gradient-threshold", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    bits_values = parse_ints(args.bits)
    rows = []
    for path in sorted(DATA_DIR.glob(args.glob)):
        row = summarize_image(
            path,
            args.crop_size,
            bits_values,
            args.transform,
            args.log_rmse_threshold,
            args.log_p99_threshold,
            args.gradient_threshold,
        )
        rows.append(row)
        print(path.name)
        for item in row["rows"]:
            print(
                f"  bits{item['bits']:02d}: "
                f"{args.transform} "
                f"{item['ratio_vs_original']:7.2f}x "
                f"bytes={item['encoded_bytes']} "
                f"enc={item['encode_seconds']:.3f}s "
                f"dec={item['decode_seconds']:.3f}s "
                f"log_rmse={item['signed_log2_rmse']:.3e} "
                f"grad={item['gradient_signed_log2_nrmse']:.3e} "
                f"hi_lost={item['display_w4_g22_highlight_1_4_lost_detail_rate']:.2%} "
                f"ex_lost={item['display_w4_g22_extreme_gt4_lost_detail_rate']:.2%} "
                f"psnr={item['psnr_peak_abs']:.2f}dB"
            )
        if row["selected_bits"] is None:
            print("  selected: none")
        else:
            print(
                f"  selected: bits{row['selected_bits']:02d} "
                f"{row['selected_ratio_vs_original']:.2f}x")
        if args.limit and len(rows) >= args.limit:
            break

    if not args.no_save:
        safe_glob = args.glob.replace("*", "star").replace(".", "_")
        output = RESULTS_DIR / (
            f"linear_index_codec_{safe_glob}"
            f"_crop{args.crop_size}_bits{args.bits.replace(',', '-')}.json"
        )
        if args.transform != "linear":
            output = RESULTS_DIR / (
                f"linear_index_codec_{safe_glob}"
                f"_crop{args.crop_size}_{args.transform}"
                f"_bits{args.bits.replace(',', '-')}.json"
            )
        output.write_text(json.dumps(rows, indent=2))
        print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
