"""Probe per-channel signed-power gamma for transform-index quantization."""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import OpenImageIO as oiio

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_linear_index_codec import error_stats  # noqa: E402
from probe_index_residual_context_entropy import med_predict_indices  # noqa: E402
from probe_power_recon_table import table_reconstruct  # noqa: E402
from probe_small_residual_escape_entropy import summarize_model  # noqa: E402


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


def quantize_channel_gamma(
    pixels: np.ndarray,
    bits: int,
    gammas: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray]:
    levels = (1 << bits) - 1
    indices = np.zeros(pixels.shape, dtype=np.uint16)
    recon = np.array(pixels, copy=True)
    for channel in range(pixels.shape[2]):
        gamma = gammas[min(channel, len(gammas) - 1)]
        values = pixels[:, :, channel].astype(np.float64)
        transformed = transform_power(values, gamma)
        range_values = transformed.astype(np.float32).astype(np.float64)
        lo = float(np.min(range_values))
        hi = float(np.max(range_values))
        if not hi > lo:
            recon[:, :, channel] = inverse_power(np.asarray(lo), gamma).astype(
                np.float32
            )
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


def residual_symbols_and_pred(indices: np.ndarray, bits: int) -> tuple[np.ndarray, np.ndarray]:
    alphabet = 1 << bits
    pred = med_predict_indices(indices, bits)
    residual = ((indices.astype(np.int32) + alphabet - pred) & (alphabet - 1)).astype(
        np.uint16
    )
    return residual, pred


def summarize_candidate(
    pixels: np.ndarray,
    bits: int,
    gammas: tuple[float, ...],
    small: int,
    category_context: str,
    detail_context: str,
    pred_bin_bits: int,
    recon_table: str,
) -> dict:
    indices, recon = quantize_channel_gamma(pixels, bits, gammas)
    if recon_table != "center":
        recon = table_reconstruct(pixels, indices, bits, recon_table)
    quality = error_stats(pixels, recon)
    residual, pred = residual_symbols_and_pred(indices, bits)
    pred_bins = None
    if pred_bin_bits > 0:
        shift = max(0, bits - pred_bin_bits)
        pred_bins = (pred.astype(np.uint16) >> np.uint16(shift)).astype(np.uint16)
    model = summarize_model(
        residual,
        bits,
        small,
        category_context,
        detail_context,
        pred_bins,
    )
    estimated_bytes = model["bits_per_sample"] * pixels.size / 8.0 + 73.0
    return {
        "bits": bits,
        "gammas": list(gammas),
        "small": small,
        "category_context": category_context,
        "detail_context": detail_context,
        "recon_table": recon_table,
        "estimated_bits_per_sample": model["bits_per_sample"],
        "estimated_bytes": estimated_bytes,
        "estimated_ratio_vs_original": pixels.nbytes / estimated_bytes,
        "escape_rate": model["escape_rate"],
        "quality_passes": quality_passes(quality),
        "quality": quality,
    }


def summarize_image(
    path: Path,
    crop_size: int,
    bits: int,
    gamma_values: tuple[float, ...],
    small: int,
    category_context: str,
    detail_context: str,
    pred_bin_bits: int,
    recon_table: str,
    max_candidates: int,
) -> dict:
    pixels = read_exr(path, crop_size)
    rows = []
    channel_count = pixels.shape[2]
    for gammas in itertools.product(gamma_values, repeat=channel_count):
        rows.append(
            summarize_candidate(
                pixels,
                bits,
                tuple(float(gamma) for gamma in gammas),
                small,
                category_context,
                detail_context,
                pred_bin_bits,
                recon_table,
            )
        )
        if max_candidates and len(rows) >= max_candidates:
            break
    rows.sort(key=lambda row: (not row["quality_passes"], row["estimated_bytes"]))
    return {
        "image": path.name,
        "shape": list(pixels.shape),
        "crop_size": crop_size,
        "bits": bits,
        "gamma_values": list(gamma_values),
        "pred_bin_bits": pred_bin_bits,
        "rows": rows,
    }


def parse_floats(text: str) -> tuple[float, ...]:
    return tuple(float(part) for part in text.split(",") if part.strip())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="sample_DSCF0009.EXR")
    parser.add_argument("--crop-size", type=int, default=1024)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--gammas", default="0.78,0.79,0.795,0.8")
    parser.add_argument("--small", type=int, default=7)
    parser.add_argument("--category-context", default="west_north_prev_channel")
    parser.add_argument("--detail-context", default="west_north_prev_channel")
    parser.add_argument("--pred-bin-bits", type=int, default=0)
    parser.add_argument("--recon-table", default="center")
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    summaries = []
    for path in sorted(DATA_DIR.glob(args.glob)):
        summary = summarize_image(
            path,
            args.crop_size,
            args.bits,
            parse_floats(args.gammas),
            args.small,
            args.category_context,
            args.detail_context,
            args.pred_bin_bits,
            args.recon_table,
            args.max_candidates,
        )
        summaries.append(summary)
        print(path.name)
        for row in summary["rows"][:16]:
            q = row["quality"]
            gamma_text = ",".join(f"{gamma:.3f}" for gamma in row["gammas"])
            print(
                f"  {'PASS' if row['quality_passes'] else 'fail'} "
                f"gammas=({gamma_text}) "
                f"{row['estimated_ratio_vs_original']:7.2f}x "
                f"bytes={row['estimated_bytes']:,.0f} "
                f"bps={row['estimated_bits_per_sample']:.3f} "
                f"escape={row['escape_rate']:.2%} "
                f"log={q['signed_log2_rmse']:.3e} "
                f"p99={q['signed_log2_p99']:.3e} "
                f"grad={q['gradient_signed_log2_nrmse']:.3e}"
            )
        if args.limit and len(summaries) >= args.limit:
            break

    if not args.no_save:
        safe_glob = args.glob.replace("*", "star").replace(".", "_")
        output = RESULTS_DIR / (
            f"channel_gamma_index_{safe_glob}"
            f"_crop{args.crop_size}_bits{args.bits}.json"
        )
        output.write_text(json.dumps(summaries, indent=2))
        print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
