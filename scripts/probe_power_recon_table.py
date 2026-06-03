"""Probe reconstruction tables for signed-power transform indices.

The index stream and entropy cost stay unchanged.  Only the decoder-side
representative value for each quantized bin is changed, using per-channel
statistics from the image.  This estimates whether a small recon table can
rescue slightly-too-aggressive gamma choices.
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

from benchmark_linear_index_codec import error_stats  # noqa: E402
from probe_parametric_transform_index import quantize_power  # noqa: E402


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


def signed_log(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.log2(1.0 + np.abs(values))


def inverse_signed_log(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * (np.exp2(np.abs(values)) - 1.0)


def table_reconstruct(
    pixels: np.ndarray,
    indices: np.ndarray,
    bits: int,
    mode: str,
) -> np.ndarray:
    levels = 1 << bits
    recon = np.array(pixels, copy=True)
    for channel in range(pixels.shape[2]):
        q = indices[:, :, channel].reshape(-1).astype(np.int64)
        values = pixels[:, :, channel].reshape(-1).astype(np.float64)
        counts = np.bincount(q, minlength=levels).astype(np.float64)
        if mode == "value-mean":
            metric_values = values
        elif mode == "signed-log-mean":
            metric_values = signed_log(values)
        else:
            raise ValueError(f"unknown mode: {mode}")
        sums = np.bincount(q, weights=metric_values, minlength=levels)
        table_metric = np.zeros(levels, dtype=np.float64)
        seen = counts > 0
        table_metric[seen] = sums[seen] / counts[seen]
        # Fill empty bins by nearest seen neighbor so table lookup is total.
        if bool(np.any(seen)):
            seen_idx = np.flatnonzero(seen)
            for idx in np.flatnonzero(~seen):
                nearest = seen_idx[np.argmin(np.abs(seen_idx - idx))]
                table_metric[idx] = table_metric[nearest]
        if mode == "signed-log-mean":
            table_values = inverse_signed_log(table_metric)
        else:
            table_values = table_metric
        recon[:, :, channel] = table_values[indices[:, :, channel]].astype(np.float32)
    return np.ascontiguousarray(recon)


def quality_passes(quality: dict[str, float]) -> bool:
    return (
        quality["signed_log2_rmse"] <= 0.004
        and quality["signed_log2_p99"] <= 0.018
        and quality["gradient_signed_log2_nrmse"] <= 0.5
    )


def summarize_image(
    path: Path,
    crop_size: int,
    bits: int,
    gammas: tuple[float, ...],
    modes: tuple[str, ...],
) -> dict:
    pixels = read_exr(path, crop_size)
    rows = []
    for gamma in gammas:
        indices, baseline = quantize_power(pixels, bits, gamma)
        baseline_quality = error_stats(pixels, baseline)
        rows.append({
            "gamma": gamma,
            "mode": "center",
            "quality_passes": quality_passes(baseline_quality),
            "quality": baseline_quality,
        })
        for mode in modes:
            recon = table_reconstruct(pixels, indices, bits, mode)
            quality = error_stats(pixels, recon)
            rows.append({
                "gamma": gamma,
                "mode": mode,
                "quality_passes": quality_passes(quality),
                "quality": quality,
            })
    return {
        "image": path.name,
        "shape": list(pixels.shape),
        "crop_size": crop_size,
        "bits": bits,
        "rows": rows,
    }


def parse_floats(text: str) -> tuple[float, ...]:
    return tuple(float(part) for part in text.split(",") if part.strip())


def parse_strings(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in text.split(",") if part.strip())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="sample_DSCF0009.EXR")
    parser.add_argument("--crop-size", type=int, default=1024)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--gammas", default="0.8,0.805,0.81")
    parser.add_argument("--modes", default="signed-log-mean,value-mean")
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
            parse_strings(args.modes),
        )
        summaries.append(summary)
        print(path.name)
        for row in summary["rows"]:
            q = row["quality"]
            print(
                f"  {'PASS' if row['quality_passes'] else 'fail'} "
                f"gamma={row['gamma']:.3f} {row['mode']:15s} "
                f"log={q['signed_log2_rmse']:.3e} "
                f"p99={q['signed_log2_p99']:.3e} "
                f"grad={q['gradient_signed_log2_nrmse']:.3e} "
                f"psnr={q['psnr_peak_abs']:.2f}dB"
            )
        if args.limit and len(summaries) >= args.limit:
            break

    if not args.no_save:
        safe_glob = args.glob.replace("*", "star").replace(".", "_")
        output = RESULTS_DIR / (
            f"power_recon_table_{safe_glob}"
            f"_crop{args.crop_size}_bits{args.bits}.json"
        )
        output.write_text(json.dumps(summaries, indent=2))
        print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
