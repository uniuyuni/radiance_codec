"""Probe deterministic residual synthesis after linear near-lossless quantization.

This is a validation script, not a production codec path.  It asks whether the
residual inside each finite linear quantization bin has reusable structure:

* center: current reconstruction from the quantized index.
* hash_uniform: deterministic white-ish jitter inside the quantization bin.
* phaseN_mean: tiny learned residual table keyed by channel and image phase.
* phaseN_indexM_mean: learned residual table keyed by channel, phase, and a
  coarse quantized-index bucket.

The learned tables are trained on some crops and evaluated on held-out crops so
the result is less likely to be pure memorization.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import OpenImageIO as oiio

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"


@dataclass
class QuantizedCrop:
    x0: int
    y0: int
    original: np.ndarray
    center: np.ndarray
    indices: np.ndarray
    step: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    finite: np.ndarray


def image_spec(path: Path) -> tuple[int, int, int]:
    image_input = oiio.ImageInput.open(str(path))
    if image_input is None:
        raise RuntimeError(f"can't open {path}: {oiio.geterror()}")
    spec = image_input.spec()
    width = int(spec.width)
    height = int(spec.height)
    channels = int(spec.nchannels)
    image_input.close()
    return width, height, channels


def read_exr_crop(path: Path, x0: int, y0: int, crop_size: int) -> np.ndarray:
    image_input = oiio.ImageInput.open(str(path))
    if image_input is None:
        raise RuntimeError(f"can't open {path}: {oiio.geterror()}")
    spec = image_input.spec()
    width = min(crop_size, spec.width - x0)
    height = min(crop_size, spec.height - y0)
    pixels = image_input.read_scanlines(
        y0,
        y0 + height,
        0,
        0,
        spec.nchannels,
        oiio.FLOAT,
    )
    image_input.close()
    if pixels is None:
        raise RuntimeError(f"can't read {path}: {oiio.geterror()}")
    return np.ascontiguousarray(
        np.asarray(pixels, dtype=np.float32).reshape(
            height,
            spec.width,
            spec.nchannels,
        )[:, x0:x0 + width, :]
    )


def crop_origins(width: int, height: int, crop_size: int, positions: str) -> list[tuple[int, int]]:
    max_x = max(0, width - crop_size)
    max_y = max(0, height - crop_size)
    if positions == "top-left":
        return [(0, 0)]
    if positions == "center":
        return [(max_x // 2, max_y // 2)]
    if positions == "grid5":
        values = [
            (0, 0),
            (max_x, 0),
            (max_x // 2, max_y // 2),
            (0, max_y),
            (max_x, max_y),
        ]
    elif positions == "grid9":
        xs = [0, max_x // 2, max_x]
        ys = [0, max_y // 2, max_y]
        values = [(x, y) for y in ys for x in xs]
    else:
        raise ValueError("positions must be top-left, center, grid5, or grid9")
    return list(dict.fromkeys(values))


def quantize_linear_tiled(pixels: np.ndarray, bits: int, tile_size: int, x0: int, y0: int) -> QuantizedCrop:
    levels = (1 << bits) - 1
    center = np.array(pixels, copy=True)
    indices = np.zeros(pixels.shape, dtype=np.uint16)
    step = np.zeros(pixels.shape, dtype=np.float32)
    lo_map = np.array(pixels, copy=True)
    hi_map = np.array(pixels, copy=True)
    finite_all = np.isfinite(pixels)

    for ty in range(0, pixels.shape[0], tile_size):
        for tx in range(0, pixels.shape[1], tile_size):
            tile = pixels[ty:ty + tile_size, tx:tx + tile_size, :]
            for channel in range(pixels.shape[2]):
                values = tile[:, :, channel].astype(np.float64)
                finite = np.isfinite(values)
                if not bool(np.any(finite)):
                    continue
                lo = float(values[finite].min())
                hi = float(values[finite].max())
                lo_map[ty:ty + tile.shape[0], tx:tx + tile.shape[1], channel] = np.float32(lo)
                hi_map[ty:ty + tile.shape[0], tx:tx + tile.shape[1], channel] = np.float32(hi)
                if hi <= lo:
                    center[ty:ty + tile.shape[0], tx:tx + tile.shape[1], channel][finite] = np.float32(lo)
                    continue
                scale = levels / (hi - lo)
                q = np.floor((values - lo) * scale + 0.5)
                q = np.clip(q, 0, levels).astype(np.uint16)
                rec = lo + q.astype(np.float64) * (hi - lo) / levels
                target = center[ty:ty + tile.shape[0], tx:tx + tile.shape[1], channel]
                target[finite] = rec[finite].astype(np.float32)
                indices[ty:ty + tile.shape[0], tx:tx + tile.shape[1], channel][finite] = q[finite]
                step[ty:ty + tile.shape[0], tx:tx + tile.shape[1], channel][finite] = np.float32((hi - lo) / levels)

    return QuantizedCrop(
        x0=x0,
        y0=y0,
        original=pixels,
        center=np.ascontiguousarray(center),
        indices=indices,
        step=step,
        lo=lo_map,
        hi=hi_map,
        finite=finite_all & (step > 0.0),
    )


def split_train_eval(crops: list[QuantizedCrop]) -> tuple[list[QuantizedCrop], list[QuantizedCrop]]:
    if len(crops) <= 1:
        return crops, crops
    train = [crop for i, crop in enumerate(crops) if i % 2 == 0]
    eval_crops = [crop for i, crop in enumerate(crops) if i % 2 == 1]
    return train, eval_crops


def global_xy(crop: QuantizedCrop) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.indices(crop.original.shape[:2], dtype=np.int32)
    return xx + crop.x0, yy + crop.y0


def normalized_residual(crop: QuantizedCrop) -> np.ndarray:
    out = np.zeros(crop.original.shape, dtype=np.float32)
    selected = crop.finite
    out[selected] = (
        (crop.original[selected].astype(np.float64)
         - crop.center[selected].astype(np.float64))
        / crop.step[selected].astype(np.float64)
    ).astype(np.float32)
    return np.clip(out, -0.5, 0.5)


def context_ids(
    crop: QuantizedCrop,
    phase: int,
    index_buckets: int,
    bits: int,
) -> tuple[np.ndarray, int]:
    xx, yy = global_xy(crop)
    phase_id = (yy % phase) * phase + (xx % phase)
    channels = crop.original.shape[2]
    channel_id = np.arange(channels, dtype=np.int32).reshape(1, 1, channels)
    context = phase_id[:, :, None] * channels + channel_id
    context_count = phase * phase * channels
    if index_buckets > 0:
        levels = (1 << bits) - 1
        bucket = (
            crop.indices.astype(np.int32) * index_buckets // max(1, levels + 1)
        )
        bucket = np.clip(bucket, 0, index_buckets - 1)
        context = context * index_buckets + bucket
        context_count *= index_buckets
    return context.astype(np.int32), context_count


def fit_residual_table(
    crops: list[QuantizedCrop],
    phase: int,
    index_buckets: int,
    bits: int,
) -> tuple[np.ndarray, np.ndarray]:
    context_count = phase * phase * crops[0].original.shape[2] * max(1, index_buckets)
    sums = np.zeros(context_count, dtype=np.float64)
    counts = np.zeros(context_count, dtype=np.int64)
    for crop in crops:
        residual = normalized_residual(crop)
        context, _ = context_ids(crop, phase, index_buckets, bits)
        selected = crop.finite.reshape(-1)
        flat_context = context.reshape(-1)[selected]
        flat_residual = residual.reshape(-1)[selected]
        sums += np.bincount(flat_context, weights=flat_residual, minlength=context_count)
        counts += np.bincount(flat_context, minlength=context_count)
    table = np.zeros(context_count, dtype=np.float32)
    valid = counts > 0
    table[valid] = (sums[valid] / counts[valid]).astype(np.float32)
    return table, counts


def apply_table(
    crop: QuantizedCrop,
    table: np.ndarray,
    phase: int,
    index_buckets: int,
    bits: int,
) -> np.ndarray:
    context, _ = context_ids(crop, phase, index_buckets, bits)
    offsets = table[context]
    recon = crop.center.astype(np.float64) + offsets.astype(np.float64) * crop.step.astype(np.float64)
    recon = np.clip(recon, crop.lo.astype(np.float64), crop.hi.astype(np.float64))
    out = np.array(crop.center, copy=True)
    out[crop.finite] = recon[crop.finite].astype(np.float32)
    return np.ascontiguousarray(out)


def hash_uniform_reconstruct(crop: QuantizedCrop, amplitude: float) -> np.ndarray:
    xx, yy = global_xy(crop)
    channels = np.arange(crop.original.shape[2], dtype=np.uint32).reshape(1, 1, -1)
    state = (
        (xx[:, :, None].astype(np.uint32) * np.uint32(0x9e3779b1))
        ^ (yy[:, :, None].astype(np.uint32) * np.uint32(0x85ebca6b))
        ^ (channels * np.uint32(0xc2b2ae35))
        ^ (crop.indices.astype(np.uint32) * np.uint32(0x27d4eb2d))
    )
    state ^= state >> np.uint32(16)
    state *= np.uint32(0x7feb352d)
    state ^= state >> np.uint32(15)
    state *= np.uint32(0x846ca68b)
    state ^= state >> np.uint32(16)
    unit = state.astype(np.float64) / float(2 ** 32)
    offsets = (unit - 0.5) * float(amplitude)
    recon = crop.center.astype(np.float64) + offsets * crop.step.astype(np.float64)
    recon = np.clip(recon, crop.lo.astype(np.float64), crop.hi.astype(np.float64))
    out = np.array(crop.center, copy=True)
    out[crop.finite] = recon[crop.finite].astype(np.float32)
    return np.ascontiguousarray(out)


def quality_stats(crops: list[QuantizedCrop], reconstructions: list[np.ndarray]) -> dict[str, float]:
    count = 0
    abs_sum = 0.0
    sq_sum = 0.0
    signed_log_sq_sum = 0.0
    signed_log_abs_values = []
    diff_abs_values = []
    grad_sq_sum = 0.0
    grad_signal_sq_sum = 0.0
    hist_original = []
    hist_recon = []

    for crop, recon in zip(crops, reconstructions):
        finite = np.isfinite(crop.original) & np.isfinite(recon)
        if not bool(np.any(finite)):
            continue
        original = crop.original[finite].astype(np.float64)
        decoded = recon[finite].astype(np.float64)
        diff = decoded - original
        diff_abs = np.abs(diff)
        count += int(diff.size)
        abs_sum += float(diff_abs.sum())
        sq_sum += float(np.dot(diff, diff))
        signed_original = np.sign(original) * np.log2(1.0 + np.abs(original))
        signed_decoded = np.sign(decoded) * np.log2(1.0 + np.abs(decoded))
        signed_abs = np.abs(signed_decoded - signed_original)
        signed_log_sq_sum += float(np.dot(signed_abs, signed_abs))
        signed_log_abs_values.append(signed_abs)
        diff_abs_values.append(diff_abs)
        hist_original.append(original)
        hist_recon.append(decoded)

        full_original = (
            np.sign(crop.original.astype(np.float64))
            * np.log2(1.0 + np.abs(crop.original.astype(np.float64)))
        )
        full_decoded = (
            np.sign(recon.astype(np.float64))
            * np.log2(1.0 + np.abs(recon.astype(np.float64)))
        )
        grad_original = np.concatenate(
            [
                np.diff(full_original, axis=0).reshape(-1),
                np.diff(full_original, axis=1).reshape(-1),
            ]
        )
        grad_decoded = np.concatenate(
            [
                np.diff(full_decoded, axis=0).reshape(-1),
                np.diff(full_decoded, axis=1).reshape(-1),
            ]
        )
        grad_finite = np.isfinite(grad_original) & np.isfinite(grad_decoded)
        if bool(np.any(grad_finite)):
            grad_diff = grad_decoded[grad_finite] - grad_original[grad_finite]
            grad_sq_sum += float(np.dot(grad_diff, grad_diff))
            grad_signal_sq_sum += float(np.dot(grad_original[grad_finite], grad_original[grad_finite]))

    if count == 0:
        return {
            "mae": 0.0,
            "rmse": 0.0,
            "signed_log2_rmse": 0.0,
            "signed_log2_p99": 0.0,
            "abs_p99": 0.0,
            "gradient_signed_log2_nrmse": 0.0,
            "histogram_ks_256": 0.0,
        }

    signed_all = np.concatenate(signed_log_abs_values)
    diff_abs_all = np.concatenate(diff_abs_values)
    originals = np.concatenate(hist_original)
    decoded = np.concatenate(hist_recon)
    hist_ks = 0.0
    lo = float(np.min(originals))
    hi = float(np.max(originals))
    if hi > lo:
        h0, edges = np.histogram(originals, bins=256, range=(lo, hi))
        h1, _ = np.histogram(decoded, bins=edges)
        c0 = np.cumsum(h0, dtype=np.float64)
        c1 = np.cumsum(h1, dtype=np.float64)
        if c0[-1] > 0 and c1[-1] > 0:
            hist_ks = float(np.max(np.abs(c0 / c0[-1] - c1 / c1[-1])))
    gradient_nrmse = 0.0
    if grad_signal_sq_sum > 0.0:
        gradient_nrmse = math.sqrt(grad_sq_sum / grad_signal_sq_sum)
    return {
        "mae": abs_sum / count,
        "rmse": math.sqrt(sq_sum / count),
        "signed_log2_rmse": math.sqrt(signed_log_sq_sum / count),
        "signed_log2_p99": float(np.percentile(signed_all, 99.0)),
        "abs_p99": float(np.percentile(diff_abs_all, 99.0)),
        "gradient_signed_log2_nrmse": gradient_nrmse,
        "histogram_ks_256": hist_ks,
    }


def table_side_bytes(entries: int) -> int:
    return int(entries * 2)


def summarize_image(
    path: Path,
    crop_size: int,
    positions: str,
    bits_values: tuple[int, ...],
    tile_size: int,
    phases: tuple[int, ...],
    index_buckets_values: tuple[int, ...],
    hash_amplitude: float,
) -> dict:
    width, height, channels = image_spec(path)
    origins = crop_origins(width, height, crop_size, positions)
    rows = []
    for bits in bits_values:
        crops = []
        for x0, y0 in origins:
            pixels = read_exr_crop(path, x0, y0, crop_size)
            crops.append(quantize_linear_tiled(pixels, bits, tile_size, x0, y0))
        train, eval_crops = split_train_eval(crops)
        center_stats = quality_stats(eval_crops, [crop.center for crop in eval_crops])
        rows.append({
            "bits": bits,
            "mode": "center",
            "phase": 0,
            "index_buckets": 0,
            "table_entries": 0,
            "table_bytes_approx": 0,
            **center_stats,
            "signed_log2_rmse_delta_vs_center": 0.0,
            "gradient_delta_vs_center": 0.0,
            "histogram_ks_delta_vs_center": 0.0,
        })

        hash_stats = quality_stats(
            eval_crops,
            [hash_uniform_reconstruct(crop, hash_amplitude) for crop in eval_crops],
        )
        rows.append({
            "bits": bits,
            "mode": "hash_uniform",
            "phase": 0,
            "index_buckets": 0,
            "table_entries": 0,
            "table_bytes_approx": 4,
            **hash_stats,
            "signed_log2_rmse_delta_vs_center": (
                hash_stats["signed_log2_rmse"] - center_stats["signed_log2_rmse"]
            ),
            "gradient_delta_vs_center": (
                hash_stats["gradient_signed_log2_nrmse"]
                - center_stats["gradient_signed_log2_nrmse"]
            ),
            "histogram_ks_delta_vs_center": (
                hash_stats["histogram_ks_256"] - center_stats["histogram_ks_256"]
            ),
        })

        for phase in phases:
            for index_buckets in (0, *index_buckets_values):
                table, counts = fit_residual_table(train, phase, index_buckets, bits)
                stats = quality_stats(
                    eval_crops,
                    [
                        apply_table(crop, table, phase, index_buckets, bits)
                        for crop in eval_crops
                    ],
                )
                mode = f"phase{phase}_mean"
                if index_buckets:
                    mode = f"phase{phase}_index{index_buckets}_mean"
                rows.append({
                    "bits": bits,
                    "mode": mode,
                    "phase": phase,
                    "index_buckets": index_buckets,
                    "table_entries": int(table.size),
                    "nonempty_table_entries": int(np.count_nonzero(counts)),
                    "table_bytes_approx": table_side_bytes(int(table.size)),
                    **stats,
                    "signed_log2_rmse_delta_vs_center": (
                        stats["signed_log2_rmse"] - center_stats["signed_log2_rmse"]
                    ),
                    "gradient_delta_vs_center": (
                        stats["gradient_signed_log2_nrmse"]
                        - center_stats["gradient_signed_log2_nrmse"]
                    ),
                    "histogram_ks_delta_vs_center": (
                        stats["histogram_ks_256"] - center_stats["histogram_ks_256"]
                    ),
                })
    return {
        "image": path.name,
        "shape": [height, width, channels],
        "crop_size": crop_size,
        "positions": positions,
        "origins": [[x, y] for x, y in origins],
        "tile_size": tile_size,
        "train_crops": len(split_train_eval([None] * len(origins))[0]),
        "eval_crops": len(split_train_eval([None] * len(origins))[1]),
        "rows": rows,
    }


def parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(",") if part.strip())


def best_rows(rows: list[dict], limit: int = 5) -> list[dict]:
    candidates = [row for row in rows if row["mode"] != "center"]
    return sorted(
        candidates,
        key=lambda row: (
            row["signed_log2_rmse_delta_vs_center"],
            row["gradient_delta_vs_center"],
            row["histogram_ks_delta_vs_center"],
        ),
    )[:limit]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="sample_DSCF0009.EXR")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--positions", default="grid9",
                        choices=["top-left", "center", "grid5", "grid9"])
    parser.add_argument("--bits", default="3,4,5,6,7")
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--phases", default="2,4,8")
    parser.add_argument("--index-buckets", default="8,16")
    parser.add_argument("--hash-amplitude", type=float, default=1.0)
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bits_values = parse_ints(args.bits)
    phases = parse_ints(args.phases)
    index_buckets = parse_ints(args.index_buckets)
    summaries = []
    for path in sorted(DATA_DIR.glob(args.glob)):
        summary = summarize_image(
            path,
            args.crop_size,
            args.positions,
            bits_values,
            args.tile_size,
            phases,
            index_buckets,
            args.hash_amplitude,
        )
        summaries.append(summary)
        print(path.name)
        for bits in bits_values:
            rows = [row for row in summary["rows"] if row["bits"] == bits]
            center = next(row for row in rows if row["mode"] == "center")
            print(
                f"  bits{bits}: center log_rmse={center['signed_log2_rmse']:.3e} "
                f"grad={center['gradient_signed_log2_nrmse']:.3e} "
                f"ks={center['histogram_ks_256']:.3e}"
            )
            for row in best_rows(rows, limit=3):
                print(
                    f"    {row['mode']:22s} "
                    f"d_log={row['signed_log2_rmse_delta_vs_center']:+.3e} "
                    f"d_grad={row['gradient_delta_vs_center']:+.3e} "
                    f"d_ks={row['histogram_ks_delta_vs_center']:+.3e} "
                    f"table={row['table_bytes_approx']}B"
                )
        if args.limit and len(summaries) >= args.limit:
            break

    if not args.no_save:
        safe_glob = args.glob.replace("*", "star").replace(".", "_")
        output = RESULTS_DIR / (
            f"noise_synthesis_quality_{safe_glob}"
            f"_crop{args.crop_size}_{args.positions}"
            f"_tile{args.tile_size}_bits{args.bits.replace(',', '-')}"
            f"_phase{args.phases.replace(',', '-')}"
            f"_idx{args.index_buckets.replace(',', '-')}.json"
        )
        output.write_text(json.dumps(summaries, indent=2))
        print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
