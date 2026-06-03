"""Probe a quality-first router across global and local linear-index modes.

This is a decision probe for the next near-lossless codec step.  It compares:

* global per-channel linear index modes, matching StageLinearIndex quality;
* local per-tile/channel linear index modes, which improve local quality but
  usually cost more bits;
* full-image quality after stitching selected tiles, so boundary gradients are
  included in the decision.

The bit counts are estimates, not a production bitstream.  They include range
metadata, per-tile mode selectors, adaptive zero-mask cost, and nonzero residual
entropy.  The goal is to answer whether a router can improve quality enough to
justify a more complex bitstream.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import OpenImageIO as oiio

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"


@dataclass(frozen=True)
class Candidate:
    mode: str
    bits: int
    indices: np.ndarray
    recon: np.ndarray
    total_bits: float
    quality: dict[str, float]


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


def entropy_bits(values: np.ndarray, alphabet: int | None = None) -> float:
    flat = values.reshape(-1).astype(np.int64)
    if flat.size == 0:
        return 0.0
    counts = np.bincount(flat, minlength=alphabet or 0)
    counts = counts[counts > 0]
    if counts.size == 0:
        return 0.0
    probs = counts / counts.sum()
    return float(-(probs * np.log2(probs)).sum() * flat.size)


def avg_predictor(indices: np.ndarray) -> np.ndarray:
    west = np.zeros_like(indices)
    north = np.zeros_like(indices)
    west[:, 1:] = indices[:, :-1]
    north[1:] = indices[:-1]
    return ((west.astype(np.uint32) + north.astype(np.uint32)) // 2).astype(
        np.uint16)


def mask_context(mask: np.ndarray, y: int, x: int, channel: int) -> int:
    context = 0
    if x:
        context |= int(mask[y, x - 1, channel])
    if y:
        context |= int(mask[y - 1, x, channel]) << 1
    if x and y:
        context |= int(mask[y - 1, x - 1, channel]) << 2
    if x + 1 < mask.shape[1] and y:
        context |= int(mask[y - 1, x + 1, channel]) << 3
    if channel:
        context |= int(mask[y, x, channel - 1]) << 4
    phase = ((y & 1) << 1) | (x & 1)
    return context + 32 * (channel + mask.shape[2] * phase)


def context_zero_mask_bits(mask: np.ndarray) -> float:
    counts = np.full((32 * mask.shape[2] * 4, 2), 0.5, dtype=np.float64)
    cost = 0.0
    for channel in range(mask.shape[2]):
        for y in range(mask.shape[0]):
            for x in range(mask.shape[1]):
                context = mask_context(mask, y, x, channel)
                bit = int(mask[y, x, channel])
                total = counts[context].sum()
                cost -= math.log2(counts[context, bit] / total)
                counts[context, bit] += 1.0
    return cost


def estimate_index_bits(indices: np.ndarray, bits: int) -> float:
    alphabet = 1 << bits
    total_bits = 0.0
    residual = np.zeros(indices.shape, dtype=np.uint16)
    for channel in range(indices.shape[2]):
        channel_indices = indices[:, :, channel]
        residual[:, :, channel] = (
            channel_indices.astype(np.int32)
            - avg_predictor(channel_indices).astype(np.int32)
        ) & (alphabet - 1)
    nonzero = residual != 0
    total_bits += context_zero_mask_bits(nonzero)
    for channel in range(indices.shape[2]):
        channel_residual = residual[:, :, channel]
        channel_nonzero = nonzero[:, :, channel]
        nonzero_values = channel_residual[channel_nonzero] - 1
        if nonzero_values.size:
            total_bits += entropy_bits(nonzero_values, alphabet - 1)
    return total_bits


def quantize_linear_global(pixels: np.ndarray, bits: int) -> tuple[np.ndarray, np.ndarray]:
    levels = (1 << bits) - 1
    indices = np.zeros(pixels.shape, dtype=np.uint16)
    recon = np.array(pixels, copy=True)
    for channel in range(pixels.shape[2]):
        values = pixels[:, :, channel].astype(np.float64)
        finite = np.isfinite(values)
        if not bool(np.any(finite)):
            continue
        lo = float(values[finite].min())
        hi = float(values[finite].max())
        if hi <= lo:
            recon[:, :, channel][finite] = np.float32(lo)
            continue
        quantized = np.floor((values - lo) / (hi - lo) * levels + 0.5)
        quantized = np.clip(quantized, 0, levels)
        indices[:, :, channel] = quantized.astype(np.uint16)
        rec = lo + quantized * (hi - lo) / levels
        recon[:, :, channel][finite] = rec[finite].astype(np.float32)
    return indices, np.ascontiguousarray(recon)


def quantize_linear_local(tile: np.ndarray, bits: int) -> tuple[np.ndarray, np.ndarray]:
    levels = (1 << bits) - 1
    indices = np.zeros(tile.shape, dtype=np.uint16)
    recon = np.array(tile, copy=True)
    for channel in range(tile.shape[2]):
        values = tile[:, :, channel].astype(np.float64)
        finite = np.isfinite(values)
        if not bool(np.any(finite)):
            continue
        lo = float(values[finite].min())
        hi = float(values[finite].max())
        if hi <= lo:
            recon[:, :, channel][finite] = np.float32(lo)
            continue
        quantized = np.floor((values - lo) / (hi - lo) * levels + 0.5)
        quantized = np.clip(quantized, 0, levels)
        indices[:, :, channel] = quantized.astype(np.uint16)
        rec = lo + quantized * (hi - lo) / levels
        recon[:, :, channel][finite] = rec[finite].astype(np.float32)
    return indices, np.ascontiguousarray(recon)


def quality_stats(original: np.ndarray, quantized: np.ndarray) -> dict[str, float]:
    finite = np.isfinite(original) & np.isfinite(quantized)
    if not bool(np.any(finite)):
        return {
            "signed_log2_rmse": 0.0,
            "signed_log2_p99": 0.0,
            "abs_p99": 0.0,
            "gradient_signed_log2_nrmse": 0.0,
            "psnr_peak_abs": math.inf,
        }
    selected_original = original[finite].astype(np.float64)
    selected_quantized = quantized[finite].astype(np.float64)
    diff = selected_quantized - selected_original
    diff_abs = np.abs(diff)
    rmse = float(np.sqrt(np.mean(diff * diff)))
    peak = float(np.max(np.abs(selected_original)))
    psnr = math.inf
    if rmse > 0.0 and peak > 0.0:
        psnr = 20.0 * math.log10(peak / rmse)

    signed_log_original = (
        np.sign(selected_original) * np.log2(1.0 + np.abs(selected_original))
    )
    signed_log_quantized = (
        np.sign(selected_quantized) * np.log2(1.0 + np.abs(selected_quantized))
    )
    signed_log_abs = np.abs(signed_log_quantized - signed_log_original)

    full_original = (
        np.sign(original.astype(np.float64))
        * np.log2(1.0 + np.abs(original.astype(np.float64)))
    )
    full_quantized = (
        np.sign(quantized.astype(np.float64))
        * np.log2(1.0 + np.abs(quantized.astype(np.float64)))
    )
    grad_original = np.concatenate(
        [
            np.diff(full_original, axis=0).reshape(-1),
            np.diff(full_original, axis=1).reshape(-1),
        ]
    )
    grad_quantized = np.concatenate(
        [
            np.diff(full_quantized, axis=0).reshape(-1),
            np.diff(full_quantized, axis=1).reshape(-1),
        ]
    )
    grad_finite = np.isfinite(grad_original) & np.isfinite(grad_quantized)
    gradient_nrmse = 0.0
    if bool(np.any(grad_finite)):
        grad_diff = grad_quantized[grad_finite] - grad_original[grad_finite]
        grad_rmse = float(np.sqrt(np.mean(grad_diff * grad_diff)))
        grad_rms = float(np.sqrt(np.mean(grad_original[grad_finite] ** 2)))
        if grad_rms > 0.0:
            gradient_nrmse = grad_rmse / grad_rms

    return {
        "signed_log2_rmse": float(np.sqrt(np.mean(signed_log_abs ** 2))),
        "signed_log2_p99": float(np.percentile(signed_log_abs, 99.0)),
        "abs_p99": float(np.percentile(diff_abs, 99.0)),
        "gradient_signed_log2_nrmse": gradient_nrmse,
        "psnr_peak_abs": psnr,
    }


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


def quality_violation(
    quality: dict[str, float],
    log_rmse_threshold: float,
    log_p99_threshold: float,
    gradient_threshold: float,
) -> float:
    values = [
        quality["signed_log2_rmse"] / max(log_rmse_threshold, 1e-12),
        quality["signed_log2_p99"] / max(log_p99_threshold, 1e-12),
        quality["gradient_signed_log2_nrmse"] / max(gradient_threshold, 1e-12),
    ]
    return max(values)


def choose_candidate(
    candidates: list[Candidate],
    log_rmse_threshold: float,
    log_p99_threshold: float,
    gradient_threshold: float,
) -> Candidate:
    passing = [
        candidate for candidate in candidates
        if quality_passes(
            candidate.quality,
            log_rmse_threshold,
            log_p99_threshold,
            gradient_threshold,
        )
    ]
    if passing:
        return min(passing, key=lambda candidate: candidate.total_bits)
    return min(
        candidates,
        key=lambda candidate: (
            quality_violation(
                candidate.quality,
                log_rmse_threshold,
                log_p99_threshold,
                gradient_threshold,
            ),
            candidate.total_bits,
        ),
    )


def make_candidates(
    tile: np.ndarray,
    global_indices: dict[int, np.ndarray],
    global_recon: dict[int, np.ndarray],
    y: int,
    x: int,
    global_bits: tuple[int, ...],
    local_bits: tuple[int, ...],
    selector_bits: float,
) -> list[Candidate]:
    candidates = []
    for bits in global_bits:
        indices = global_indices[bits][y:y + tile.shape[0], x:x + tile.shape[1], :]
        recon = global_recon[bits][y:y + tile.shape[0], x:x + tile.shape[1], :]
        candidates.append(Candidate(
            mode="global",
            bits=bits,
            indices=indices,
            recon=np.ascontiguousarray(recon),
            total_bits=estimate_index_bits(indices, bits) + selector_bits,
            quality=quality_stats(tile, recon),
        ))
    for bits in local_bits:
        indices, recon = quantize_linear_local(tile, bits)
        metadata_bits = tile.shape[2] * 2 * 32
        candidates.append(Candidate(
            mode="local",
            bits=bits,
            indices=indices,
            recon=recon,
            total_bits=estimate_index_bits(indices, bits) + metadata_bits + selector_bits,
            quality=quality_stats(tile, recon),
        ))
    return candidates


def summarize_image(
    path: Path,
    crop_size: int,
    tile_size: int,
    global_bits: tuple[int, ...],
    local_bits: tuple[int, ...],
    log_rmse_threshold: float,
    log_p99_threshold: float,
    gradient_threshold: float,
) -> dict:
    pixels = read_exr(path, crop_size)
    global_indices = {}
    global_recon = {}
    for bits in global_bits:
        global_indices[bits], global_recon[bits] = quantize_linear_global(pixels, bits)

    selected_recon = np.array(pixels, copy=True)
    selected_counter: Counter[str] = Counter()
    tile_rows = []
    total_bits = float(pixels.shape[2] * 2 * 32) if global_bits else 0.0
    selector_bits = 8.0
    worst_tile_violation = 0.0
    fallback_tiles = 0

    for y in range(0, pixels.shape[0], tile_size):
        for x in range(0, pixels.shape[1], tile_size):
            tile = pixels[y:y + tile_size, x:x + tile_size, :]
            candidates = make_candidates(
                tile,
                global_indices,
                global_recon,
                y,
                x,
                global_bits,
                local_bits,
                selector_bits,
            )
            selected = choose_candidate(
                candidates,
                log_rmse_threshold,
                log_p99_threshold,
                gradient_threshold,
            )
            selected_recon[y:y + tile.shape[0], x:x + tile.shape[1], :] = selected.recon
            total_bits += selected.total_bits
            key = f"{selected.mode}{selected.bits}"
            selected_counter[key] += 1
            violation = quality_violation(
                selected.quality,
                log_rmse_threshold,
                log_p99_threshold,
                gradient_threshold,
            )
            worst_tile_violation = max(worst_tile_violation, violation)
            if not quality_passes(
                selected.quality,
                log_rmse_threshold,
                log_p99_threshold,
                gradient_threshold,
            ):
                fallback_tiles += 1
            tile_rows.append({
                "x": x,
                "y": y,
                "mode": selected.mode,
                "bits": selected.bits,
                "estimated_bits": selected.total_bits,
                "ratio_vs_float32": (
                    tile.nbytes * 8.0 / selected.total_bits
                    if selected.total_bits > 0.0 else math.inf
                ),
                **selected.quality,
                "quality_violation": violation,
            })

    full_quality = quality_stats(pixels, selected_recon)
    ratio = pixels.nbytes * 8.0 / total_bits if total_bits > 0.0 else math.inf
    baselines = []
    for bits in global_bits:
        q = quality_stats(pixels, global_recon[bits])
        bps = (
            estimate_index_bits(global_indices[bits], bits)
            + pixels.shape[2] * 2 * 32
        ) / global_indices[bits].size
        baselines.append({
            "mode": "global",
            "bits": bits,
            "estimated_ratio_vs_float32": 32.0 / bps if bps > 0.0 else math.inf,
            **q,
        })

    return {
        "image": path.name,
        "shape": list(pixels.shape),
        "crop_size": crop_size,
        "tile_size": tile_size,
        "global_bits": list(global_bits),
        "local_bits": list(local_bits),
        "log_rmse_threshold": log_rmse_threshold,
        "log_p99_threshold": log_p99_threshold,
        "gradient_threshold": gradient_threshold,
        "estimated_bits": total_bits,
        "estimated_bytes": math.ceil(total_bits / 8.0),
        "estimated_ratio_vs_float32": ratio,
        "selected_modes": dict(sorted(selected_counter.items())),
        "fallback_tiles": fallback_tiles,
        "worst_tile_quality_violation": worst_tile_violation,
        "full_quality": full_quality,
        "baselines": baselines,
        "tiles": tile_rows,
    }


def parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(",") if part.strip())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="sample_DSCF0009.EXR")
    parser.add_argument("--crop-size", type=int, default=1024)
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--global-bits", default="7,8,10")
    parser.add_argument("--local-bits", default="4,5,6,7,8,10,12")
    parser.add_argument("--log-rmse-threshold", type=float, default=0.004)
    parser.add_argument("--log-p99-threshold", type=float, default=0.018)
    parser.add_argument("--gradient-threshold", type=float, default=0.5)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    global_bits = parse_ints(args.global_bits)
    local_bits = parse_ints(args.local_bits)
    rows = []
    for path in sorted(DATA_DIR.glob(args.glob)):
        row = summarize_image(
            path,
            args.crop_size,
            args.tile_size,
            global_bits,
            local_bits,
            args.log_rmse_threshold,
            args.log_p99_threshold,
            args.gradient_threshold,
        )
        rows.append(row)
        q = row["full_quality"]
        print(path.name)
        print(
            f"  router={row['estimated_ratio_vs_float32']:.2f}x "
            f"bytes={row['estimated_bytes']} "
            f"modes={row['selected_modes']} "
            f"fallback_tiles={row['fallback_tiles']} "
            f"log_rmse={q['signed_log2_rmse']:.3e} "
            f"log_p99={q['signed_log2_p99']:.3e} "
            f"grad={q['gradient_signed_log2_nrmse']:.3e} "
            f"psnr={q['psnr_peak_abs']:.2f}dB")
        for baseline in row["baselines"]:
            print(
                f"    baseline {baseline['mode']}{baseline['bits']}: "
                f"{baseline['estimated_ratio_vs_float32']:.2f}x "
                f"log_rmse={baseline['signed_log2_rmse']:.3e} "
                f"log_p99={baseline['signed_log2_p99']:.3e} "
                f"grad={baseline['gradient_signed_log2_nrmse']:.3e} "
                f"psnr={baseline['psnr_peak_abs']:.2f}dB")

    if not args.no_save:
        safe_glob = args.glob.replace("*", "star").replace(".", "_")
        output = RESULTS_DIR / (
            f"quality_router_modes_{safe_glob}"
            f"_crop{args.crop_size}_tile{args.tile_size}"
            f"_g{args.global_bits.replace(',', '-')}"
            f"_l{args.local_bits.replace(',', '-')}"
            f"_lr{args.log_rmse_threshold:g}"
            f"_lp{args.log_p99_threshold:g}"
            f"_gr{args.gradient_threshold:g}.json"
        )
        output.write_text(json.dumps(rows, indent=2))
        print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
