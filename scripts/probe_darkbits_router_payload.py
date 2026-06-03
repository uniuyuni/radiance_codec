"""Estimate payload for base signed-log bits plus dark-region refinement.

This is a sizing probe, not a bitstream implementation.  It assumes a route
that stores a normal StageLinearIndex stream for all pixels, then adds a binary
dark mask and a small correction stream that refines only dark pixels to a
higher bit depth.
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
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
sys.path.insert(0, str(ROOT / "codec" / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

import radiance_codec  # noqa: E402
from benchmark_linear_index_codec import error_stats  # noqa: E402
from probe_power_recon_table import table_reconstruct  # noqa: E402


PROB_BITS = 14
PROB_SCALE = 1 << PROB_BITS
STREAM_FLUSH_BYTES = 4
ROUTER_HEADER_BYTES = 64


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
    values64 = values.astype(np.float64)
    return np.sign(values64) * np.log2(1.0 + np.abs(values64))


def inverse_signed_log(values: np.ndarray) -> np.ndarray:
    values64 = values.astype(np.float64)
    return np.sign(values64) * (np.exp2(np.abs(values64)) - 1.0)


def transform_values(values: np.ndarray, transform: str, power_gamma: float) -> np.ndarray:
    values64 = values.astype(np.float64)
    if transform == "linear":
        return values64
    if transform == "signed-log":
        return signed_log(values64)
    if transform == "gamma075":
        return np.sign(values64) * np.power(np.abs(values64), 0.75)
    if transform == "power":
        return np.sign(values64) * np.power(np.abs(values64), float(power_gamma))
    if transform == "asinh":
        return np.arcsinh(values64)
    raise ValueError(f"unknown transform: {transform}")


def inverse_transform_values(values: np.ndarray, transform: str, power_gamma: float) -> np.ndarray:
    values64 = values.astype(np.float64)
    if transform == "linear":
        return values64
    if transform == "signed-log":
        return inverse_signed_log(values64)
    if transform == "gamma075":
        return np.sign(values64) * np.power(np.abs(values64), 1.0 / 0.75)
    if transform == "power":
        return np.sign(values64) * np.power(np.abs(values64), 1.0 / float(power_gamma))
    if transform == "asinh":
        return np.sinh(values64)
    raise ValueError(f"unknown transform: {transform}")


def dark_luma_mask(pixels: np.ndarray, dark_max: float) -> np.ndarray:
    rgb = pixels[:, :, :3].astype(np.float64)
    luma = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    return luma <= float(dark_max)


def local_std(values: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return np.zeros(values.shape, dtype=np.float64)
    padded = np.pad(values.astype(np.float64, copy=False), radius, mode="reflect")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    squared = padded * padded
    integral_sq = np.pad(squared, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    size = 2 * radius + 1
    total = (
        integral[size:, size:]
        - integral[:-size, size:]
        - integral[size:, :-size]
        + integral[:-size, :-size]
    )
    total_sq = (
        integral_sq[size:, size:]
        - integral_sq[:-size, size:]
        - integral_sq[size:, :-size]
        + integral_sq[:-size, :-size]
    )
    mean = total / float(size * size)
    mean_sq = total_sq / float(size * size)
    return np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))


def dark_router_mask(
    pixels: np.ndarray,
    base_bits: int,
    dark_max: float,
    mode: str,
    radius: int,
    std_step_threshold: float,
    transform: str,
    power_gamma: float,
) -> np.ndarray:
    luma_mask = dark_luma_mask(pixels, dark_max)
    if mode == "luma":
        return luma_mask
    if mode != "banding":
        raise ValueError(f"unknown mask mode: {mode}")

    levels = (1 << base_bits) - 1
    risk = np.zeros(pixels.shape[:2], dtype=bool)
    for channel in range(pixels.shape[2]):
        transformed = transform_values(pixels[:, :, channel], transform, power_gamma)
        range_values = transformed.astype(np.float32).astype(np.float64)
        lo = float(np.min(range_values))
        hi = float(np.max(range_values))
        if not hi > lo:
            continue
        step = (hi - lo) / levels
        risk |= local_std(transformed, radius) < float(std_step_threshold) * float(step)
    return luma_mask & risk


def quantize_signed_log(
    pixels: np.ndarray,
    bits: int,
    transform: str,
    power_gamma: float,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    levels = (1 << bits) - 1
    indices = np.zeros(pixels.shape, dtype=np.uint16)
    decoded = np.zeros(pixels.shape, dtype=np.float32)
    ranges = []
    for channel in range(pixels.shape[2]):
        transformed = transform_values(pixels[:, :, channel], transform, power_gamma)
        range_values = transformed.astype(np.float32).astype(np.float64)
        lo = float(np.min(range_values))
        hi = float(np.max(range_values))
        ranges.append({"lo": lo, "hi": hi})
        if not hi > lo:
            decoded[:, :, channel] = inverse_transform_values(
                np.asarray(lo),
                transform,
                power_gamma,
            ).astype(np.float32)
            continue
        q = np.floor((transformed - lo) / (hi - lo) * levels + 0.5)
        q = np.clip(q, 0, levels).astype(np.uint16)
        indices[:, :, channel] = q
        rec_t = lo + q.astype(np.float64) * (hi - lo) / levels
        decoded[:, :, channel] = inverse_transform_values(
            rec_t,
            transform,
            power_gamma,
        ).astype(np.float32)
    return indices, np.ascontiguousarray(decoded), ranges


def reconstruct_mixed(
    pixels: np.ndarray,
    base_bits: int,
    dark_bits: int,
    dark_max: float,
    mask_mode: str,
    mask_radius: int,
    std_step_threshold: float,
    transform: str,
    power_gamma: float,
    recon_table: str,
) -> tuple[np.ndarray, dict]:
    base_indices, base_decoded, base_ranges = quantize_signed_log(
        pixels,
        base_bits,
        transform,
        power_gamma,
    )
    dark_indices, dark_decoded, dark_ranges = quantize_signed_log(
        pixels,
        dark_bits,
        transform,
        power_gamma,
    )
    if recon_table != "center":
        base_decoded = table_reconstruct(pixels, base_indices, base_bits, recon_table)
        dark_decoded = table_reconstruct(pixels, dark_indices, dark_bits, recon_table)
    mask = dark_router_mask(
        pixels,
        base_bits,
        dark_max,
        mask_mode,
        mask_radius,
        std_step_threshold,
        transform,
        power_gamma,
    )
    decoded = np.array(base_decoded, copy=True)
    for channel in range(pixels.shape[2]):
        decoded[:, :, channel][mask] = dark_decoded[:, :, channel][mask]
    return np.ascontiguousarray(decoded), {
        "base_indices": base_indices,
        "dark_indices": dark_indices,
        "mask": mask,
        "base_ranges": base_ranges,
        "dark_ranges": dark_ranges,
    }


def quantized_finite_bits(counts: np.ndarray) -> float:
    total = int(counts.sum())
    if total <= 0:
        return 0.0
    seen = counts > 0
    freqs = np.zeros(counts.shape, dtype=np.int64)
    raw = (counts[seen] * PROB_SCALE + total // 2) // total
    raw = np.clip(raw, 1, PROB_SCALE - 1)
    freqs[seen] = raw
    assigned = int(freqs.sum())
    while assigned > PROB_SCALE:
        best = int(np.argmax(freqs))
        freqs[best] -= 1
        assigned -= 1
    while assigned < PROB_SCALE:
        best = int(np.argmax(freqs))
        freqs[best] += 1
        assigned += 1
    nz = counts > 0
    return float(
        (counts[nz].astype(np.float64)
         * (PROB_BITS - np.log2(freqs[nz].astype(np.float64)))).sum()
    )


def summarize_counts(counts: np.ndarray, model_bytes: int) -> dict:
    sample_count = int(counts.sum())
    finite_bits = quantized_finite_bits(counts)
    payload_bytes = int(math.ceil(finite_bits / 8.0)) + STREAM_FLUSH_BYTES
    nz = counts[counts > 0].astype(np.float64)
    ideal_bits = 0.0
    if sample_count:
        ideal_bits = float(-(nz * np.log2(nz / float(sample_count))).sum())
    return {
        "sample_count": sample_count,
        "ideal_bits": ideal_bits,
        "finite_bits": finite_bits,
        "bits_per_sample": 0.0 if sample_count == 0 else finite_bits / sample_count,
        "payload_bytes": payload_bytes,
        "model_bytes": model_bytes,
        "total_bytes": payload_bytes + model_bytes,
    }


def summarize_binary_mask(mask: np.ndarray) -> dict:
    mask_u8 = mask.astype(np.uint8)
    order0 = np.bincount(mask_u8.reshape(-1), minlength=2).astype(np.int64)

    west = np.zeros(mask_u8.shape, dtype=np.uint8)
    north = np.zeros(mask_u8.shape, dtype=np.uint8)
    west[:, 1:] = mask_u8[:, :-1]
    north[1:, :] = mask_u8[:-1, :]
    context = west + 2 * north
    joint = np.bincount(
        (context.reshape(-1).astype(np.int64) * 2) + mask_u8.reshape(-1).astype(np.int64),
        minlength=8,
    ).reshape(4, 2)

    context_bits = 0.0
    for row in joint:
        context_bits += quantized_finite_bits(row)
    context_payload = int(math.ceil(context_bits / 8.0)) + STREAM_FLUSH_BYTES
    return {
        "dark_pixel_rate": float(np.mean(mask)),
        "order0": summarize_counts(order0, model_bytes=2 * 2),
        "west_north": {
            "sample_count": int(joint.sum()),
            "finite_bits": context_bits,
            "bits_per_sample": context_bits / float(joint.sum()),
            "payload_bytes": context_payload,
            "model_bytes": 4 * 2 * 2,
            "total_bytes": context_payload + 4 * 2 * 2,
        },
    }


def summarize_refinement(
    base_indices: np.ndarray,
    dark_indices: np.ndarray,
    mask: np.ndarray,
    base_bits: int,
    dark_bits: int,
) -> dict:
    base_levels = (1 << base_bits) - 1
    dark_levels = (1 << dark_bits) - 1
    predicted_dark = np.rint(
        base_indices.astype(np.float64) * dark_levels / float(base_levels)
    ).astype(np.int32)
    correction = dark_indices.astype(np.int32) - predicted_dark
    selected = correction[mask, :].reshape(-1)
    if selected.size == 0:
        counts = np.zeros(1, dtype=np.int64)
    else:
        lo = int(np.min(selected))
        hi = int(np.max(selected))
        counts = np.bincount(selected - lo, minlength=hi - lo + 1).astype(np.int64)
    model_bytes = int(counts.size * 2 + 8)
    summary = summarize_counts(counts, model_bytes=model_bytes)
    summary.update({
        "correction_min": None if selected.size == 0 else int(np.min(selected)),
        "correction_max": None if selected.size == 0 else int(np.max(selected)),
        "correction_alphabet": int(counts.size),
    })
    return summary


def summarize_case(
    path: Path,
    pixels: np.ndarray,
    base_bits: int,
    dark_bits: int,
    dark_max: float,
    mask_mode: str,
    mask_radius: int,
    std_step_threshold: float,
    transform: str,
    power_gamma: float,
    recon_table: str,
    encode_base: bool,
    base_estimated_bytes: int,
) -> dict:
    started = time.perf_counter()
    decoded, route = reconstruct_mixed(
        pixels,
        base_bits,
        dark_bits,
        dark_max,
        mask_mode,
        mask_radius,
        std_step_threshold,
        transform,
        power_gamma,
        recon_table,
    )
    base_encoded_bytes = base_estimated_bytes if base_estimated_bytes > 0 else None
    base_encode_seconds = None
    if encode_base:
        if transform == "power":
            raise ValueError("--no-encode-base is required for --transform power")
        t0 = time.perf_counter()
        encoded = radiance_codec.encode_linear_index_near_lossless(
            pixels,
            bits=base_bits,
            effort=9,
            transform=transform,
        )
        base_encode_seconds = time.perf_counter() - t0
        base_encoded_bytes = len(encoded)

    mask_summary = summarize_binary_mask(route["mask"])
    refinement = summarize_refinement(
        route["base_indices"],
        route["dark_indices"],
        route["mask"],
        base_bits,
        dark_bits,
    )
    mask_best = min(
        mask_summary["order0"],
        mask_summary["west_north"],
        key=lambda row: row["total_bytes"],
    )
    estimate_bytes = None
    if base_encoded_bytes is not None:
        estimate_bytes = (
            base_encoded_bytes
            + ROUTER_HEADER_BYTES
            + int(mask_best["total_bytes"])
            + int(refinement["total_bytes"])
        )
    raw_bytes = int(pixels.nbytes)
    return {
        "image": path.name,
        "shape": list(pixels.shape),
        "raw_bytes": raw_bytes,
        "base_bits": base_bits,
        "dark_bits": dark_bits,
        "dark_max": dark_max,
        "transform": transform,
        "power_gamma": power_gamma,
        "recon_table": recon_table,
        "mask_mode": mask_mode,
        "mask_radius": mask_radius,
        "std_step_threshold": std_step_threshold,
        "quality": error_stats(pixels, decoded),
        "base_encoded_bytes": base_encoded_bytes,
        "base_encode_seconds": base_encode_seconds,
        "mask": mask_summary,
        "refinement": refinement,
        "router_header_bytes": ROUTER_HEADER_BYTES,
        "estimated_bytes": estimate_bytes,
        "estimated_ratio": None if estimate_bytes is None else raw_bytes / estimate_bytes,
        "elapsed_sec": time.perf_counter() - started,
    }


def parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(",") if part.strip())


def parse_floats(text: str) -> tuple[float, ...]:
    return tuple(float(part) for part in text.split(",") if part.strip())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="sample_DSCF0009.EXR")
    parser.add_argument("--crop-size", type=int, default=1024)
    parser.add_argument("--base-bits", default="8,9")
    parser.add_argument("--dark-bits", type=int, default=10)
    parser.add_argument("--dark-max", default="0.18,0.25")
    parser.add_argument("--transform", choices=("linear", "signed-log", "gamma075", "asinh", "power"), default="signed-log")
    parser.add_argument("--power-gamma", type=float, default=0.8)
    parser.add_argument(
        "--recon-table",
        choices=("center", "value-mean", "signed-log-mean"),
        default="center",
    )
    parser.add_argument("--base-estimated-bytes", type=int, default=0)
    parser.add_argument("--mask-mode", choices=("luma", "banding"), default="luma")
    parser.add_argument("--mask-radius", type=int, default=2)
    parser.add_argument("--std-step-threshold", type=float, default=0.25)
    parser.add_argument("--no-encode-base", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    rows = []
    base_bits_values = parse_ints(args.base_bits)
    dark_max_values = parse_floats(args.dark_max)
    for path in sorted(DATA_DIR.glob(args.glob)):
        pixels = read_exr(path, args.crop_size)
        print(path.name, flush=True)
        for base_bits in base_bits_values:
            for dark_max in dark_max_values:
                row = summarize_case(
                    path,
                    pixels,
                    base_bits,
                    args.dark_bits,
                    dark_max,
                    args.mask_mode,
                    args.mask_radius,
                    args.std_step_threshold,
                    args.transform,
                    args.power_gamma,
                    args.recon_table,
                    not args.no_encode_base,
                    args.base_estimated_bytes,
                )
                rows.append(row)
                q = row["quality"]
                estimate = row["estimated_bytes"]
                estimate_text = "n/a" if estimate is None else f"{estimate:,}"
                ratio_text = "n/a" if row["estimated_ratio"] is None else f"{row['estimated_ratio']:.2f}x"
                mask_best = min(
                    row["mask"]["order0"],
                    row["mask"]["west_north"],
                    key=lambda item: item["total_bytes"],
                )
                print(
                    f"  bits{base_bits}+dark{args.dark_bits} max{dark_max:g}: "
                    f"{args.transform} "
                    f"gamma={args.power_gamma:g} "
                    f"recon={args.recon_table} "
                    f"mask={args.mask_mode} "
                    f"est={estimate_text} ratio={ratio_text} "
                    f"dark={row['mask']['dark_pixel_rate']:.2%} "
                    f"mask={mask_best['total_bytes']:,} "
                    f"refine={row['refinement']['total_bytes']:,} "
                    f"log={q['signed_log2_rmse']:.3e} "
                    f"grad={q['gradient_signed_log2_nrmse']:.3e}",
                    flush=True,
                )
        if args.limit and len(rows) >= args.limit:
            break

    if not args.no_save:
        safe_glob = args.glob.replace("*", "star").replace(".", "_")
        output = RESULTS_DIR / (
            f"darkbits_router_payload_{safe_glob}"
            f"_crop{args.crop_size}_base{args.base_bits.replace(',', '-')}"
            f"_{args.transform}_g{args.power_gamma:g}_{args.recon_table}_dark{args.dark_bits}"
            f"_max{args.dark_max.replace(',', '-')}.json"
        )
        if args.mask_mode != "luma":
            output = RESULTS_DIR / (
                f"darkbits_router_payload_{safe_glob}"
                f"_crop{args.crop_size}_base{args.base_bits.replace(',', '-')}"
                f"_{args.transform}_g{args.power_gamma:g}_{args.recon_table}_dark{args.dark_bits}"
                f"_max{args.dark_max.replace(',', '-')}"
                f"_mask{args.mask_mode}_std{args.std_step_threshold:g}.json"
            )
        output.write_text(json.dumps(rows, indent=2))
        print(f"\nSaved: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
