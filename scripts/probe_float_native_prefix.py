"""Probe near-lossless float32-native mantissa prefix routes.

This revisits the original float32 bit split from a near-lossless angle.  The
lossless tail studies showed that raw low mantissa bits are the hard wall, but
near-lossless coding can choose not to store that tail.  This script asks:

* How many high mantissa bits must remain for the proxy visual judge?
* How much entropy is left in sign/exponent/mantissa-prefix streams?
* Do order-preserving float bits, spatial prediction, or channel deltas expose
  enough structure to be worth implementing?
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
sys.path.insert(0, str(ROOT / "codec" / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

import radiance_codec  # noqa: E402
from audit_display_quality_regions import read_exr  # noqa: E402
from audit_visual_gate import (  # noqa: E402
    candidate_metrics,
    evaluate_gate,
    summarize_key_metrics,
)
from benchmark_linear_index_codec import error_stats  # noqa: E402


def parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(",") if part.strip())


def entropy_bits(values: np.ndarray) -> float:
    flat = np.ascontiguousarray(values).reshape(-1)
    if flat.size == 0:
        return 0.0
    _, counts = np.unique(flat, return_counts=True)
    probs = counts.astype(np.float64) / float(flat.size)
    return float(-np.sum(counts.astype(np.float64) * np.log2(probs)))


def entropy_bps(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return entropy_bits(values) / float(values.size)


def ordered_float_bits(bits: np.ndarray) -> np.ndarray:
    sign = (bits >> np.uint32(31)) != 0
    return np.where(sign, ~bits, bits ^ np.uint32(0x80000000)).astype(np.uint32)


def jpeg_ls_predict(values: np.ndarray) -> np.ndarray:
    work = values.astype(np.int64, copy=False)
    west = np.zeros(work.shape, dtype=np.int64)
    north = np.zeros(work.shape, dtype=np.int64)
    northwest = np.zeros(work.shape, dtype=np.int64)
    west[:, 1:, :] = work[:, :-1, :]
    north[1:, :, :] = work[:-1, :, :]
    northwest[1:, 1:, :] = work[:-1, :-1, :]
    lo = np.minimum(west, north)
    hi = np.maximum(west, north)
    return np.where(northwest >= hi, lo, np.where(northwest <= lo, hi, west + north - northwest))


def med_residual(values: np.ndarray, width: int) -> np.ndarray:
    if width >= 63:
        raise ValueError("width too large for int64 residuals")
    modulus_mask = np.int64((1 << width) - 1)
    prediction = jpeg_ls_predict(values)
    residual = (values.astype(np.int64, copy=False) - prediction) & modulus_mask
    return residual.astype(np.uint32)


def green_delta(values: np.ndarray, width: int) -> np.ndarray:
    if values.shape[2] < 3:
        return values
    modulus_mask = np.int64((1 << width) - 1)
    work = values.astype(np.int64, copy=False)
    out = np.array(work, copy=True)
    green = work[:, :, 1]
    out[:, :, 0] = (work[:, :, 0] - green) & modulus_mask
    out[:, :, 1] = green
    out[:, :, 2] = (work[:, :, 2] - green) & modulus_mask
    return out.astype(np.uint32)


def mantissa_prefix_reconstruct(
    pixels: np.ndarray,
    keep_bits: int,
    recon: str,
) -> np.ndarray:
    keep_bits = min(23, max(0, int(keep_bits)))
    low_bits = 23 - keep_bits
    bits = np.ascontiguousarray(pixels, dtype=np.float32).view(np.uint32)
    if low_bits == 0:
        return bits.copy().view(np.float32).reshape(pixels.shape)
    low_mask = np.uint32((1 << low_bits) - 1)
    out = bits & np.uint32(0xFFFFFFFF ^ int(low_mask))
    if recon == "center":
        exponent = (bits >> np.uint32(23)) & np.uint32(0xFF)
        finite_normal = (exponent != np.uint32(0xFF)) & (exponent != np.uint32(0))
        out[finite_normal] |= np.uint32(1 << (low_bits - 1))
    elif recon != "zero":
        raise ValueError(f"unknown recon mode: {recon}")
    return np.ascontiguousarray(out.view(np.float32).reshape(pixels.shape))


def prefix_payloads(bits: np.ndarray, keep_bits: int) -> dict[str, np.ndarray]:
    low_bits = 23 - keep_bits
    raw_prefix = (bits >> np.uint32(low_bits)).astype(np.uint32)
    ordered_prefix = (ordered_float_bits(bits) >> np.uint32(low_bits)).astype(np.uint32)
    return {
        "raw": raw_prefix,
        "ordered": ordered_prefix,
    }


def estimate_prefix_costs(bits: np.ndarray, keep_bits: int) -> dict:
    width = 9 + int(keep_bits)
    payloads = prefix_payloads(bits, keep_bits)
    raw = payloads["raw"]
    ordered = payloads["ordered"]
    ordered_green = green_delta(ordered, width)
    rows = {
        "raw_order0": entropy_bps(raw),
        "ordered_order0": entropy_bps(ordered),
        "ordered_med": entropy_bps(med_residual(ordered, width)),
        "ordered_green_delta_order0": entropy_bps(ordered_green),
        "ordered_green_delta_med": entropy_bps(med_residual(ordered_green, width)),
    }
    best_mode = min(rows, key=rows.get)
    return {
        "prefix_width_bits": width,
        "entropy_bits_per_sample": rows,
        "best_entropy_mode": best_mode,
        "best_bits_per_sample": rows[best_mode],
    }


def tail_entropy(bits: np.ndarray, keep_bits: int) -> dict:
    low_bits = 23 - int(keep_bits)
    if low_bits <= 0:
        return {
            "tail_width_bits": 0,
            "tail_bits_per_sample": 0.0,
            "tail_zero_rate": 1.0,
        }
    mask = np.uint32((1 << low_bits) - 1)
    tail = bits & mask
    return {
        "tail_width_bits": low_bits,
        "tail_bits_per_sample": entropy_bps(tail),
        "tail_zero_rate": float(np.mean(tail == 0)),
    }


def sign_exponent_summary(bits: np.ndarray) -> dict:
    sign = ((bits >> np.uint32(31)) & np.uint32(1)).astype(np.uint8)
    exponent = ((bits >> np.uint32(23)) & np.uint32(0xFF)).astype(np.uint16)
    finite = exponent != np.uint16(0xFF)
    positive = int(np.count_nonzero(sign == 0))
    negative = int(np.count_nonzero(sign == 1))
    return {
        "sign_class": (
            "all_positive" if negative == 0 else "all_negative" if positive == 0 else "mixed"
        ),
        "positive_samples": positive,
        "negative_samples": negative,
        "finite_rate": float(np.mean(finite)),
        "zero_or_subnormal_rate": float(np.mean(exponent == 0)),
        "sign_entropy_bits_per_sample": entropy_bps(sign),
        "exponent_entropy_bits_per_sample": entropy_bps(exponent),
        "sign_exponent_entropy_bits_per_sample": entropy_bps(
            ((sign.astype(np.uint16) << np.uint16(8)) | exponent).astype(np.uint16)
        ),
    }


def gate_defaults() -> SimpleNamespace:
    return SimpleNamespace(
        dark_lost_warn=0.04,
        dark_lost_reject=0.10,
        dark_grad_ratio_warn=1.45,
        dark_grad_ratio_reject=2.1,
        dark_grad_abs_warn=0.035,
        dark_grad_abs_reject=0.08,
        mid_lost_warn=0.06,
        mid_lost_reject=0.12,
        highlight_lost_warn=0.035,
        highlight_lost_reject=0.075,
        extreme_lost_warn=0.06,
        extreme_lost_reject=0.14,
        highlight_energy_min=0.94,
        highlight_energy_max=1.06,
        min_gradient_samples=1024,
    )


def summarize_keep_bits(
    pixels: np.ndarray,
    bits: np.ndarray,
    keep_bits: int,
    recon: str,
    anchor_regions: dict,
    white: float,
    gamma: float,
) -> dict:
    reconstructed = mantissa_prefix_reconstruct(pixels, keep_bits, recon)
    regions = candidate_metrics(pixels, reconstructed, white, gamma)
    gate = evaluate_gate(regions, anchor_regions, gate_defaults())
    quality = error_stats(pixels, reconstructed)
    costs = estimate_prefix_costs(bits, keep_bits)
    raw_bytes = int(pixels.nbytes)
    estimated_bytes = int(math.ceil(costs["best_bits_per_sample"] * pixels.size / 8.0))
    return {
        "keep_mantissa_bits": int(keep_bits),
        "dropped_mantissa_bits": int(23 - keep_bits),
        "recon": recon,
        "estimated_best_prefix_bytes": estimated_bytes,
        "estimated_ratio_vs_float32": (
            raw_bytes / float(estimated_bytes) if estimated_bytes > 0 else math.inf
        ),
        "prefix_costs": costs,
        "discarded_tail": tail_entropy(bits, keep_bits),
        "gate": gate,
        "key_metrics": summarize_key_metrics(regions, anchor_regions),
        "quality": quality,
    }


def summarize_image(
    path: Path,
    crop_size: int,
    keep_values: tuple[int, ...],
    recon: str,
    white: float,
    gamma: float,
) -> dict:
    started = time.perf_counter()
    pixels = read_exr(path, crop_size)
    pixels = np.ascontiguousarray(pixels, dtype=np.float32)
    bits = pixels.view(np.uint32)
    anchor = radiance_codec.quantize_linear_index(
        pixels,
        bits=10,
        transform="signed-log",
    )
    anchor_regions = candidate_metrics(pixels, anchor, white, gamma)
    rows = [
        summarize_keep_bits(
            pixels,
            bits,
            keep_bits,
            recon,
            anchor_regions,
            white,
            gamma,
        )
        for keep_bits in keep_values
    ]
    best_by_gate = next(
        (
            row
            for row in sorted(rows, key=lambda item: item["estimated_best_prefix_bytes"])
            if row["gate"]["decision"] != "reject"
        ),
        None,
    )
    return {
        "image": path.name,
        "shape": list(pixels.shape),
        "crop_size": crop_size,
        "white": white,
        "gamma": gamma,
        "anchor": "quant:signed-log:10",
        "field_summary": sign_exponent_summary(bits),
        "best_non_reject_keep_bits": (
            None if best_by_gate is None else best_by_gate["keep_mantissa_bits"]
        ),
        "best_non_reject_estimated_bytes": (
            None if best_by_gate is None else best_by_gate["estimated_best_prefix_bytes"]
        ),
        "elapsed_sec": time.perf_counter() - started,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="sample_*.EXR")
    parser.add_argument("--crop-size", type=int, default=1024)
    parser.add_argument("--keep-bits", default="6,7,8,9,10,11,12,13,14")
    parser.add_argument("--recon", choices=("center", "zero"), default="center")
    parser.add_argument("--white", type=float, default=4.0)
    parser.add_argument("--gamma", type=float, default=2.2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    keep_values = parse_ints(args.keep_bits)
    summaries = []
    for path in sorted(DATA_DIR.glob(args.glob)):
        summary = summarize_image(
            path,
            args.crop_size,
            keep_values,
            args.recon,
            args.white,
            args.gamma,
        )
        summaries.append(summary)
        fields = summary["field_summary"]
        print(
            f"{path.name}: sign={fields['sign_class']} "
            f"expH={fields['exponent_entropy_bits_per_sample']:.3f} "
            f"best_non_reject={summary['best_non_reject_keep_bits']}",
            flush=True,
        )
        for row in summary["rows"]:
            gate = row["gate"]["decision"]
            costs = row["prefix_costs"]
            dark = row["key_metrics"]["dark_0_0.25"]
            high = row["key_metrics"]["highlight_1_4"]
            print(
                f"  {gate.upper():6s} keep{row['keep_mantissa_bits']:02d} "
                f"est={row['estimated_best_prefix_bytes']:,} "
                f"ratio={row['estimated_ratio_vs_float32']:.2f}x "
                f"bps={costs['best_bits_per_sample']:.3f} "
                f"mode={costs['best_entropy_mode']} "
                f"dark_lost_delta={dark['lost_detail_delta_vs_anchor']:.2%} "
                f"hi_lost_delta={high['lost_detail_delta_vs_anchor']:.2%}",
                flush=True,
            )
        if args.limit and len(summaries) >= args.limit:
            break

    if not summaries:
        print(f"no files matched: {args.glob}", flush=True)
        return 1

    if not args.no_save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        output = RESULTS_DIR / (
            f"float_native_prefix_{args.glob.replace('*', 'star').replace('.', '_')}"
            f"_crop{args.crop_size}_keep{args.keep_bits.replace(',', '-')}"
            f"_{args.recon}.json"
        )
        output.write_text(json.dumps(summaries, indent=2))
        print(f"\nSaved: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
