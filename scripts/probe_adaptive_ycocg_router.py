"""Probe adaptive dark-centered YCoCg asymmetric routing.

The full-image asymmetric YCoCg route is promising on sample_DSCF0009, but it
is not universally compact.  This probe tests a router:

* base route for the whole image or non-dark region;
* dark route using luma-priority YCoCg asymmetric quantization;
* deterministic mask driven by luma and optional smoothness.

It estimates a split payload and judges the stitched RGB reconstruction with
the proxy visual gate.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
sys.path.insert(0, str(ROOT / "codec" / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

import radiance_codec  # noqa: E402
from audit_linear_index_payload import predict_for_channel  # noqa: E402
from audit_visual_gate import (  # noqa: E402
    candidate_metrics,
    default_gate_args,
    evaluate_gate,
    summarize_key_metrics,
)
from probe_asymmetric_color_index import quantize_asymmetric  # noqa: E402
from probe_color_tile_transform_index import forward_color, inverse_color  # noqa: E402
from probe_darkbits_router_payload import summarize_binary_mask  # noqa: E402
from probe_darkbits_router_payload import read_exr  # noqa: E402


def parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(",") if part.strip())


def parse_floats(text: str) -> tuple[float, ...]:
    return tuple(float(part) for part in text.split(",") if part.strip())


def safe_name(text: str) -> str:
    return text.replace("*", "star").replace(".", "_").replace(",", "-")


def luma_linear(pixels: np.ndarray) -> np.ndarray:
    rgb = pixels[:, :, :3].astype(np.float64)
    return 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]


def box_mean(values: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return values.astype(np.float64, copy=False)
    padded = np.pad(values.astype(np.float64, copy=False), radius, mode="reflect")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    size = 2 * radius + 1
    total = (
        integral[size:, size:]
        - integral[:-size, size:]
        - integral[size:, :-size]
        + integral[:-size, :-size]
    )
    return total / float(size * size)


def local_std(values: np.ndarray, radius: int) -> np.ndarray:
    mean = box_mean(values, radius)
    mean_sq = box_mean(values * values, radius)
    return np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))


def build_mask(
    pixels: np.ndarray,
    mode: str,
    dark_max: float,
    smooth_radius: int,
    smooth_threshold: float,
) -> np.ndarray:
    y = luma_linear(pixels)
    dark = y <= float(dark_max)
    if mode == "dark":
        return dark
    # Smoothness is measured in display-ish log luma so exposure-lift banding is
    # treated as visible risk, while textured regions can stay cheaper.
    log_y = np.log2(1.0 + np.maximum(y, 0.0))
    smooth = local_std(log_y, smooth_radius) <= float(smooth_threshold)
    if mode == "dark-smooth":
        return dark & smooth
    if mode == "dark-or-smooth":
        return dark | smooth
    raise ValueError(f"unknown mask mode: {mode}")


def quantize_asymmetric_allow_zero(
    pixels: np.ndarray,
    color: str,
    bits: tuple[int, int, int],
    transforms: tuple[str, str, str],
    power_gamma: float,
) -> tuple[np.ndarray, np.ndarray]:
    if min(bits) > 0:
        indices, decoded, _ = quantize_asymmetric(
            pixels,
            color,
            bits,
            transforms,
            power_gamma,
        )
        return indices, decoded
    nonzero_bits = tuple(max(1, bit) for bit in bits)
    planes = forward_color(pixels, color)
    indices, decoded, _ = quantize_asymmetric(
        pixels,
        color,
        nonzero_bits,
        transforms,
        power_gamma,
    )
    decoded_planes = forward_color(decoded, color)
    for channel, bit in enumerate(bits):
        if bit == 0:
            indices[:, :, channel] = 0
            decoded_planes[:, :, channel] = 0.0
    return indices, inverse_color(decoded_planes, color)


def entropy_bits_from_counts(counts: np.ndarray) -> float:
    total = int(counts.sum())
    if total <= 0:
        return 0.0
    nz = counts[counts > 0].astype(np.float64)
    probs = nz / float(total)
    return float(-np.sum(nz * np.log2(probs)))


def selected_residual_cost(
    indices: np.ndarray,
    bits: tuple[int, int, int],
    mask: np.ndarray,
) -> dict:
    total_bits = 0.0
    rows = []
    for channel, bit_depth in enumerate(bits):
        selected = mask
        if bit_depth <= 0 or not np.any(selected):
            rows.append({
                "channel": channel,
                "bits": bit_depth,
                "samples": int(np.count_nonzero(selected)),
                "estimated_bits": 0.0,
                "bits_per_sample": 0.0,
            })
            continue
        alphabet = 1 << bit_depth
        channel_indices = indices[:, :, channel:channel + 1]
        pred = predict_for_channel(channel_indices, 0, "med")
        residual = (
            channel_indices[:, :, 0].astype(np.int32) + alphabet - pred
        ) & (alphabet - 1)
        values = residual[selected]
        counts = np.bincount(values.astype(np.int64), minlength=alphabet)
        bits_cost = entropy_bits_from_counts(counts)
        total_bits += bits_cost
        rows.append({
            "channel": channel,
            "bits": bit_depth,
            "samples": int(values.size),
            "estimated_bits": bits_cost,
            "bits_per_sample": bits_cost / float(values.size) if values.size else 0.0,
        })
    return {
        "estimated_bits": total_bits,
        "estimated_bytes": int(math.ceil(total_bits / 8.0) + 32),
        "channels": rows,
    }


def summarize_case(
    pixels: np.ndarray,
    anchor_regions: dict,
    mask: np.ndarray,
    base_bits: tuple[int, int, int],
    dark_bits: tuple[int, int, int],
    base_color: str,
    dark_color: str,
    base_transform: str,
    dark_transform: str,
    power_gamma: float,
    white: float,
    gamma: float,
    mask_mode: str,
    dark_max: float,
) -> dict:
    started = time.perf_counter()
    transforms_base = (base_transform, base_transform, base_transform)
    transforms_dark = (dark_transform, dark_transform, dark_transform)
    base_indices, base_decoded = quantize_asymmetric_allow_zero(
        pixels,
        base_color,
        base_bits,
        transforms_base,
        power_gamma,
    )
    dark_indices, dark_decoded = quantize_asymmetric_allow_zero(
        pixels,
        dark_color,
        dark_bits,
        transforms_dark,
        power_gamma,
    )
    decoded = np.array(base_decoded, copy=True)
    decoded[mask, :] = dark_decoded[mask, :]

    non_mask = ~mask
    base_payload = selected_residual_cost(base_indices, base_bits, non_mask)
    dark_payload = selected_residual_cost(dark_indices, dark_bits, mask)
    mask_summary = summarize_binary_mask(mask)
    mask_payload = min(
        mask_summary["order0"],
        mask_summary["west_north"],
        key=lambda row: row["total_bytes"],
    )
    estimated_bytes = (
        base_payload["estimated_bytes"]
        + dark_payload["estimated_bytes"]
        + int(mask_payload["total_bytes"])
        + 128
    )
    regions = candidate_metrics(pixels, decoded, white, gamma)
    gate = evaluate_gate(regions, anchor_regions, default_gate_args())
    key_metrics = summarize_key_metrics(regions, anchor_regions)
    return {
        "label": (
            f"base-{base_color}{base_bits[0]}-{base_bits[1]}-{base_bits[2]}_{base_transform}"
            f"__dark-{dark_color}{dark_bits[0]}-{dark_bits[1]}-{dark_bits[2]}_{dark_transform}"
            f"__{mask_mode}{dark_max:g}"
        ),
        "base_color": base_color,
        "dark_color": dark_color,
        "base_bits": list(base_bits),
        "dark_bits": list(dark_bits),
        "base_transform": base_transform,
        "dark_transform": dark_transform,
        "mask_mode": mask_mode,
        "dark_max": dark_max,
        "mask_pixel_rate": float(np.mean(mask)),
        "raw_bytes": int(pixels.nbytes),
        "estimated_bytes": int(estimated_bytes),
        "estimated_ratio": pixels.nbytes / float(estimated_bytes),
        "base_payload": base_payload,
        "dark_payload": dark_payload,
        "mask_payload": mask_payload,
        "gate": gate,
        "key_metrics": key_metrics,
        "elapsed_sec": time.perf_counter() - started,
    }


def decision_rank(decision: str) -> int:
    return {"pass": 0, "maybe": 1, "reject": 2}.get(decision, 9)


def print_row(row: dict, prefix: str = "") -> None:
    dark = row["key_metrics"]["dark_0_0.25"]
    high = row["key_metrics"]["highlight_1_4"]
    reasons = "; ".join(
        f"{item['region']} {item['metric']}={item['value']:.2e}"
        for item in row["gate"]["failures"][:2]
    )
    if not reasons:
        reasons = "ok"
    print(
        f"{prefix}{row['gate']['decision'].upper():6s} {row['label']:58s} "
        f"est={row['estimated_bytes']:,} ratio={row['estimated_ratio']:.2f}x "
        f"mask={row['mask_pixel_rate']:.2%} "
        f"darkD={dark['lost_detail_delta_vs_anchor']:.2%} "
        f"hiD={high['lost_detail_delta_vs_anchor']:.2%} - {reasons}",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="sample_DSCF0009.EXR")
    parser.add_argument("--crop-size", type=int, default=1024)
    parser.add_argument("--mask-mode", choices=("dark", "dark-smooth", "dark-or-smooth"), default="dark")
    parser.add_argument("--dark-max", default="0.03,0.05,0.08,0.12")
    parser.add_argument("--smooth-radius", type=int, default=2)
    parser.add_argument("--smooth-threshold", type=float, default=0.004)
    parser.add_argument("--base-color", default="ycocg")
    parser.add_argument("--dark-color", default="ycocg")
    parser.add_argument("--base-y-bits", default="8,9")
    parser.add_argument("--base-chroma-bits", default="4,5,6")
    parser.add_argument("--dark-y-bits", default="10")
    parser.add_argument("--dark-chroma-bits", default="0,4,5,6")
    parser.add_argument("--base-transform", default="gamma075")
    parser.add_argument("--dark-transform", default="gamma075")
    parser.add_argument("--power-gamma", type=float, default=0.75)
    parser.add_argument("--anchor-transform", default="signed-log")
    parser.add_argument("--anchor-bits", type=int, default=10)
    parser.add_argument("--white", type=float, default=4.0)
    parser.add_argument("--gamma", type=float, default=2.2)
    parser.add_argument("--top", type=int, default=18)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    summaries = []
    for path in sorted(DATA_DIR.glob(args.glob)):
        print(path.name, flush=True)
        pixels = read_exr(path, args.crop_size)
        anchor = radiance_codec.quantize_linear_index(
            pixels,
            bits=args.anchor_bits,
            transform=args.anchor_transform,
        )
        anchor_regions = candidate_metrics(pixels, anchor, args.white, args.gamma)
        rows = []
        for dark_max in parse_floats(args.dark_max):
            mask = build_mask(
                pixels,
                args.mask_mode,
                dark_max,
                args.smooth_radius,
                args.smooth_threshold,
            )
            for by in parse_ints(args.base_y_bits):
                for bc in parse_ints(args.base_chroma_bits):
                    base_bits = (by, bc, bc)
                    for dy in parse_ints(args.dark_y_bits):
                        for dc in parse_ints(args.dark_chroma_bits):
                            dark_bits = (dy, dc, dc)
                            rows.append(
                                summarize_case(
                                    pixels,
                                    anchor_regions,
                                    mask,
                                    base_bits,
                                    dark_bits,
                                    args.base_color,
                                    args.dark_color,
                                    args.base_transform,
                                    args.dark_transform,
                                    args.power_gamma,
                                    args.white,
                                    args.gamma,
                                    args.mask_mode,
                                    dark_max,
                                )
                            )
        rows.sort(key=lambda row: (decision_rank(row["gate"]["decision"]), row["estimated_bytes"]))
        summary = {
            "image": path.name,
            "shape": list(pixels.shape),
            "crop_size": args.crop_size,
            "mask_mode": args.mask_mode,
            "dark_max": args.dark_max,
            "anchor": f"quant:{args.anchor_transform}:{args.anchor_bits}",
            "rows": rows,
        }
        summaries.append(summary)
        for row in rows[:args.top]:
            print_row(row, prefix="  ")
        if args.limit and len(summaries) >= args.limit:
            break
    if not summaries:
        print(f"no files matched: {args.glob}", flush=True)
        return 1
    if not args.no_save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        output = RESULTS_DIR / (
            f"adaptive_ycocg_router_{safe_name(args.glob)}"
            f"_crop{args.crop_size}_{args.mask_mode}"
            f"_dark{args.dark_max.replace(',', '-')}.json"
        )
        output.write_text(json.dumps(summaries, indent=2))
        print(f"\nSaved: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
