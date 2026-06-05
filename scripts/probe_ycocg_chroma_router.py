"""Probe a Y9 C6/C7 chroma router for YCoCg transform-index previews.

The user-visible checkpoint is:

* Y9 C7 is visually acceptable on sample_DSCF0009 crop1024.
* Y9 C6 is close, but has a tiny black-lift / tone shift.
* Y8 C7 is too aggressive.

This probe asks whether we can keep Y9 everywhere, use C6 for most pixels, and
spend C7 only where it matters.  Some masks are oracle masks; they are useful as
upper bounds before designing a cheaper deterministic selector.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
OUTPUT_DIR = ROOT / "outputs" / "previews" / "ycocg_chroma_router"
sys.path.insert(0, str(ROOT / "codec" / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

import radiance_codec  # noqa: E402
from audit_display_quality_regions import display_map, luminance  # noqa: E402
from audit_linear_index_payload import predict_for_channel  # noqa: E402
from audit_visual_gate import (  # noqa: E402
    candidate_metrics,
    default_gate_args,
    evaluate_gate,
    summarize_key_metrics,
)
from export_linear_index_preview_png import to_preview_u16, write_png  # noqa: E402
from probe_adaptive_ycocg_router import build_mask, entropy_bits_from_counts  # noqa: E402
from probe_asymmetric_color_index import quantize_plane  # noqa: E402
from probe_color_tile_transform_index import forward_color, inverse_color  # noqa: E402
from probe_darkbits_router_payload import read_exr, summarize_binary_mask  # noqa: E402


def parse_floats(text: str) -> tuple[float, ...]:
    return tuple(float(part) for part in text.split(",") if part.strip())


def parse_strings(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in text.split(",") if part.strip())


def safe_name(text: str) -> str:
    return text.replace("*", "star").replace(".", "_").replace(",", "-").replace("/", "_")


def quantize_planes(
    pixels: np.ndarray,
    y_bits: int,
    chroma_bits: int,
    transform: str,
    power_gamma: float,
) -> tuple[np.ndarray, np.ndarray]:
    planes = forward_color(pixels, "ycocg")
    bits = (y_bits, chroma_bits, chroma_bits)
    indices = np.zeros(planes[:, :, :3].shape, dtype=np.uint16)
    decoded_planes = np.array(planes[:, :, :3], copy=True)
    for channel, bit_depth in enumerate(bits):
        q, decoded, _ = quantize_plane(
            planes[:, :, channel],
            bit_depth,
            transform,
            power_gamma,
        )
        indices[:, :, channel] = q
        decoded_planes[:, :, channel] = decoded
    return indices, decoded_planes


def selected_channel_cost(indices: np.ndarray, channel: int, bits: int, mask: np.ndarray) -> dict:
    samples = int(np.count_nonzero(mask))
    if bits <= 0 or samples == 0:
        return {
            "channel": channel,
            "bits": bits,
            "samples": samples,
            "estimated_bits": 0.0,
            "bits_per_sample": 0.0,
        }
    alphabet = 1 << bits
    channel_indices = indices[:, :, channel:channel + 1]
    pred = predict_for_channel(channel_indices, 0, "med")
    residual = (channel_indices[:, :, 0].astype(np.int32) + alphabet - pred) & (alphabet - 1)
    values = residual[mask]
    counts = np.bincount(values.astype(np.int64), minlength=alphabet)
    bits_cost = entropy_bits_from_counts(counts)
    return {
        "channel": channel,
        "bits": bits,
        "samples": samples,
        "estimated_bits": bits_cost,
        "bits_per_sample": bits_cost / float(samples),
    }


def payload_estimate(
    idx_c6: np.ndarray,
    idx_c7: np.ndarray,
    mask_c7: np.ndarray,
    y_bits: int,
    c6_bits: int,
    c7_bits: int,
) -> dict:
    non_mask = ~mask_c7
    rows = [
        selected_channel_cost(idx_c6, 0, y_bits, np.ones(mask_c7.shape, dtype=bool)),
        selected_channel_cost(idx_c6, 1, c6_bits, non_mask),
        selected_channel_cost(idx_c6, 2, c6_bits, non_mask),
        selected_channel_cost(idx_c7, 1, c7_bits, mask_c7),
        selected_channel_cost(idx_c7, 2, c7_bits, mask_c7),
    ]
    mask_summary = summarize_binary_mask(mask_c7)
    mask_payload = min(
        mask_summary["order0"],
        mask_summary["west_north"],
        key=lambda row: row["total_bytes"],
    )
    estimated_bits = sum(row["estimated_bits"] for row in rows)
    estimated_bytes = int(math.ceil(estimated_bits / 8.0) + int(mask_payload["total_bytes"]) + 128)
    return {
        "estimated_bits": estimated_bits,
        "estimated_bytes": estimated_bytes,
        "channels": rows,
        "mask_payload": mask_payload,
    }


def display_error(original: np.ndarray, decoded: np.ndarray, white: float, gamma: float) -> np.ndarray:
    orig = display_map(original, white, gamma)
    dec = display_map(decoded, white, gamma)
    luma_diff = luminance(dec) - luminance(orig)
    rgb_diff = dec - orig
    return np.sqrt(np.mean(rgb_diff * rgb_diff, axis=2) + luma_diff * luma_diff)


def top_rate_mask(score: np.ndarray, rate: float) -> np.ndarray:
    rate = float(np.clip(rate, 0.0, 1.0))
    if rate <= 0.0:
        return np.zeros(score.shape, dtype=bool)
    if rate >= 1.0:
        return np.ones(score.shape, dtype=bool)
    threshold = float(np.quantile(score.reshape(-1), 1.0 - rate))
    return score >= threshold


def build_router_masks(
    pixels: np.ndarray,
    decoded_c6: np.ndarray,
    decoded_c7: np.ndarray,
    white: float,
    lift_white: float,
    gamma: float,
    oracle_rates: tuple[float, ...],
    dark_max_values: tuple[float, ...],
) -> list[tuple[str, np.ndarray]]:
    masks: list[tuple[str, np.ndarray]] = []
    for dark_max in dark_max_values:
        masks.append((
            f"dark{dark_max:g}",
            build_mask(pixels, "dark", dark_max, 2, 0.002),
        ))
        masks.append((
            f"dark-smooth{dark_max:g}",
            build_mask(pixels, "dark-smooth", dark_max, 2, 0.002),
        ))
    neg = np.any(decoded_c6[:, :, :3] < 0.0, axis=2)
    masks.append(("c6-rgb-negative", neg))
    dark = build_mask(pixels, "dark", 0.25, 2, 0.002)
    masks.append(("dark0.25-or-c6-negative", dark | neg))

    err_c6 = display_error(pixels, decoded_c6, white, gamma)
    err_c7 = display_error(pixels, decoded_c7, white, gamma)
    err_lift_c6 = display_error(pixels, decoded_c6, lift_white, gamma)
    err_lift_c7 = display_error(pixels, decoded_c7, lift_white, gamma)
    benefit = np.maximum(err_c6 - err_c7, 0.0)
    benefit_lift = np.maximum(err_lift_c6 - err_lift_c7, 0.0)
    combined = benefit + benefit_lift
    for rate in oracle_rates:
        masks.append((f"oracle-benefit-top{rate:g}", top_rate_mask(combined, rate)))
        masks.append((f"oracle-lift-error-top{rate:g}", top_rate_mask(err_lift_c6, rate)))
    return masks


def summarize_case(
    pixels: np.ndarray,
    anchor_regions: dict,
    anchor_regions_lift: dict,
    idx_c6: np.ndarray,
    planes_c6: np.ndarray,
    idx_c7: np.ndarray,
    planes_c7: np.ndarray,
    mask: np.ndarray,
    label: str,
    y_bits: int,
    c6_bits: int,
    c7_bits: int,
    white: float,
    lift_white: float,
    gamma: float,
) -> dict:
    stitched_planes = np.array(planes_c6, copy=True)
    stitched_planes[:, :, 1][mask] = planes_c7[:, :, 1][mask]
    stitched_planes[:, :, 2][mask] = planes_c7[:, :, 2][mask]
    decoded = inverse_color(stitched_planes, "ycocg")
    payload = payload_estimate(idx_c6, idx_c7, mask, y_bits, c6_bits, c7_bits)
    regions = candidate_metrics(pixels, decoded, white, gamma)
    regions_lift = candidate_metrics(pixels, decoded, lift_white, gamma)
    gate = evaluate_gate(regions, anchor_regions, default_gate_args())
    gate_lift = evaluate_gate(regions_lift, anchor_regions_lift, default_gate_args())
    key_metrics = summarize_key_metrics(regions, anchor_regions)
    key_metrics_lift = summarize_key_metrics(regions_lift, anchor_regions_lift)
    return {
        "label": label,
        "mask_pixel_rate": float(np.mean(mask)),
        "raw_bytes": int(pixels.nbytes),
        "estimated_bytes": int(payload["estimated_bytes"]),
        "estimated_ratio": pixels.nbytes / float(payload["estimated_bytes"]),
        "payload": payload,
        "gate": gate,
        "gate_lift": gate_lift,
        "key_metrics": key_metrics,
        "key_metrics_lift": key_metrics_lift,
        "decoded": decoded,
    }


def decision_rank(decision: str) -> int:
    return {"pass": 0, "maybe": 1, "reject": 2}.get(decision, 9)


def print_row(row: dict, prefix: str = "") -> None:
    dark = row["key_metrics"]["dark_0_0.25"]
    lift_dark = row["key_metrics_lift"]["dark_0_0.25"]
    print(
        f"{prefix}{row['gate']['decision'].upper():6s}/"
        f"{row['gate_lift']['decision'].upper():6s} "
        f"{row['label']:30s} "
        f"est={row['estimated_bytes']:,} ratio={row['estimated_ratio']:.2f}x "
        f"mask={row['mask_pixel_rate']:.2%} "
        f"darkD={dark['lost_detail_delta_vs_anchor']:.2%} "
        f"liftDarkD={lift_dark['lost_detail_delta_vs_anchor']:.2%}",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="sample_DSCF0009.EXR")
    parser.add_argument("--crop-size", type=int, default=1024)
    parser.add_argument("--y-bits", type=int, default=9)
    parser.add_argument("--c6-bits", type=int, default=6)
    parser.add_argument("--c7-bits", type=int, default=7)
    parser.add_argument("--transform", default="gamma075")
    parser.add_argument("--power-gamma", type=float, default=0.75)
    parser.add_argument("--white", type=float, default=4.0)
    parser.add_argument("--lift-white", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=2.2)
    parser.add_argument("--oracle-rates", default="0.05,0.1,0.2,0.35,0.5")
    parser.add_argument("--dark-max", default="0.03,0.05,0.1,0.25")
    parser.add_argument("--anchor-transform", default="signed-log")
    parser.add_argument("--anchor-bits", type=int, default=10)
    parser.add_argument("--top", type=int, default=18)
    parser.add_argument("--only-labels", default="")
    parser.add_argument("--export-labels", default="")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    summaries = []
    only_labels = set(parse_strings(args.only_labels))
    export_labels = set(parse_strings(args.export_labels))
    for path in sorted(DATA_DIR.glob(args.glob)):
        print(path.name, flush=True)
        pixels = read_exr(path, args.crop_size)
        anchor = radiance_codec.quantize_linear_index(
            pixels,
            bits=args.anchor_bits,
            transform=args.anchor_transform,
        )
        anchor_regions = candidate_metrics(pixels, anchor, args.white, args.gamma)
        anchor_regions_lift = candidate_metrics(pixels, anchor, args.lift_white, args.gamma)
        idx_c6, planes_c6 = quantize_planes(
            pixels,
            args.y_bits,
            args.c6_bits,
            args.transform,
            args.power_gamma,
        )
        idx_c7, planes_c7 = quantize_planes(
            pixels,
            args.y_bits,
            args.c7_bits,
            args.transform,
            args.power_gamma,
        )
        decoded_c6 = inverse_color(planes_c6, "ycocg")
        decoded_c7 = inverse_color(planes_c7, "ycocg")
        masks = build_router_masks(
            pixels,
            decoded_c6,
            decoded_c7,
            args.white,
            args.lift_white,
            args.gamma,
            parse_floats(args.oracle_rates),
            parse_floats(args.dark_max),
        )
        rows = []
        for label, mask in masks:
            if only_labels and label not in only_labels:
                continue
            rows.append(
                summarize_case(
                    pixels,
                    anchor_regions,
                    anchor_regions_lift,
                    idx_c6,
                    planes_c6,
                    idx_c7,
                    planes_c7,
                    mask,
                    label,
                    args.y_bits,
                    args.c6_bits,
                    args.c7_bits,
                    args.white,
                    args.lift_white,
                    args.gamma,
                )
            )
        rows.sort(
            key=lambda row: (
                decision_rank(row["gate"]["decision"]),
                decision_rank(row["gate_lift"]["decision"]),
                row["estimated_bytes"],
            )
        )
        stem = safe_name(path.stem)
        crop_label = "full" if args.crop_size == 0 else f"crop{args.crop_size}"
        original_png = args.output_dir / f"{stem}_{crop_label}_original_w{args.white:g}_g{args.gamma:g}.png"
        if export_labels and not original_png.exists():
            write_png(original_png, to_preview_u16(pixels, args.white, args.gamma))
        for row in rows:
            if row["label"] in export_labels:
                decoded_png = (
                    args.output_dir
                    / (
                        f"{stem}_{crop_label}_Y{args.y_bits}"
                        f"_C{args.c6_bits}-{args.c7_bits}_{safe_name(row['label'])}"
                        f"_w{args.white:g}_g{args.gamma:g}_decoded.png"
                    )
                )
                write_png(decoded_png, to_preview_u16(row["decoded"], args.white, args.gamma))
                row["original_png"] = str(original_png)
                row["decoded_png"] = str(decoded_png)
        for row in rows[:args.top]:
            print_row(row, prefix="  ")
            if "decoded_png" in row:
                print(f"    decoded_png={row['decoded_png']}", flush=True)
        serializable_rows = []
        for row in rows:
            row_copy = dict(row)
            row_copy.pop("decoded", None)
            serializable_rows.append(row_copy)
        summaries.append({
            "image": path.name,
            "shape": list(pixels.shape),
            "crop_size": args.crop_size,
            "y_bits": args.y_bits,
            "c6_bits": args.c6_bits,
            "c7_bits": args.c7_bits,
            "white": args.white,
            "lift_white": args.lift_white,
            "gamma": args.gamma,
            "rows": serializable_rows,
        })
        if args.limit and len(summaries) >= args.limit:
            break
    if not summaries:
        print(f"no files matched: {args.glob}", flush=True)
        return 1
    if not args.no_save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        output = RESULTS_DIR / (
            f"ycocg_chroma_router_{safe_name(args.glob)}"
            f"_crop{args.crop_size}_Y{args.y_bits}_C{args.c6_bits}-{args.c7_bits}.json"
        )
        output.write_text(json.dumps(summaries, indent=2))
        print(f"Saved: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
