"""Probe tile-local YCoCg transform-index ranges.

Global YCoCg ranges made the full sample_DSCF0009 visibly warmer than the
crop1024 preview because window/highlight chroma expanded Co/Cg bin width.  This
probe tests whether tile-local min/max ranges can preserve the crop-like visual
quality while keeping range metadata under control.
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
OUTPUT_DIR = ROOT / "outputs" / "previews" / "ycocg_tile_range"
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
from export_linear_index_preview_png import to_preview_u16, write_png  # noqa: E402
from probe_adaptive_ycocg_router import entropy_bits_from_counts  # noqa: E402
from probe_color_tile_transform_index import forward_color, inverse_color  # noqa: E402
from probe_darkbits_router_payload import (  # noqa: E402
    inverse_transform_values,
    read_exr,
    transform_values,
)


def parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(",") if part.strip())


def safe_name(text: str) -> str:
    return text.replace("*", "star").replace(".", "_").replace(",", "-").replace("/", "_")


def quantize_with_range(
    values: np.ndarray,
    bits: int,
    transform: str,
    power_gamma: float,
    lo: float,
    hi: float,
) -> tuple[np.ndarray, np.ndarray]:
    levels = (1 << bits) - 1
    transformed = transform_values(values, transform, power_gamma)
    if not hi > lo:
        decoded_value = inverse_transform_values(np.asarray(lo), transform, power_gamma)
        return (
            np.zeros(values.shape, dtype=np.uint16),
            np.full(values.shape, decoded_value, dtype=np.float32),
        )
    q = np.floor((transformed - lo) / (hi - lo) * levels + 0.5)
    q = np.clip(q, 0, levels).astype(np.uint16)
    rec_t = lo + q.astype(np.float64) * (hi - lo) / levels
    decoded = inverse_transform_values(rec_t, transform, power_gamma).astype(np.float32)
    return q, decoded


def residual_entropy_bytes(indices: np.ndarray, bits: tuple[int, int, int]) -> dict:
    total_bits = 0.0
    rows = []
    for channel, bit_depth in enumerate(bits):
        alphabet = 1 << bit_depth
        channel_indices = indices[:, :, channel:channel + 1]
        pred = predict_for_channel(channel_indices, 0, "med")
        residual = (
            channel_indices[:, :, 0].astype(np.int32) + alphabet - pred
        ) & (alphabet - 1)
        counts = np.bincount(residual.reshape(-1).astype(np.int64), minlength=alphabet)
        bit_cost = entropy_bits_from_counts(counts)
        total_bits += bit_cost
        rows.append({
            "channel": channel,
            "bits": bit_depth,
            "estimated_bits": bit_cost,
            "bits_per_sample": bit_cost / float(residual.size),
        })
    return {
        "estimated_bits": total_bits,
        "estimated_bytes": int(math.ceil(total_bits / 8.0)),
        "channels": rows,
    }


def quantize_tile_local(
    pixels: np.ndarray,
    tile_size: int,
    bits: tuple[int, int, int],
    transform: str,
    power_gamma: float,
    range_bytes_per_value: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    planes = forward_color(pixels, "ycocg")[:, :, :3]
    h, w, _ = planes.shape
    indices = np.zeros(planes.shape, dtype=np.uint16)
    decoded_planes = np.zeros(planes.shape, dtype=np.float32)
    ranges = []
    started = time.perf_counter()
    for y0 in range(0, h, tile_size):
        for x0 in range(0, w, tile_size):
            y1 = min(y0 + tile_size, h)
            x1 = min(x0 + tile_size, w)
            tile_ranges = []
            for channel in range(3):
                values = planes[y0:y1, x0:x1, channel]
                transformed = transform_values(values, transform, power_gamma)
                lo = float(np.min(transformed))
                hi = float(np.max(transformed))
                q, decoded = quantize_with_range(
                    values,
                    bits[channel],
                    transform,
                    power_gamma,
                    lo,
                    hi,
                )
                indices[y0:y1, x0:x1, channel] = q
                decoded_planes[y0:y1, x0:x1, channel] = decoded
                tile_ranges.append([lo, hi])
            ranges.append({"x": x0, "y": y0, "ranges": tile_ranges})
    decoded = inverse_color(decoded_planes, "ycocg")
    entropy_payload = residual_entropy_bytes(indices, bits)
    tiles_x = math.ceil(w / tile_size)
    tiles_y = math.ceil(h / tile_size)
    tile_count = tiles_x * tiles_y
    range_bytes = tile_count * 3 * 2 * int(range_bytes_per_value)
    estimated_bytes = int(entropy_payload["estimated_bytes"] + range_bytes + 128)
    return indices, np.ascontiguousarray(decoded), {
        "tile_size": tile_size,
        "tile_count": tile_count,
        "range_bytes_per_value": range_bytes_per_value,
        "range_bytes": range_bytes,
        "entropy_payload": entropy_payload,
        "estimated_bytes": estimated_bytes,
        "ranges": ranges[:16],
        "elapsed_sec": time.perf_counter() - started,
    }


def summarize_case(
    pixels: np.ndarray,
    anchor_regions: dict,
    anchor_regions_lift: dict,
    tile_size: int,
    bits: tuple[int, int, int],
    transform: str,
    power_gamma: float,
    range_bytes_per_value: int,
    white: float,
    lift_white: float,
    gamma: float,
) -> dict:
    _, decoded, payload = quantize_tile_local(
        pixels,
        tile_size,
        bits,
        transform,
        power_gamma,
        range_bytes_per_value,
    )
    regions = candidate_metrics(pixels, decoded, white, gamma)
    regions_lift = candidate_metrics(pixels, decoded, lift_white, gamma)
    gate = evaluate_gate(regions, anchor_regions, default_gate_args())
    gate_lift = evaluate_gate(regions_lift, anchor_regions_lift, default_gate_args())
    key_metrics = summarize_key_metrics(regions, anchor_regions)
    key_metrics_lift = summarize_key_metrics(regions_lift, anchor_regions_lift)
    estimated_bytes = int(payload["estimated_bytes"])
    return {
        "label": f"ycocg_tile{tile_size}_Y{bits[0]}C{bits[1]}_{transform}",
        "tile_size": tile_size,
        "bits": list(bits),
        "transform": transform,
        "raw_bytes": int(pixels.nbytes),
        "estimated_bytes": estimated_bytes,
        "estimated_ratio": pixels.nbytes / float(estimated_bytes),
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
    high = row["key_metrics"]["highlight_1_4"]
    lift_dark = row["key_metrics_lift"]["dark_0_0.25"]
    print(
        f"{prefix}{row['gate']['decision'].upper():6s}/"
        f"{row['gate_lift']['decision'].upper():6s} "
        f"{row['label']:30s} "
        f"est={row['estimated_bytes']:,} ratio={row['estimated_ratio']:.2f}x "
        f"tiles={row['payload']['tile_count']} range={row['payload']['range_bytes']:,}B "
        f"darkD={dark['lost_detail_delta_vs_anchor']:.2%} "
        f"liftDarkD={lift_dark['lost_detail_delta_vs_anchor']:.2%} "
        f"hiD={high['lost_detail_delta_vs_anchor']:.2%}",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="sample_DSCF0009.EXR")
    parser.add_argument("--crop-size", type=int, default=1024)
    parser.add_argument("--tile-sizes", default="64,128,256,512")
    parser.add_argument("--y-bits", type=int, default=9)
    parser.add_argument("--chroma-bits", type=int, default=7)
    parser.add_argument("--transform", default="gamma075")
    parser.add_argument("--power-gamma", type=float, default=0.75)
    parser.add_argument("--range-bytes-per-value", type=int, default=2)
    parser.add_argument("--white", type=float, default=4.0)
    parser.add_argument("--lift-white", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=2.2)
    parser.add_argument("--anchor-transform", default="signed-log")
    parser.add_argument("--anchor-bits", type=int, default=10)
    parser.add_argument("--export-labels", default="")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    export_labels = set(part.strip() for part in args.export_labels.split(",") if part.strip())
    summaries = []
    bits = (args.y_bits, args.chroma_bits, args.chroma_bits)
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
        rows = []
        for tile_size in parse_ints(args.tile_sizes):
            row = summarize_case(
                pixels,
                anchor_regions,
                anchor_regions_lift,
                tile_size,
                bits,
                args.transform,
                args.power_gamma,
                args.range_bytes_per_value,
                args.white,
                args.lift_white,
                args.gamma,
            )
            rows.append(row)
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
                    / f"{stem}_{crop_label}_{safe_name(row['label'])}_w{args.white:g}_g{args.gamma:g}_decoded.png"
                )
                write_png(decoded_png, to_preview_u16(row["decoded"], args.white, args.gamma))
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
            f"ycocg_tile_range_{safe_name(args.glob)}"
            f"_crop{args.crop_size}_Y{args.y_bits}C{args.chroma_bits}"
            f"_tile{safe_name(args.tile_sizes)}.json"
        )
        output.write_text(json.dumps(summaries, indent=2))
        print(f"Saved: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
