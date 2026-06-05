"""Probe clipped global YCoCg ranges plus sparse escape channels.

Full-image global YCoCg ranges preserve index coherence, but rare window /
highlight chroma outliers make Co/Cg bins too wide.  Tile-local ranges recover
quality but destroy index entropy.  This probe tests the middle ground:

* keep one global range per channel;
* choose robust percentile bounds for chroma;
* mark values outside the range as sparse escapes;
* preview escaped channels by restoring original plane values as an upper bound.
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
OUTPUT_DIR = ROOT / "outputs" / "previews" / "ycocg_clipped_range"
sys.path.insert(0, str(ROOT / "codec" / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

import radiance_codec  # noqa: E402
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
from probe_ycocg_tile_range import residual_entropy_bytes  # noqa: E402


def parse_floats(text: str) -> tuple[float, ...]:
    return tuple(float(part) for part in text.split(",") if part.strip())


def safe_name(text: str) -> str:
    return (
        text.replace("*", "star")
        .replace(".", "_")
        .replace(",", "-")
        .replace("/", "_")
        .replace(":", "_")
    )


def binary_entropy_bits(ones: int, total: int) -> float:
    if total <= 0 or ones <= 0 or ones >= total:
        return 0.0
    p = ones / float(total)
    return -float(total) * (p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


def quantize_clipped_channel(
    values: np.ndarray,
    bits: int,
    transform: str,
    power_gamma: float,
    lo: float,
    hi: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    levels = (1 << bits) - 1
    transformed = transform_values(values, transform, power_gamma)
    escape = (transformed < lo) | (transformed > hi)
    if not hi > lo:
        decoded_value = inverse_transform_values(np.asarray(lo), transform, power_gamma)
        return (
            np.zeros(values.shape, dtype=np.uint16),
            np.full(values.shape, decoded_value, dtype=np.float32),
            escape,
        )
    clipped = np.clip(transformed, lo, hi)
    q = np.floor((clipped - lo) / (hi - lo) * levels + 0.5)
    q = np.clip(q, 0, levels).astype(np.uint16)
    rec_t = lo + q.astype(np.float64) * (hi - lo) / levels
    decoded = inverse_transform_values(rec_t, transform, power_gamma).astype(np.float32)
    return q, decoded, escape


def summarize_escape_masks(escape_masks: np.ndarray, escape_value_bits: int) -> dict:
    total_samples = int(escape_masks.shape[0] * escape_masks.shape[1])
    mask_bits = 0.0
    channels = []
    escape_values = 0
    for channel in range(escape_masks.shape[2]):
        mask = escape_masks[:, :, channel]
        count = int(np.count_nonzero(mask))
        bits = binary_entropy_bits(count, total_samples)
        mask_bits += bits
        escape_values += count
        channels.append({
            "channel": channel,
            "escape_count": count,
            "escape_rate": count / float(total_samples),
            "mask_bits": bits,
        })
    value_bits = escape_values * int(escape_value_bits)
    return {
        "channels": channels,
        "escape_values": escape_values,
        "mask_bits": mask_bits,
        "value_bits": value_bits,
        "estimated_bits": mask_bits + value_bits,
        "estimated_bytes": int(math.ceil((mask_bits + value_bits) / 8.0)),
        "escape_value_bits": escape_value_bits,
    }


def quantize_clipped_ycocg(
    pixels: np.ndarray,
    bits: tuple[int, int, int],
    transform: str,
    power_gamma: float,
    low_percentile: float,
    high_percentile: float,
    clip_y: bool,
    escape_preview: str,
    escape_value_bits: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    planes = forward_color(pixels, "ycocg")[:, :, :3]
    indices = np.zeros(planes.shape, dtype=np.uint16)
    decoded_planes = np.zeros(planes.shape, dtype=np.float32)
    escape_masks = np.zeros(planes.shape, dtype=bool)
    ranges = []
    steps = []
    for channel in range(3):
        values = planes[:, :, channel]
        transformed = transform_values(values, transform, power_gamma)
        if channel == 0 and not clip_y:
            lo = float(np.min(transformed))
            hi = float(np.max(transformed))
        else:
            lo, hi = np.percentile(transformed.reshape(-1), [low_percentile, high_percentile])
            lo = float(lo)
            hi = float(hi)
        q, decoded, escape = quantize_clipped_channel(
            values,
            bits[channel],
            transform,
            power_gamma,
            lo,
            hi,
        )
        indices[:, :, channel] = q
        decoded_planes[:, :, channel] = decoded
        escape_masks[:, :, channel] = escape
        if escape_preview == "restore-plane":
            decoded_planes[:, :, channel][escape] = values[escape].astype(np.float32)
        ranges.append([lo, hi])
        steps.append((hi - lo) / float((1 << bits[channel]) - 1) if hi > lo else 0.0)
    decoded = inverse_color(decoded_planes, "ycocg")
    index_payload = residual_entropy_bytes(indices, bits)
    escape_payload = summarize_escape_masks(escape_masks, escape_value_bits)
    range_bytes = 3 * 2 * 4
    estimated_bytes = int(
        index_payload["estimated_bytes"]
        + escape_payload["estimated_bytes"]
        + range_bytes
        + 128
    )
    return indices, np.ascontiguousarray(decoded), {
        "ranges": ranges,
        "steps": steps,
        "index_payload": index_payload,
        "escape_payload": escape_payload,
        "range_bytes": range_bytes,
        "estimated_bytes": estimated_bytes,
    }


def summarize_case(
    pixels: np.ndarray,
    anchor_regions: dict | None,
    anchor_regions_lift: dict | None,
    bits: tuple[int, int, int],
    transform: str,
    power_gamma: float,
    low_percentile: float,
    high_percentile: float,
    clip_y: bool,
    escape_preview: str,
    escape_value_bits: int,
    white: float,
    lift_white: float,
    gamma: float,
    run_metrics: bool,
) -> dict:
    _, decoded, payload = quantize_clipped_ycocg(
        pixels,
        bits,
        transform,
        power_gamma,
        low_percentile,
        high_percentile,
        clip_y,
        escape_preview,
        escape_value_bits,
    )
    row = {
        "label": (
            f"ycocg_clip{low_percentile:g}-{high_percentile:g}"
            f"_Y{bits[0]}C{bits[1]}_{transform}_{escape_preview}"
        ),
        "low_percentile": low_percentile,
        "high_percentile": high_percentile,
        "clip_y": clip_y,
        "escape_preview": escape_preview,
        "bits": list(bits),
        "raw_bytes": int(pixels.nbytes),
        "estimated_bytes": int(payload["estimated_bytes"]),
        "estimated_ratio": pixels.nbytes / float(payload["estimated_bytes"]),
        "payload": payload,
        "decoded": decoded,
    }
    if run_metrics and anchor_regions is not None and anchor_regions_lift is not None:
        regions = candidate_metrics(pixels, decoded, white, gamma)
        regions_lift = candidate_metrics(pixels, decoded, lift_white, gamma)
        row["gate"] = evaluate_gate(regions, anchor_regions, default_gate_args())
        row["gate_lift"] = evaluate_gate(regions_lift, anchor_regions_lift, default_gate_args())
        row["key_metrics"] = summarize_key_metrics(regions, anchor_regions)
        row["key_metrics_lift"] = summarize_key_metrics(regions_lift, anchor_regions_lift)
    return row


def print_row(row: dict, prefix: str = "") -> None:
    escape = row["payload"]["escape_payload"]
    escape_rate = escape["escape_values"] / float(np.prod(row["shape"][:2]) * 3) if "shape" in row else 0.0
    gate = row.get("gate", {}).get("decision", "n/a").upper()
    gate_lift = row.get("gate_lift", {}).get("decision", "n/a").upper()
    print(
        f"{prefix}{gate:6s}/{gate_lift:6s} {row['label']:48s} "
        f"est={row['estimated_bytes']:,} ratio={row['estimated_ratio']:.2f}x "
        f"esc={escape['escape_values']:,} ({escape_rate:.3%}) "
        f"steps=({row['payload']['steps'][0]:.4g},"
        f"{row['payload']['steps'][1]:.4g},"
        f"{row['payload']['steps'][2]:.4g})",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="sample_DSCF0009.EXR")
    parser.add_argument("--crop-size", type=int, default=1024)
    parser.add_argument("--percentiles", default="0.1:99.9,0.25:99.75,0.5:99.5,1:99")
    parser.add_argument("--y-bits", type=int, default=9)
    parser.add_argument("--chroma-bits", type=int, default=7)
    parser.add_argument("--transform", default="gamma075")
    parser.add_argument("--power-gamma", type=float, default=0.75)
    parser.add_argument("--clip-y", action="store_true")
    parser.add_argument("--escape-preview", choices=("clip", "restore-plane"), default="restore-plane")
    parser.add_argument("--escape-value-bits", type=int, default=16)
    parser.add_argument("--white", type=float, default=4.0)
    parser.add_argument("--lift-white", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=2.2)
    parser.add_argument("--anchor-transform", default="signed-log")
    parser.add_argument("--anchor-bits", type=int, default=10)
    parser.add_argument("--run-metrics", action="store_true")
    parser.add_argument("--export-labels", default="")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    percentile_pairs = []
    for item in args.percentiles.split(","):
        if not item.strip():
            continue
        lo_text, hi_text = item.split(":", 1)
        percentile_pairs.append((float(lo_text), float(hi_text)))
    export_labels = set(part.strip() for part in args.export_labels.split(",") if part.strip())
    bits = (args.y_bits, args.chroma_bits, args.chroma_bits)
    summaries = []
    for path in sorted(DATA_DIR.glob(args.glob)):
        print(path.name, flush=True)
        pixels = read_exr(path, args.crop_size)
        anchor_regions = None
        anchor_regions_lift = None
        if args.run_metrics:
            anchor = radiance_codec.quantize_linear_index(
                pixels,
                bits=args.anchor_bits,
                transform=args.anchor_transform,
            )
            anchor_regions = candidate_metrics(pixels, anchor, args.white, args.gamma)
            anchor_regions_lift = candidate_metrics(pixels, anchor, args.lift_white, args.gamma)
        rows = []
        for low, high in percentile_pairs:
            row = summarize_case(
                pixels,
                anchor_regions,
                anchor_regions_lift,
                bits,
                args.transform,
                args.power_gamma,
                low,
                high,
                args.clip_y,
                args.escape_preview,
                args.escape_value_bits,
                args.white,
                args.lift_white,
                args.gamma,
                args.run_metrics,
            )
            row["shape"] = list(pixels.shape)
            rows.append(row)
        rows.sort(key=lambda row: row["estimated_bytes"])
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
        serializable = []
        for row in rows:
            copy = dict(row)
            copy.pop("decoded", None)
            serializable.append(copy)
        summaries.append({
            "image": path.name,
            "shape": list(pixels.shape),
            "crop_size": args.crop_size,
            "rows": serializable,
        })
        if args.limit and len(summaries) >= args.limit:
            break
    if not summaries:
        print(f"no files matched: {args.glob}", flush=True)
        return 1
    if not args.no_save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        output = RESULTS_DIR / (
            f"ycocg_clipped_range_{safe_name(args.glob)}"
            f"_crop{args.crop_size}_Y{args.y_bits}C{args.chroma_bits}.json"
        )
        output.write_text(json.dumps(summaries, indent=2))
        print(f"Saved: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
