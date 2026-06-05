"""Probe YCoCg reconstruction tables and cheap RGB bias correction.

This keeps the transform-index payload unchanged and only changes the decoded
representative value for each Y/Co/Cg quantization bin.  The goal is to reduce
the subtle global brightness / color shift seen in YCoCg previews without
spending another full bit everywhere.
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
OUTPUT_DIR = ROOT / "outputs" / "previews" / "ycocg_recon_table"
sys.path.insert(0, str(ROOT / "codec" / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

import radiance_codec  # noqa: E402
from audit_display_quality_regions import display_map, luminance, region_masks  # noqa: E402
from audit_visual_gate import (  # noqa: E402
    candidate_metrics,
    default_gate_args,
    evaluate_gate,
    summarize_key_metrics,
)
from export_linear_index_preview_png import to_preview_u16, write_png  # noqa: E402
from probe_asymmetric_color_index import estimate_index_bytes, quantize_plane  # noqa: E402
from probe_color_tile_transform_index import forward_color, inverse_color  # noqa: E402
from probe_darkbits_router_payload import (  # noqa: E402
    inverse_transform_values,
    read_exr,
    transform_values,
)


def parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(",") if part.strip())


def parse_strings(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in text.split(",") if part.strip())


def safe_name(text: str) -> str:
    return (
        text.replace(" ", "_")
        .replace("*", "star")
        .replace(".", "_")
        .replace(",", "-")
        .replace("+", "plus")
        .replace("/", "_")
    )


def nearest_fill(values: np.ndarray, seen: np.ndarray) -> np.ndarray:
    filled = np.array(values, copy=True)
    if not bool(np.any(seen)):
        return filled
    seen_idx = np.flatnonzero(seen)
    for idx in np.flatnonzero(~seen):
        nearest = seen_idx[np.argmin(np.abs(seen_idx - idx))]
        filled[idx] = filled[nearest]
    return filled


def table_decode_plane(
    values: np.ndarray,
    indices: np.ndarray,
    bits: int,
    transform: str,
    power_gamma: float,
    mode: str,
) -> tuple[np.ndarray, int]:
    levels = 1 << bits
    flat_q = indices.reshape(-1).astype(np.int64)
    flat_values = values.reshape(-1).astype(np.float64)
    counts = np.bincount(flat_q, minlength=levels).astype(np.float64)
    seen = counts > 0
    if mode == "value-mean":
        sums = np.bincount(flat_q, weights=flat_values, minlength=levels)
        table = np.zeros(levels, dtype=np.float64)
        table[seen] = sums[seen] / counts[seen]
        table = nearest_fill(table, seen)
    elif mode == "transform-mean":
        transformed = transform_values(flat_values, transform, power_gamma)
        sums = np.bincount(flat_q, weights=transformed, minlength=levels)
        table_t = np.zeros(levels, dtype=np.float64)
        table_t[seen] = sums[seen] / counts[seen]
        table_t = nearest_fill(table_t, seen)
        table = inverse_transform_values(table_t, transform, power_gamma)
    else:
        raise ValueError(f"unknown table mode: {mode}")
    decoded = table[indices].astype(np.float32)
    return decoded, levels * 4


def quantize_ycocg(
    pixels: np.ndarray,
    bits: tuple[int, int, int],
    transform: str,
    power_gamma: float,
    recon_mode: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    mode_parts = recon_mode.split("+")
    base_mode = mode_parts[0]
    options = set(mode_parts[1:])
    planes = forward_color(pixels, "ycocg")
    indices = np.zeros(planes[:, :, :3].shape, dtype=np.uint16)
    decoded_planes = np.array(planes, copy=True)
    table_bytes = 0
    for channel in range(3):
        q, center_decoded, _ = quantize_plane(
            planes[:, :, channel],
            bits[channel],
            transform,
            power_gamma,
        )
        indices[:, :, channel] = q
        if base_mode == "center":
            decoded_planes[:, :, channel] = center_decoded
        else:
            decoded, channel_table_bytes = table_decode_plane(
                planes[:, :, channel],
                q,
                bits[channel],
                transform,
                power_gamma,
                base_mode,
            )
            decoded_planes[:, :, channel] = decoded
            table_bytes += channel_table_bytes
    if "gamut-project" in options:
        decoded = inverse_ycocg_nonnegative(decoded_planes)
    else:
        decoded = inverse_color(decoded_planes, "ycocg")
    if "rgb-affine" in options:
        decoded, affine_bytes = apply_rgb_affine(pixels, decoded)
        table_bytes += affine_bytes
    return indices, np.ascontiguousarray(decoded), table_bytes


def inverse_ycocg_nonnegative(planes: np.ndarray) -> np.ndarray:
    out = np.array(planes, dtype=np.float32, copy=True)
    y = planes[:, :, 0].astype(np.float64)
    co = planes[:, :, 1].astype(np.float64)
    cg = planes[:, :, 2].astype(np.float64)
    deltas = (
        -0.5 * cg + 0.5 * co,
        0.5 * cg,
        -0.5 * cg - 0.5 * co,
    )
    t = np.ones(y.shape, dtype=np.float64)
    for delta in deltas:
        shrinking = delta < 0.0
        limit = np.ones(y.shape, dtype=np.float64)
        limit[shrinking] = np.maximum(y[shrinking], 0.0) / np.maximum(-delta[shrinking], 1e-30)
        t = np.minimum(t, limit)
    t = np.clip(t, 0.0, 1.0)
    out[:, :, 0] = (y + t * deltas[0]).astype(np.float32)
    out[:, :, 1] = (y + t * deltas[1]).astype(np.float32)
    out[:, :, 2] = (y + t * deltas[2]).astype(np.float32)
    return np.ascontiguousarray(out)


def apply_rgb_affine(original: np.ndarray, decoded: np.ndarray) -> tuple[np.ndarray, int]:
    corrected = np.array(decoded, copy=True)
    for channel in range(3):
        x = decoded[:, :, channel].reshape(-1).astype(np.float64)
        y = original[:, :, channel].reshape(-1).astype(np.float64)
        finite = np.isfinite(x) & np.isfinite(y)
        if not bool(np.any(finite)):
            continue
        design = np.stack([x[finite], np.ones(int(np.count_nonzero(finite)))], axis=1)
        scale, offset = np.linalg.lstsq(design, y[finite], rcond=None)[0]
        corrected[:, :, channel] = (
            decoded[:, :, channel].astype(np.float64) * scale + offset
        ).astype(np.float32)
    return np.ascontiguousarray(corrected), 6 * 4


def display_bias_stats(original: np.ndarray, decoded: np.ndarray, white: float, gamma: float) -> dict:
    original_display = display_map(original, white, gamma)
    decoded_display = display_map(decoded, white, gamma)
    original_luma = luminance(original_display)
    decoded_luma = luminance(decoded_display)
    masks = region_masks(original)
    rows = {}
    for name in ("dark_0_0.25", "mid_0.25_1", "highlight_1_4", "all"):
        mask = masks[name]
        if not bool(np.any(mask)):
            rows[name] = {
                "samples": 0,
                "mean_luma_delta": 0.0,
                "mean_rgb_delta": [0.0, 0.0, 0.0],
                "abs_mean_rgb_delta": 0.0,
            }
            continue
        rgb_delta = decoded_display[mask] - original_display[mask]
        luma_delta = decoded_luma[mask] - original_luma[mask]
        mean_rgb = np.mean(rgb_delta, axis=0)
        rows[name] = {
            "samples": int(np.count_nonzero(mask)),
            "mean_luma_delta": float(np.mean(luma_delta)),
            "mean_rgb_delta": [float(value) for value in mean_rgb],
            "abs_mean_rgb_delta": float(np.mean(np.abs(mean_rgb))),
        }
    return rows


def summarize_case(
    pixels: np.ndarray,
    anchor_regions: dict,
    anchor_regions_lift: dict,
    y_bits: int,
    chroma_bits: int,
    transform: str,
    power_gamma: float,
    recon_mode: str,
    white: float,
    lift_white: float,
    gamma: float,
) -> dict:
    bits = (y_bits, chroma_bits, chroma_bits)
    indices, decoded, table_bytes = quantize_ycocg(
        pixels,
        bits,
        transform,
        power_gamma,
        recon_mode,
    )
    payload = estimate_index_bytes(indices, bits)
    estimated_bytes = int(payload["estimated_bytes"] + table_bytes)
    regions = candidate_metrics(pixels, decoded, white, gamma)
    regions_lift = candidate_metrics(pixels, decoded, lift_white, gamma)
    gate = evaluate_gate(regions, anchor_regions, default_gate_args())
    gate_lift = evaluate_gate(regions_lift, anchor_regions_lift, default_gate_args())
    key_metrics = summarize_key_metrics(regions, anchor_regions)
    key_metrics_lift = summarize_key_metrics(regions_lift, anchor_regions_lift)
    return {
        "label": f"ycocg_Y{y_bits}C{chroma_bits}_{transform}_{recon_mode}",
        "bits": list(bits),
        "transform": transform,
        "power_gamma": power_gamma,
        "recon_mode": recon_mode,
        "raw_bytes": int(pixels.nbytes),
        "estimated_bytes": estimated_bytes,
        "estimated_ratio": pixels.nbytes / float(estimated_bytes),
        "index_payload": payload,
        "table_bytes": table_bytes,
        "gate": gate,
        "gate_lift": gate_lift,
        "key_metrics": key_metrics,
        "key_metrics_lift": key_metrics_lift,
        "bias": display_bias_stats(pixels, decoded, white, gamma),
        "bias_lift": display_bias_stats(pixels, decoded, lift_white, gamma),
        "decoded": decoded,
    }


def decision_rank(decision: str) -> int:
    return {"pass": 0, "maybe": 1, "reject": 2}.get(decision, 9)


def print_row(row: dict, prefix: str = "") -> None:
    dark = row["key_metrics"]["dark_0_0.25"]
    lift_dark = row["key_metrics_lift"]["dark_0_0.25"]
    bias = row["bias"]["all"]
    print(
        f"{prefix}{row['gate']['decision'].upper():6s}/"
        f"{row['gate_lift']['decision'].upper():6s} "
        f"{row['label']:42s} "
        f"est={row['estimated_bytes']:,} ratio={row['estimated_ratio']:.2f}x "
        f"darkD={dark['lost_detail_delta_vs_anchor']:.2%} "
        f"liftDarkD={lift_dark['lost_detail_delta_vs_anchor']:.2%} "
        f"meanL={bias['mean_luma_delta']:+.2e} "
        f"meanRGB=({bias['mean_rgb_delta'][0]:+.2e},"
        f"{bias['mean_rgb_delta'][1]:+.2e},"
        f"{bias['mean_rgb_delta'][2]:+.2e})",
        flush=True,
    )


def export_preview(
    path: Path,
    pixels: np.ndarray,
    row: dict,
    crop_size: int,
    white: float,
    gamma: float,
    output_dir: Path,
    decoded_only: bool,
) -> dict:
    stem = safe_name(path.stem)
    crop_label = "full" if crop_size == 0 else f"crop{crop_size}"
    original_png = output_dir / f"{stem}_{crop_label}_original_w{white:g}_g{gamma:g}.png"
    decoded_png = output_dir / f"{stem}_{crop_label}_{safe_name(row['label'])}_w{white:g}_g{gamma:g}_decoded.png"
    if not decoded_only and not original_png.exists():
        write_png(original_png, to_preview_u16(pixels, white, gamma))
    write_png(decoded_png, to_preview_u16(row["decoded"], white, gamma))
    return {
        "original_png": None if decoded_only else str(original_png),
        "decoded_png": str(decoded_png),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="sample_DSCF0009.EXR")
    parser.add_argument("--crop-size", type=int, default=1024)
    parser.add_argument("--y-bits", default="9,10")
    parser.add_argument("--chroma-bits", default="6")
    parser.add_argument("--transform", default="gamma075")
    parser.add_argument("--power-gamma", type=float, default=0.75)
    parser.add_argument(
        "--recon-modes",
        default=(
            "center,center+gamut-project,transform-mean,value-mean,"
            "value-mean+gamut-project,center+rgb-affine,value-mean+rgb-affine"
        ),
    )
    parser.add_argument("--anchor-transform", default="signed-log")
    parser.add_argument("--anchor-bits", type=int, default=10)
    parser.add_argument("--white", type=float, default=4.0)
    parser.add_argument("--lift-white", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=2.2)
    parser.add_argument("--export-modes", default="")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--decoded-only", action="store_true")
    parser.add_argument("--top", type=int, default=18)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    rows_out = []
    export_modes = set(parse_strings(args.export_modes))
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
        for y_bits in parse_ints(args.y_bits):
            for chroma_bits in parse_ints(args.chroma_bits):
                for recon_mode in parse_strings(args.recon_modes):
                    row = summarize_case(
                        pixels,
                        anchor_regions,
                        anchor_regions_lift,
                        y_bits,
                        chroma_bits,
                        args.transform,
                        args.power_gamma,
                        recon_mode,
                        args.white,
                        args.lift_white,
                        args.gamma,
                    )
                    if recon_mode in export_modes:
                        row.update(
                            export_preview(
                                path,
                                pixels,
                                row,
                                args.crop_size,
                                args.white,
                                args.gamma,
                                args.output_dir,
                                args.decoded_only,
                            )
                        )
                    rows.append(row)
        rows.sort(
            key=lambda row: (
                decision_rank(row["gate"]["decision"]),
                decision_rank(row["gate_lift"]["decision"]),
                row["estimated_bytes"],
                row["bias"]["all"]["abs_mean_rgb_delta"],
            )
        )
        for row in rows[:args.top]:
            print_row(row, prefix="  ")
        serializable_rows = []
        for row in rows:
            row_copy = dict(row)
            row_copy.pop("decoded", None)
            serializable_rows.append(row_copy)
        rows_out.append({
            "image": path.name,
            "shape": list(pixels.shape),
            "crop_size": args.crop_size,
            "white": args.white,
            "lift_white": args.lift_white,
            "gamma": args.gamma,
            "anchor": f"quant:{args.anchor_transform}:{args.anchor_bits}",
            "rows": serializable_rows,
        })
        if args.limit and len(rows_out) >= args.limit:
            break
    if not rows_out:
        print(f"no files matched: {args.glob}", flush=True)
        return 1
    if not args.no_save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        output = RESULTS_DIR / (
            f"ycocg_recon_table_{safe_name(args.glob)}"
            f"_crop{args.crop_size}_Y{safe_name(args.y_bits)}_C{safe_name(args.chroma_bits)}.json"
        )
        output.write_text(json.dumps(rows_out, indent=2))
        print(f"Saved: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
