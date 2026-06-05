"""Probe an explicitly denoised VST profile targeting very high ratios.

This branch intentionally changes the image.  It is not near-lossless faithful
compression.  The route separates both luma and chroma in VST/YCoCg space:

* low-frequency Y and Co/Cg are guided-filtered, downsampled, and quantized;
* high-frequency residuals are soft-thresholded sparse escapes;
* discarded high-frequency residual is treated as denoising, not lossless error.

The goal is to see whether a Lightroom-like denoised profile can approach
10 MB on the full DSCF sample while staying visually pleasant.
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
OUTPUT_DIR = ROOT / "outputs" / "previews" / "vst_denoised_profile"
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
from probe_color_tile_transform_index import forward_color, inverse_color  # noqa: E402
from probe_darkbits_router_payload import read_exr, summarize_binary_mask  # noqa: E402
from probe_vst_chroma_nr import (  # noqa: E402
    block_mean_downsample,
    forward_vst,
    guided_filter_gray,
    inverse_vst,
    parse_floats,
    parse_ints,
    parse_strings,
    parse_thresholds,
    quantize_range,
    residual_entropy_indices,
    safe_name,
    selected_entropy_indices,
    shrink_highpass,
    upsample_repeat,
)
from probe_ycocg_recon_table import display_bias_stats  # noqa: E402


def quantized_low_plane(values: np.ndarray, scale: int, bits: int) -> tuple[np.ndarray, np.ndarray, dict]:
    coarse = block_mean_downsample(values, scale)
    q, rec, value_range = quantize_range(coarse, bits)
    decoded = upsample_repeat(rec, values.shape, scale)
    payload = residual_entropy_indices(q[:, :, None], (bits,))
    return decoded, q, {"range": value_range, "payload": payload}


def sparse_high_payload(high: np.ndarray, bits: int, mask: np.ndarray) -> tuple[np.ndarray, dict]:
    high_3d = high[:, :, None].astype(np.float32)
    payload = selected_entropy_indices(high_3d, bits, mask)
    decoded = payload["decoded"][:, :, 0]
    if bool(np.any(mask)):
        mask_summary = summarize_binary_mask(mask)
        mask_payload = min(
            mask_summary["order0"],
            mask_summary["west_north"],
            key=lambda row: row["total_bytes"],
        )
    else:
        mask_payload = {"total_bytes": 0, "estimated_bits": 0.0, "mode": "empty"}
    payload = dict(payload)
    payload.pop("decoded", None)
    payload["mask_payload"] = mask_payload
    payload["estimated_bytes_with_mask"] = int(payload["estimated_bytes"] + int(mask_payload["total_bytes"]))
    return decoded, payload


def encode_decode_case(
    pixels: np.ndarray,
    vst_mode: str,
    vst_a: float,
    vst_b: float,
    vst_eps: float,
    y_low_bits: int,
    y_low_scale: int,
    y_radius: int,
    y_guide_eps: float,
    y_threshold_mult: float | None,
    y_high_bits: int,
    chroma_low_bits: int,
    chroma_low_scale: int,
    chroma_radius: int,
    chroma_guide_eps: float,
    chroma_threshold_mult: float | None,
    chroma_high_bits: int,
) -> tuple[np.ndarray, dict]:
    rgb = pixels[:, :, :3]
    t_rgb = forward_vst(rgb, vst_mode, vst_a, vst_b, vst_eps).astype(np.float32)
    planes = forward_color(t_rgb, "ycocg")[:, :, :3]
    y_plane = planes[:, :, 0]
    chroma = planes[:, :, 1:3]

    y_low_full = guided_filter_gray(
        y_plane.astype(np.float64),
        y_plane.astype(np.float64),
        y_radius,
        y_guide_eps,
    )
    y_low_decoded, _, y_low_payload = quantized_low_plane(
        y_low_full,
        y_low_scale,
        y_low_bits,
    )
    y_high = y_plane - y_low_full
    y_high_kept, y_thresholds = shrink_highpass(y_high[:, :, None], y_threshold_mult)
    y_high_mask = np.abs(y_high_kept[:, :, 0]) > 0.0
    y_high_decoded, y_high_payload = sparse_high_payload(
        y_high_kept[:, :, 0],
        y_high_bits,
        y_high_mask,
    )

    guide = y_plane.astype(np.float64)
    chroma_low_full = np.zeros(chroma.shape, dtype=np.float32)
    for channel in range(2):
        chroma_low_full[:, :, channel] = guided_filter_gray(
            guide,
            chroma[:, :, channel].astype(np.float64),
            chroma_radius,
            chroma_guide_eps,
        )
    chroma_low_indices = []
    chroma_low_decoded = np.zeros(chroma.shape, dtype=np.float32)
    chroma_ranges = []
    for channel in range(2):
        decoded, q, payload = quantized_low_plane(
            chroma_low_full[:, :, channel],
            chroma_low_scale,
            chroma_low_bits,
        )
        chroma_low_decoded[:, :, channel] = decoded
        chroma_low_indices.append(q)
        chroma_ranges.append(payload["range"])
    chroma_idx_stack = np.stack(chroma_low_indices, axis=2)
    chroma_low_payload = residual_entropy_indices(
        chroma_idx_stack,
        (chroma_low_bits, chroma_low_bits),
    )

    chroma_high = chroma - chroma_low_full
    chroma_high_kept, chroma_thresholds = shrink_highpass(chroma_high, chroma_threshold_mult)
    chroma_high_mask = np.any(np.abs(chroma_high_kept) > 0.0, axis=2)
    chroma_high_payload = selected_entropy_indices(
        chroma_high_kept,
        chroma_high_bits,
        chroma_high_mask,
    )
    chroma_high_decoded = chroma_high_payload["decoded"]
    if bool(np.any(chroma_high_mask)):
        mask_summary = summarize_binary_mask(chroma_high_mask)
        chroma_mask_payload = min(
            mask_summary["order0"],
            mask_summary["west_north"],
            key=lambda row: row["total_bytes"],
        )
    else:
        chroma_mask_payload = {"total_bytes": 0, "estimated_bits": 0.0, "mode": "empty"}
    chroma_high_payload = dict(chroma_high_payload)
    chroma_high_payload.pop("decoded", None)
    chroma_high_payload["mask_payload"] = chroma_mask_payload
    chroma_high_payload["estimated_bytes_with_mask"] = int(
        chroma_high_payload["estimated_bytes"] + int(chroma_mask_payload["total_bytes"])
    )

    decoded_planes = np.zeros(planes.shape, dtype=np.float32)
    decoded_planes[:, :, 0] = y_low_decoded + y_high_decoded
    decoded_planes[:, :, 1:3] = chroma_low_decoded + chroma_high_decoded
    decoded_t_rgb = inverse_color(decoded_planes, "ycocg")[:, :, :3]
    decoded_rgb = inverse_vst(decoded_t_rgb, vst_mode, vst_a, vst_b, vst_eps)
    decoded = np.array(pixels, copy=True)
    decoded[:, :, :3] = decoded_rgb.astype(np.float32)

    metadata_bytes = 320
    estimated_bytes = int(
        y_low_payload["payload"]["estimated_bytes"]
        + y_high_payload["estimated_bytes_with_mask"]
        + chroma_low_payload["estimated_bytes"]
        + chroma_high_payload["estimated_bytes_with_mask"]
        + metadata_bytes
    )
    return np.ascontiguousarray(decoded), {
        "vst_mode": vst_mode,
        "vst_a": vst_a,
        "vst_b": vst_b,
        "vst_eps": vst_eps,
        "y_low_bits": y_low_bits,
        "y_low_scale": y_low_scale,
        "y_radius": y_radius,
        "y_guide_eps": y_guide_eps,
        "y_threshold_mult": y_threshold_mult,
        "y_thresholds": y_thresholds,
        "y_high_bits": y_high_bits,
        "y_high_pixel_rate": float(np.mean(y_high_mask)),
        "chroma_low_bits": chroma_low_bits,
        "chroma_low_scale": chroma_low_scale,
        "chroma_radius": chroma_radius,
        "chroma_guide_eps": chroma_guide_eps,
        "chroma_threshold_mult": chroma_threshold_mult,
        "chroma_thresholds": chroma_thresholds,
        "chroma_high_bits": chroma_high_bits,
        "chroma_high_pixel_rate": float(np.mean(chroma_high_mask)),
        "estimated_bytes": estimated_bytes,
        "payload": {
            "y_low": y_low_payload,
            "y_high": y_high_payload,
            "chroma_low": {
                "payload": chroma_low_payload,
                "ranges": chroma_ranges,
            },
            "chroma_high": chroma_high_payload,
            "metadata_bytes": metadata_bytes,
        },
    }


def summarize_case(
    pixels: np.ndarray,
    anchor_regions: dict,
    anchor_regions_lift: dict,
    args: argparse.Namespace,
    y_low_bits: int,
    y_low_scale: int,
    y_radius: int,
    y_guide_eps: float,
    y_threshold_mult: float | None,
    y_high_bits: int,
    chroma_low_bits: int,
    chroma_low_scale: int,
    chroma_radius: int,
    chroma_guide_eps: float,
    chroma_threshold_mult: float | None,
    chroma_high_bits: int,
) -> dict:
    decoded, payload = encode_decode_case(
        pixels,
        args.vst_mode,
        args.vst_a,
        args.vst_b,
        args.vst_eps,
        y_low_bits,
        y_low_scale,
        y_radius,
        y_guide_eps,
        y_threshold_mult,
        y_high_bits,
        chroma_low_bits,
        chroma_low_scale,
        chroma_radius,
        chroma_guide_eps,
        chroma_threshold_mult,
        chroma_high_bits,
    )
    regions = candidate_metrics(pixels, decoded, args.white, args.gamma)
    regions_lift = candidate_metrics(pixels, decoded, args.lift_white, args.gamma)
    gate = evaluate_gate(regions, anchor_regions, default_gate_args())
    gate_lift = evaluate_gate(regions_lift, anchor_regions_lift, default_gate_args())
    key_metrics = summarize_key_metrics(regions, anchor_regions)
    key_metrics_lift = summarize_key_metrics(regions_lift, anchor_regions_lift)
    yt = "discard" if y_threshold_mult is None else f"{y_threshold_mult:g}"
    ct = "discard" if chroma_threshold_mult is None else f"{chroma_threshold_mult:g}"
    label = (
        f"vstdenoise_{args.vst_mode}_YL{y_low_bits}s{y_low_scale}"
        f"_YH{y_high_bits}t{yt}_CL{chroma_low_bits}s{chroma_low_scale}"
        f"_CH{chroma_high_bits}t{ct}_yr{y_radius}_cr{chroma_radius}"
    )
    return {
        "label": label,
        "raw_bytes": int(pixels.nbytes),
        "estimated_bytes": int(payload["estimated_bytes"]),
        "estimated_ratio": pixels.nbytes / float(payload["estimated_bytes"]),
        "payload": payload,
        "gate": gate,
        "gate_lift": gate_lift,
        "key_metrics": key_metrics,
        "key_metrics_lift": key_metrics_lift,
        "bias": display_bias_stats(pixels, decoded, args.white, args.gamma),
        "bias_lift": display_bias_stats(pixels, decoded, args.lift_white, args.gamma),
        "decoded": decoded,
    }


def decision_rank(decision: str) -> int:
    return {"pass": 0, "maybe": 1, "reject": 2}.get(decision, 9)


def print_row(row: dict, prefix: str = "") -> None:
    dark = row["key_metrics"]["dark_0_0.25"]
    lift_dark = row["key_metrics_lift"]["dark_0_0.25"]
    high = row["key_metrics"]["highlight_1_4"]
    payload = row["payload"]
    print(
        f"{prefix}{row['gate']['decision'].upper():6s}/"
        f"{row['gate_lift']['decision'].upper():6s} "
        f"{row['label']:64s} "
        f"est={row['estimated_bytes']:,} ratio={row['estimated_ratio']:.2f}x "
        f"yh={payload['y_high_pixel_rate']:.2%} "
        f"ch={payload['chroma_high_pixel_rate']:.2%} "
        f"darkD={dark['lost_detail_delta_vs_anchor']:.2%} "
        f"liftDarkD={lift_dark['lost_detail_delta_vs_anchor']:.2%} "
        f"hiD={high['lost_detail_delta_vs_anchor']:.2%}",
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
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = safe_name(path.stem)
    crop_label = "full" if crop_size == 0 else f"crop{crop_size}"
    original_png = output_dir / f"{stem}_{crop_label}_original_w{white:g}_g{gamma:g}.png"
    decoded_png = output_dir / f"{stem}_{crop_label}_{safe_name(row['label'])}_w{white:g}_g{gamma:g}_decoded.png"
    if not original_png.exists():
        write_png(original_png, to_preview_u16(pixels, white, gamma))
    write_png(decoded_png, to_preview_u16(row["decoded"], white, gamma))
    return {"original_png": str(original_png), "decoded_png": str(decoded_png)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="sample_DSCF0009.EXR")
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--vst-mode", default="gamma075")
    parser.add_argument("--vst-a", type=float, default=1.0)
    parser.add_argument("--vst-b", type=float, default=0.01)
    parser.add_argument("--vst-eps", type=float, default=1.0e-4)
    parser.add_argument("--y-low-bits", default="8")
    parser.add_argument("--y-low-scales", default="2,4")
    parser.add_argument("--y-radii", default="2")
    parser.add_argument("--y-guide-eps", default="0.05,0.1")
    parser.add_argument("--y-threshold-mults", default="2,3")
    parser.add_argument("--y-high-bits", default="5,6")
    parser.add_argument("--chroma-low-bits", default="6,7")
    parser.add_argument("--chroma-low-scales", default="4")
    parser.add_argument("--chroma-radii", default="2")
    parser.add_argument("--chroma-guide-eps", default="0.1")
    parser.add_argument("--chroma-threshold-mults", default="3,4")
    parser.add_argument("--chroma-high-bits", default="4,5")
    parser.add_argument("--anchor-transform", default="signed-log")
    parser.add_argument("--anchor-bits", type=int, default=10)
    parser.add_argument("--white", type=float, default=4.0)
    parser.add_argument("--lift-white", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=2.2)
    parser.add_argument("--export-labels", default="")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--top", type=int, default=18)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    export_labels = set(part.strip() for part in args.export_labels.split(",") if part.strip())
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
        anchor_regions_lift = candidate_metrics(pixels, anchor, args.lift_white, args.gamma)
        rows = []
        for y_low_bits in parse_ints(args.y_low_bits):
            for y_low_scale in parse_ints(args.y_low_scales):
                for y_radius in parse_ints(args.y_radii):
                    for y_guide_eps in parse_floats(args.y_guide_eps):
                        for y_threshold_mult in parse_thresholds(args.y_threshold_mults):
                            for y_high_bits in parse_ints(args.y_high_bits):
                                for chroma_low_bits in parse_ints(args.chroma_low_bits):
                                    for chroma_low_scale in parse_ints(args.chroma_low_scales):
                                        for chroma_radius in parse_ints(args.chroma_radii):
                                            for chroma_guide_eps in parse_floats(args.chroma_guide_eps):
                                                for chroma_threshold_mult in parse_thresholds(args.chroma_threshold_mults):
                                                    for chroma_high_bits in parse_ints(args.chroma_high_bits):
                                                        row = summarize_case(
                                                            pixels,
                                                            anchor_regions,
                                                            anchor_regions_lift,
                                                            args,
                                                            y_low_bits,
                                                            y_low_scale,
                                                            y_radius,
                                                            y_guide_eps,
                                                            y_threshold_mult,
                                                            y_high_bits,
                                                            chroma_low_bits,
                                                            chroma_low_scale,
                                                            chroma_radius,
                                                            chroma_guide_eps,
                                                            chroma_threshold_mult,
                                                            chroma_high_bits,
                                                        )
                                                        if row["label"] in export_labels:
                                                            row.update(
                                                                export_preview(
                                                                    path,
                                                                    pixels,
                                                                    row,
                                                                    args.crop_size,
                                                                    args.white,
                                                                    args.gamma,
                                                                    args.output_dir,
                                                                )
                                                            )
                                                        rows.append(row)
        rows.sort(
            key=lambda row: (
                decision_rank(row["gate"]["decision"]),
                decision_rank(row["gate_lift"]["decision"]),
                row["estimated_bytes"],
            )
        )
        for row in rows[:args.top]:
            print_row(row, prefix="  ")
            if "decoded_png" in row:
                print(f"    decoded_png={row['decoded_png']}", flush=True)
        serializable = []
        for row in rows:
            row_copy = dict(row)
            row_copy.pop("decoded", None)
            serializable.append(row_copy)
        summaries.append({
            "image": path.name,
            "shape": list(pixels.shape),
            "crop_size": args.crop_size,
            "white": args.white,
            "lift_white": args.lift_white,
            "gamma": args.gamma,
            "anchor": f"quant:{args.anchor_transform}:{args.anchor_bits}",
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
            f"vst_denoised_profile_{safe_name(args.glob)}"
            f"_crop{args.crop_size}.json"
        )
        output.write_text(json.dumps(summaries, indent=2))
        print(f"Saved: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
