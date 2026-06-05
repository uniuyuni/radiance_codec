"""Probe a VST-space chroma-only NR route for visual compression.

This is a codec research probe, not a production bitstream.  It deliberately
avoids cloning a vendor NR pipeline.  The experiment uses general components:

* sign-preserving variance-stabilizing transforms on RGB;
* YCoCg in that transformed space;
* guided low-pass only for Co/Cg, using Y as guide;
* sparse soft-thresholded chroma high-pass escapes.

The point is to test whether chroma noise can be reduced/represented cheaply
without touching luma enough to trigger dark banding.
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
OUTPUT_DIR = ROOT / "outputs" / "previews" / "vst_chroma_nr"
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
from probe_darkbits_router_payload import read_exr, summarize_binary_mask  # noqa: E402
from probe_log_signal_noise_resynth import box_mean_2d  # noqa: E402
from probe_ycocg_recon_table import display_bias_stats  # noqa: E402


def parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(",") if part.strip())


def parse_floats(text: str) -> tuple[float, ...]:
    return tuple(float(part) for part in text.split(",") if part.strip())


def parse_strings(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in text.split(",") if part.strip())


def parse_thresholds(text: str) -> tuple[float | None, ...]:
    out: list[float | None] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if part in {"discard", "none", "inf"}:
            out.append(None)
        else:
            out.append(float(part))
    return tuple(out)


def safe_name(text: str) -> str:
    return (
        text.replace("*", "star")
        .replace(".", "_")
        .replace(",", "-")
        .replace("/", "_")
        .replace(":", "_")
    )


def forward_vst(values: np.ndarray, mode: str, a: float, b: float, eps: float) -> np.ndarray:
    x = values.astype(np.float64)
    sign = np.sign(x)
    ax = np.abs(x)
    if mode == "linear":
        return x
    if mode == "log":
        return sign * np.log2(1.0 + ax / float(eps))
    if mode == "asinh":
        return np.arcsinh(x / float(eps))
    if mode == "gamma075":
        return sign * np.power(ax, 0.75)
    if mode == "sqrtvst":
        aa = max(float(a), 1.0e-12)
        bb = max(float(b), 1.0e-12)
        return sign * (2.0 / aa) * (np.sqrt(aa * ax + bb) - math.sqrt(bb))
    raise ValueError(f"unknown vst mode: {mode}")


def inverse_vst(values: np.ndarray, mode: str, a: float, b: float, eps: float) -> np.ndarray:
    y = values.astype(np.float64)
    sign = np.sign(y)
    ay = np.abs(y)
    if mode == "linear":
        return y
    if mode == "log":
        return sign * float(eps) * (np.exp2(ay) - 1.0)
    if mode == "asinh":
        return float(eps) * np.sinh(y)
    if mode == "gamma075":
        return sign * np.power(ay, 1.0 / 0.75)
    if mode == "sqrtvst":
        aa = max(float(a), 1.0e-12)
        bb = max(float(b), 1.0e-12)
        root = ay * aa * 0.5 + math.sqrt(bb)
        return sign * np.maximum((root * root - bb) / aa, 0.0)
    raise ValueError(f"unknown vst mode: {mode}")


def guided_filter_gray(guide: np.ndarray, values: np.ndarray, radius: int, eps: float) -> np.ndarray:
    mean_i = box_mean_2d(guide, radius)
    mean_p = box_mean_2d(values, radius)
    corr_i = box_mean_2d(guide * guide, radius)
    corr_ip = box_mean_2d(guide * values, radius)
    var_i = corr_i - mean_i * mean_i
    cov_ip = corr_ip - mean_i * mean_p
    a = cov_ip / (var_i + float(eps))
    b = mean_p - a * mean_i
    mean_a = box_mean_2d(a, radius)
    mean_b = box_mean_2d(b, radius)
    return (mean_a * guide + mean_b).astype(np.float32)


def block_mean_downsample(values: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return values.astype(np.float32, copy=True)
    h, w = values.shape
    by = math.ceil(h / scale)
    bx = math.ceil(w / scale)
    out = np.zeros((by, bx), dtype=np.float32)
    for yb in range(by):
        for xb in range(bx):
            tile = values[yb * scale:min((yb + 1) * scale, h), xb * scale:min((xb + 1) * scale, w)]
            out[yb, xb] = np.float32(np.mean(tile))
    return out


def upsample_repeat(values: np.ndarray, shape: tuple[int, int], scale: int) -> np.ndarray:
    if scale <= 1:
        return values.astype(np.float32, copy=True)
    out = np.repeat(np.repeat(values, scale, axis=0), scale, axis=1)
    return out[:shape[0], :shape[1]].astype(np.float32)


def quantize_range(values: np.ndarray, bits: int) -> tuple[np.ndarray, np.ndarray, dict]:
    levels = (1 << bits) - 1
    vals = values.astype(np.float64)
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    q = np.zeros(vals.shape, dtype=np.uint16)
    if not hi > lo:
        return q, np.full(vals.shape, lo, dtype=np.float32), {"lo": lo, "hi": hi}
    qf = np.floor((vals - lo) / (hi - lo) * levels + 0.5)
    q = np.clip(qf, 0, levels).astype(np.uint16)
    rec = lo + q.astype(np.float64) * (hi - lo) / levels
    return q, rec.astype(np.float32), {"lo": lo, "hi": hi}


def residual_entropy_indices(indices: np.ndarray, bits: tuple[int, ...]) -> dict:
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
            "samples": int(residual.size),
            "estimated_bits": bit_cost,
            "bits_per_sample": bit_cost / float(residual.size),
        })
    return {
        "estimated_bits": total_bits,
        "estimated_bytes": int(math.ceil(total_bits / 8.0)),
        "channels": rows,
    }


def selected_entropy_indices(values: np.ndarray, bits: int, mask: np.ndarray) -> dict:
    samples = int(np.count_nonzero(mask))
    if samples == 0:
        return {
            "estimated_bits": 0.0,
            "estimated_bytes": 0,
            "ranges": [],
            "channels": [],
            "decoded": np.zeros(values.shape, dtype=np.float32),
        }
    selected = values[mask]
    decoded = np.zeros(values.shape, dtype=np.float32)
    indices = np.zeros(selected.shape, dtype=np.uint16)
    ranges = []
    rows = []
    total_bits = 0.0
    for channel in range(values.shape[2]):
        q, rec, channel_range = quantize_range(selected[:, channel], bits)
        indices[:, channel] = q
        decoded[:, :, channel][mask] = rec
        counts = np.bincount(q.astype(np.int64), minlength=1 << bits)
        bit_cost = entropy_bits_from_counts(counts)
        total_bits += bit_cost
        ranges.append(channel_range)
        rows.append({
            "channel": channel,
            "bits": bits,
            "samples": samples,
            "estimated_bits": bit_cost,
            "bits_per_sample": bit_cost / float(samples),
        })
    return {
        "estimated_bits": total_bits,
        "estimated_bytes": int(math.ceil(total_bits / 8.0)),
        "ranges": ranges,
        "channels": rows,
        "decoded": decoded,
    }


def robust_sigma(values: np.ndarray) -> float:
    flat = values.reshape(-1).astype(np.float64)
    med = float(np.median(flat))
    mad = float(np.median(np.abs(flat - med)))
    return max(1.4826 * mad, 1.0e-12)


def shrink_highpass(high: np.ndarray, threshold_mult: float | None) -> tuple[np.ndarray, list[float]]:
    out = np.zeros(high.shape, dtype=np.float32)
    thresholds = []
    if threshold_mult is None:
        return out, [math.inf for _ in range(high.shape[2])]
    for channel in range(high.shape[2]):
        sigma = robust_sigma(high[:, :, channel])
        threshold = float(threshold_mult) * sigma
        thresholds.append(threshold)
        vals = high[:, :, channel].astype(np.float64)
        mag = np.maximum(np.abs(vals) - threshold, 0.0)
        out[:, :, channel] = (np.sign(vals) * mag).astype(np.float32)
    return out, thresholds


def encode_decode_case(
    pixels: np.ndarray,
    vst_mode: str,
    vst_a: float,
    vst_b: float,
    vst_eps: float,
    y_bits: int,
    chroma_low_bits: int,
    high_bits: int,
    low_scale: int,
    guide_radius: int,
    guide_eps: float,
    threshold_mult: float | None,
) -> tuple[np.ndarray, dict]:
    rgb = pixels[:, :, :3]
    t_rgb = forward_vst(rgb, vst_mode, vst_a, vst_b, vst_eps).astype(np.float32)
    planes = forward_color(t_rgb, "ycocg")[:, :, :3]
    y_plane = planes[:, :, 0]
    chroma = planes[:, :, 1:3]

    y_idx, y_decoded, y_range = quantize_range(y_plane, y_bits)
    guide = y_plane.astype(np.float64)
    low_full = np.zeros(chroma.shape, dtype=np.float32)
    for channel in range(2):
        low_full[:, :, channel] = guided_filter_gray(
            guide,
            chroma[:, :, channel].astype(np.float64),
            guide_radius,
            guide_eps,
        )
    high = chroma - low_full
    high_kept, high_thresholds = shrink_highpass(high, threshold_mult)

    low_indices = []
    low_decoded_full = np.zeros(chroma.shape, dtype=np.float32)
    low_ranges = []
    for channel in range(2):
        coarse = block_mean_downsample(low_full[:, :, channel], low_scale)
        q, rec, channel_range = quantize_range(coarse, chroma_low_bits)
        low_indices.append(q)
        low_ranges.append(channel_range)
        low_decoded_full[:, :, channel] = upsample_repeat(rec, y_plane.shape, low_scale)
    low_idx_stack = np.stack(low_indices, axis=2)

    high_mask = np.any(np.abs(high_kept) > 0.0, axis=2)
    high_payload = selected_entropy_indices(high_kept, high_bits, high_mask)
    high_decoded = high_payload["decoded"]

    decoded_planes = np.zeros(planes.shape, dtype=np.float32)
    decoded_planes[:, :, 0] = y_decoded
    decoded_planes[:, :, 1:3] = low_decoded_full + high_decoded
    decoded_t_rgb = inverse_color(decoded_planes, "ycocg")[:, :, :3]
    decoded_rgb = inverse_vst(decoded_t_rgb, vst_mode, vst_a, vst_b, vst_eps)
    decoded = np.array(pixels, copy=True)
    decoded[:, :, :3] = decoded_rgb.astype(np.float32)

    y_payload = residual_entropy_indices(y_idx[:, :, None], (y_bits,))
    low_payload = residual_entropy_indices(
        low_idx_stack,
        (chroma_low_bits, chroma_low_bits),
    )
    if bool(np.any(high_mask)):
        mask_summary = summarize_binary_mask(high_mask)
        high_mask_payload = min(
            mask_summary["order0"],
            mask_summary["west_north"],
            key=lambda row: row["total_bytes"],
        )
    else:
        high_mask_payload = {"total_bytes": 0, "estimated_bits": 0.0, "mode": "empty"}
    metadata_bytes = 256
    high_bytes = int(high_payload["estimated_bytes"] + int(high_mask_payload["total_bytes"]))
    estimated_bytes = int(
        y_payload["estimated_bytes"]
        + low_payload["estimated_bytes"]
        + high_bytes
        + metadata_bytes
    )
    return np.ascontiguousarray(decoded), {
        "vst_mode": vst_mode,
        "vst_a": vst_a,
        "vst_b": vst_b,
        "vst_eps": vst_eps,
        "y_bits": y_bits,
        "chroma_low_bits": chroma_low_bits,
        "high_bits": high_bits,
        "low_scale": low_scale,
        "guide_radius": guide_radius,
        "guide_eps": guide_eps,
        "threshold_mult": threshold_mult,
        "high_thresholds": high_thresholds,
        "high_pixel_rate": float(np.mean(high_mask)),
        "estimated_bytes": estimated_bytes,
        "payload": {
            "y": y_payload,
            "y_range": y_range,
            "chroma_low": low_payload,
            "chroma_low_ranges": low_ranges,
            "high": {
                "estimated_bytes": high_payload["estimated_bytes"],
                "channels": high_payload["channels"],
                "ranges": high_payload["ranges"],
                "mask_payload": high_mask_payload,
            },
            "metadata_bytes": metadata_bytes,
        },
    }


def summarize_case(
    pixels: np.ndarray,
    anchor_regions: dict,
    anchor_regions_lift: dict,
    vst_mode: str,
    vst_a: float,
    vst_b: float,
    vst_eps: float,
    y_bits: int,
    chroma_low_bits: int,
    high_bits: int,
    low_scale: int,
    guide_radius: int,
    guide_eps: float,
    threshold_mult: float | None,
    white: float,
    lift_white: float,
    gamma: float,
) -> dict:
    decoded, payload = encode_decode_case(
        pixels,
        vst_mode,
        vst_a,
        vst_b,
        vst_eps,
        y_bits,
        chroma_low_bits,
        high_bits,
        low_scale,
        guide_radius,
        guide_eps,
        threshold_mult,
    )
    regions = candidate_metrics(pixels, decoded, white, gamma)
    regions_lift = candidate_metrics(pixels, decoded, lift_white, gamma)
    gate = evaluate_gate(regions, anchor_regions, default_gate_args())
    gate_lift = evaluate_gate(regions_lift, anchor_regions_lift, default_gate_args())
    key_metrics = summarize_key_metrics(regions, anchor_regions)
    key_metrics_lift = summarize_key_metrics(regions_lift, anchor_regions_lift)
    threshold_label = "discard" if threshold_mult is None else f"{threshold_mult:g}"
    label = (
        f"vstchroma_{vst_mode}_Y{y_bits}_CL{chroma_low_bits}"
        f"_H{high_bits}_s{low_scale}_r{guide_radius}"
        f"_ge{guide_eps:g}_tm{threshold_label}"
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
        "bias": display_bias_stats(pixels, decoded, white, gamma),
        "bias_lift": display_bias_stats(pixels, decoded, lift_white, gamma),
        "decoded": decoded,
    }


def decision_rank(decision: str) -> int:
    return {"pass": 0, "maybe": 1, "reject": 2}.get(decision, 9)


def print_row(row: dict, prefix: str = "") -> None:
    dark = row["key_metrics"]["dark_0_0.25"]
    lift_dark = row["key_metrics_lift"]["dark_0_0.25"]
    high = row["key_metrics"]["highlight_1_4"]
    bias = row["bias"]["all"]
    high_rate = row["payload"]["high_pixel_rate"]
    print(
        f"{prefix}{row['gate']['decision'].upper():6s}/"
        f"{row['gate_lift']['decision'].upper():6s} "
        f"{row['label']:58s} "
        f"est={row['estimated_bytes']:,} ratio={row['estimated_ratio']:.2f}x "
        f"high={high_rate:.2%} "
        f"darkD={dark['lost_detail_delta_vs_anchor']:.2%} "
        f"liftDarkD={lift_dark['lost_detail_delta_vs_anchor']:.2%} "
        f"hiD={high['lost_detail_delta_vs_anchor']:.2%} "
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
    parser.add_argument("--vst-modes", default="sqrtvst,gamma075")
    parser.add_argument("--vst-a", type=float, default=1.0)
    parser.add_argument("--vst-b", type=float, default=0.01)
    parser.add_argument("--vst-eps", type=float, default=1.0e-4)
    parser.add_argument("--y-bits", default="9")
    parser.add_argument("--chroma-low-bits", default="6,7")
    parser.add_argument("--high-bits", default="7")
    parser.add_argument("--low-scales", default="1,2,4")
    parser.add_argument("--guide-radii", default="2,4")
    parser.add_argument("--guide-eps", default="0.01,0.05,0.1")
    parser.add_argument("--threshold-mults", default="discard,1,2")
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
        for vst_mode in parse_strings(args.vst_modes):
            for y_bits in parse_ints(args.y_bits):
                for chroma_low_bits in parse_ints(args.chroma_low_bits):
                    for high_bits in parse_ints(args.high_bits):
                        for low_scale in parse_ints(args.low_scales):
                            for guide_radius in parse_ints(args.guide_radii):
                                for guide_eps in parse_floats(args.guide_eps):
                                    for threshold_mult in parse_thresholds(args.threshold_mults):
                                        row = summarize_case(
                                            pixels,
                                            anchor_regions,
                                            anchor_regions_lift,
                                            vst_mode,
                                            args.vst_a,
                                            args.vst_b,
                                            args.vst_eps,
                                            y_bits,
                                            chroma_low_bits,
                                            high_bits,
                                            low_scale,
                                            guide_radius,
                                            guide_eps,
                                            threshold_mult,
                                            args.white,
                                            args.lift_white,
                                            args.gamma,
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
                row["bias"]["all"]["abs_mean_rgb_delta"],
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
            f"vst_chroma_nr_{safe_name(args.glob)}"
            f"_crop{args.crop_size}_Y{safe_name(args.y_bits)}"
            f"_CL{safe_name(args.chroma_low_bits)}.json"
        )
        output.write_text(json.dumps(summaries, indent=2))
        print(f"Saved: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
