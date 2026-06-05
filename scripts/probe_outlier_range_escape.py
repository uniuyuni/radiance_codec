"""Probe percentile/step driven highlight outlier escape for VST Y quantization.

The current Y8 VST-chroma route uses a global Y range.  Extremely wide HDR
values can make one Y8 step too large for mid tones.  This probe escapes
outlier pixels to signed-log10 and quantizes the VST base range only from the
remaining pixels.
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
PREVIEW_DIR = ROOT / "outputs" / "previews" / "outlier_range_escape"
sys.path.insert(0, str(ROOT / "codec" / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

import radiance_codec  # noqa: E402
from audit_linear_index_payload import predict_for_channel, quantize_indices  # noqa: E402
from audit_near_lossless_quality import (  # noqa: E402
    build_region_masks,
    evaluate,
    export_quality_pngs,
    key_table,
    signed_log_stats,
    summarize_regions,
)
from estimate_vst_signedlog_route import residual_selected_cost  # noqa: E402
from probe_adaptive_ycocg_router import build_mask, entropy_bits_from_counts  # noqa: E402
from probe_color_tile_transform_index import forward_color, inverse_color  # noqa: E402
from probe_darkbits_router_payload import read_exr, summarize_binary_mask  # noqa: E402
from probe_vst_chroma_nr import (  # noqa: E402
    block_mean_downsample,
    forward_vst,
    guided_filter_gray,
    inverse_vst,
    quantize_range,
    safe_name,
    selected_entropy_indices,
    shrink_highpass,
    upsample_repeat,
)


def parse_floats(text: str) -> tuple[float, ...]:
    return tuple(float(part) for part in text.split(",") if part.strip())


def quantize_range_from_mask(values: np.ndarray, bits: int, range_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    levels = (1 << bits) - 1
    vals = values.astype(np.float64)
    selected = vals[range_mask]
    if selected.size == 0:
        selected = vals.reshape(-1)
    lo = float(np.min(selected))
    hi = float(np.max(selected))
    q = np.zeros(vals.shape, dtype=np.uint16)
    if not hi > lo:
        return q, np.full(vals.shape, lo, dtype=np.float32), {"lo": lo, "hi": hi, "step": 0.0}
    qf = np.floor((vals - lo) / (hi - lo) * levels + 0.5)
    q = np.clip(qf, 0, levels).astype(np.uint16)
    rec = lo + q.astype(np.float64) * (hi - lo) / levels
    return q, rec.astype(np.float32), {"lo": lo, "hi": hi, "step": (hi - lo) / levels}


def residual_selected_cost_indices(indices: np.ndarray, bits: tuple[int, ...], mask: np.ndarray, predictor: str) -> dict:
    total_bits = 0.0
    rows = []
    samples = int(np.count_nonzero(mask))
    for channel, bit_depth in enumerate(bits):
        if samples == 0:
            rows.append({
                "channel": channel,
                "bits": bit_depth,
                "predictor": predictor,
                "samples": 0,
                "estimated_bits": 0.0,
                "bits_per_sample": 0.0,
            })
            continue
        alphabet = 1 << bit_depth
        pred = predict_for_channel(indices, channel, predictor)
        residual = (indices[:, :, channel].astype(np.int32) + alphabet - pred) & (alphabet - 1)
        values = residual[mask]
        counts = np.bincount(values.astype(np.int64), minlength=alphabet)
        bit_cost = entropy_bits_from_counts(counts)
        total_bits += bit_cost
        rows.append({
            "channel": channel,
            "bits": bit_depth,
            "predictor": predictor,
            "samples": samples,
            "estimated_bits": bit_cost,
            "bits_per_sample": bit_cost / float(samples),
        })
    return {
        "predictor": predictor,
        "estimated_bits": total_bits,
        "estimated_bytes": int(math.ceil(total_bits / 8.0) + 32),
        "channels": rows,
    }


def downsample_mask_any(mask: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return mask.astype(bool, copy=True)
    h, w = mask.shape
    by = math.ceil(h / scale)
    bx = math.ceil(w / scale)
    out = np.zeros((by, bx), dtype=bool)
    for yb in range(by):
        for xb in range(bx):
            tile = mask[yb * scale:min((yb + 1) * scale, h), xb * scale:min((xb + 1) * scale, w)]
            out[yb, xb] = bool(np.any(tile))
    return out


def reconstruct_candidate(pixels: np.ndarray, route_mask: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, dict]:
    started = time.perf_counter()
    rgb = pixels[:, :, :3]
    t_rgb = forward_vst(rgb, args.vst_mode, args.vst_a, args.vst_b, args.vst_eps).astype(np.float32)
    planes = forward_color(t_rgb, "ycocg")[:, :, :3]
    y_plane = planes[:, :, 0]
    chroma = planes[:, :, 1:3]
    nonmask = ~route_mask

    y_idx, y_decoded, y_range = quantize_range_from_mask(y_plane, args.y_bits, nonmask)
    guide = y_plane.astype(np.float64)
    low_full = np.zeros(chroma.shape, dtype=np.float32)
    for channel in range(2):
        low_full[:, :, channel] = guided_filter_gray(
            guide,
            chroma[:, :, channel].astype(np.float64),
            args.guide_radius,
            args.guide_eps,
        )
    high = chroma - low_full
    high_kept, high_thresholds = shrink_highpass(high, args.threshold_mult)

    low_indices = []
    low_decoded_full = np.zeros(chroma.shape, dtype=np.float32)
    low_ranges = []
    low_nonmask = ~downsample_mask_any(route_mask, args.low_scale)
    for channel in range(2):
        coarse = block_mean_downsample(low_full[:, :, channel], args.low_scale)
        q, rec, channel_range = quantize_range_from_mask(coarse, args.chroma_low_bits, low_nonmask)
        low_indices.append(q)
        low_ranges.append(channel_range)
        low_decoded_full[:, :, channel] = upsample_repeat(rec, y_plane.shape, args.low_scale)
    low_idx_stack = np.stack(low_indices, axis=2)

    high_mask = nonmask & np.any(np.abs(high_kept) > 0.0, axis=2)
    high_payload = selected_entropy_indices(high_kept, args.high_bits, high_mask)
    high_decoded = high_payload["decoded"]

    decoded_planes = np.zeros(planes.shape, dtype=np.float32)
    decoded_planes[:, :, 0] = y_decoded
    decoded_planes[:, :, 1:3] = low_decoded_full + high_decoded
    decoded_t_rgb = inverse_color(decoded_planes, "ycocg")[:, :, :3]
    decoded_rgb = inverse_vst(decoded_t_rgb, args.vst_mode, args.vst_a, args.vst_b, args.vst_eps)
    candidate = np.array(pixels, copy=True)
    candidate[:, :, :3] = decoded_rgb.astype(np.float32)

    anchor = radiance_codec.quantize_linear_index(
        pixels,
        bits=args.anchor_bits,
        transform=args.anchor_transform,
    )
    candidate[route_mask, :3] = anchor[route_mask, :3]

    y_payload = residual_selected_cost_indices(y_idx[:, :, None], (args.y_bits,), nonmask, "med")
    low_payload = residual_selected_cost_indices(
        low_idx_stack,
        (args.chroma_low_bits, args.chroma_low_bits),
        low_nonmask,
        "med",
    )
    signed_idx = quantize_indices(
        pixels[:, :, :3],
        args.anchor_bits,
        args.anchor_transform,
    )
    signed_payload = residual_selected_cost(
        signed_idx,
        (args.anchor_bits, args.anchor_bits, args.anchor_bits),
        route_mask,
        args.signedlog_predictor,
    )
    mask_summary = summarize_binary_mask(route_mask)
    mask_payload = min(mask_summary["order0"], mask_summary["west_north"], key=lambda row: row["total_bytes"])
    high_mask_summary = summarize_binary_mask(high_mask)
    high_mask_payload = min(
        high_mask_summary["order0"],
        high_mask_summary["west_north"],
        key=lambda row: row["total_bytes"],
    )
    estimated_bytes = int(
        y_payload["estimated_bytes"]
        + low_payload["estimated_bytes"]
        + signed_payload["estimated_bytes"]
        + high_payload["estimated_bytes"]
        + int(mask_payload["total_bytes"])
        + int(high_mask_payload["total_bytes"])
        + 384
    )
    payload = {
        "estimated_bytes": estimated_bytes,
        "estimated_ratio_raw": pixels.nbytes / float(estimated_bytes),
        "reconstruct_seconds": time.perf_counter() - started,
        "y_range": y_range,
        "chroma_low_ranges": low_ranges,
        "high_thresholds": high_thresholds,
        "high_pixel_rate": float(np.mean(high_mask)),
        "components": {
            "y": y_payload,
            "chroma_low": low_payload,
            "signedlog_escape": signed_payload,
            "route_mask": mask_payload,
            "high": {
                "estimated_bytes": high_payload["estimated_bytes"],
                "mask": high_mask_payload,
            },
        },
    }
    return candidate, anchor, payload


def audit_candidate(pixels: np.ndarray, candidate: np.ndarray, anchor: np.ndarray, route_mask: np.ndarray, args: argparse.Namespace) -> dict:
    masks = build_region_masks(pixels, route_mask)
    candidate_regions = {}
    anchor_regions = {}
    for ev in (0.0, 3.0, 5.0):
        key = f"ev{int(ev)}"
        candidate_regions[key] = summarize_regions(
            pixels,
            candidate,
            masks,
            ev,
            args.white,
            args.gamma,
            args.low_radius,
            args.structure_radius,
        )
        anchor_regions[key] = summarize_regions(
            pixels,
            anchor,
            masks,
            ev,
            args.white,
            args.gamma,
            args.low_radius,
            args.structure_radius,
        )
    return {
        "decision": evaluate(candidate_regions, anchor_regions),
        "signed_log_maxabs": {
            "candidate": signed_log_stats(pixels, candidate, masks),
            "anchor": signed_log_stats(pixels, anchor, masks),
        },
        "key_metrics": key_table(candidate_regions, anchor_regions),
        "candidate_regions": candidate_regions,
        "anchor_regions": anchor_regions,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="sample_hilberts-mill-conference-room_2K.exr")
    parser.add_argument("--crop-size", type=int, default=0)
    parser.add_argument("--percentiles", default="99,99.5,99.9,99.95,99.99,100")
    parser.add_argument("--target-y-step", type=float, default=0.01)
    parser.add_argument("--vst-mode", default="gamma075")
    parser.add_argument("--vst-a", type=float, default=1.0)
    parser.add_argument("--vst-b", type=float, default=0.01)
    parser.add_argument("--vst-eps", type=float, default=1.0e-4)
    parser.add_argument("--y-bits", type=int, default=8)
    parser.add_argument("--chroma-low-bits", type=int, default=8)
    parser.add_argument("--high-bits", type=int, default=5)
    parser.add_argument("--low-scale", type=int, default=2)
    parser.add_argument("--guide-radius", type=int, default=2)
    parser.add_argument("--guide-eps", type=float, default=0.1)
    parser.add_argument("--threshold-mult", type=float, default=2.5)
    parser.add_argument("--anchor-bits", type=int, default=10)
    parser.add_argument("--anchor-transform", default="signed-log")
    parser.add_argument("--signedlog-predictor", default="med")
    parser.add_argument("--mask-mode", default="dark-smooth")
    parser.add_argument("--dark-max", type=float, default=0.5)
    parser.add_argument("--mask-radius", type=int, default=2)
    parser.add_argument("--smooth-threshold", type=float, default=0.0025)
    parser.add_argument("--white", type=float, default=4.0)
    parser.add_argument("--gamma", type=float, default=2.2)
    parser.add_argument("--low-radius", type=int, default=16)
    parser.add_argument("--structure-radius", type=int, default=2)
    parser.add_argument("--export-best-png", action="store_true")
    parser.add_argument("--export-ev3-png", action="store_true")
    parser.add_argument("--preview-dir", type=Path, default=PREVIEW_DIR)
    parser.add_argument("--phase", default="percentile_step_escape")
    args = parser.parse_args()

    pixels = read_exr(DATA_DIR / args.input, args.crop_size)
    base_mask = build_mask(
        pixels,
        args.mask_mode,
        args.dark_max,
        args.mask_radius,
        args.smooth_threshold,
    )
    max_rgb = np.max(pixels[:, :, :3].astype(np.float32), axis=2)
    percentiles = parse_floats(args.percentiles)
    rows = []
    best = None
    best_payload = None
    best_candidate = None
    best_anchor = None
    best_mask = None
    for percentile in percentiles:
        threshold = float(np.percentile(max_rgb.reshape(-1).astype(np.float64), percentile))
        outlier_mask = max_rgb > np.float32(threshold)
        route_mask = base_mask | outlier_mask
        candidate, anchor, payload = reconstruct_candidate(pixels, route_mask, args)
        audit = audit_candidate(pixels, candidate, anchor, route_mask, args)
        reject_count = sum(1 for item in audit["decision"]["failures"] if item["severity"] == "reject")
        maybe_count = len(audit["decision"]["failures"]) - reject_count
        row = {
            "image": args.input,
            "percentile": percentile,
            "threshold_maxrgb": threshold,
            "outlier_pixel_rate": float(np.mean(outlier_mask)),
            "route_mask_rate": float(np.mean(route_mask)),
            "target_y_step": args.target_y_step,
            "y_step": payload["y_range"]["step"],
            "y_step_ok": payload["y_range"]["step"] <= args.target_y_step,
            "estimated_bytes": payload["estimated_bytes"],
            "estimated_ratio_raw": payload["estimated_ratio_raw"],
            "decision": audit["decision"],
            "reject_count": reject_count,
            "maybe_count": maybe_count,
            "payload": payload,
            "key_metrics": audit["key_metrics"],
        }
        rows.append(row)
        score = (
            reject_count,
            maybe_count,
            0 if row["y_step_ok"] else 1,
            payload["estimated_bytes"],
            row["route_mask_rate"],
        )
        if best is None or score < best:
            best = score
            best_payload = row
            best_candidate = candidate
            best_anchor = anchor
            best_mask = route_mask
        print(
            f"p{percentile:g} thr={threshold:.6g} route={row['route_mask_rate']:.2%} "
            f"y_step={row['y_step']:.6g} size={row['estimated_bytes']:,} "
            f"ratio={row['estimated_ratio_raw']:.2f}x decision={audit['decision']['decision']} "
            f"reject={reject_count} maybe={maybe_count}",
            flush=True,
        )

    summary = {
        "image": args.input,
        "shape": list(pixels.shape),
        "route": {
            "base": "VST Y range from non-escape pixels",
            "escape": f"{args.anchor_transform}{args.anchor_bits}",
            "percentiles": list(percentiles),
            "target_y_step": args.target_y_step,
        },
        "rows": rows,
        "best": best_payload,
    }
    if args.export_best_png and best_payload is not None:
        args.label = f"outlier_p{best_payload['percentile']:g}_Y{args.y_bits}_CL{args.chroma_low_bits}"
        summary["best_pngs"] = export_quality_pngs(pixels, best_candidate, best_anchor, best_mask, args)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    crop_label = "full" if args.crop_size == 0 else f"crop{args.crop_size}"
    output = RESULTS_DIR / (
        f"outlier_range_escape_{safe_name(Path(args.input).stem)}_{crop_label}"
        f"_Y{args.y_bits}_CL{args.chroma_low_bits}.json"
    )
    output.write_text(json.dumps(summary, indent=2))
    print(f"summary={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
