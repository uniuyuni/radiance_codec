"""Audit the current visual-guard near-lossless route.

This is the RAW-proxy quality gate for the research route:

* VST-chroma base in most pixels;
* signed-log10 escape inside a route mask;
* optional display-diff guard mask for visual edge/banding fixes.

The script reconstructs linear decoded pixels, then compares against the
original and a faithful signed-log10 anchor at EV0/+3/+5 display exposures.
It is intentionally metric-heavy and bitstream-light: reconstructed pixels are
fixed, and this is only a quality audit.
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
from export_linear_index_preview_png import to_preview_u16, write_png  # noqa: E402
from estimate_vst_signedlog_route import read_mask_png  # noqa: E402
from probe_adaptive_ycocg_router import build_mask  # noqa: E402
from probe_color_tile_transform_index import forward_color, inverse_color  # noqa: E402
from probe_darkbits_router_payload import read_exr  # noqa: E402
from probe_log_signal_noise_resynth import box_mean_2d  # noqa: E402
from probe_vst_chroma_nr import (  # noqa: E402
    block_mean_downsample,
    forward_vst,
    guided_filter_gray,
    inverse_vst,
    safe_name,
    selected_entropy_indices,
    shrink_highpass,
    upsample_repeat,
)

PREVIEW_DIR = ROOT / "outputs" / "previews" / "near_lossless_quality_audit"
DSCF_VISUAL_GUARD_MASK = (
    ROOT
    / "outputs"
    / "previews"
    / "vst_chroma_dark_protect"
    / "candidate_Y8_slog10_diffguard_L0_01_R0_d0_mask.png"
)


def display_rgb(pixels: np.ndarray, ev: float, white: float, gamma: float) -> np.ndarray:
    scale = float(2.0 ** ev)
    rgb = pixels[:, :, :3].astype(np.float32, copy=False) * np.float32(scale / white)
    rgb = np.clip(rgb, 0.0, 1.0)
    return np.power(rgb, np.float32(1.0 / gamma)).astype(np.float32)


def luma(display: np.ndarray) -> np.ndarray:
    return (
        display[:, :, 0] * np.float32(0.2126)
        + display[:, :, 1] * np.float32(0.7152)
        + display[:, :, 2] * np.float32(0.0722)
    ).astype(np.float32)


def nearest_percentile(values: np.ndarray, percentile: float) -> float:
    if values.size == 0:
        return 0.0
    kth = int(math.ceil((percentile / 100.0) * values.size)) - 1
    kth = max(0, min(values.size - 1, kth))
    return float(np.partition(values, kth)[kth])


def scalar_stats(diff: np.ndarray, mask: np.ndarray) -> dict:
    if not bool(np.any(mask)):
        return {
            "samples": 0,
            "rmse": 0.0,
            "mae": 0.0,
            "p95_abs": 0.0,
            "p99_abs": 0.0,
            "p999_abs": 0.0,
            "max_abs": 0.0,
            "psnr_db": math.inf,
        }
    selected = diff[mask].astype(np.float64)
    abs_diff = np.abs(selected)
    rmse = float(np.sqrt(np.mean(selected * selected)))
    psnr = math.inf if rmse == 0.0 else 20.0 * math.log10(1.0 / rmse)
    return {
        "samples": int(np.count_nonzero(mask)),
        "rmse": rmse,
        "mae": float(np.mean(abs_diff)),
        "p95_abs": nearest_percentile(abs_diff, 95.0),
        "p99_abs": nearest_percentile(abs_diff, 99.0),
        "p999_abs": nearest_percentile(abs_diff, 99.9),
        "max_abs": float(np.max(abs_diff)),
        "psnr_db": psnr,
    }


def gradient_fields(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gx = values[:, 1:] - values[:, :-1]
    gy = values[1:, :] - values[:-1, :]
    return gx.astype(np.float32), gy.astype(np.float32)


def gradient_stats_from_fields(
    original_grads: tuple[np.ndarray, np.ndarray],
    decoded_grads: tuple[np.ndarray, np.ndarray],
    mask: np.ndarray,
) -> dict:
    ox, oy = original_grads
    dx, dy = decoded_grads
    masks = [mask[:, 1:] & mask[:, :-1], mask[1:, :] & mask[:-1, :]]
    orig_parts = []
    dec_parts = []
    for grad_mask, orig_grad, dec_grad in zip(masks, (ox, oy), (dx, dy)):
        if bool(np.any(grad_mask)):
            orig_parts.append(orig_grad[grad_mask].astype(np.float64))
            dec_parts.append(dec_grad[grad_mask].astype(np.float64))
    if not orig_parts:
        return {
            "gradient_samples": 0,
            "grad_rmse": 0.0,
            "grad_p99_abs": 0.0,
            "grad_nrmse": 0.0,
            "grad_energy_ratio": 1.0,
            "grad_correlation": 1.0,
            "lost_detail_rate": 0.0,
        }
    orig = np.concatenate(orig_parts)
    dec = np.concatenate(dec_parts)
    delta = dec - orig
    signal = float(np.sqrt(np.mean(orig * orig)))
    dec_signal = float(np.sqrt(np.mean(dec * dec)))
    rmse = float(np.sqrt(np.mean(delta * delta)))
    denom = math.sqrt(float(np.sum(orig * orig)) * float(np.sum(dec * dec)))
    visible = np.abs(orig) > (1.0 / 4096.0)
    lost = visible & (np.abs(dec) < np.abs(orig) * 0.5)
    return {
        "gradient_samples": int(orig.size),
        "grad_rmse": rmse,
        "grad_p99_abs": nearest_percentile(np.abs(delta), 99.0),
        "grad_nrmse": 0.0 if signal == 0.0 else rmse / signal,
        "grad_energy_ratio": 1.0 if signal == 0.0 else dec_signal / signal,
        "grad_correlation": 1.0 if denom == 0.0 else float(np.sum(orig * dec) / denom),
        "lost_detail_rate": float(np.mean(lost)) if bool(np.any(visible)) else 0.0,
    }


def gradient_stats(original_luma: np.ndarray, decoded_luma: np.ndarray, mask: np.ndarray) -> dict:
    return gradient_stats_from_fields(
        gradient_fields(original_luma),
        gradient_fields(decoded_luma),
        mask,
    )


def chroma_fields(display: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = display.astype(np.float32, copy=False)
    co = values[:, :, 0] - values[:, :, 2]
    cg = values[:, :, 1] - (values[:, :, 0] + values[:, :, 2]) * np.float32(0.5)
    return co.astype(np.float32), cg.astype(np.float32)


def chroma_stats_from_fields(
    original_chroma: tuple[np.ndarray, np.ndarray],
    decoded_chroma: tuple[np.ndarray, np.ndarray],
    mask: np.ndarray,
) -> dict:
    if not bool(np.any(mask)):
        return {"samples": 0, "co_p99_abs": 0.0, "cg_p99_abs": 0.0, "co_rmse": 0.0, "cg_rmse": 0.0}
    orig_co, orig_cg = original_chroma
    dec_co, dec_cg = decoded_chroma
    co = (dec_co - orig_co)[mask].astype(np.float64)
    cg = (dec_cg - orig_cg)[mask].astype(np.float64)
    return {
        "samples": int(np.count_nonzero(mask)),
        "co_rmse": float(np.sqrt(np.mean(co * co))),
        "cg_rmse": float(np.sqrt(np.mean(cg * cg))),
        "co_p99_abs": nearest_percentile(np.abs(co), 99.0),
        "cg_p99_abs": nearest_percentile(np.abs(cg), 99.0),
    }


def chroma_stats(original_display: np.ndarray, decoded_display: np.ndarray, mask: np.ndarray) -> dict:
    return chroma_stats_from_fields(
        chroma_fields(original_display),
        chroma_fields(decoded_display),
        mask,
    )


def low_frequency_field(original_luma: np.ndarray, decoded_luma: np.ndarray, radius: int) -> np.ndarray:
    residual = decoded_luma.astype(np.float32) - original_luma.astype(np.float32)
    return box_mean_2d(residual.astype(np.float64), radius).astype(np.float32)


def low_frequency_stats(low: np.ndarray, mask: np.ndarray, radius: int) -> dict:
    stats = scalar_stats(low, mask)
    max_idx = int(np.argmax(np.where(mask, np.abs(low), -1.0)))
    y, x = np.unravel_index(max_idx, low.shape)
    stats["radius"] = radius
    stats["max_abs_x"] = int(x)
    stats["max_abs_y"] = int(y)
    stats["max_abs_signed"] = float(low[y, x])
    return stats


def build_region_masks(pixels: np.ndarray, route_mask: np.ndarray) -> dict[str, np.ndarray]:
    rgb = pixels[:, :, :3].astype(np.float32)
    value = np.max(rgb, axis=2)
    linear_luma = (
        rgb[:, :, 0] * np.float32(0.2126)
        + rgb[:, :, 1] * np.float32(0.7152)
        + rgb[:, :, 2] * np.float32(0.0722)
    )
    log_luma = np.log2(np.maximum(linear_luma, 0.0) + np.float32(1.0e-4))
    gx = np.zeros(log_luma.shape, dtype=np.float32)
    gy = np.zeros(log_luma.shape, dtype=np.float32)
    gx[:, 1:] = np.abs(log_luma[:, 1:] - log_luma[:, :-1])
    gy[1:, :] = np.abs(log_luma[1:, :] - log_luma[:-1, :])
    grad = np.maximum(gx, gy)
    local_mean = box_mean_2d(log_luma.astype(np.float64), 3)
    local_sq = box_mean_2d((log_luma.astype(np.float64) ** 2), 3)
    local_std = np.sqrt(np.maximum(local_sq - local_mean * local_mean, 0.0)).astype(np.float32)
    grad_q35 = float(np.quantile(grad.reshape(-1), 0.35))
    grad_q95 = float(np.quantile(grad.reshape(-1), 0.95))
    std_q40 = float(np.quantile(local_std.reshape(-1), 0.40))
    std_q75 = float(np.quantile(local_std.reshape(-1), 0.75))
    smooth = (grad <= grad_q35) & (local_std <= std_q40)
    edge = grad >= grad_q95
    texture = local_std >= std_q75
    return {
        "all": np.ones(value.shape, dtype=bool),
        "dark_0_0.25": value <= 0.25,
        "mid_0.25_1": (value > 0.25) & (value <= 1.0),
        "highlight_1_4": (value > 1.0) & (value <= 4.0),
        "extreme_gt4": value > 4.0,
        "smooth": smooth,
        "dark_smooth": smooth & (value <= 0.25),
        "edge": edge,
        "texture": texture,
        "route_mask": route_mask,
        "nonmask": ~route_mask,
    }


def summarize_regions(
    original: np.ndarray,
    decoded: np.ndarray,
    masks: dict[str, np.ndarray],
    ev: float,
    white: float,
    gamma: float,
    low_radius: int,
    structure_radius: int,
) -> dict:
    original_display = display_rgb(original, ev, white, gamma)
    decoded_display = display_rgb(decoded, ev, white, gamma)
    original_luma = luma(original_display)
    decoded_luma = luma(decoded_display)
    luma_diff = decoded_luma - original_luma
    low_luma_diff = low_frequency_field(original_luma, decoded_luma, low_radius)
    original_structure = box_mean_2d(original_luma.astype(np.float64), structure_radius).astype(np.float32)
    decoded_structure = box_mean_2d(decoded_luma.astype(np.float64), structure_radius).astype(np.float32)
    original_grads = gradient_fields(original_luma)
    decoded_grads = gradient_fields(decoded_luma)
    original_structure_grads = gradient_fields(original_structure)
    decoded_structure_grads = gradient_fields(decoded_structure)
    original_chroma = chroma_fields(original_display)
    decoded_chroma = chroma_fields(decoded_display)
    rgb_abs = np.max(np.abs(decoded_display - original_display), axis=2)
    regions = {}
    for name, mask in masks.items():
        regions[name] = {
            "samples": int(np.count_nonzero(mask)),
            "display_luma": scalar_stats(luma_diff, mask),
            "display_rgb_maxabs": scalar_stats(rgb_abs, mask),
            "display_luma_gradient": gradient_stats_from_fields(original_grads, decoded_grads, mask),
            "display_luma_structure_gradient": gradient_stats_from_fields(
                original_structure_grads,
                decoded_structure_grads,
                mask,
            ),
            "display_luma_low_frequency": low_frequency_stats(low_luma_diff, mask, low_radius),
            "display_chroma": chroma_stats_from_fields(original_chroma, decoded_chroma, mask),
        }
    return regions


def signed_log_stats(original: np.ndarray, decoded: np.ndarray, masks: dict[str, np.ndarray]) -> dict:
    def sl(values: np.ndarray) -> np.ndarray:
        rgb = values[:, :, :3].astype(np.float32)
        return np.sign(rgb) * np.log2(np.abs(rgb) + np.float32(1.0))

    diff = sl(decoded) - sl(original)
    max_abs = np.max(np.abs(diff), axis=2)
    rows = {}
    for name, mask in masks.items():
        rows[name] = scalar_stats(max_abs, mask)
    return rows


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


def choose_outlier_mask(
    pixels: np.ndarray,
    base_mask: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict | None]:
    if not args.enable_outlier_router:
        return np.zeros(base_mask.shape, dtype=bool), None

    rgb = pixels[:, :, :3].astype(np.float32)
    max_rgb = np.max(rgb, axis=2)
    p97, p99, p100 = np.percentile(max_rgb.reshape(-1).astype(np.float64), (97.0, 99.0, 100.0))
    max_over_p99 = float(p100 / max(p99, 1.0e-12))
    p99_over_p97 = float(p99 / max(p97, 1.0e-12))
    if max(max_over_p99, p99_over_p97) < args.outlier_activation_ratio:
        return np.zeros(base_mask.shape, dtype=bool), {
            "enabled": True,
            "active": False,
            "reason": "range ratios below activation threshold",
            "activation_ratio": args.outlier_activation_ratio,
            "max_over_p99": max_over_p99,
            "p99_over_p97": p99_over_p97,
            "p97": float(p97),
            "p99": float(p99),
            "p100": float(p100),
        }
    t_rgb = forward_vst(rgb, args.vst_mode, args.vst_a, args.vst_b, args.vst_eps).astype(np.float32)
    y_plane = forward_color(t_rgb, "ycocg")[:, :, 0]
    percentiles = parse_floats(args.outlier_percentiles)
    rows = []
    chosen = None
    for percentile in sorted(percentiles, reverse=True):
        threshold = float(np.percentile(max_rgb.reshape(-1).astype(np.float64), percentile))
        outlier = max_rgb > np.float32(threshold)
        route_mask = base_mask | outlier
        nonmask = ~route_mask
        _q, _rec, y_range = quantize_range_from_mask(y_plane, args.y_bits, nonmask)
        row = {
            "percentile": percentile,
            "threshold_maxrgb": threshold,
            "outlier_pixel_rate": float(np.mean(outlier)),
            "route_mask_rate": float(np.mean(route_mask)),
            "y_range": y_range,
            "y_step_ok": y_range["step"] <= args.target_y_step,
        }
        rows.append(row)
        if chosen is None and row["y_step_ok"]:
            chosen = row

    if chosen is None:
        chosen = rows[-1]
    outlier_mask = max_rgb > np.float32(chosen["threshold_maxrgb"])
    summary = {
        "enabled": True,
        "active": True,
        "activation_ratio": args.outlier_activation_ratio,
        "max_over_p99": max_over_p99,
        "p99_over_p97": p99_over_p97,
        "p97": float(p97),
        "p99": float(p99),
        "p100": float(p100),
        "target_y_step": args.target_y_step,
        "percentiles": list(percentiles),
        "chosen": chosen,
        "rows": rows,
    }
    return outlier_mask, summary


def encode_decode_case_masked_range(
    pixels: np.ndarray,
    route_mask: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict]:
    rgb = pixels[:, :, :3]
    t_rgb = forward_vst(rgb, args.vst_mode, args.vst_a, args.vst_b, args.vst_eps).astype(np.float32)
    planes = forward_color(t_rgb, "ycocg")[:, :, :3]
    y_plane = planes[:, :, 0]
    chroma = planes[:, :, 1:3]
    nonmask = ~route_mask

    _y_idx, y_decoded, y_range = quantize_range_from_mask(y_plane, args.y_bits, nonmask)
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

    low_nonmask = ~downsample_mask_any(route_mask, args.low_scale)
    low_decoded_full = np.zeros(chroma.shape, dtype=np.float32)
    low_ranges = []
    for channel in range(2):
        coarse = block_mean_downsample(low_full[:, :, channel], args.low_scale)
        _q, rec, channel_range = quantize_range_from_mask(coarse, args.chroma_low_bits, low_nonmask)
        low_ranges.append(channel_range)
        low_decoded_full[:, :, channel] = upsample_repeat(rec, y_plane.shape, args.low_scale)

    high_mask = nonmask & np.any(np.abs(high_kept) > 0.0, axis=2)
    high_payload = selected_entropy_indices(high_kept, args.high_bits, high_mask)
    high_decoded = high_payload["decoded"]

    decoded_planes = np.zeros(planes.shape, dtype=np.float32)
    decoded_planes[:, :, 0] = y_decoded
    decoded_planes[:, :, 1:3] = low_decoded_full + high_decoded
    decoded_t_rgb = inverse_color(decoded_planes, "ycocg")[:, :, :3]
    decoded_rgb = inverse_vst(decoded_t_rgb, args.vst_mode, args.vst_a, args.vst_b, args.vst_eps)
    decoded = np.array(pixels, copy=True)
    decoded[:, :, :3] = decoded_rgb.astype(np.float32)
    return np.ascontiguousarray(decoded), {
        "vst_mode": args.vst_mode,
        "y_bits": args.y_bits,
        "chroma_low_bits": args.chroma_low_bits,
        "high_bits": args.high_bits,
        "low_scale": args.low_scale,
        "guide_radius": args.guide_radius,
        "guide_eps": args.guide_eps,
        "threshold_mult": args.threshold_mult,
        "range_source": "non_escape_pixels",
        "y_range": y_range,
        "chroma_low_ranges": low_ranges,
        "high_thresholds": high_thresholds,
        "high_pixel_rate": float(np.mean(high_mask)),
        "high_estimated_bytes": int(high_payload["estimated_bytes"]),
    }


def write_mask_png(path: Path, mask: np.ndarray) -> None:
    image = np.repeat(mask[:, :, None], 3, axis=2).astype(np.uint16) * np.uint16(65535)
    write_png(path, image)


def preview_diff_u16(original: np.ndarray, decoded: np.ndarray, ev: float, white: float, gamma: float) -> np.ndarray:
    original_display = display_rgb(original, ev, white, gamma)
    decoded_display = display_rgb(decoded, ev, white, gamma)
    diff = decoded_display - original_display
    # Neutral gray means no difference. Positive error is brighter, negative is darker.
    mapped = np.clip(diff * np.float32(4.0) + np.float32(0.5), 0.0, 1.0)
    return np.rint(mapped * np.float32(65535.0)).astype(np.uint16)


def chroma_diff_u16(original: np.ndarray, decoded: np.ndarray, ev: float, white: float, gamma: float) -> np.ndarray:
    original_display = display_rgb(original, ev, white, gamma)
    decoded_display = display_rgb(decoded, ev, white, gamma)
    orig_co, orig_cg = chroma_fields(original_display)
    dec_co, dec_cg = chroma_fields(decoded_display)
    co = dec_co - orig_co
    cg = dec_cg - orig_cg
    image = np.zeros((*co.shape, 3), dtype=np.float32)
    image[:, :, 0] = np.clip(co * np.float32(4.0) + np.float32(0.5), 0.0, 1.0)
    image[:, :, 1] = np.clip(cg * np.float32(4.0) + np.float32(0.5), 0.0, 1.0)
    image[:, :, 2] = np.float32(0.5)
    return np.rint(image * np.float32(65535.0)).astype(np.uint16)


def export_quality_pngs(
    original: np.ndarray,
    candidate: np.ndarray,
    anchor: np.ndarray,
    route_mask: np.ndarray,
    args: argparse.Namespace,
) -> dict:
    crop_label = "full" if args.crop_size == 0 else f"crop{args.crop_size}"
    stem = safe_name(Path(args.input).stem)
    label = safe_name(args.label)
    output_dir = args.preview_dir / safe_name(args.phase)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{stem}_{crop_label}_{label}"
    original_png = output_dir / f"{prefix}_original_w{args.white:g}_g{args.gamma:g}.png"
    candidate_png = output_dir / f"{prefix}_candidate_w{args.white:g}_g{args.gamma:g}.png"
    anchor_png = output_dir / f"{prefix}_signedlog{args.anchor_bits}_anchor_w{args.white:g}_g{args.gamma:g}.png"
    diff_png = output_dir / f"{prefix}_displaydiff_ev0_x4_neutralgray.png"
    chroma_png = output_dir / f"{prefix}_chromadiff_ev0_x4_rg.png"
    mask_png = output_dir / f"{prefix}_route_mask.png"

    if not original_png.exists():
        write_png(original_png, to_preview_u16(original, args.white, args.gamma))
    write_png(candidate_png, to_preview_u16(candidate, args.white, args.gamma))
    write_png(anchor_png, to_preview_u16(anchor, args.white, args.gamma))
    write_png(diff_png, preview_diff_u16(original, candidate, 0.0, args.white, args.gamma))
    write_png(chroma_png, chroma_diff_u16(original, candidate, 0.0, args.white, args.gamma))
    write_mask_png(mask_png, route_mask)

    paths = {
        "original_png": str(original_png),
        "candidate_png": str(candidate_png),
        "signedlog_anchor_png": str(anchor_png),
        "display_diff_ev0_x4_png": str(diff_png),
        "chroma_diff_ev0_x4_png": str(chroma_png),
        "route_mask_png": str(mask_png),
    }
    if args.export_ev3_png:
        ev3_candidate_png = output_dir / f"{prefix}_candidate_ev3_w{args.white:g}_g{args.gamma:g}.png"
        ev3_diff_png = output_dir / f"{prefix}_displaydiff_ev3_x4_neutralgray.png"
        write_png(ev3_candidate_png, np.rint(display_rgb(candidate, 3.0, args.white, args.gamma) * 65535.0).astype(np.uint16))
        write_png(ev3_diff_png, preview_diff_u16(original, candidate, 3.0, args.white, args.gamma))
        paths["candidate_ev3_png"] = str(ev3_candidate_png)
        paths["display_diff_ev3_x4_png"] = str(ev3_diff_png)
    return paths


def evaluate(candidate_regions: dict, anchor_regions: dict) -> dict:
    failures: list[dict] = []

    def add(severity: str, exposure: str, region: str, metric: str, value: float, threshold: float) -> None:
        failures.append({
            "severity": severity,
            "exposure": exposure,
            "region": region,
            "metric": metric,
            "value": value,
            "threshold": threshold,
        })

    for exposure in ("ev0", "ev3"):
        cand = candidate_regions[exposure]
        anchor = anchor_regions[exposure]
        for region in ("dark_0_0.25", "dark_smooth"):
            c_lost = cand[region]["display_luma_structure_gradient"]["lost_detail_rate"]
            a_lost = anchor[region]["display_luma_structure_gradient"]["lost_detail_rate"]
            delta = c_lost - a_lost
            if delta > 0.10:
                add("reject", exposure, region, "structure_lost_detail_delta_vs_anchor", delta, 0.10)
            elif delta > 0.04:
                add("maybe", exposure, region, "structure_lost_detail_delta_vs_anchor", delta, 0.04)

            c_low = cand[region]["display_luma_low_frequency"]["p99_abs"]
            a_low = anchor[region]["display_luma_low_frequency"]["p99_abs"]
            limit = max(a_low * 2.0, a_low + 0.01)
            if c_low > limit:
                add("maybe", exposure, region, "low_frequency_p99_abs", c_low, limit)

        for region in ("highlight_1_4", "extreme_gt4"):
            samples = cand[region]["samples"]
            if samples < 1024:
                continue
            c_grad = cand[region]["display_luma_structure_gradient"]
            a_grad = anchor[region]["display_luma_structure_gradient"]
            lost_delta = c_grad["lost_detail_rate"] - a_grad["lost_detail_rate"]
            if lost_delta > 0.075:
                add("reject", exposure, region, "structure_lost_detail_delta_vs_anchor", lost_delta, 0.075)
            elif lost_delta > 0.035:
                add("maybe", exposure, region, "structure_lost_detail_delta_vs_anchor", lost_delta, 0.035)
            energy = c_grad["grad_energy_ratio"]
            if energy < 0.94 or energy > 1.06:
                add("maybe", exposure, region, "structure_grad_energy_ratio", energy, 0.94 if energy < 0.94 else 1.06)

        chroma_mid = cand["mid_0.25_1"]["display_chroma"]
        anchor_mid = anchor["mid_0.25_1"]["display_chroma"]
        chroma_limit = max(anchor_mid["co_p99_abs"], anchor_mid["cg_p99_abs"]) + 0.03
        chroma_value = max(chroma_mid["co_p99_abs"], chroma_mid["cg_p99_abs"])
        if chroma_value > chroma_limit:
            add("maybe", exposure, "mid_0.25_1", "chroma_p99_abs", chroma_value, chroma_limit)

    if any(item["severity"] == "reject" for item in failures):
        decision = "reject"
    elif failures:
        decision = "maybe"
    else:
        decision = "pass"
    return {"decision": decision, "failures": failures}


def key_table(candidate_regions: dict, anchor_regions: dict) -> dict:
    table = {}
    for exposure in ("ev0", "ev3", "ev5"):
        rows = {}
        for region in ("dark_smooth", "dark_0_0.25", "mid_0.25_1", "highlight_1_4", "extreme_gt4", "all"):
            cand = candidate_regions[exposure][region]
            anchor = anchor_regions[exposure][region]
            rows[region] = {
                "samples": cand["samples"],
                "luma_p99": cand["display_luma"]["p99_abs"],
                "anchor_luma_p99": anchor["display_luma"]["p99_abs"],
                "lowfreq_p99": cand["display_luma_low_frequency"]["p99_abs"],
                "anchor_lowfreq_p99": anchor["display_luma_low_frequency"]["p99_abs"],
                "grad_nrmse": cand["display_luma_gradient"]["grad_nrmse"],
                "anchor_grad_nrmse": anchor["display_luma_gradient"]["grad_nrmse"],
                "grad_energy": cand["display_luma_gradient"]["grad_energy_ratio"],
                "lost_detail_delta": (
                    cand["display_luma_gradient"]["lost_detail_rate"]
                    - anchor["display_luma_gradient"]["lost_detail_rate"]
                ),
                "structure_grad_nrmse": cand["display_luma_structure_gradient"]["grad_nrmse"],
                "anchor_structure_grad_nrmse": anchor["display_luma_structure_gradient"]["grad_nrmse"],
                "structure_grad_energy": cand["display_luma_structure_gradient"]["grad_energy_ratio"],
                "structure_lost_detail_delta": (
                    cand["display_luma_structure_gradient"]["lost_detail_rate"]
                    - anchor["display_luma_structure_gradient"]["lost_detail_rate"]
                ),
            }
        table[exposure] = rows
    return table


def reconstruct_visual_guard(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    pixels = read_exr(DATA_DIR / args.input, args.crop_size)
    if pixels.shape[2] < 3:
        raise RuntimeError("expected at least RGB channels")
    started = time.perf_counter()
    anchor = radiance_codec.quantize_linear_index(
        pixels,
        bits=args.anchor_bits,
        transform=args.anchor_transform,
    )
    route_mask = build_mask(
        pixels,
        args.mask_mode,
        args.dark_max,
        args.mask_radius,
        args.smooth_threshold,
    )
    outlier_mask, outlier_summary = choose_outlier_mask(pixels, route_mask, args)
    if outlier_summary is not None:
        route_mask |= outlier_mask
    additional_summary = None
    if args.additional_mask_png is not None:
        additional = read_mask_png(args.additional_mask_png, route_mask.shape)
        additional_summary = {
            "path": str(args.additional_mask_png),
            "pixel_rate": float(np.mean(additional)),
            "new_pixel_rate": float(np.mean(additional & ~route_mask)),
        }
        route_mask |= additional
    if args.cpp_reconstruct:
        if args.additional_mask_png is not None:
            raise RuntimeError("--cpp-reconstruct does not yet support additional-mask-png")
        candidate, cpp_report = radiance_codec.reconstruct_near_lossless_router_v1(
            pixels,
            y_bits=args.y_bits,
            chroma_low_bits=args.chroma_low_bits,
            high_bits=args.high_bits,
            anchor_bits=args.anchor_bits,
            low_scale=args.low_scale,
            guide_radius=args.guide_radius,
            guide_eps=args.guide_eps,
            threshold_mult=args.threshold_mult,
            dark_max=args.dark_max,
            mask_radius=args.mask_radius,
            smooth_threshold=args.smooth_threshold,
            target_y_step=args.target_y_step,
            outlier_activation_ratio=args.outlier_activation_ratio,
        )
        base_payload = {
            "backend": "cpp",
            "cpp_report": cpp_report,
        }
    else:
        base, base_payload = encode_decode_case_masked_range(pixels, route_mask, args)
        candidate = base
        candidate[route_mask, :3] = anchor[route_mask, :3]
    meta = {
        "reconstruct_seconds": time.perf_counter() - started,
        "base_payload": base_payload,
        "route_mask_rate": float(np.mean(route_mask)),
        "outlier_router": outlier_summary,
        "additional_mask": additional_summary,
    }
    return pixels, candidate, anchor, route_mask, meta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="sample_DSCF0009.EXR")
    parser.add_argument("--crop-size", type=int, default=0)
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
    parser.add_argument("--mask-mode", default="dark-smooth")
    parser.add_argument("--dark-max", type=float, default=0.5)
    parser.add_argument("--mask-radius", type=int, default=2)
    parser.add_argument("--smooth-threshold", type=float, default=0.0025)
    parser.add_argument("--enable-outlier-router", action="store_true")
    parser.add_argument("--outlier-percentiles", default="100,99.9,99.5,99,98.5,98,97,95")
    parser.add_argument("--target-y-step", type=float, default=0.0032)
    parser.add_argument("--outlier-activation-ratio", type=float, default=4.0)
    parser.add_argument(
        "--additional-mask-png",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--no-default-visual-guard",
        action="store_true",
        help="Do not auto-apply the sample_DSCF0009 display-diff guard mask.",
    )
    parser.add_argument("--white", type=float, default=4.0)
    parser.add_argument("--gamma", type=float, default=2.2)
    parser.add_argument("--low-radius", type=int, default=16)
    parser.add_argument("--structure-radius", type=int, default=2)
    parser.add_argument("--export-png", action="store_true")
    parser.add_argument("--export-ev3-png", action="store_true")
    parser.add_argument("--cpp-reconstruct", action="store_true")
    parser.add_argument("--preview-dir", type=Path, default=PREVIEW_DIR)
    parser.add_argument("--phase", default="visual_guard_quality_audit")
    parser.add_argument("--label", default="visual_guard_Y8_L001")
    args = parser.parse_args()

    if (
        args.additional_mask_png is None
        and not args.no_default_visual_guard
        and Path(args.input).stem == "sample_DSCF0009"
        and DSCF_VISUAL_GUARD_MASK.exists()
    ):
        args.additional_mask_png = DSCF_VISUAL_GUARD_MASK

    pixels, candidate, anchor, route_mask, meta = reconstruct_visual_guard(args)
    print(
        f"{args.label} reconstructed mask={meta['route_mask_rate']:.2%} "
        f"reconstruct={meta['reconstruct_seconds']:.2f}s",
        flush=True,
    )
    masks = build_region_masks(pixels, route_mask)
    candidate_regions = {}
    anchor_regions = {}
    for ev in (0.0, 3.0, 5.0):
        key = f"ev{int(ev)}"
        print(f" evaluating {key} candidate", flush=True)
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
        print(f" evaluating {key} signed-log{args.anchor_bits} anchor", flush=True)
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

    evaluation = evaluate(candidate_regions, anchor_regions)
    summary = {
        "image": args.input,
        "shape": list(pixels.shape),
        "label": args.label,
        "route": {
            "base": "vst-chroma",
            "vst_mode": args.vst_mode,
            "y_bits": args.y_bits,
            "chroma_low_bits": args.chroma_low_bits,
            "high_bits": args.high_bits,
            "threshold_mult": args.threshold_mult,
            "escape": f"{args.anchor_transform}{args.anchor_bits}",
            "mask_mode": args.mask_mode,
            "dark_max": args.dark_max,
            "smooth_threshold": args.smooth_threshold,
            "structure_radius": args.structure_radius,
            "outlier_router": {
                "enabled": args.enable_outlier_router,
                "target_y_step": args.target_y_step,
                "percentiles": args.outlier_percentiles,
            },
        },
        "meta": meta,
        "decision": evaluation,
        "signed_log_maxabs": {
            "candidate": signed_log_stats(pixels, candidate, masks),
            "anchor": signed_log_stats(pixels, anchor, masks),
        },
        "key_metrics": key_table(candidate_regions, anchor_regions),
        "candidate_regions": candidate_regions,
        "anchor_regions": anchor_regions,
        "pngs": {},
        "note": (
            "Quality audit for the current visual-guard research route. "
            "Candidate is compared against original and signed-log10 anchor."
        ),
    }
    if args.export_png:
        summary["pngs"] = export_quality_pngs(pixels, candidate, anchor, route_mask, args)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    crop_label = "full" if args.crop_size == 0 else f"crop{args.crop_size}"
    output = RESULTS_DIR / f"near_lossless_quality_{safe_name(args.label)}_{safe_name(Path(args.input).stem)}_{crop_label}.json"
    output.write_text(json.dumps(summary, indent=2))

    print(
        f"{args.label} decision={evaluation['decision']} "
        f"mask={meta['route_mask_rate']:.2%} "
        f"reconstruct={meta['reconstruct_seconds']:.2f}s"
    )
    for exposure in ("ev0", "ev3", "ev5"):
        dark = summary["key_metrics"][exposure]["dark_smooth"]
        hi = summary["key_metrics"][exposure]["highlight_1_4"]
        print(
            f" {exposure}: darkSmooth lumaP99={dark['luma_p99']:.6f} "
            f"lowP99={dark['lowfreq_p99']:.6f} "
            f"structureLostDelta={dark['structure_lost_detail_delta']:.4f}; "
            f"highlight structureEnergy={hi['structure_grad_energy']:.4f} "
            f"structureLostDelta={hi['structure_lost_detail_delta']:.4f}"
        )
    if evaluation["failures"]:
        print(" failures:")
        for failure in evaluation["failures"]:
            print(
                f"  {failure['severity']} {failure['exposure']} "
                f"{failure['region']} {failure['metric']}="
                f"{failure['value']:.6f} > {failure['threshold']:.6f}"
            )
    print(f"summary={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
