"""Probe lossless coding improvements for the current VST+signed-log route.

This does not change reconstructed pixels.  It keeps the current route masks
and quantized indices fixed, then asks whether better predictors or
decoder-visible entropy contexts can reduce the payload estimate.
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
sys.path.insert(0, str(ROOT / "scripts"))

from audit_linear_index_payload import predict_for_channel, quantize_indices  # noqa: E402
from estimate_vst_signedlog_route import (  # noqa: E402
    downsample_mask_any,
    quantize_vst_base_indices,
    read_mask_png,
)
from probe_adaptive_ycocg_router import build_mask, entropy_bits_from_counts  # noqa: E402
from probe_darkbits_router_payload import read_exr  # noqa: E402
from probe_vst_chroma_nr import quantize_range, safe_name, shrink_highpass  # noqa: E402


def context_maps(mask: np.ndarray, names: set[str] | None = None) -> dict[str, np.ndarray]:
    height, width = mask.shape
    yy, xx = np.indices((height, width), dtype=np.int32)
    mask_u8 = mask.astype(np.int32)
    west = np.zeros(mask.shape, dtype=np.int32)
    north = np.zeros(mask.shape, dtype=np.int32)
    northwest = np.zeros(mask.shape, dtype=np.int32)
    west[:, 1:] = mask_u8[:, :-1]
    north[1:, :] = mask_u8[:-1, :]
    northwest[1:, 1:] = mask_u8[:-1, :-1]
    maps = {
        "order0": np.zeros(mask.shape, dtype=np.int32),
        "phase2": (xx & 1) + 2 * (yy & 1),
        "phase4": (xx & 3) + 4 * (yy & 3),
        "xtrans6": (xx % 6) + 6 * (yy % 6),
        "mask_wn": west + 2 * north,
        "mask_wnw": west + 2 * north + 4 * northwest,
        "phase2_mask_wn": (xx & 1) + 2 * (yy & 1) + 4 * (west + 2 * north),
    }
    if names is None:
        return maps
    missing = names - set(maps)
    if missing:
        raise ValueError(f"unknown contexts: {sorted(missing)}")
    return {name: maps[name] for name in maps if name in names}


def conditional_bits(symbols: np.ndarray, contexts: np.ndarray, alphabet: int) -> float:
    if symbols.size == 0:
        return 0.0
    flat_symbols = symbols.astype(np.int64).reshape(-1)
    flat_contexts = contexts.astype(np.int64).reshape(-1)
    context_count = int(flat_contexts.max()) + 1 if flat_contexts.size else 1
    joint = np.bincount(
        flat_contexts * int(alphabet) + flat_symbols,
        minlength=context_count * int(alphabet),
    ).reshape(context_count, int(alphabet))
    total = 0.0
    for row in joint:
        total += entropy_bits_from_counts(row)
    return total


def model_bytes(context_count: int, alphabet: int) -> int:
    # Rough static model table size.  This is intentionally conservative enough
    # to penalize over-fragmented contexts in the probe.
    return context_count * alphabet * 2


def residual_context_rows(
    label: str,
    indices: np.ndarray,
    bits: tuple[int, ...],
    mask: np.ndarray,
    predictors: tuple[str, ...],
    context_names: set[str],
) -> list[dict]:
    rows = []
    contexts_by_name = context_maps(mask, context_names)
    for predictor in predictors:
        for context_name, context in contexts_by_name.items():
            estimated_bits = 0.0
            stream_model_bytes = 0
            channels = []
            for channel, bit_depth in enumerate(bits):
                alphabet = 1 << bit_depth
                pred = predict_for_channel(indices, channel, predictor)
                residual = (
                    indices[:, :, channel].astype(np.int32) + alphabet - pred
                ) & (alphabet - 1)
                selected_symbols = residual[mask]
                selected_contexts = context[mask]
                bit_cost = conditional_bits(selected_symbols, selected_contexts, alphabet)
                context_count = int(selected_contexts.max()) + 1 if selected_contexts.size else 1
                channel_model_bytes = model_bytes(context_count, alphabet)
                estimated_bits += bit_cost
                stream_model_bytes += channel_model_bytes
                channels.append({
                    "channel": channel,
                    "bits": bit_depth,
                    "samples": int(selected_symbols.size),
                    "estimated_bits": bit_cost,
                    "bits_per_sample": (
                        0.0 if selected_symbols.size == 0 else bit_cost / float(selected_symbols.size)
                    ),
                    "context_count": context_count,
                    "model_bytes": channel_model_bytes,
                })
            estimated_bytes = int(math.ceil(estimated_bits / 8.0) + 32 + stream_model_bytes)
            rows.append({
                "stream": label,
                "predictor": predictor,
                "context": context_name,
                "estimated_bits": estimated_bits,
                "estimated_bytes_with_model": estimated_bytes,
                "model_bytes": stream_model_bytes,
                "channels": channels,
            })
    rows.sort(key=lambda row: row["estimated_bytes_with_model"])
    return rows


def high_indices(values: np.ndarray, bits: int, mask: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    indices = np.zeros(values.shape, dtype=np.uint16)
    ranges = []
    if not np.any(mask):
        return indices, ranges
    selected = values[mask]
    for channel in range(values.shape[2]):
        q, _rec, channel_range = quantize_range(selected[:, channel], bits)
        indices[:, :, channel][mask] = q
        ranges.append(channel_range)
    return indices, ranges


def best_summary(name: str, rows: list[dict], baseline_predictor: str, baseline_context: str) -> dict:
    baseline = next(
        row for row in rows
        if row["predictor"] == baseline_predictor and row["context"] == baseline_context
    )
    best = rows[0]
    return {
        "stream": name,
        "baseline": {
            "predictor": baseline["predictor"],
            "context": baseline["context"],
            "estimated_bytes_with_model": baseline["estimated_bytes_with_model"],
        },
        "best": {
            "predictor": best["predictor"],
            "context": best["context"],
            "estimated_bytes_with_model": best["estimated_bytes_with_model"],
        },
        "gain_bytes": baseline["estimated_bytes_with_model"] - best["estimated_bytes_with_model"],
        "top_rows": rows[:8],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="sample_DSCF0009.EXR")
    parser.add_argument("--crop-size", type=int, default=0)
    parser.add_argument("--y-bits", type=int, default=8)
    parser.add_argument("--chroma-low-bits", type=int, default=8)
    parser.add_argument("--signedlog-bits", type=int, default=10)
    parser.add_argument("--base-high-bits", type=int, default=5)
    parser.add_argument("--base-threshold", type=float, default=2.5)
    parser.add_argument("--mask-mode", default="dark-smooth")
    parser.add_argument("--dark-max", type=float, default=0.5)
    parser.add_argument("--mask-radius", type=int, default=2)
    parser.add_argument("--smooth-threshold", type=float, default=0.0025)
    parser.add_argument("--low-scale", type=int, default=2)
    parser.add_argument("--additional-mask-png", type=Path, default=None)
    parser.add_argument("--label", default="route_coding_contexts")
    parser.add_argument(
        "--streams",
        default="vst_y_nonmask,signedlog10_mask",
        help="Comma-separated streams, or all.",
    )
    parser.add_argument(
        "--contexts",
        default="order0,phase2,xtrans6,mask_wn",
        help="Comma-separated contexts to test.",
    )
    args = parser.parse_args()
    requested_streams = {
        item.strip()
        for item in ("vst_y_nonmask,vst_chroma_low_nonmask,vst_chroma_high_nonmask,signedlog10_mask"
                     if args.streams == "all" else args.streams).split(",")
        if item.strip()
    }
    requested_contexts = {item.strip() for item in args.contexts.split(",") if item.strip()}

    pixels = read_exr(DATA_DIR / args.input, args.crop_size)
    route_mask = build_mask(
        pixels,
        args.mask_mode,
        args.dark_max,
        args.mask_radius,
        args.smooth_threshold,
    )
    additional_summary = None
    if args.additional_mask_png is not None:
        additional = read_mask_png(args.additional_mask_png, route_mask.shape)
        additional_summary = {
            "path": str(args.additional_mask_png),
            "pixel_rate": float(np.mean(additional)),
            "new_pixel_rate": float(np.mean(additional & ~route_mask)),
        }
        route_mask |= additional
    nonmask = ~route_mask
    low_nonmask = ~downsample_mask_any(route_mask, args.low_scale)

    y_idx, low_idx, high = quantize_vst_base_indices(
        pixels,
        "gamma075",
        1.0,
        0.01,
        1.0e-4,
        args.y_bits,
        args.chroma_low_bits,
        args.low_scale,
        2,
        0.1,
    )
    signed_idx = quantize_indices(pixels[:, :, :3], args.signedlog_bits, "signed-log")
    high_thresholds = None
    high_ranges = None
    high_mask = None
    high_idx = None
    if "vst_chroma_high_nonmask" in requested_streams:
        high_kept, high_thresholds = shrink_highpass(high, args.base_threshold)
        high_mask = nonmask & np.any(np.abs(high_kept) > 0.0, axis=2)
        high_idx, high_ranges = high_indices(high_kept, args.base_high_bits, high_mask)

    spatial_predictors = ("zero", "west", "north", "avg", "med")
    signed_predictors = (
        "zero",
        "west",
        "north",
        "avg",
        "med",
        "prev-channel-med0",
        "prev-channel-avg0",
        "med-prev-avg",
    )

    streams = []
    if "vst_y_nonmask" in requested_streams:
        y_rows = residual_context_rows(
            "vst_y_nonmask",
            y_idx,
            (args.y_bits,),
            nonmask,
            spatial_predictors,
            requested_contexts,
        )
        streams.append(best_summary("vst_y_nonmask", y_rows, "med", "order0"))

    if "vst_chroma_low_nonmask" in requested_streams:
        low_rows = residual_context_rows(
            "vst_chroma_low_nonmask",
            low_idx,
            (args.chroma_low_bits, args.chroma_low_bits),
            low_nonmask,
            spatial_predictors,
            requested_contexts,
        )
        streams.append(best_summary("vst_chroma_low_nonmask", low_rows, "med", "order0"))

    if "vst_chroma_high_nonmask" in requested_streams:
        if high_idx is None or high_mask is None:
            raise RuntimeError("internal error: high stream was not prepared")
        high_rows = residual_context_rows(
            "vst_chroma_high_nonmask",
            high_idx,
            (args.base_high_bits, args.base_high_bits),
            high_mask,
            spatial_predictors,
            requested_contexts,
        )
        streams.append(best_summary("vst_chroma_high_nonmask", high_rows, "zero", "order0"))

    if "signedlog10_mask" in requested_streams:
        signed_rows = residual_context_rows(
            "signedlog10_mask",
            signed_idx,
            (args.signedlog_bits, args.signedlog_bits, args.signedlog_bits),
            route_mask,
            signed_predictors,
            requested_contexts,
        )
        streams.append(best_summary("signedlog10_mask", signed_rows, "med", "order0"))

    baseline_total = sum(row["baseline"]["estimated_bytes_with_model"] for row in streams)
    best_total = sum(row["best"]["estimated_bytes_with_model"] for row in streams)
    summary = {
        "image": args.input,
        "shape": list(pixels.shape),
        "raw_bytes": int(pixels.nbytes),
        "route_mask_rate": float(np.mean(route_mask)),
        "additional_mask": additional_summary,
        "requested_streams": sorted(requested_streams),
        "requested_contexts": sorted(requested_contexts),
        "high_thresholds": high_thresholds,
        "high_ranges": high_ranges,
        "streams": streams,
        "baseline_value_bytes_with_models": int(baseline_total),
        "best_value_bytes_with_models": int(best_total),
        "best_gain_bytes": int(baseline_total - best_total),
        "note": (
            "Coding-only probe. Reconstructed pixels and route masks are fixed. "
            "Model byte estimates are approximate and penalize large context tables."
        ),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    crop_label = "full" if args.crop_size == 0 else f"crop{args.crop_size}"
    output = RESULTS_DIR / f"{safe_name(args.label)}_{safe_name(Path(args.input).stem)}_{crop_label}.json"
    output.write_text(json.dumps(summary, indent=2))
    print(
        f"baseline_values={baseline_total:,} best_values={best_total:,} "
        f"gain={baseline_total - best_total:,.0f} bytes"
    )
    for stream in streams:
        print(
            f"{stream['stream']}: base={stream['baseline']['estimated_bytes_with_model']:,} "
            f"{stream['baseline']['predictor']}/{stream['baseline']['context']} -> "
            f"best={stream['best']['estimated_bytes_with_model']:,} "
            f"{stream['best']['predictor']}/{stream['best']['context']} "
            f"gain={stream['gain_bytes']:,}"
        )
    print(f"summary={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
