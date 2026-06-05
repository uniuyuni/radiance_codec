"""Estimate a VST-chroma base plus signed-log10 masked escape route.

This is a sizing probe, not a bitstream implementation.  It measures the
current diagnostic that looked visually OK:

* non-risk pixels use the VST-chroma denoised base;
* risk pixels are corrected toward the faithful signed-log10 reconstruction.

The estimate uses selected-region residual entropy for the VST base components
and a selected signed-log index stream for the mask.  It is intentionally
conservative enough for research direction, but not a final codec byte count.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import OpenImageIO as oiio

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
sys.path.insert(0, str(ROOT / "scripts"))

from audit_linear_index_payload import predict_for_channel, quantize_indices  # noqa: E402
from probe_adaptive_ycocg_router import build_mask, entropy_bits_from_counts  # noqa: E402
from probe_darkbits_router_payload import read_exr, summarize_binary_mask  # noqa: E402
from probe_vst_chroma_nr import (  # noqa: E402
    block_mean_downsample,
    forward_vst,
    guided_filter_gray,
    quantize_range,
    safe_name,
    selected_entropy_indices,
    shrink_highpass,
)
from probe_color_tile_transform_index import forward_color  # noqa: E402


def residual_selected_cost(
    indices: np.ndarray,
    bits: tuple[int, ...],
    mask: np.ndarray,
    predictor: str = "med",
) -> dict:
    total_bits = 0.0
    rows = []
    for channel, bit_depth in enumerate(bits):
        selected = mask
        samples = int(np.count_nonzero(selected))
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
        residual = (
            indices[:, :, channel].astype(np.int32) + alphabet - pred
        ) & (alphabet - 1)
        values = residual[selected]
        counts = np.bincount(values.astype(np.int64), minlength=alphabet)
        bits_cost = entropy_bits_from_counts(counts)
        total_bits += bits_cost
        rows.append({
            "channel": channel,
            "bits": bit_depth,
            "predictor": predictor,
            "samples": samples,
            "estimated_bits": bits_cost,
            "bits_per_sample": bits_cost / float(samples),
        })
    return {
        "predictor": predictor,
        "estimated_bits": total_bits,
        "estimated_bytes": int(math.ceil(total_bits / 8.0) + 32),
        "channels": rows,
    }


def strip_arrays(row: dict) -> dict:
    return {key: value for key, value in row.items() if not isinstance(value, np.ndarray)}


def quantize_vst_base_indices(
    pixels: np.ndarray,
    vst_mode: str,
    vst_a: float,
    vst_b: float,
    vst_eps: float,
    y_bits: int,
    chroma_low_bits: int,
    low_scale: int,
    guide_radius: int,
    guide_eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb = pixels[:, :, :3]
    t_rgb = forward_vst(rgb, vst_mode, vst_a, vst_b, vst_eps).astype(np.float32)
    planes = forward_color(t_rgb, "ycocg")[:, :, :3]
    y_plane = planes[:, :, 0]
    chroma = planes[:, :, 1:3]
    y_idx, _, _ = quantize_range(y_plane, y_bits)

    guide = y_plane.astype(np.float64)
    low_full = np.zeros(chroma.shape, dtype=np.float32)
    for channel in range(2):
        low_full[:, :, channel] = guided_filter_gray(
            guide,
            chroma[:, :, channel].astype(np.float64),
            guide_radius,
            guide_eps,
        )
    low_indices = []
    low_masks = []
    for channel in range(2):
        coarse = block_mean_downsample(low_full[:, :, channel], low_scale)
        q, _, _ = quantize_range(coarse, chroma_low_bits)
        low_indices.append(q)
        low_masks.append(np.ones(q.shape, dtype=bool))
    high = chroma - low_full
    return y_idx[:, :, None], np.stack(low_indices, axis=2), high


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


def read_mask_png(path: Path, shape: tuple[int, int]) -> np.ndarray:
    image_input = oiio.ImageInput.open(str(path))
    if image_input is None:
        raise RuntimeError(f"can't open {path}: {oiio.geterror()}")
    spec = image_input.spec()
    pixels = image_input.read_image(format=oiio.UINT16)
    image_input.close()
    if pixels is None:
        raise RuntimeError(f"can't read {path}: {oiio.geterror()}")
    image = np.asarray(pixels, dtype=np.uint16).reshape(
        spec.height,
        spec.width,
        spec.nchannels,
    )
    if image.shape[:2] != shape:
        raise RuntimeError(f"mask shape mismatch: mask={image.shape[:2]} expected={shape}")
    return np.any(image[:, :, : min(3, spec.nchannels)] > 0, axis=2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="sample_DSCF0009.EXR")
    parser.add_argument("--crop-size", type=int, default=0)
    parser.add_argument("--vst-mode", default="gamma075")
    parser.add_argument("--vst-a", type=float, default=1.0)
    parser.add_argument("--vst-b", type=float, default=0.01)
    parser.add_argument("--vst-eps", type=float, default=1.0e-4)
    parser.add_argument("--y-bits", type=int, default=10)
    parser.add_argument("--chroma-low-bits", type=int, default=8)
    parser.add_argument("--low-scale", type=int, default=2)
    parser.add_argument("--guide-radius", type=int, default=2)
    parser.add_argument("--guide-eps", type=float, default=0.1)
    parser.add_argument("--signedlog-bits", type=int, default=10)
    parser.add_argument("--signedlog-predictor", default="med")
    parser.add_argument("--base-high-bits", type=int, default=5)
    parser.add_argument("--base-threshold", type=float, default=2.5)
    parser.add_argument("--mask-mode", default="dark-smooth")
    parser.add_argument("--dark-max", type=float, default=0.5)
    parser.add_argument("--mask-radius", type=int, default=2)
    parser.add_argument("--smooth-threshold", type=float, default=0.004)
    parser.add_argument(
        "--additional-mask-png",
        type=Path,
        default=None,
        help="Optional extra guard mask PNG to OR with the generated mask.",
    )
    args = parser.parse_args()

    path = DATA_DIR / args.input
    pixels = read_exr(path, args.crop_size)
    mask = build_mask(
        pixels,
        args.mask_mode,
        args.dark_max,
        args.mask_radius,
        args.smooth_threshold,
    )
    additional_mask = None
    additional_mask_summary = None
    if args.additional_mask_png is not None:
        additional_mask = read_mask_png(args.additional_mask_png, mask.shape)
        additional_mask_summary = {
            "path": str(args.additional_mask_png),
            "pixel_rate": float(np.mean(additional_mask)),
            "overlap_with_base_mask_rate": float(np.mean(additional_mask & mask)),
            "new_pixel_rate": float(np.mean(additional_mask & ~mask)),
        }
        mask = mask | additional_mask
    nonmask = ~mask
    low_nonmask = ~downsample_mask_any(mask, args.low_scale)

    y_idx, low_idx, high = quantize_vst_base_indices(
        pixels,
        args.vst_mode,
        args.vst_a,
        args.vst_b,
        args.vst_eps,
        args.y_bits,
        args.chroma_low_bits,
        args.low_scale,
        args.guide_radius,
        args.guide_eps,
    )
    signed_idx = quantize_indices(pixels[:, :, :3], args.signedlog_bits, "signed-log")
    high_kept, high_thresholds = shrink_highpass(high, args.base_threshold)
    high_mask_nonrisk = nonmask & np.any(np.abs(high_kept) > 0.0, axis=2)

    y_payload = residual_selected_cost(y_idx, (args.y_bits,), nonmask)
    low_payload = residual_selected_cost(
        low_idx,
        (args.chroma_low_bits, args.chroma_low_bits),
        low_nonmask,
    )
    signedlog_payload = residual_selected_cost(
        signed_idx,
        (args.signedlog_bits, args.signedlog_bits, args.signedlog_bits),
        mask,
        args.signedlog_predictor,
    )
    high_payload = strip_arrays(
        selected_entropy_indices(high_kept, args.base_high_bits, high_mask_nonrisk)
    )
    high_mask_summary = summarize_binary_mask(high_mask_nonrisk)
    high_mask_payload = min(
        high_mask_summary["order0"],
        high_mask_summary["west_north"],
        key=lambda row: row["total_bytes"],
    )
    mask_summary = summarize_binary_mask(mask)
    mask_payload = min(
        mask_summary["order0"],
        mask_summary["west_north"],
        key=lambda row: row["total_bytes"],
    )
    metadata_bytes = 512
    estimated_bytes = (
        y_payload["estimated_bytes"]
        + low_payload["estimated_bytes"]
        + high_payload["estimated_bytes"]
        + int(high_mask_payload["total_bytes"])
        + signedlog_payload["estimated_bytes"]
        + int(mask_payload["total_bytes"])
        + metadata_bytes
    )
    summary = {
        "image": args.input,
        "shape": list(pixels.shape),
        "raw_bytes": int(pixels.nbytes),
        "mask_mode": args.mask_mode,
        "dark_max": args.dark_max,
        "mask_radius": args.mask_radius,
        "smooth_threshold": args.smooth_threshold,
        "additional_mask": additional_mask_summary,
        "signedlog_predictor": args.signedlog_predictor,
        "mask_pixel_rate": float(np.mean(mask)),
        "low_nonmask_pixel_rate": float(np.mean(low_nonmask)),
        "estimated_bytes": int(estimated_bytes),
        "estimated_ratio": pixels.nbytes / float(estimated_bytes),
        "payload": {
            "vst_y_nonmask": y_payload,
            "vst_chroma_low_nonmask": low_payload,
            "vst_chroma_high_nonmask": high_payload,
            "vst_chroma_high_nonmask_mask": high_mask_payload,
            "vst_chroma_high_thresholds": high_thresholds,
            "vst_chroma_high_nonmask_pixel_rate": float(np.mean(high_mask_nonrisk)),
            "signedlog10_mask": signedlog_payload,
            "mask": mask_payload,
            "metadata_bytes": metadata_bytes,
        },
        "note": (
            "Sizing probe for VST nonmask + signed-log masked route. "
            "Includes VST Y/low/high for nonmask and signed-log indices for mask. "
            "This is an entropy probe, not a final bitstream byte count."
        ),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    crop_label = "full" if args.crop_size == 0 else f"crop{args.crop_size}"
    output = RESULTS_DIR / (
        f"vst_signedlog_route_{safe_name(path.stem)}_{crop_label}"
        f"_Y{args.y_bits}_CL{args.chroma_low_bits}"
        f"_slog{args.signedlog_bits}"
        f"_spred{safe_name(args.signedlog_predictor)}"
        f"_{args.mask_mode}{args.dark_max:g}_st{args.smooth_threshold:g}.json"
    )
    if args.additional_mask_png is not None:
        output = RESULTS_DIR / (
            f"vst_signedlog_route_{safe_name(path.stem)}_{crop_label}"
            f"_Y{args.y_bits}_CL{args.chroma_low_bits}"
            f"_slog{args.signedlog_bits}"
            f"_spred{safe_name(args.signedlog_predictor)}"
            f"_{args.mask_mode}{args.dark_max:g}_st{args.smooth_threshold:g}"
            f"_plus_{safe_name(args.additional_mask_png.stem)}.json"
        )
    output.write_text(json.dumps(summary, indent=2))
    print(
        f"mask={summary['mask_pixel_rate']:.2%} "
        f"est={summary['estimated_bytes']:,} "
        f"ratio={summary['estimated_ratio']:.2f}x"
    )
    print(
        f" y={y_payload['estimated_bytes']:,} "
        f"low={low_payload['estimated_bytes']:,} "
        f"high={high_payload['estimated_bytes']:,}+{int(high_mask_payload['total_bytes']):,} "
        f"signedlog_mask={signedlog_payload['estimated_bytes']:,} "
        f"mask_bytes={int(mask_payload['total_bytes']):,}"
    )
    print(f"summary={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
