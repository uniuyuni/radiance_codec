"""Probe low-resolution dark correction maps for bits8 banding.

Per-pixel dark refinement preserves quality but costs too much.  This probe
tries a cheaper side channel: store a low-resolution correction field in
signed-log space and apply it only to deterministic dark/banding-risk pixels.

If dark banding is mostly low-frequency quantization drift, this should rescue
gradation at a fraction of the bits10 refinement cost.  If the missing signal is
mostly per-pixel high entropy, the proxy gate will still reject it.
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
from audit_linear_index_payload import residual_stats  # noqa: E402
from audit_visual_gate import (  # noqa: E402
    candidate_metrics,
    default_gate_args,
    evaluate_gate,
    summarize_key_metrics,
)
from probe_darkbits_router_payload import (  # noqa: E402
    dark_luma_mask,
    dark_router_mask,
    inverse_signed_log,
    quantize_signed_log,
    read_exr,
    signed_log,
)


BUILTIN_TRANSFORMS = {"linear", "signed-log", "gamma075", "asinh"}


def parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(",") if part.strip())


def parse_floats(text: str) -> tuple[float, ...]:
    return tuple(float(part) for part in text.split(",") if part.strip())


def safe_name(text: str) -> str:
    return text.replace("*", "star").replace(".", "_").replace(",", "-")


def entropy_bits(values: np.ndarray) -> float:
    flat = np.ascontiguousarray(values).reshape(-1)
    if flat.size == 0:
        return 0.0
    _, counts = np.unique(flat, return_counts=True)
    probs = counts.astype(np.float64) / float(flat.size)
    return float(-np.sum(counts.astype(np.float64) * np.log2(probs)))


def upsample_grid(grid: np.ndarray, height: int, width: int, block_size: int) -> np.ndarray:
    rows, cols, channels = grid.shape
    y_centers = np.minimum(np.arange(rows) * block_size + block_size * 0.5, height - 1)
    x_centers = np.minimum(np.arange(cols) * block_size + block_size * 0.5, width - 1)
    x_target = np.arange(width, dtype=np.float64)
    y_target = np.arange(height, dtype=np.float64)
    out = np.zeros((height, width, channels), dtype=np.float64)
    for channel in range(channels):
        row_interp = np.zeros((rows, width), dtype=np.float64)
        for row in range(rows):
            row_interp[row, :] = np.interp(x_target, x_centers, grid[row, :, channel])
        for x in range(width):
            out[:, x, channel] = np.interp(y_target, y_centers, row_interp[:, x])
    return out


def correction_mask(
    base_decoded: np.ndarray,
    base_bits: int,
    dark_max: float,
    mask_mode: str,
    mask_radius: int,
    std_step_threshold: float,
    transform: str,
    power_gamma: float,
) -> np.ndarray:
    if mask_mode == "luma":
        return dark_luma_mask(base_decoded, dark_max)
    if mask_mode == "banding":
        return dark_router_mask(
            base_decoded,
            base_bits,
            dark_max,
            "banding",
            mask_radius,
            std_step_threshold,
            transform,
            power_gamma,
        )
    raise ValueError(f"unknown mask mode: {mask_mode}")


def build_correction_grid(
    original: np.ndarray,
    base_decoded: np.ndarray,
    mask: np.ndarray,
    block_size: int,
) -> np.ndarray:
    height, width, channels = original.shape
    rows = math.ceil(height / block_size)
    cols = math.ceil(width / block_size)
    residual = signed_log(original) - signed_log(base_decoded)
    grid = np.zeros((rows, cols, channels), dtype=np.float64)
    for gy in range(rows):
        y0 = gy * block_size
        y1 = min(height, y0 + block_size)
        for gx in range(cols):
            x0 = gx * block_size
            x1 = min(width, x0 + block_size)
            block_mask = mask[y0:y1, x0:x1]
            if not np.any(block_mask):
                continue
            for channel in range(channels):
                values = residual[y0:y1, x0:x1, channel][block_mask]
                if values.size:
                    grid[gy, gx, channel] = float(np.mean(values))
    return grid


def quantize_grid(grid: np.ndarray, corr_bits: int) -> tuple[np.ndarray, np.ndarray, dict]:
    levels = (1 << corr_bits) - 1
    decoded = np.zeros(grid.shape, dtype=np.float64)
    quantized = np.zeros(grid.shape, dtype=np.uint16)
    ranges = []
    for channel in range(grid.shape[2]):
        values = grid[:, :, channel]
        lo = float(np.min(values))
        hi = float(np.max(values))
        ranges.append({"lo": lo, "hi": hi})
        if not hi > lo:
            decoded[:, :, channel] = lo
            continue
        q = np.floor((values - lo) / (hi - lo) * levels + 0.5)
        q = np.clip(q, 0, levels).astype(np.uint16)
        quantized[:, :, channel] = q
        decoded[:, :, channel] = lo + q.astype(np.float64) * (hi - lo) / levels
    fixed_bytes = int(math.ceil(quantized.size * corr_bits / 8.0) + 8 * len(ranges))
    entropy_bytes = int(math.ceil(entropy_bits(quantized) / 8.0) + 8 * len(ranges) + 16)
    return quantized, decoded, {
        "ranges": ranges,
        "fixed_bytes": fixed_bytes,
        "entropy_bytes": entropy_bytes,
    }


def estimate_base_bytes(
    pixels: np.ndarray,
    indices: np.ndarray,
    base_bits: int,
    transform: str,
    power_gamma: float,
    encode_base: bool,
) -> dict:
    stats = residual_stats(indices, base_bits, "med")
    estimate = int(
        math.ceil(stats["residual_order0_entropy_bits_per_sample"] * pixels.size / 8.0)
        + 128
    )
    encoded = None
    if encode_base and transform in BUILTIN_TRANSFORMS:
        payload = radiance_codec.encode_linear_index_near_lossless(
            pixels,
            bits=base_bits,
            effort=9,
            transform=transform,
        )
        encoded = len(payload)
        estimate = encoded
    return {
        "estimated_bytes": estimate,
        "encoded_bytes": encoded,
        "estimated_bps": stats["residual_order0_entropy_bits_per_sample"],
        "nonzero_rate": stats["nonzero_rate"],
    }


def summarize_case(
    pixels: np.ndarray,
    anchor_regions: dict,
    base_bits: int,
    transform: str,
    power_gamma: float,
    block_size: int,
    corr_bits: int,
    dark_max: float,
    mask_mode: str,
    mask_radius: int,
    std_step_threshold: float,
    encode_base: bool,
    white: float,
    gamma: float,
) -> dict:
    started = time.perf_counter()
    indices, base_decoded, _ = quantize_signed_log(
        pixels,
        base_bits,
        transform,
        power_gamma,
    )
    mask = correction_mask(
        base_decoded,
        base_bits,
        dark_max,
        mask_mode,
        mask_radius,
        std_step_threshold,
        transform,
        power_gamma,
    )
    grid = build_correction_grid(pixels, base_decoded, mask, block_size)
    _, decoded_grid, grid_payload = quantize_grid(grid, corr_bits)
    correction = upsample_grid(
        decoded_grid,
        pixels.shape[0],
        pixels.shape[1],
        block_size,
    )
    corrected_log = signed_log(base_decoded)
    corrected_log[mask, :] += correction[mask, :]
    corrected = inverse_signed_log(corrected_log).astype(np.float32)
    base = estimate_base_bytes(
        pixels,
        indices,
        base_bits,
        transform,
        power_gamma,
        encode_base,
    )
    correction_bytes = min(grid_payload["fixed_bytes"], grid_payload["entropy_bytes"])
    estimated_bytes = int(base["estimated_bytes"]) + correction_bytes + 64
    regions = candidate_metrics(pixels, corrected, white, gamma)
    gate = evaluate_gate(regions, anchor_regions, default_gate_args())
    key_metrics = summarize_key_metrics(regions, anchor_regions)
    return {
        "label": (
            f"{transform}_g{power_gamma:g}_b{base_bits}"
            f"_corr{corr_bits}_blk{block_size}"
            f"_max{dark_max:g}_{mask_mode}_std{std_step_threshold:g}"
        ),
        "base_bits": base_bits,
        "transform": transform,
        "power_gamma": power_gamma,
        "block_size": block_size,
        "corr_bits": corr_bits,
        "dark_max": dark_max,
        "mask_mode": mask_mode,
        "mask_radius": mask_radius,
        "std_step_threshold": std_step_threshold,
        "raw_bytes": int(pixels.nbytes),
        "estimated_bytes": estimated_bytes,
        "estimated_ratio": pixels.nbytes / float(estimated_bytes),
        "base": base,
        "mask_pixel_rate": float(np.mean(mask)),
        "grid_shape": list(grid.shape),
        "grid_payload": grid_payload,
        "correction_bytes": correction_bytes,
        "gate": gate,
        "key_metrics": key_metrics,
        "elapsed_sec": time.perf_counter() - started,
    }


def decision_rank(decision: str) -> int:
    return {"pass": 0, "maybe": 1, "reject": 2}.get(decision, 9)


def print_row(row: dict, prefix: str = "") -> None:
    dark = row["key_metrics"]["dark_0_0.25"]
    high = row["key_metrics"]["highlight_1_4"]
    reasons = "; ".join(
        f"{item['region']} {item['metric']}={item['value']:.2e}"
        for item in row["gate"]["failures"][:2]
    )
    if not reasons:
        reasons = "ok"
    print(
        f"{prefix}{row['gate']['decision'].upper():6s} {row['label']:46s} "
        f"est={row['estimated_bytes']:,} ratio={row['estimated_ratio']:.2f}x "
        f"mask={row['mask_pixel_rate']:.2%} corr={row['correction_bytes']:,} "
        f"darkD={dark['lost_detail_delta_vs_anchor']:.2%} "
        f"hiD={high['lost_detail_delta_vs_anchor']:.2%} - {reasons}",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="sample_DSCF0009.EXR")
    parser.add_argument("--crop-size", type=int, default=1024)
    parser.add_argument("--base-bits", type=int, default=8)
    parser.add_argument("--transform", default="gamma075")
    parser.add_argument("--power-gamma", type=float, default=0.9)
    parser.add_argument("--block-sizes", default="16,32,64")
    parser.add_argument("--corr-bits", default="8,10,12")
    parser.add_argument("--dark-max", default="0.03,0.05,0.08")
    parser.add_argument("--mask-mode", choices=("luma", "banding"), default="luma")
    parser.add_argument("--mask-radius", type=int, default=2)
    parser.add_argument("--std-step-threshold", type=float, default=0.25)
    parser.add_argument("--anchor-transform", default="signed-log")
    parser.add_argument("--anchor-bits", type=int, default=10)
    parser.add_argument("--white", type=float, default=4.0)
    parser.add_argument("--gamma", type=float, default=2.2)
    parser.add_argument("--encode-base", action="store_true")
    parser.add_argument("--top", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

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
        rows = []
        for block_size in parse_ints(args.block_sizes):
            for corr_bits in parse_ints(args.corr_bits):
                for dark_max in parse_floats(args.dark_max):
                    rows.append(
                        summarize_case(
                            pixels,
                            anchor_regions,
                            args.base_bits,
                            args.transform,
                            args.power_gamma,
                            block_size,
                            corr_bits,
                            dark_max,
                            args.mask_mode,
                            args.mask_radius,
                            args.std_step_threshold,
                            args.encode_base,
                            args.white,
                            args.gamma,
                        )
                    )
        rows.sort(key=lambda row: (decision_rank(row["gate"]["decision"]), row["estimated_bytes"]))
        summary = {
            "image": path.name,
            "shape": list(pixels.shape),
            "crop_size": args.crop_size,
            "anchor": f"quant:{args.anchor_transform}:{args.anchor_bits}",
            "rows": rows,
        }
        summaries.append(summary)
        for row in rows[:args.top]:
            print_row(row, prefix="  ")
        if args.limit and len(summaries) >= args.limit:
            break

    if not summaries:
        print(f"no files matched: {args.glob}", flush=True)
        return 1
    if not args.no_save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        output = RESULTS_DIR / (
            f"lowres_dark_correction_{safe_name(args.glob)}"
            f"_crop{args.crop_size}_{args.transform}_b{args.base_bits}"
            f"_blk{args.block_sizes.replace(',', '-')}"
            f"_corr{args.corr_bits.replace(',', '-')}.json"
        )
        output.write_text(json.dumps(summaries, indent=2))
        print(f"\nSaved: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
