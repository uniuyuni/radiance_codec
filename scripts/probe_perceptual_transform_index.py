"""Probe perceptual transform-index quantization curves.

signed-log/gamma/asinh already move quantization away from raw linear space.
This script compares them with PQ-like and ACEScct-like curves using the proxy
visual gate, so we can separate "perceptually safer" from "payload affordable".
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
from probe_darkbits_router_payload import read_exr  # noqa: E402


PQ_M1 = 2610.0 / 16384.0
PQ_M2 = 2523.0 / 32.0
PQ_C1 = 3424.0 / 4096.0
PQ_C2 = 2413.0 / 128.0
PQ_C3 = 2392.0 / 128.0

ACES_CUT = 0.0078125
ACES_A = 10.5402377416545
ACES_B = 0.0729055341958355
ACES_LOG_A = 9.72
ACES_LOG_B = 17.52


def parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(",") if part.strip())


def parse_strings(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in text.split(",") if part.strip())


def safe_name(text: str) -> str:
    return text.replace("*", "star").replace(".", "_").replace(",", "-")


def pq_oetf(x: np.ndarray) -> np.ndarray:
    x = np.maximum(x, 0.0)
    xm = np.power(x, PQ_M1)
    return np.power((PQ_C1 + PQ_C2 * xm) / (1.0 + PQ_C3 * xm), PQ_M2)


def pq_eotf(y: np.ndarray) -> np.ndarray:
    ym = np.power(np.maximum(y, 0.0), 1.0 / PQ_M2)
    num = np.maximum(ym - PQ_C1, 0.0)
    den = np.maximum(PQ_C2 - PQ_C3 * ym, 1.0e-12)
    return np.power(num / den, 1.0 / PQ_M1)


def acescct_encode_positive(x: np.ndarray) -> np.ndarray:
    x = np.maximum(x, 0.0)
    out = ACES_A * x + ACES_B
    mask = x >= ACES_CUT
    out[mask] = (np.log2(x[mask]) + ACES_LOG_A) / ACES_LOG_B
    return out


def acescct_decode_positive(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    y_cut = ACES_A * ACES_CUT + ACES_B
    return np.where(
        y >= y_cut,
        np.exp2(y * ACES_LOG_B - ACES_LOG_A),
        (y - ACES_B) / ACES_A,
    )


def transform_values(values: np.ndarray, mode: str, scale: float) -> np.ndarray:
    v = values.astype(np.float64)
    av = np.abs(v)
    sign = np.sign(v)
    scale = max(float(scale), 1.0e-12)
    if mode == "linear":
        return v
    if mode == "signed-log":
        return sign * np.log2(1.0 + av)
    if mode == "gamma075":
        return sign * np.power(av, 0.75)
    if mode == "gamma09":
        return sign * np.power(av, 0.9)
    if mode == "asinh":
        return np.arcsinh(v)
    if mode == "pq":
        return sign * pq_oetf(av / scale)
    if mode == "acescct":
        return sign * acescct_encode_positive(av / scale)
    raise ValueError(f"unknown transform: {mode}")


def inverse_transform_values(values: np.ndarray, mode: str, scale: float) -> np.ndarray:
    v = values.astype(np.float64)
    av = np.abs(v)
    sign = np.sign(v)
    scale = max(float(scale), 1.0e-12)
    if mode == "linear":
        return v
    if mode == "signed-log":
        return sign * (np.exp2(av) - 1.0)
    if mode == "gamma075":
        return sign * np.power(av, 1.0 / 0.75)
    if mode == "gamma09":
        return sign * np.power(av, 1.0 / 0.9)
    if mode == "asinh":
        return np.sinh(v)
    if mode == "pq":
        return sign * pq_eotf(av) * scale
    if mode == "acescct":
        return sign * acescct_decode_positive(av) * scale
    raise ValueError(f"unknown transform: {mode}")


def quantize_transform(
    pixels: np.ndarray,
    bits: int,
    mode: str,
    scale: float,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    levels = (1 << bits) - 1
    indices = np.zeros(pixels.shape, dtype=np.uint16)
    decoded = np.zeros(pixels.shape, dtype=np.float32)
    ranges = []
    for channel in range(pixels.shape[2]):
        values = pixels[:, :, channel].astype(np.float64)
        transformed = transform_values(values, mode, scale)
        range_values = transformed.astype(np.float32).astype(np.float64)
        lo = float(np.min(range_values))
        hi = float(np.max(range_values))
        ranges.append({"lo": lo, "hi": hi})
        if not hi > lo:
            decoded[:, :, channel] = inverse_transform_values(
                np.asarray(lo),
                mode,
                scale,
            ).astype(np.float32)
            continue
        q = np.floor((transformed - lo) / (hi - lo) * levels + 0.5)
        q = np.clip(q, 0, levels).astype(np.uint16)
        indices[:, :, channel] = q
        rec_t = lo + q.astype(np.float64) * (hi - lo) / levels
        decoded[:, :, channel] = inverse_transform_values(rec_t, mode, scale).astype(np.float32)
    return indices, np.ascontiguousarray(decoded), ranges


def estimate_payload(indices: np.ndarray, bits: int) -> dict:
    stats = residual_stats(indices, bits, "med")
    estimated_bytes = int(
        math.ceil(stats["residual_order0_entropy_bits_per_sample"] * indices.size / 8.0)
        + 128
    )
    return {
        "estimated_bytes": estimated_bytes,
        "estimated_bps": stats["residual_order0_entropy_bits_per_sample"],
        "nonzero_rate": stats["nonzero_rate"],
    }


def summarize_case(
    pixels: np.ndarray,
    anchor_regions: dict,
    bits: int,
    transform: str,
    scale: float,
    white: float,
    gamma: float,
) -> dict:
    started = time.perf_counter()
    indices, decoded, ranges = quantize_transform(pixels, bits, transform, scale)
    payload = estimate_payload(indices, bits)
    regions = candidate_metrics(pixels, decoded, white, gamma)
    gate = evaluate_gate(regions, anchor_regions, default_gate_args())
    key_metrics = summarize_key_metrics(regions, anchor_regions)
    return {
        "label": f"{transform}_bits{bits}_scale{scale:g}",
        "bits": bits,
        "transform": transform,
        "scale": scale,
        "ranges": ranges,
        "raw_bytes": int(pixels.nbytes),
        "estimated_bytes": payload["estimated_bytes"],
        "estimated_ratio": pixels.nbytes / float(payload["estimated_bytes"]),
        "payload": payload,
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
        f"{prefix}{row['gate']['decision'].upper():6s} {row['label']:28s} "
        f"est={row['estimated_bytes']:,} ratio={row['estimated_ratio']:.2f}x "
        f"bps={row['payload']['estimated_bps']:.3f} "
        f"darkD={dark['lost_detail_delta_vs_anchor']:.2%} "
        f"hiD={high['lost_detail_delta_vs_anchor']:.2%} - {reasons}",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="sample_DSCF0009.EXR")
    parser.add_argument("--crop-size", type=int, default=1024)
    parser.add_argument("--bits", default="7,8,9,10")
    parser.add_argument(
        "--transforms",
        default="signed-log,gamma075,gamma09,asinh,pq,acescct",
    )
    parser.add_argument("--scale", type=float, default=4.0)
    parser.add_argument("--anchor-transform", default="signed-log")
    parser.add_argument("--anchor-bits", type=int, default=10)
    parser.add_argument("--white", type=float, default=4.0)
    parser.add_argument("--gamma", type=float, default=2.2)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    rows_by_image = []
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
        for bits in parse_ints(args.bits):
            for transform in parse_strings(args.transforms):
                rows.append(
                    summarize_case(
                        pixels,
                        anchor_regions,
                        bits,
                        transform,
                        args.scale,
                        args.white,
                        args.gamma,
                    )
                )
        rows.sort(key=lambda row: (decision_rank(row["gate"]["decision"]), row["estimated_bytes"]))
        rows_by_image.append({
            "image": path.name,
            "shape": list(pixels.shape),
            "crop_size": args.crop_size,
            "anchor": f"quant:{args.anchor_transform}:{args.anchor_bits}",
            "rows": rows,
        })
        for row in rows[:args.top]:
            print_row(row, prefix="  ")
        if args.limit and len(rows_by_image) >= args.limit:
            break

    if not rows_by_image:
        print(f"no files matched: {args.glob}", flush=True)
        return 1
    if not args.no_save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        output = RESULTS_DIR / (
            f"perceptual_transform_index_{safe_name(args.glob)}"
            f"_crop{args.crop_size}_bits{args.bits.replace(',', '-')}"
            f"_scale{args.scale:g}.json"
        )
        output.write_text(json.dumps(rows_by_image, indent=2))
        print(f"\nSaved: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
