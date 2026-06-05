"""Probe luma-priority asymmetric color transform-index quantization.

Prior color probes used the same bit depth for all transformed planes.  This
one tests the perceptual idea directly: keep the luma-like plane precise and
compress chroma-like planes more aggressively, then judge the RGB result with
the proxy visual gate.
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
    return text.replace("*", "star").replace(".", "_").replace(",", "-")


def quantize_plane(
    values: np.ndarray,
    bits: int,
    transform: str,
    power_gamma: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    levels = (1 << bits) - 1
    transformed = transform_values(values, transform, power_gamma)
    range_values = transformed.astype(np.float32).astype(np.float64)
    lo = float(np.min(range_values))
    hi = float(np.max(range_values))
    indices = np.zeros(values.shape, dtype=np.uint16)
    if not hi > lo:
        decoded = inverse_transform_values(
            np.asarray(lo),
            transform,
            power_gamma,
        ).astype(np.float32)
        return indices, np.full(values.shape, decoded, dtype=np.float32), {"lo": lo, "hi": hi}
    q = np.floor((transformed - lo) / (hi - lo) * levels + 0.5)
    q = np.clip(q, 0, levels).astype(np.uint16)
    rec_t = lo + q.astype(np.float64) * (hi - lo) / levels
    decoded = inverse_transform_values(rec_t, transform, power_gamma).astype(np.float32)
    return q, decoded, {"lo": lo, "hi": hi}


def quantize_asymmetric(
    pixels: np.ndarray,
    color: str,
    bits: tuple[int, int, int],
    transforms: tuple[str, str, str],
    power_gamma: float,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    planes = forward_color(pixels, color)
    indices = np.zeros(planes.shape, dtype=np.uint16)
    decoded_planes = np.array(planes, copy=True)
    ranges = []
    for channel in range(min(3, planes.shape[2])):
        q, decoded, channel_range = quantize_plane(
            planes[:, :, channel],
            bits[channel],
            transforms[channel],
            power_gamma,
        )
        indices[:, :, channel] = q
        decoded_planes[:, :, channel] = decoded
        ranges.append(channel_range)
    if planes.shape[2] > 3:
        decoded_planes[:, :, 3:] = planes[:, :, 3:]
    return indices, inverse_color(decoded_planes, color), ranges


def estimate_index_bytes(indices: np.ndarray, bits: tuple[int, int, int]) -> dict:
    total_bits = 0.0
    rows = []
    for channel, bit_depth in enumerate(bits):
        channel_indices = indices[:, :, channel:channel + 1]
        stats = residual_stats(channel_indices, bit_depth, "med")
        channel_bits = stats["residual_order0_entropy_bits_per_sample"] * channel_indices.size
        total_bits += channel_bits
        rows.append({
            "channel": channel,
            "bits": bit_depth,
            "estimated_bits_per_sample": stats["residual_order0_entropy_bits_per_sample"],
            "nonzero_rate": stats["nonzero_rate"],
            "estimated_bits": channel_bits,
        })
    metadata_bytes = 128 + 8 * len(bits)
    return {
        "estimated_bits": total_bits,
        "estimated_bytes": int(math.ceil(total_bits / 8.0) + metadata_bytes),
        "channels": rows,
        "metadata_bytes": metadata_bytes,
    }


def summarize_case(
    pixels: np.ndarray,
    anchor_regions: dict,
    color: str,
    bits: tuple[int, int, int],
    transforms: tuple[str, str, str],
    power_gamma: float,
    white: float,
    gamma: float,
) -> dict:
    started = time.perf_counter()
    indices, decoded, ranges = quantize_asymmetric(
        pixels,
        color,
        bits,
        transforms,
        power_gamma,
    )
    payload = estimate_index_bytes(indices, bits)
    regions = candidate_metrics(pixels, decoded, white, gamma)
    gate = evaluate_gate(regions, anchor_regions, default_gate_args())
    key_metrics = summarize_key_metrics(regions, anchor_regions)
    return {
        "label": (
            f"{color}_bits{bits[0]}-{bits[1]}-{bits[2]}"
            f"_{transforms[0]}-{transforms[1]}-{transforms[2]}"
        ),
        "color": color,
        "bits": list(bits),
        "transforms": list(transforms),
        "power_gamma": power_gamma,
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
        f"{prefix}{row['gate']['decision'].upper():6s} {row['label']:48s} "
        f"est={row['estimated_bytes']:,} ratio={row['estimated_ratio']:.2f}x "
        f"darkD={dark['lost_detail_delta_vs_anchor']:.2%} "
        f"hiD={high['lost_detail_delta_vs_anchor']:.2%} - {reasons}",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="sample_DSCF0009.EXR")
    parser.add_argument("--crop-size", type=int, default=1024)
    parser.add_argument("--color", default="ycocg")
    parser.add_argument("--y-bits", default="8,9,10")
    parser.add_argument("--chroma-bits", default="4,5,6,7,8")
    parser.add_argument("--y-transforms", default="signed-log,gamma075,asinh")
    parser.add_argument("--chroma-transforms", default="linear,signed-log,gamma075")
    parser.add_argument("--power-gamma", type=float, default=0.75)
    parser.add_argument("--anchor-transform", default="signed-log")
    parser.add_argument("--anchor-bits", type=int, default=10)
    parser.add_argument("--white", type=float, default=4.0)
    parser.add_argument("--gamma", type=float, default=2.2)
    parser.add_argument("--top", type=int, default=18)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    y_bits_values = parse_ints(args.y_bits)
    chroma_bits_values = parse_ints(args.chroma_bits)
    y_transforms = parse_strings(args.y_transforms)
    chroma_transforms = parse_strings(args.chroma_transforms)
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
        for y_bits in y_bits_values:
            for chroma_bits in chroma_bits_values:
                bits = (y_bits, chroma_bits, chroma_bits)
                for y_transform in y_transforms:
                    for chroma_transform in chroma_transforms:
                        rows.append(
                            summarize_case(
                                pixels,
                                anchor_regions,
                                args.color,
                                bits,
                                (y_transform, chroma_transform, chroma_transform),
                                args.power_gamma,
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
            f"asymmetric_color_index_{safe_name(args.glob)}"
            f"_crop{args.crop_size}_{args.color}"
            f"_y{args.y_bits.replace(',', '-')}"
            f"_c{args.chroma_bits.replace(',', '-')}.json"
        )
        output.write_text(json.dumps(summaries, indent=2))
        print(f"\nSaved: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
