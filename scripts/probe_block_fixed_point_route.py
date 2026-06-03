"""Probe exact block fixed-point routes for float32 HDR samples.

This route changes the representation instead of squeezing mantissa tail bits:
within each tile, finite float32 values are mapped exactly to signed integers
under a shared power-of-two scale.  The shared-scale integer field is then
tested with byte-plane zstd and simple spatial residuals.

The probe is intentionally conservative: sign is still accounted for, and each
tile pays route/scale/width metadata.  It is a triage tool for deciding whether
this representation is worth moving toward a C++ bitstream mode.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_structural_predictors import DATA_DIR, float_to_bits, read_exr  # noqa: E402
from probe_grouped_tail_mlp import CONTEXT_FAMILIES, context_family_cost  # noqa: E402
from structural_delta_grouped_roundtrip import choose_group_payloads  # noqa: E402

RESULTS_DIR = ROOT / "results"


def entropy_bits_from_counts(counts: np.ndarray) -> float:
    total = float(np.sum(counts))
    if total == 0.0:
        return 0.0
    probs = counts.astype(np.float64) / total
    probs = probs[probs > 0.0]
    return float(-np.sum(probs * np.log2(probs)) * total)


def byte_entropy_cost(values: np.ndarray, byte_count: int) -> float:
    if values.size == 0 or byte_count <= 0:
        return 0.0
    packed = np.ascontiguousarray(values.reshape(-1), dtype=np.uint64)
    byte_view = packed.view(np.uint8).reshape(-1, 8)
    total = 0.0
    for index in range(byte_count):
        counts = np.bincount(byte_view[:, index], minlength=256)
        total += entropy_bits_from_counts(counts)
    return total


def byteplane_bytes(values: np.ndarray, byte_count: int) -> bytes:
    if values.size == 0 or byte_count <= 0:
        return b""
    packed = np.ascontiguousarray(values.reshape(-1), dtype=np.uint64)
    byte_view = packed.view(np.uint8).reshape(-1, 8)
    return b"".join(
        np.ascontiguousarray(byte_view[:, index]).tobytes()
        for index in range(byte_count)
    )


def zstd_cost(values: np.ndarray, byte_count: int, zstd_path: str | None, level: int) -> float:
    if zstd_path is None:
        return float("inf")
    data = byteplane_bytes(values, byte_count)
    if not data:
        return 0.0
    result = subprocess.run(
        [zstd_path, "-q", f"-{level}", "-c"],
        input=data,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return float(len(result.stdout) * 8)


def best_family_cost(payload: np.ndarray, bits: range | tuple[int, ...]) -> float:
    total = 0.0
    for bit in bits:
        total += min(
            context_family_cost(payload, bit, family)
            for family in CONTEXT_FAMILIES
        )
    return total


def current_channel_cost(
    body_payload: np.ndarray,
    sign_payload: np.ndarray,
    channels: list[int],
) -> float:
    if not channels:
        return 0.0
    return best_family_cost(body_payload[..., channels], range(31)) + best_family_cost(
        sign_payload[..., channels],
        (31,),
    )


def zigzag_i64(values: np.ndarray) -> np.ndarray:
    signed = np.ascontiguousarray(values, dtype=np.int64)
    return ((signed << np.int64(1)) ^ (signed >> np.int64(63))).astype(np.uint64)


def bit_length_u64(value: int) -> int:
    return int(value).bit_length()


def shift_zero(values: np.ndarray, dy: int, dx: int) -> np.ndarray:
    out = np.zeros_like(values)
    height, width, _channels = values.shape
    dst_y0 = max(0, -dy)
    dst_y1 = min(height, height - dy)
    dst_x0 = max(0, -dx)
    dst_x1 = min(width, width - dx)
    src_y0 = dst_y0 + dy
    src_y1 = dst_y1 + dy
    src_x0 = dst_x0 + dx
    src_x1 = dst_x1 + dx
    out[dst_y0:dst_y1, dst_x0:dst_x1] = values[src_y0:src_y1, src_x0:src_x1]
    return out


def paeth(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    p = a + b - c
    pa = np.abs(p - a)
    pb = np.abs(p - b)
    pc = np.abs(p - c)
    return np.where((pa <= pb) & (pa <= pc), a, np.where(pb <= pc, b, c))


def residual_candidates(values: np.ndarray) -> dict[str, np.ndarray]:
    west = shift_zero(values, 0, -1)
    north = shift_zero(values, -1, 0)
    northwest = shift_zero(values, -1, -1)
    previous_channel = np.zeros_like(values)
    if values.shape[2] > 1:
        previous_channel[..., 1:] = values[..., :-1]
    green_lift = values.copy()
    green_reorder = values.copy()
    ycocg = values.copy()
    if values.shape[2] >= 3:
        green_lift[..., 0] = values[..., 0] - values[..., 1]
        green_lift[..., 1] = values[..., 1]
        green_lift[..., 2] = values[..., 2] - values[..., 1]
        if values.shape[2] > 3:
            green_lift[..., 3:] = values[..., 3:]

        green_reorder[..., 0] = values[..., 1]
        green_reorder[..., 1] = values[..., 0] - values[..., 1]
        green_reorder[..., 2] = values[..., 2] - values[..., 1]
        if values.shape[2] > 3:
            green_reorder[..., 3:] = values[..., 3:]

        red = values[..., 0]
        green_channel = values[..., 1]
        blue = values[..., 2]
        co = red - blue
        tmp = blue + (co >> np.int64(1))
        cg = green_channel - tmp
        y = tmp + (cg >> np.int64(1))
        ycocg[..., 0] = y
        ycocg[..., 1] = co
        ycocg[..., 2] = cg
        if values.shape[2] > 3:
            ycocg[..., 3:] = values[..., 3:]
    average = (west + north) // np.int64(2)
    gradient = west + north - northwest
    return {
        "raw": values,
        "west_delta": values - west,
        "north_delta": values - north,
        "average_delta": values - average,
        "gradient_delta": values - gradient,
        "paeth_delta": values - paeth(west, north, northwest),
        "previous_channel_delta": values - previous_channel,
        "green_lift": green_lift,
        "green_reorder_lift": green_reorder,
        "ycocg_lift": ycocg,
    }


def fixed_point_integer_tile(bits_tile: np.ndarray) -> tuple[np.ndarray, int, int, bool]:
    sign = ((bits_tile >> np.uint32(31)) & np.uint32(1)).astype(np.uint8)
    exponent = ((bits_tile >> np.uint32(23)) & np.uint32(0xFF)).astype(np.uint16)
    mantissa = (bits_tile & np.uint32(0x7FFFFF)).astype(np.uint64)
    finite = exponent != 0xFF
    if not bool(np.all(finite)):
        raise ValueError("non-finite values are not supported by this route")

    scale_field = np.where(exponent == 0, np.uint16(1), exponent).astype(np.uint16)
    significand = np.where(
        exponent == 0,
        mantissa,
        mantissa | np.uint64(1 << 23),
    ).astype(np.uint64)
    nonzero = significand != 0
    min_scale = int(np.min(scale_field[nonzero])) if bool(np.any(nonzero)) else 1
    shifts = (scale_field.astype(np.int32) - min_scale).astype(np.int32)
    if int(np.max(shifts, initial=0)) > 39:
        # 24 + 39 keeps unsigned integer magnitude within int64 after sign.
        raise OverflowError("shared-scale integer would exceed int64 budget")

    unsigned = significand << shifts.astype(np.uint64)
    max_unsigned = int(np.max(unsigned, initial=np.uint64(0)))
    if max_unsigned >= (1 << 62):
        raise OverflowError("shared-scale integer would exceed signed int64 budget")
    signed = unsigned.astype(np.int64)
    signed = np.where(sign != 0, -signed, signed).astype(np.int64)
    max_abs = int(np.max(np.abs(signed), initial=np.int64(0)))
    magnitude_width = bit_length_u64(max_abs)
    return signed, min_scale, magnitude_width, bool(np.all(sign == 0))


def reconstruct_bits_from_fixed(values: np.ndarray, min_scale: int, signs: np.ndarray) -> np.ndarray:
    out = np.zeros(values.shape, dtype=np.uint32)
    flat_values = values.reshape(-1)
    flat_signs = signs.reshape(-1)
    flat_out = out.reshape(-1)
    for index, value in enumerate(flat_values):
        sign_bit = int(flat_signs[index])
        abs_value = abs(int(value))
        if abs_value == 0:
            flat_out[index] = np.uint32(sign_bit << 31)
            continue
        length = abs_value.bit_length()
        if min_scale == 1 and length <= 23:
            exponent = 0
            mantissa = abs_value
        else:
            exponent = min_scale + length - 24
            shift = length - 24
            if shift < 0:
                raise RuntimeError("invalid normal reconstruction shift")
            if shift and (abs_value & ((1 << shift) - 1)):
                raise RuntimeError("shared integer is not canonical float32")
            significand = abs_value >> shift
            if significand < (1 << 23) or significand >= (1 << 24):
                raise RuntimeError("invalid normal significand")
            mantissa = significand - (1 << 23)
        flat_out[index] = np.uint32(
            (sign_bit << 31) | (int(exponent) << 23) | int(mantissa),
        )
    return out


def summarize_tile(
    bits_tile: np.ndarray,
    body_payload_tile: np.ndarray,
    sign_payload_tile: np.ndarray,
    metadata_bits: float,
    zstd_path: str | None,
    zstd_level: int,
    verify: bool,
) -> tuple[dict[str, float], dict[str, object]]:
    sign_cost = best_family_cost(sign_payload_tile, (31,))
    body_cost = best_family_cost(body_payload_tile, range(31))
    current = sign_cost + body_cost
    sign_bits = ((bits_tile >> np.uint32(31)) & np.uint32(1)).astype(np.uint8)
    try:
        integers, min_scale, magnitude_width, all_sign_zero = fixed_point_integer_tile(
            bits_tile,
        )
        if verify:
            restored = reconstruct_bits_from_fixed(integers, min_scale, sign_bits)
            if not np.array_equal(restored, bits_tile):
                raise RuntimeError("fixed-point reconstruction mismatch")
    except (OverflowError, ValueError, RuntimeError) as error:
        return (
            {
                "current": current,
                "best_fixed_zstd": float("inf"),
                "best_fixed_entropy": float("inf"),
            },
            {
                "eligible": False,
                "reason": str(error),
            },
        )

    best_zstd = float("inf")
    best_entropy = float("inf")
    best_zstd_name = ""
    best_entropy_name = ""
    best_width = 0
    route_costs = {}
    for name, residual in residual_candidates(integers).items():
        encoded = zigzag_i64(residual)
        max_encoded = int(np.max(encoded, initial=np.uint64(0)))
        width = max(1, bit_length_u64(max_encoded))
        byte_count = max(1, (width + 7) // 8)
        entropy = metadata_bits + byte_entropy_cost(encoded, byte_count)
        zstd = metadata_bits + zstd_cost(encoded, byte_count, zstd_path, zstd_level)
        route_costs[f"{name}_entropy"] = entropy
        route_costs[f"{name}_zstd"] = zstd
        if entropy < best_entropy:
            best_entropy = entropy
            best_entropy_name = name
        if zstd < best_zstd:
            best_zstd = zstd
            best_zstd_name = name
            best_width = width

    if integers.shape[2] >= 3 and all_sign_zero:
        ref_channels = [1]
        if integers.shape[2] > 3:
            ref_channels.extend(range(3, integers.shape[2]))
        ref_cost = current_channel_cost(body_payload_tile, sign_payload_tile, ref_channels)
        rb_residual = np.stack(
            (
                integers[..., 0] - integers[..., 1],
                integers[..., 2] - integers[..., 1],
            ),
            axis=2,
        )
        encoded = zigzag_i64(rb_residual)
        max_encoded = int(np.max(encoded, initial=np.uint64(0)))
        width = max(1, bit_length_u64(max_encoded))
        byte_count = max(1, (width + 7) // 8)
        entropy = metadata_bits + ref_cost + byte_entropy_cost(encoded, byte_count)
        zstd = metadata_bits + ref_cost + zstd_cost(
            encoded,
            byte_count,
            zstd_path,
            zstd_level,
        )
        route_costs["green_ref_rb_entropy"] = entropy
        route_costs["green_ref_rb_zstd"] = zstd
        if entropy < best_entropy:
            best_entropy = entropy
            best_entropy_name = "green_ref_rb"
        if zstd < best_zstd:
            best_zstd = zstd
            best_zstd_name = "green_ref_rb"
            best_width = width

    return (
        {
            "current": current,
            "sign_cost": sign_cost,
            "body_cost": body_cost,
            "best_fixed_zstd": best_zstd,
            "best_fixed_entropy": best_entropy,
        },
        {
            "eligible": True,
            "min_scale": min_scale,
            "magnitude_width": magnitude_width,
            "all_sign_zero": all_sign_zero,
            "best_zstd": best_zstd_name,
            "best_entropy": best_entropy_name,
            "best_zstd_width": best_width,
            "route_costs": route_costs,
        },
    )


def summarize_image(
    path: Path,
    crop_size: int,
    tile_size: int,
    previous_bits: int,
    metadata_bits: float,
    zstd_path: str | None,
    zstd_level: int,
    verify: bool,
) -> dict:
    pixels = read_exr(path)
    if crop_size:
        pixels = pixels[:crop_size, :crop_size]
    pixels = np.ascontiguousarray(pixels, dtype=np.float32)
    bits = float_to_bits(pixels).copy()
    _modes, payloads, _carries = choose_group_payloads(
        bits,
        pixels,
        tile_size,
        previous_bits,
        "body",
    )

    totals = Counter()
    selected_zstd = Counter()
    selected_entropy = Counter()
    hybrid_zstd = Counter()
    hybrid_entropy = Counter()
    width_histogram = Counter()
    scale_histogram = Counter()
    tiles = []
    for y in range(0, bits.shape[0], tile_size):
        for x in range(0, bits.shape[1], tile_size):
            sl = np.s_[y:y + tile_size, x:x + tile_size]
            costs, details = summarize_tile(
                bits[sl],
                payloads["body"][sl],
                payloads["sign"][sl],
                metadata_bits,
                zstd_path,
                zstd_level,
                verify,
            )
            totals["current_bits"] += costs["current"]
            totals["best_fixed_zstd_bits"] += costs["best_fixed_zstd"]
            totals["best_fixed_entropy_bits"] += costs["best_fixed_entropy"]
            if costs["best_fixed_zstd"] < costs["current"]:
                totals["hybrid_zstd_bits"] += costs["best_fixed_zstd"]
                hybrid_zstd["fixed"] += 1
            else:
                totals["hybrid_zstd_bits"] += costs["current"]
                hybrid_zstd["current"] += 1
            if costs["best_fixed_entropy"] < costs["current"]:
                totals["hybrid_entropy_bits"] += costs["best_fixed_entropy"]
                hybrid_entropy["fixed"] += 1
            else:
                totals["hybrid_entropy_bits"] += costs["current"]
                hybrid_entropy["current"] += 1
            if details["eligible"]:
                selected_zstd[str(details["best_zstd"])] += 1
                selected_entropy[str(details["best_entropy"])] += 1
                width_histogram[str(details["magnitude_width"])] += 1
                scale_histogram[str(details["min_scale"])] += 1
            else:
                selected_zstd["ineligible"] += 1
                selected_entropy["ineligible"] += 1
            tiles.append({"y": y, "x": x, "costs": costs, "details": details})

    raw_bits = int(bits.size * 32)
    current = float(totals["current_bits"])
    fixed_zstd = float(totals["best_fixed_zstd_bits"])
    fixed_entropy = float(totals["best_fixed_entropy_bits"])
    hybrid_zstd_bits = float(totals["hybrid_zstd_bits"])
    hybrid_entropy_bits = float(totals["hybrid_entropy_bits"])
    return {
        "image": path.name,
        "shape": list(bits.shape),
        "crop_size": crop_size,
        "tile_size": tile_size,
        "previous_bits": previous_bits,
        "metadata_bits": metadata_bits,
        "zstd_level": zstd_level,
        "raw_bits": raw_bits,
        "current_bits": current,
        "best_fixed_zstd_bits": fixed_zstd,
        "best_fixed_entropy_bits": fixed_entropy,
        "hybrid_zstd_bits": hybrid_zstd_bits,
        "hybrid_entropy_bits": hybrid_entropy_bits,
        "current_ratio": raw_bits / current if current else 1.0,
        "best_fixed_zstd_ratio": raw_bits / fixed_zstd if fixed_zstd else 1.0,
        "best_fixed_entropy_ratio": raw_bits / fixed_entropy if fixed_entropy else 1.0,
        "hybrid_zstd_ratio": raw_bits / hybrid_zstd_bits if hybrid_zstd_bits else 1.0,
        "hybrid_entropy_ratio": (
            raw_bits / hybrid_entropy_bits if hybrid_entropy_bits else 1.0
        ),
        "zstd_gain": current / fixed_zstd if fixed_zstd else 1.0,
        "entropy_gain": current / fixed_entropy if fixed_entropy else 1.0,
        "hybrid_zstd_gain": current / hybrid_zstd_bits if hybrid_zstd_bits else 1.0,
        "hybrid_entropy_gain": (
            current / hybrid_entropy_bits if hybrid_entropy_bits else 1.0
        ),
        "selected_zstd": dict(selected_zstd),
        "selected_entropy": dict(selected_entropy),
        "hybrid_zstd": dict(hybrid_zstd),
        "hybrid_entropy": dict(hybrid_entropy),
        "width_histogram": dict(width_histogram),
        "scale_histogram": dict(scale_histogram),
        "tiles": tiles,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="sample_hilberts-mill-conference-room_2K.exr")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--previous-bits", type=int, default=4)
    parser.add_argument("--metadata-bits", type=float, default=24.0)
    parser.add_argument("--zstd", action="store_true")
    parser.add_argument("--zstd-level", type=int, default=3)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    zstd_path = shutil.which("zstd") if args.zstd else None
    if args.zstd and zstd_path is None:
        raise RuntimeError("zstd was requested but no zstd executable was found")
    rows = []
    for path in sorted(DATA_DIR.glob(args.glob)):
        row = summarize_image(
            path,
            args.crop_size,
            args.tile_size,
            args.previous_bits,
            args.metadata_bits,
            zstd_path,
            args.zstd_level,
            args.verify,
        )
        rows.append(row)
        print(
            f"{path.stem:43s} "
            f"current={row['current_ratio']:.3f}x "
            f"fixed_zstd={row['best_fixed_zstd_ratio']:.3f}x "
            f"zstd_gain={row['zstd_gain']:.4f} "
            f"hybrid_zstd={row['hybrid_zstd_ratio']:.3f}x "
            f"hybrid_gain={row['hybrid_zstd_gain']:.4f} "
            f"fixed_entropy={row['best_fixed_entropy_ratio']:.3f}x "
            f"entropy_gain={row['entropy_gain']:.4f} "
            f"zstd={row['selected_zstd']} hybrid={row['hybrid_zstd']} "
            f"width={row['width_histogram']}"
        )
        if args.limit and len(rows) >= args.limit:
            break

    if not args.no_save:
        safe_glob = args.glob.replace("*", "star").replace(".", "_")
        output = RESULTS_DIR / (
            f"block_fixed_point_route_{safe_glob}_tile{args.tile_size}"
            f"_prev{args.previous_bits}_crop{args.crop_size}.json"
        )
        output.write_text(json.dumps(rows, indent=2))
        print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
