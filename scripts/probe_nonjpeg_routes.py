"""Probe non-JPEG lossless routes on top of current grouped-delta payloads.

This is a routing/upper-bound probe, not a new bitstream. It keeps the current
GDX2 grouped ordered-delta payload generation and asks whether non-JPEG
scientific-compression ideas would improve the entropy stage:

* raw fallback for nearly random planes;
* ndzip-like bit-matrix zero-word elimination;
* bit-matrix constant-word elimination;
* typed byte-stream entropy as a diagnostic lower bound.

If a route cannot beat the current GDX2 adaptive context estimate on the already
chosen payload, it is unlikely to be worth moving to C++ before a deeper
candidate-mode search.
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
from audit_float_tail_structure import entropy_bits  # noqa: E402
from evaluate_structural_delta_context import (  # noqa: E402
    bits_to_ordered,
    delta_predictors,
    ordered_xor_predictors,
)
from evaluate_structural_delta_grouped import (  # noqa: E402
    GROUP_PRESETS,
    field_mask,
    filter_group_predictors,
    group_bit_indices,
)
from evaluate_structural_delta_sidecarry import field_layout  # noqa: E402
from probe_grouped_tail_mlp import CONTEXT_FAMILIES, context_family_cost, kt_cost  # noqa: E402
from structural_delta_grouped_roundtrip import choose_group_payloads, mode_payload  # noqa: E402

RESULTS_DIR = ROOT / "results"


def dirichlet_multinomial_cost(counts: tuple[int, ...], alpha: float = 0.5) -> float:
    total = sum(counts)
    if total == 0:
        return 0.0
    log_probability = math.lgamma(len(counts) * alpha) - math.lgamma(
        total + len(counts) * alpha
    )
    for count in counts:
        log_probability += math.lgamma(count + alpha) - math.lgamma(alpha)
    return -log_probability / math.log(2.0)


def gdx2_family_cost(payload: np.ndarray, bits: tuple[int, ...]) -> float:
    return sum(
        min(context_family_cost(payload, bit, family) for family in CONTEXT_FAMILIES)
        for bit in bits
    )


def raw_bit_cost(payload: np.ndarray, bits: tuple[int, ...]) -> float:
    return float(payload.size * len(bits))


def bit_chunks(payload: np.ndarray, bit: int, word_bits: int) -> np.ndarray:
    plane = ((payload.reshape(-1) >> np.uint32(bit)) & np.uint32(1)).astype(np.uint8)
    if plane.size == 0:
        return np.zeros((0, word_bits), dtype=np.uint8)
    pad = (-plane.size) % word_bits
    if pad:
        plane = np.pad(plane, (0, pad), constant_values=0)
    return plane.reshape(-1, word_bits)


def bitmatrix_zero_word_cost(
    payload: np.ndarray,
    bits: tuple[int, ...],
    word_bits: int,
) -> float:
    """One adaptive nonzero mask per bitplane, raw words for nonzero chunks."""

    cost = 0.0
    for bit in bits:
        chunks = bit_chunks(payload, bit, word_bits)
        if chunks.size == 0:
            continue
        ones = chunks.sum(axis=1)
        nonzero = int(np.count_nonzero(ones))
        total = int(ones.size)
        cost += kt_cost(nonzero, total)
        cost += nonzero * word_bits
    return cost


def bitmatrix_constant_word_cost(
    payload: np.ndarray,
    bits: tuple[int, ...],
    word_bits: int,
) -> float:
    """Adaptive {all-zero, all-one, mixed} mask; raw words only for mixed chunks."""

    cost = 0.0
    for bit in bits:
        chunks = bit_chunks(payload, bit, word_bits)
        if chunks.size == 0:
            continue
        ones = chunks.sum(axis=1)
        zero_count = int(np.count_nonzero(ones == 0))
        one_count = int(np.count_nonzero(ones == word_bits))
        mixed_count = int(ones.size - zero_count - one_count)
        cost += dirichlet_multinomial_cost((zero_count, one_count, mixed_count))
        cost += mixed_count * word_bits
    return cost


def bitmatrix_fixed_mask_cost(
    payload: np.ndarray,
    bits: tuple[int, ...],
    word_bits: int,
) -> float:
    """Simpler ndzip-like fixed one-bit nonzero mask; useful as speed baseline."""

    cost = 0.0
    for bit in bits:
        chunks = bit_chunks(payload, bit, word_bits)
        if chunks.size == 0:
            continue
        ones = chunks.sum(axis=1)
        nonzero = int(np.count_nonzero(ones))
        cost += ones.size
        cost += nonzero * word_bits
    return float(cost)


def typed_byte_entropy_cost(payload: np.ndarray, bits: tuple[int, ...]) -> float:
    """Optimistic byte-position entropy lower bound for diagnostics only."""

    if not bits:
        return 0.0
    mask = np.uint32(0)
    for bit in bits:
        mask |= np.uint32(1) << np.uint32(bit)
    values = np.ascontiguousarray(payload.reshape(-1) & mask, dtype=np.uint32)
    byte_view = values.view(np.uint8).reshape(-1, 4)
    return sum(entropy_bits(byte_view[:, index], 8) * values.size for index in range(4))


def typed_byteplane_bytes(payload: np.ndarray, bits: tuple[int, ...]) -> bytes:
    if not bits:
        return b""
    mask = np.uint32(0)
    for bit in bits:
        mask |= np.uint32(1) << np.uint32(bit)
    low_byte = min(bits) // 8
    high_byte = max(bits) // 8
    values = np.ascontiguousarray(payload.reshape(-1) & mask, dtype=np.uint32)
    byte_view = values.view(np.uint8).reshape(-1, 4)
    planes = [
        np.ascontiguousarray(byte_view[:, index])
        for index in range(low_byte, high_byte + 1)
    ]
    return b"".join(plane.tobytes() for plane in planes)


def zstd_byteplane_cost(
    payload: np.ndarray,
    bits: tuple[int, ...],
    zstd_path: str | None,
    zstd_level: int,
) -> float:
    if zstd_path is None:
        return float("inf")
    data = typed_byteplane_bytes(payload, bits)
    if not data:
        return 0.0
    result = subprocess.run(
        [zstd_path, "-q", f"-{zstd_level}", "-c"],
        input=data,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return float(len(result.stdout) * 8)


def zigzag_group_payload(payload: np.ndarray, bits: tuple[int, ...]) -> np.ndarray | None:
    """Bijectively map a contiguous modular residual group to centered zigzag."""

    low_bit, width, mask = field_layout(bits)
    if width <= 1:
        return None
    expected = tuple(range(low_bit + width - 1, low_bit - 1, -1))
    if bits != expected:
        return None
    digit = (payload >> np.uint32(low_bit)) & mask
    sign = digit >> np.uint32(width - 1)
    zigzag = ((digit << np.uint32(1)) ^ (np.uint32(0) - sign)) & mask
    return (zigzag << np.uint32(low_bit)).astype(np.uint32)


def common_low_zero_bits(values: np.ndarray, max_bits: int) -> int:
    if values.size == 0:
        return 0
    best = 0
    flat = values.reshape(-1)
    for width in range(1, max_bits + 1):
        mask = np.uint32((1 << width) - 1)
        if bool(np.all((flat & mask) == 0)):
            best = width
        else:
            break
    return best


def route_costs(
    payload: np.ndarray,
    bits: tuple[int, ...],
    word_bits: int,
    route_header_bits: int,
    zstd_path: str | None,
    zstd_level: int,
) -> tuple[dict[str, float], str, float]:
    current = gdx2_family_cost(payload, bits)
    costs = {
        "gdx2": current,
        "raw": route_header_bits + raw_bit_cost(payload, bits),
        "bitmatrix_zero_kt": route_header_bits
        + bitmatrix_zero_word_cost(payload, bits, word_bits),
        "bitmatrix_const_kt": route_header_bits
        + bitmatrix_constant_word_cost(payload, bits, word_bits),
        "bitmatrix_fixed_mask": route_header_bits
        + bitmatrix_fixed_mask_cost(payload, bits, word_bits),
        "typed_byte_entropy_lower": typed_byte_entropy_cost(payload, bits),
    }
    if zstd_path is not None:
        costs["typed_byteplane_zstd"] = route_header_bits + zstd_byteplane_cost(
            payload,
            bits,
            zstd_path,
            zstd_level,
        )
    zigzag = zigzag_group_payload(payload, bits)
    if zigzag is not None:
        costs["zigzag_bitmatrix_zero_kt"] = route_header_bits + bitmatrix_zero_word_cost(
            zigzag,
            bits,
            word_bits,
        )
        costs["zigzag_bitmatrix_const_kt"] = (
            route_header_bits
            + bitmatrix_constant_word_cost(
                zigzag,
                bits,
                word_bits,
            )
        )
        costs["zigzag_typed_byte_entropy_lower"] = typed_byte_entropy_cost(
            zigzag,
            bits,
        )
        if zstd_path is not None:
            costs["zigzag_typed_byteplane_zstd"] = (
                route_header_bits
                + zstd_byteplane_cost(
                    zigzag,
                    bits,
                    zstd_path,
                    zstd_level,
                )
            )
    candidate_costs = {
        key: value
        for key, value in costs.items()
        if not key.endswith("_entropy_lower")
    }
    route, best = min(candidate_costs.items(), key=lambda item: item[1])
    return costs, route, best


def candidate_modes(group_name: str, xor_tiles: dict[str, np.ndarray], delta_tiles: dict[str, np.ndarray]) -> list[str]:
    group_xor_tiles, group_delta_tiles = filter_group_predictors(
        group_name,
        xor_tiles,
        delta_tiles,
    )
    modes = ["raw"]
    modes.extend(f"xor_{name}" for name in group_xor_tiles)
    for name in group_delta_tiles:
        modes.append(name)
        modes.append(f"local_{name}")
    return modes


def best_candidate_route_costs(
    ordered_tile: np.ndarray,
    xor_tiles: dict[str, np.ndarray],
    delta_tiles: dict[str, np.ndarray],
    group_name: str,
    fields: tuple[str, ...],
    word_bits: int,
    route_header_bits: int,
    zstd_path: str | None,
    zstd_level: int,
) -> tuple[float, float, str, str, dict[str, float]]:
    bit_indices = group_bit_indices(fields)
    group_xor_tiles, group_delta_tiles = filter_group_predictors(
        group_name,
        xor_tiles,
        delta_tiles,
    )
    mask = field_mask(bit_indices)
    current_best = float("inf")
    hybrid_best = float("inf")
    current_mode = ""
    hybrid_route = ""
    best_costs: dict[str, float] = {}

    for mode in candidate_modes(group_name, xor_tiles, delta_tiles):
        payload, carry = mode_payload(
            mode,
            ordered_tile,
            group_xor_tiles,
            group_delta_tiles,
            bit_indices,
        )
        if carry is not None:
            # The current body/sign preset does not need carry side bits.  For
            # other presets, leave a visible penalty instead of silently
            # producing an optimistic estimate.
            carry_cost = float(carry.size)
        else:
            carry_cost = 0.0
        payload = payload & mask
        costs, route, best = route_costs(
            payload,
            bit_indices,
            word_bits,
            route_header_bits,
            zstd_path,
            zstd_level,
        )
        current = costs["gdx2"] + carry_cost
        hybrid = best + carry_cost
        if current < current_best:
            current_best = current
            current_mode = mode
        if hybrid < hybrid_best:
            hybrid_best = hybrid
            hybrid_route = f"{mode}/{route}"
            best_costs = costs
    return current_best, hybrid_best, current_mode, hybrid_route, best_costs


def summarize_image(
    path: Path,
    crop_size: int,
    tile_size: int,
    previous_bits: int,
    preset: str,
    word_bits: int,
    route_header_bits: int,
    search_candidates: bool,
    zstd_path: str | None,
    zstd_level: int,
) -> dict:
    pixels = read_exr(path)
    if crop_size:
        pixels = pixels[:crop_size, :crop_size]
    pixels = np.ascontiguousarray(pixels, dtype=np.float32)
    bits = float_to_bits(pixels).copy()
    ordered = bits_to_ordered(bits)
    xor_predictor_bits = ordered_xor_predictors(bits, pixels, causal=True)
    delta_prediction_bits = delta_predictors(bits, pixels, causal=True)
    raw_mantissa = bits & np.uint32(0x7FFFFF)
    _modes, payloads, _carries = choose_group_payloads(
        bits,
        pixels,
        tile_size,
        previous_bits,
        preset,
    )

    height, width, _channels = bits.shape
    totals = Counter()
    route_histogram = Counter()
    zero_histogram = Counter()
    tiles = []

    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            sl = np.s_[y:y + tile_size, x:x + tile_size]
            raw_low_zero = common_low_zero_bits(raw_mantissa[sl], 23)
            zero_histogram[str(raw_low_zero)] += 1
            tile_row = {
                "y": y,
                "x": x,
                "raw_mantissa_low_zero_bits": raw_low_zero,
                "groups": {},
            }
            for group_name, fields in GROUP_PRESETS[preset]:
                group_bits = group_bit_indices(fields)
                if search_candidates:
                    current, best, current_mode, route, costs = best_candidate_route_costs(
                        ordered[sl],
                        {name: value[sl] for name, value in xor_predictor_bits.items()},
                        {name: value[sl] for name, value in delta_prediction_bits.items()},
                        group_name,
                        fields,
                        word_bits,
                        route_header_bits,
                        zstd_path,
                        zstd_level,
                    )
                else:
                    payload_tile = payloads[group_name][sl]
                    costs, route, best = route_costs(
                        payload_tile,
                        group_bits,
                        word_bits,
                        route_header_bits,
                        zstd_path,
                        zstd_level,
                    )
                    current = costs["gdx2"]
                    current_mode = "chosen"
                totals["current_bits"] += current
                totals["hybrid_bits"] += best
                totals["typed_byte_entropy_lower"] += costs["typed_byte_entropy_lower"]
                for key, value in costs.items():
                    totals[f"route_{key}"] += value
                route_key = f"{group_name}:{route}"
                route_histogram[route_key] += 1
                tile_row["groups"][group_name] = {
                    "route": route,
                    "current_mode": current_mode,
                    "gain": current / best if best else 1.0,
                    "costs": costs,
                }
            tiles.append(tile_row)

    raw_bits = int(bits.size * 32)
    current_bits = float(totals["current_bits"])
    hybrid_bits = float(totals["hybrid_bits"])
    lower_bits = float(totals["typed_byte_entropy_lower"])
    return {
        "image": path.name,
        "shape": list(bits.shape),
        "crop_size": crop_size,
        "tile_size": tile_size,
        "previous_bits": previous_bits,
        "preset": preset,
        "word_bits": word_bits,
        "route_header_bits": route_header_bits,
        "search_candidates": search_candidates,
        "raw_bits": raw_bits,
        "current_bits": current_bits,
        "hybrid_bits": hybrid_bits,
        "typed_byte_entropy_lower_bits": lower_bits,
        "current_ratio": raw_bits / current_bits if current_bits else 1.0,
        "hybrid_ratio": raw_bits / hybrid_bits if hybrid_bits else 1.0,
        "hybrid_gain": current_bits / hybrid_bits if hybrid_bits else 1.0,
        "typed_byte_lower_gain": current_bits / lower_bits if lower_bits else 1.0,
        "route_histogram": dict(route_histogram),
        "raw_mantissa_low_zero_histogram": dict(zero_histogram),
        "totals": dict(totals),
        "tiles": tiles,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="*.exr")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--crop-size", type=int, default=0)
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--previous-bits", type=int, default=4)
    parser.add_argument("--preset", choices=sorted(GROUP_PRESETS), default="body")
    parser.add_argument("--word-bits", type=int, default=64)
    parser.add_argument("--route-header-bits", type=int, default=4)
    parser.add_argument("--search-candidates", action="store_true")
    parser.add_argument("--zstd", action="store_true")
    parser.add_argument("--zstd-level", type=int, default=3)
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
            args.preset,
            args.word_bits,
            args.route_header_bits,
            args.search_candidates,
            zstd_path,
            args.zstd_level,
        )
        rows.append(row)
        print(
            f"{path.stem:43s} "
            f"current={row['current_ratio']:.3f}x "
            f"hybrid={row['hybrid_ratio']:.3f}x "
            f"gain={row['hybrid_gain']:.4f} "
            f"routes={row['route_histogram']}"
        )
        if args.limit and len(rows) >= args.limit:
            break
    safe_glob = args.glob.replace("*", "star").replace(".", "_")
    output = RESULTS_DIR / (
        f"nonjpeg_routes_{safe_glob}_tile{args.tile_size}_prev{args.previous_bits}"
        f"_word{args.word_bits}_crop{args.crop_size}.json"
    )
    output.write_text(json.dumps(rows, indent=2))
    print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
