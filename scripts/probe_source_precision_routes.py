"""Probe exact source-precision routes for float32 HDR tiles.

This extends the half16 route idea with a bf16/coarse-float lane.  The codec
still restores the original float32 bits exactly; a route is eligible only when
the tile's float32 bit pattern can be represented by the smaller source domain.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_structural_predictors import DATA_DIR, float_to_bits, read_exr  # noqa: E402
from probe_half16_route import (  # noqa: E402
    best_family_cost,
    current_tile_cost,
    half_bits,
    half_exact,
    half_to_ordered,
    predictors,
)
from structural_delta_grouped_roundtrip import choose_group_payloads  # noqa: E402

RESULTS_DIR = ROOT / "results"
U16_MASK = np.uint32(0xFFFF)


def ordered16_from_float_high16(bits: np.ndarray) -> np.ndarray:
    high = (bits >> np.uint32(16)) & U16_MASK
    sign = (high >> np.uint32(15)) != 0
    return np.where(sign, (~high) & U16_MASK, high ^ np.uint32(0x8000)).astype(np.uint32)


def bf16_exact(bits: np.ndarray) -> bool:
    return bool(np.all((bits & np.uint32(0xFFFF)) == 0))


def ordered16_route_cost(
    ordered: np.ndarray,
    route_header_bits: int,
    predictor_modes: tuple[str, ...] | None,
) -> tuple[float, str]:
    best_cost = float("inf")
    best_mode = ""
    payloads = predictors(ordered)
    modes = predictor_modes if predictor_modes is not None else tuple(payloads)
    for mode in modes:
        payload = payloads[mode]
        cost = best_family_cost(payload, range(15, -1, -1))
        if cost < best_cost:
            best_cost = cost
            best_mode = mode
    return route_header_bits + best_cost, best_mode


def best_source_route(
    pixels: np.ndarray,
    bits: np.ndarray,
    route_header_bits: int,
    predictor_modes: tuple[str, ...] | None,
) -> tuple[float, str, dict[str, str]]:
    best = float("inf")
    best_route = ""
    diagnostics: dict[str, str] = {}

    if half_exact(pixels):
        cost, mode = ordered16_route_cost(
            half_to_ordered(half_bits(pixels)),
            route_header_bits,
            predictor_modes,
        )
        diagnostics["half_mode"] = mode
        if cost < best:
            best = cost
            best_route = "half16"

    if bf16_exact(bits):
        cost, mode = ordered16_route_cost(
            ordered16_from_float_high16(bits),
            route_header_bits,
            predictor_modes,
        )
        diagnostics["bf16_mode"] = mode
        if cost < best:
            best = cost
            best_route = "bf16"

    return best, best_route, diagnostics


def summarize_image(
    path: Path,
    crop_size: int,
    tile_size: int,
    previous_bits: int,
    route_header_bits: int,
    predictor_modes: tuple[str, ...] | None,
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

    height, width, _channels = pixels.shape
    current_bits = 0.0
    routed_bits = 0.0
    eligible = Counter()
    selected = Counter()
    mode_histogram = Counter()
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            sl = np.s_[y:y + tile_size, x:x + tile_size]
            current = current_tile_cost(payloads["body"][sl], payloads["sign"][sl])
            current_bits += current
            best = current
            route_cost, route, diagnostics = best_source_route(
                pixels[sl],
                bits[sl],
                route_header_bits,
                predictor_modes,
            )
            if "half_mode" in diagnostics:
                eligible["half16"] += 1
            if "bf16_mode" in diagnostics:
                eligible["bf16"] += 1
            if route and route_cost < best:
                best = route_cost
                selected[route] += 1
                mode_key = "half_mode" if route == "half16" else "bf16_mode"
                mode_histogram[f"{route}:{diagnostics[mode_key]}"] += 1
            else:
                selected["gdx"] += 1
            routed_bits += best

    raw_bits = int(pixels.size * 32)
    return {
        "image": path.name,
        "shape": list(pixels.shape),
        "crop_size": crop_size,
        "tile_size": tile_size,
        "previous_bits": previous_bits,
        "route_header_bits": route_header_bits,
        "predictor_modes": list(predictor_modes) if predictor_modes is not None else "all",
        "raw_bits": raw_bits,
        "current_bits": current_bits,
        "routed_bits": routed_bits,
        "current_ratio": raw_bits / current_bits if current_bits else 1.0,
        "routed_ratio": raw_bits / routed_bits if routed_bits else 1.0,
        "gain": current_bits / routed_bits if routed_bits else 1.0,
        "eligible": dict(eligible),
        "selected": dict(selected),
        "mode_histogram": dict(mode_histogram),
    }


def geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="*.exr")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--previous-bits", type=int, default=4)
    parser.add_argument("--route-header-bits", type=int, default=8)
    parser.add_argument(
        "--predictor-modes",
        default="",
        help=(
            "Comma-separated half/bf16 predictor modes. Empty means all modes. "
            "Useful for lightweight smoke tests."
        ),
    )
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def validate_predictor_modes(predictor_modes: tuple[str, ...] | None) -> None:
    if predictor_modes is None:
        return
    sample = np.zeros((1, 1, 1), dtype=np.uint32)
    valid = set(predictors(sample))
    unknown = sorted(set(predictor_modes) - valid)
    if unknown:
        raise ValueError(
            f"unknown predictor mode(s): {', '.join(unknown)}; "
            f"valid modes are: {', '.join(sorted(valid))}"
        )


def main() -> int:
    args = parse_args()
    paths = sorted(DATA_DIR.glob(args.glob))
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        raise RuntimeError(f"no images matched {DATA_DIR / args.glob}")

    rows = []
    predictor_modes = (
        tuple(part for part in args.predictor_modes.split(",") if part)
        if args.predictor_modes
        else None
    )
    validate_predictor_modes(predictor_modes)
    print(
        f"{'image':43s} {'current':>9s} {'routed':>9s} "
        f"{'gain':>8s} eligible selected"
    )
    print("-" * 112)
    for path in paths:
        row = summarize_image(
            path,
            args.crop_size,
            args.tile_size,
            args.previous_bits,
            args.route_header_bits,
            predictor_modes,
        )
        rows.append(row)
        print(
            f"{path.stem:43s} "
            f"{row['current_ratio']:8.3f}x "
            f"{row['routed_ratio']:8.3f}x "
            f"{row['gain']:7.4f} {row['eligible']} {row['selected']}"
        )

    print("-" * 112)
    print(
        f"geomean current={geomean([row['current_ratio'] for row in rows]):.3f}x "
        f"routed={geomean([row['routed_ratio'] for row in rows]):.3f}x"
    )

    if not args.no_save:
        safe_glob = args.glob.replace("*", "star").replace(".", "_")
        output = RESULTS_DIR / (
            f"source_precision_routes_{safe_glob}_tile{args.tile_size}"
            f"_crop{args.crop_size}.json"
        )
        output.write_text(json.dumps(rows, indent=2))
        print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
