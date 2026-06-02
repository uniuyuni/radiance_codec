"""Audit how far current GDX3 is from an exact-lossless ratio target.

The goal is not to be discouraging; it is to make the target concrete. For each
image this reports:

* actual current C++ grouped-delta size;
* the exact target bit budget;
* how much must still disappear;
* current grouped-delta model cost by rough bit region.

The per-region model cost is estimated with the Python grouped-delta payload and
the current context-family list.  It is close enough to show which regions block
16x, while the actual encoded size comes from the C++ codec.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import OpenImageIO as oiio

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "codec" / "python"))

import radiance_codec  # noqa: E402
from analyze_structural_predictors import DATA_DIR, float_to_bits  # noqa: E402
from audit_source_precision import classify_tile  # noqa: E402
from probe_grouped_tail_mlp import CONTEXT_FAMILIES, context_family_cost  # noqa: E402
from structural_delta_grouped_roundtrip import choose_group_payloads  # noqa: E402

RESULTS_DIR = ROOT / "results"


def read_exr(path: Path) -> np.ndarray:
    image_input = oiio.ImageInput.open(str(path))
    if image_input is None:
        raise RuntimeError(f"can't open {path}: {oiio.geterror()}")
    spec = image_input.spec()
    pixels = image_input.read_image(format=oiio.FLOAT)
    image_input.close()
    if pixels is None:
        raise RuntimeError(f"can't read {path}: {oiio.geterror()}")
    return np.asarray(pixels, dtype=np.float32).reshape(
        spec.height,
        spec.width,
        spec.nchannels,
    )


def family_candidates(group_name: str, bit: int, effort: int) -> tuple[str, ...]:
    if effort >= 10:
        return CONTEXT_FAMILIES
    if group_name == "sign":
        return ("base",)
    return CONTEXT_FAMILIES


def region_for(group_name: str, bit: int) -> str:
    if group_name == "sign":
        return "sign"
    if bit <= 14:
        return "body_low_0_14"
    if bit <= 20:
        return "body_mid_15_20"
    return "body_high_21_30"


def best_family_cost(payload: np.ndarray, bit: int, group_name: str, effort: int) -> float:
    return min(
        context_family_cost(payload, bit, family)
        for family in family_candidates(group_name, bit, effort)
    )


def estimate_regions(
    pixels: np.ndarray,
    tile_size: int,
    previous_bits: int,
    effort: int,
) -> dict[str, float]:
    bits = float_to_bits(pixels).copy()
    _modes, payloads, _carries = choose_group_payloads(
        bits,
        pixels,
        tile_size,
        previous_bits,
        "body",
    )
    height, width, _channels = bits.shape
    totals = Counter()
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            sl = np.s_[y:y + tile_size, x:x + tile_size]
            body = payloads["body"][sl]
            for bit in range(30, -1, -1):
                totals[region_for("body", bit)] += best_family_cost(
                    body,
                    bit,
                    "body",
                    effort,
                )
            sign = payloads["sign"][sl]
            totals["sign"] += best_family_cost(sign, 31, "sign", effort)
    return {key: float(value) for key, value in totals.items()}


def geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def summarize_image(path: Path, crop_size: int, effort: int, target_ratio: float) -> dict:
    pixels = read_exr(path)
    if crop_size:
        pixels = pixels[:crop_size, :crop_size]
    pixels = np.ascontiguousarray(pixels, dtype=np.float32)
    raw_bits = pixels.nbytes * 8
    encoded = radiance_codec.encode(
        pixels,
        stages=radiance_codec.Stage.GROUPED_DELTA,
        effort=effort,
    )
    encoded_bits = len(encoded) * 8
    target_bits = raw_bits / target_ratio
    target8_bits = raw_bits / 8.0
    regions = estimate_regions(pixels, 128, 4, effort)
    source = classify_tile(pixels, float_to_bits(pixels).copy())
    return {
        "image": path.name,
        "shape": list(pixels.shape),
        "raw_bits": raw_bits,
        "encoded_bits": encoded_bits,
        "ratio": raw_bits / encoded_bits,
        "target_ratio": target_ratio,
        "target8_bits": target8_bits,
        "target_bits": target_bits,
        "bits_to_remove_for_8x": max(0.0, encoded_bits - target8_bits),
        "bits_to_remove_for_target": max(0.0, encoded_bits - target_bits),
        "current_over_target_budget": encoded_bits / target_bits,
        "region_bits": regions,
        "region_bits_per_sample": {
            key: value / float(pixels.size)
            for key, value in regions.items()
        },
        "source_precision": source,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="*.exr")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--crop-size", type=int, default=0)
    parser.add_argument("--effort", type=int, default=9)
    parser.add_argument("--target-ratio", type=float, default=16.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = []
    print(
        f"{'image':43s} {'ratio':>8s} {f'x{args.target_ratio:g}_gap':>10s} "
        f"{'low':>8s} {'mid':>8s} {'high':>8s} {'class':>22s}"
    )
    print("-" * 116)
    for path in sorted(DATA_DIR.glob(args.glob)):
        row = summarize_image(path, args.crop_size, args.effort, args.target_ratio)
        rows.append(row)
        r = row["region_bits_per_sample"]
        source_class = row["source_precision"]["class"]
        print(
            f"{path.stem:43s} "
            f"{row['ratio']:7.2f}x "
            f"{row['bits_to_remove_for_target'] / 8 / 1024:9.1f}K "
            f"{r.get('body_low_0_14', 0.0):7.2f} "
            f"{r.get('body_mid_15_20', 0.0):7.2f} "
            f"{r.get('body_high_21_30', 0.0):7.2f} "
            f"{source_class:>22s}"
        )
        if args.limit and len(rows) >= args.limit:
            break
    if rows:
        print("-" * 116)
        print(f"geomean ratio: {geomean([row['ratio'] for row in rows]):.3f}x")
        print(
            f"total bits to remove for {args.target_ratio:g}x: "
            f"{sum(row['bits_to_remove_for_target'] for row in rows) / 8 / 1024:.1f} KiB"
        )
    safe_glob = args.glob.replace("*", "star").replace(".", "_")
    output = RESULTS_DIR / (
        f"budget_x{args.target_ratio:g}_{safe_glob}_effort{args.effort}"
        f"_crop{args.crop_size}.json"
    )
    output.write_text(json.dumps(rows, indent=2))
    print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
