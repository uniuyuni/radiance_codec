"""Audit float32 HDR entropy by field, spatial hierarchy, and channel lifting."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import OpenImageIO as oiio

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ml"))

from hdr_hierarchy import (  # noqa: E402
    FIELD_GROUPS,
    float_to_bits,
    lift_channels,
    nearest_context_residuals,
)

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
LIFTING_MODES = ["none", "green_delta", "green_xor"]


def read_exr(path: Path) -> np.ndarray:
    image_input = oiio.ImageInput.open(str(path))
    if image_input is None:
        raise RuntimeError(f"can't open {path}: {oiio.geterror()}")
    spec = image_input.spec()
    pixels = image_input.read_image(format=oiio.FLOAT)
    image_input.close()
    return np.asarray(pixels, dtype=np.float32).reshape(
        spec.height, spec.width, spec.nchannels
    )


def binary_entropy(probability: float) -> float:
    if probability <= 0.0 or probability >= 1.0:
        return 0.0
    return (
        -probability * math.log2(probability)
        - (1.0 - probability) * math.log2(1.0 - probability)
    )


def summarize_planes(bits: np.ndarray) -> dict:
    flat = bits.reshape(-1)
    plane_entropy = []
    for bit_index in range(31, -1, -1):
        probability = float(np.mean((flat >> bit_index) & 1))
        plane_entropy.append(binary_entropy(probability))
    by_bit = {
        str(bit_index): plane_entropy[31 - bit_index]
        for bit_index in range(31, -1, -1)
    }
    fields = {
        field: sum(by_bit[str(bit_index)] for bit_index in bit_indices)
        for field, bit_indices in FIELD_GROUPS.items()
    }
    total = sum(plane_entropy)
    return {
        "bits_per_value": total,
        "ratio_vs_float32": 32.0 / total if total > 0 else float("inf"),
        "field_bits_per_value": fields,
        "plane_bits_per_value_msb_to_lsb": plane_entropy,
    }


def analyze(path: Path, max_step: int) -> dict:
    pixels = read_exr(path)
    raw_bits = float_to_bits(pixels)
    hierarchy_bits, passes = nearest_context_residuals(raw_bits, max_step)
    representations = {}
    for prefix, values in [("raw", raw_bits), ("hier_nearest", hierarchy_bits)]:
        for mode in LIFTING_MODES:
            name = f"{prefix}_{mode}"
            representations[name] = summarize_planes(lift_channels(values, mode))

    per_pass = {}
    for hierarchy_pass in passes:
        values = hierarchy_bits[hierarchy_pass.mask]
        per_pass[hierarchy_pass.name] = {
            "pixels": int(np.count_nonzero(hierarchy_pass.mask)),
            "none": summarize_planes(values),
        }
        if raw_bits.shape[2] >= 3:
            pass_image = hierarchy_bits[hierarchy_pass.mask].reshape(
                -1, 1, raw_bits.shape[2]
            )
            per_pass[hierarchy_pass.name]["green_xor"] = summarize_planes(
                lift_channels(pass_image, "green_xor")
            )

    best_name = min(
        representations,
        key=lambda name: representations[name]["bits_per_value"],
    )
    return {
        "image": path.name,
        "shape": list(pixels.shape),
        "max_step": max_step,
        "representations": representations,
        "per_pass": per_pass,
        "best_representation": best_name,
        "best_bits_per_value": representations[best_name]["bits_per_value"],
        "best_ratio_vs_float32": representations[best_name]["ratio_vs_float32"],
    }


def geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="ph_*.exr")
    parser.add_argument("--max-step", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = sorted(DATA_DIR.glob(args.glob))
    if not paths:
        raise RuntimeError(f"no images matched {DATA_DIR / args.glob}")
    RESULTS_DIR.mkdir(exist_ok=True)
    results = [analyze(path, args.max_step) for path in paths]

    print(
        f"{'image':43s} {'raw':>7s} {'hier':>7s} {'best':>22s} {'ratio':>8s}"
    )
    print("-" * 94)
    for result in results:
        representations = result["representations"]
        raw = representations["raw_none"]["ratio_vs_float32"]
        hierarchy = representations["hier_nearest_none"]["ratio_vs_float32"]
        print(
            f"{Path(result['image']).stem:43s} {raw:6.2f}x {hierarchy:6.2f}x "
            f"{result['best_representation']:>22s} "
            f"{result['best_ratio_vs_float32']:7.2f}x"
        )
    best_ratios = [result["best_ratio_vs_float32"] for result in results]
    print("-" * 94)
    print(f"{'geomean best independent-plane bound':68s} {geomean(best_ratios):7.2f}x")
    print(
        "\nThese are order-0 independent bit-plane bounds, not final codec "
        "ratios. A learned conditional model may improve on them."
    )

    output = RESULTS_DIR / "v2_float_field_audit.json"
    output.write_text(json.dumps(results, indent=2))
    print(f"Saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
