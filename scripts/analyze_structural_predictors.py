"""Quick audit for PIC-like structural predictors on float32 HDR images.

This is a feasibility probe, not a codec. It estimates independent bit-plane
entropy after several reversible predictors. The most important experiment is
tile-wise, field-wise predictor selection:

  sign | exponent | mantissa_hi | mantissa_mid | mantissa_lo

Each tile and field group may choose a different predictor. This mimics a
structure-aware codec that stores a cheap mode map and then entropy-codes exact
corrections.
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
sys.path.insert(0, str(ROOT / "ml"))

from hdr_hierarchy import FIELD_GROUPS, float_to_bits  # noqa: E402

DATA_DIR = ROOT / "data"
RESULTS_JSON = ROOT / "results" / "results.json"
OUTPUT_JSON = ROOT / "results" / "structural_predictor_audit.json"


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
        spec.height, spec.width, spec.nchannels
    )


def binary_entropy(probability: float) -> float:
    if probability <= 0.0 or probability >= 1.0:
        return 0.0
    return (
        -probability * math.log2(probability)
        - (1.0 - probability) * math.log2(1.0 - probability)
    )


def bit_entropy(bits: np.ndarray, bit_indices: tuple[int, ...]) -> float:
    flat = bits.reshape(-1)
    total = 0.0
    for bit_index in bit_indices:
        probability = float(np.mean((flat >> bit_index) & 1))
        total += binary_entropy(probability)
    return total


def field_cost(bits: np.ndarray) -> dict[str, float]:
    return {
        name: bit_entropy(bits, indices)
        for name, indices in FIELD_GROUPS.items()
    }


def shifted(bits: np.ndarray, dy: int, dx: int) -> np.ndarray:
    out = np.zeros_like(bits)
    height, width, _ = bits.shape
    src_y0 = max(0, -dy)
    src_y1 = min(height, height - dy)
    src_x0 = max(0, -dx)
    src_x1 = min(width, width - dx)
    dst_y0 = src_y0 + dy
    dst_y1 = src_y1 + dy
    dst_x0 = src_x0 + dx
    dst_x1 = src_x1 + dx
    out[dst_y0:dst_y1, dst_x0:dst_x1] = bits[src_y0:src_y1, src_x0:src_x1]
    return out


def float_med_prediction(pixels: np.ndarray) -> np.ndarray:
    height, width, channels = pixels.shape
    pred = np.zeros_like(pixels)
    for y in range(height):
        if y == 0:
            pred[y, 1:] = pixels[y, :-1]
            continue
        pred[y, 0] = pixels[y - 1, 0]
        w = pixels[y, :-1]
        n = pixels[y - 1, 1:]
        nw = pixels[y - 1, :-1]
        mx = np.maximum(n, w)
        mn = np.minimum(n, w)
        planar = n + w - nw
        row_pred = np.where(nw >= mx, mn, np.where(nw <= mn, mx, planar))
        pred[y, 1:] = row_pred
    return float_to_bits(pred)


def float_planar_prediction(pixels: np.ndarray) -> np.ndarray:
    pred = np.zeros_like(pixels)
    pred[0, 1:] = pixels[0, :-1]
    pred[1:, 0] = pixels[:-1, 0]
    pred[1:, 1:] = pixels[1:, :-1] + pixels[:-1, 1:] - pixels[:-1, :-1]
    return float_to_bits(pred.astype(np.float32, copy=False))


def channel_green_prediction(bits: np.ndarray) -> np.ndarray:
    pred = np.zeros_like(bits)
    if bits.shape[2] >= 3:
        pred[..., 0] = bits[..., 1]
        pred[..., 2] = bits[..., 1]
    return pred


def channel_previous_prediction(bits: np.ndarray) -> np.ndarray:
    pred = np.zeros_like(bits)
    if bits.shape[2] >= 2:
        pred[..., 1:] = bits[..., :-1]
    return pred


def bit_xor_planar_prediction(bits: np.ndarray) -> np.ndarray:
    west = shifted(bits, 0, -1)
    north = shifted(bits, -1, 0)
    northwest = shifted(bits, -1, -1)
    return west ^ north ^ northwest


def bit_majority_prediction(bits: np.ndarray) -> np.ndarray:
    west = shifted(bits, 0, -1)
    north = shifted(bits, -1, 0)
    northwest = shifted(bits, -1, -1)
    return (west & north) | (west & northwest) | (north & northwest)


def predictors(pixels: np.ndarray) -> dict[str, np.ndarray]:
    bits = float_to_bits(pixels)
    return {
        "zero": np.zeros_like(bits),
        "west": shifted(bits, 0, -1),
        "north": shifted(bits, -1, 0),
        "northwest": shifted(bits, -1, -1),
        "northeast": shifted(bits, -1, 1),
        "channel_green": channel_green_prediction(bits),
        "channel_previous": channel_previous_prediction(bits),
        "bit_xor_planar": bit_xor_planar_prediction(bits),
        "bit_majority": bit_majority_prediction(bits),
        "float_med": float_med_prediction(pixels),
        "float_planar": float_planar_prediction(pixels),
    }


def global_best_by_field(
    bits: np.ndarray,
    predictor_bits: dict[str, np.ndarray],
) -> tuple[float, dict[str, str], dict[str, float]]:
    choices = {}
    costs = {}
    for field, indices in FIELD_GROUPS.items():
        best_name = ""
        best_cost = float("inf")
        for name, pred in predictor_bits.items():
            residual = bits ^ pred
            cost = bit_entropy(residual, indices)
            if cost < best_cost:
                best_name = name
                best_cost = cost
        choices[field] = best_name
        costs[field] = best_cost
    return sum(costs.values()), choices, costs


def tile_best_by_field(
    bits: np.ndarray,
    predictor_bits: dict[str, np.ndarray],
    tile_size: int,
) -> tuple[float, dict[str, int]]:
    height, width, channels = bits.shape
    total_bits = 0.0
    mode_histogram: dict[str, int] = {name: 0 for name in predictor_bits}
    mode_bits = math.ceil(math.log2(len(predictor_bits)))
    n_values = 0
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            sl = np.s_[y:y + tile_size, x:x + tile_size]
            tile_values = bits[sl].size
            n_values += tile_values
            for indices in FIELD_GROUPS.values():
                best_name = ""
                best_cost = float("inf")
                for name, pred in predictor_bits.items():
                    residual = bits[sl] ^ pred[sl]
                    cost = bit_entropy(residual, indices) * tile_values
                    if cost < best_cost:
                        best_name = name
                        best_cost = cost
                total_bits += best_cost + mode_bits
                mode_histogram[best_name] += 1
    return total_bits / n_values, mode_histogram


def ratio_from_bits_per_value(bits_per_value: float) -> float:
    return 32.0 / bits_per_value if bits_per_value > 0 else float("inf")


def load_blosc_ratios() -> dict[str, float]:
    if not RESULTS_JSON.exists():
        return {}
    rows = json.loads(RESULTS_JSON.read_text())
    return {
        Path(row["image"]).stem: float(row["ratio"])
        for row in rows
        if row["method"] == "blosc_zstd9_bitshuf"
    }


def geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="ph_*.exr")
    parser.add_argument("--tile-size", type=int, default=32)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = sorted(DATA_DIR.glob(args.glob))
    if not paths:
        raise RuntimeError(f"no images matched {DATA_DIR / args.glob}")
    blosc = load_blosc_ratios()
    results = []
    print(
        f"{'image':43s} {'raw':>7s} {'global':>8s} "
        f"{'tile':>8s} {'Blosc':>8s}"
    )
    print("-" * 82)
    for path in paths:
        pixels = read_exr(path)
        bits = float_to_bits(pixels)
        predictor_bits = predictors(pixels)
        raw_bits = sum(field_cost(bits).values())
        global_bits, choices, costs = global_best_by_field(bits, predictor_bits)
        tile_bits, mode_histogram = tile_best_by_field(
            bits, predictor_bits, args.tile_size
        )
        row = {
            "image": path.name,
            "shape": list(pixels.shape),
            "tile_size": args.tile_size,
            "raw_bits_per_value": raw_bits,
            "global_best_bits_per_value": global_bits,
            "tile_best_bits_per_value": tile_bits,
            "global_choices": choices,
            "global_field_costs": costs,
            "tile_mode_histogram": mode_histogram,
            "ratios": {
                "raw": ratio_from_bits_per_value(raw_bits),
                "global_best": ratio_from_bits_per_value(global_bits),
                "tile_best": ratio_from_bits_per_value(tile_bits),
                "blosc": blosc.get(path.stem),
            },
        }
        results.append(row)
        blosc_text = (
            f"{row['ratios']['blosc']:7.2f}x"
            if row["ratios"]["blosc"] is not None
            else "      -"
        )
        print(
            f"{path.stem:43s} "
            f"{row['ratios']['raw']:6.2f}x "
            f"{row['ratios']['global_best']:7.2f}x "
            f"{row['ratios']['tile_best']:7.2f}x "
            f"{blosc_text}"
        )
    print("-" * 82)
    for key in ["raw", "global_best", "tile_best", "blosc"]:
        values = [
            row["ratios"][key]
            for row in results
            if row["ratios"][key] is not None
        ]
        print(f"geomean {key:13s}: {geomean(values):.2f}x")
    OUTPUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {OUTPUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
