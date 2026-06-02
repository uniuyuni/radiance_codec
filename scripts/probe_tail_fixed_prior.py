"""Probe decoder-bundled fixed priors for exact low-tail bits.

Several puresky probes show a large conditional-entropy gap for raw low
mantissa tails, but dictionary/static-table routes lose the gain to side
information.  This probe removes transmitted tables: it trains fixed binary
context counts on other images and evaluates held-out low-tail coding cost.

If leave-one-out wins, the model could be compiled into the decoder or shipped
as a format-level profile.  If only self-oracle wins, the structure is
image-local and needs a better low-side-information adaptation route.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_structural_predictors import DATA_DIR, float_to_bits, read_exr  # noqa: E402
from evaluate_structural_delta_context import bits_to_ordered  # noqa: E402
from probe_grouped_tail_mlp import CONTEXT_FAMILIES, context_family_cost  # noqa: E402
from probe_tail_static_prob_table import context_seed, fold_hash  # noqa: E402
from structural_delta_grouped_roundtrip import choose_group_payloads  # noqa: E402

RESULTS_DIR = ROOT / "results"


@dataclass
class ImageData:
    path: Path
    shape: tuple[int, int, int]
    raw_bits: int
    bits: np.ndarray
    ordered: np.ndarray
    body_payload: np.ndarray
    sign_payload: np.ndarray


def geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(",") if part)


def load_image(path: Path, crop_size: int, tile_size: int, previous_bits: int) -> ImageData:
    pixels = read_exr(path)
    if crop_size:
        pixels = pixels[:crop_size, :crop_size]
    pixels = np.ascontiguousarray(pixels, dtype=np.float32)
    bits = float_to_bits(pixels).copy()
    ordered = bits_to_ordered(bits)
    _modes, payloads, _carries = choose_group_payloads(
        bits,
        pixels,
        tile_size,
        previous_bits,
        "body",
    )
    return ImageData(
        path=path,
        shape=tuple(int(v) for v in pixels.shape),
        raw_bits=int(pixels.size * 32),
        bits=bits,
        ordered=ordered,
        body_payload=payloads["body"],
        sign_payload=payloads["sign"],
    )


def best_family_cost(payload: np.ndarray, bit_indices: range | tuple[int, ...]) -> float:
    total = 0.0
    for bit in bit_indices:
        total += min(
            context_family_cost(payload, bit, family)
            for family in CONTEXT_FAMILIES
        )
    return total


def current_costs(data: ImageData, tail_width: int) -> dict[str, float]:
    low = best_family_cost(data.body_payload, range(tail_width))
    high = best_family_cost(data.body_payload, range(tail_width, 31))
    sign = best_family_cost(data.sign_payload, (31,))
    return {
        "sign": sign,
        "high": high,
        "low": low,
        "total": sign + high + low,
    }


def tail_values(data: ImageData, tail_width: int, tail_source: str) -> np.ndarray:
    mask = np.uint32((1 << tail_width) - 1)
    if tail_source == "ordered":
        return data.ordered & mask
    if tail_source == "mantissa":
        return (data.bits & np.uint32(0x7fffff)) & mask
    raise ValueError(f"unknown tail source: {tail_source}")


def context_symbols(
    data: ImageData,
    tail_width: int,
    context_bits: int,
    previous_tail_bits: int,
    xy_bits: int,
    context_mode: str,
    tail_source: str,
) -> list[tuple[np.ndarray, np.ndarray]]:
    tail = tail_values(data, tail_width, tail_source)
    seed = context_seed(
        data.bits,
        data.ordered,
        data.body_payload,
        tail_width,
        xy_bits,
        context_mode,
    )
    rows = []
    for bit in range(tail_width - 1, -1, -1):
        symbols = ((tail >> np.uint32(bit)) & np.uint32(1)).astype(np.uint8)
        code = seed ^ (np.uint64(bit) << np.uint64(45))
        if previous_tail_bits > 0 and bit + 1 < tail_width:
            available = min(previous_tail_bits, tail_width - bit - 1)
            previous = (tail >> np.uint32(bit + 1)) & np.uint32(
                (1 << available) - 1
            )
            code ^= previous.astype(np.uint64) << np.uint64(49)
        rows.append(
            (
                fold_hash(code, context_bits).reshape(-1),
                symbols.reshape(-1),
            )
        )
    return rows


def train_counts(
    images: list[ImageData],
    tail_width: int,
    context_bits: int,
    previous_tail_bits: int,
    xy_bits: int,
    context_mode: str,
    tail_source: str,
) -> tuple[np.ndarray, np.ndarray]:
    context_count = 1 << context_bits
    ones = np.zeros((tail_width, context_count), dtype=np.int64)
    totals = np.zeros((tail_width, context_count), dtype=np.int64)
    for data in images:
        for bit_index, (contexts, symbols) in enumerate(
            context_symbols(
                data,
                tail_width,
                context_bits,
                previous_tail_bits,
                xy_bits,
                context_mode,
                tail_source,
            )
        ):
            totals[bit_index] += np.bincount(
                contexts,
                minlength=context_count,
            ).astype(np.int64)
            ones[bit_index] += np.bincount(
                contexts,
                weights=symbols,
                minlength=context_count,
            ).astype(np.int64)
    return ones, totals


def learned_cost(
    data: ImageData,
    ones: np.ndarray,
    totals: np.ndarray,
    tail_width: int,
    context_bits: int,
    previous_tail_bits: int,
    xy_bits: int,
    context_mode: str,
    tail_source: str,
    alpha: float,
) -> float:
    cost = 0.0
    for bit_index, (contexts, symbols) in enumerate(
        context_symbols(
            data,
            tail_width,
            context_bits,
            previous_tail_bits,
            xy_bits,
            context_mode,
            tail_source,
        )
    ):
        probability = (
            ones[bit_index][contexts].astype(np.float64) + alpha
        ) / (totals[bit_index][contexts].astype(np.float64) + 2.0 * alpha)
        probability = np.clip(probability, 1.0e-9, 1.0 - 1.0e-9)
        cost += float(
            np.sum(
                np.where(
                    symbols != 0,
                    -np.log2(probability),
                    -np.log2(1.0 - probability),
                )
            )
        )
    return cost


def train_set_for(
    images: list[ImageData],
    test: ImageData,
    policy: str,
) -> list[ImageData]:
    if policy == "leave-one-out":
        return [image for image in images if image.path != test.path]
    if policy == "self-oracle":
        return [test]
    if policy == "all":
        return images
    raise ValueError(f"unknown train policy: {policy}")


def summarize(
    images: list[ImageData],
    tail_widths: tuple[int, ...],
    context_bits_values: tuple[int, ...],
    previous_tail_bits_values: tuple[int, ...],
    context_modes: tuple[str, ...],
    xy_bits_values: tuple[int, ...],
    train_policy: str,
    tail_source: str,
    alpha: float,
    route_header_bits: float,
) -> list[dict]:
    rows = []
    for test in images:
        image_rows = []
        train_images = train_set_for(images, test, train_policy)
        for tail_width in tail_widths:
            current = current_costs(test, tail_width)
            best_row = None
            for context_bits in context_bits_values:
                for previous_tail_bits in previous_tail_bits_values:
                    for xy_bits in xy_bits_values:
                        for context_mode in context_modes:
                            ones, totals = train_counts(
                                train_images,
                                tail_width,
                                context_bits,
                                previous_tail_bits,
                                xy_bits,
                                context_mode,
                                tail_source,
                            )
                            low = learned_cost(
                                test,
                                ones,
                                totals,
                                tail_width,
                                context_bits,
                                previous_tail_bits,
                                xy_bits,
                                context_mode,
                                tail_source,
                                alpha,
                            ) + route_header_bits
                            selected_low = min(current["low"], low)
                            total = current["sign"] + current["high"] + selected_low
                            row = {
                                "tail_width": tail_width,
                                "context_bits": context_bits,
                                "previous_tail_bits": previous_tail_bits,
                                "xy_bits": xy_bits,
                                "context_mode": context_mode,
                                "current_bits": current["total"],
                                "hybrid_bits": total,
                                "current_ratio": test.raw_bits / current["total"],
                                "hybrid_ratio": test.raw_bits / total,
                                "current_low_bits_per_sample": (
                                    current["low"] / np.prod(test.shape)
                                ),
                                "learned_low_bits_per_sample": low / np.prod(test.shape),
                                "gain": current["total"] / total,
                                "route": (
                                    "fixed_prior"
                                    if low < current["low"]
                                    else "current_low"
                                ),
                                "train_images": [image.path.name for image in train_images],
                            }
                            if best_row is None or row["hybrid_bits"] < best_row["hybrid_bits"]:
                                best_row = row
            image_rows.append(best_row)
        rows.append(
            {
                "image": test.path.name,
                "shape": list(test.shape),
                "raw_bits": test.raw_bits,
                "rows": image_rows,
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="ph_*puresky*.exr")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--previous-bits", type=int, default=4)
    parser.add_argument("--tail-widths", default="8,10,12,15")
    parser.add_argument("--context-bits", default="8,10,12")
    parser.add_argument("--previous-tail-bits", default="0,2,4")
    parser.add_argument(
        "--context-modes",
        default="ordered_high,float_high,payload_high,payload_exp_high",
    )
    parser.add_argument("--xy-bits", default="0,2,3")
    parser.add_argument(
        "--train-policy",
        choices=("leave-one-out", "self-oracle", "all"),
        default="leave-one-out",
    )
    parser.add_argument(
        "--tail-source",
        choices=("ordered", "mantissa"),
        default="ordered",
    )
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--route-header-bits", type=float, default=8.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = sorted(DATA_DIR.glob(args.glob))
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit(f"no images matched {args.glob!r}")

    images = [
        load_image(path, args.crop_size, args.tile_size, args.previous_bits)
        for path in paths
    ]
    tail_widths = parse_ints(args.tail_widths)
    context_bits_values = parse_ints(args.context_bits)
    previous_tail_bits_values = parse_ints(args.previous_tail_bits)
    context_modes = tuple(part for part in args.context_modes.split(",") if part)
    xy_bits_values = parse_ints(args.xy_bits)

    rows = summarize(
        images,
        tail_widths,
        context_bits_values,
        previous_tail_bits_values,
        context_modes,
        xy_bits_values,
        args.train_policy,
        args.tail_source,
        args.alpha,
        args.route_header_bits,
    )

    for image in rows:
        print(image["image"])
        for row in image["rows"]:
            print(
                f"  low{row['tail_width']:02d}/ctx{row['context_bits']:02d}"
                f"/pt{row['previous_tail_bits']:02d}/xy{row['xy_bits']}"
                f"/{row['context_mode']}: "
                f"current={row['current_ratio']:.3f}x "
                f"hybrid={row['hybrid_ratio']:.3f}x "
                f"low={row['current_low_bits_per_sample']:.3f}->"
                f"{row['learned_low_bits_per_sample']:.3f}bps "
                f"route={row['route']}"
            )

    if rows:
        print("\ngeomean:")
        for tail_width in tail_widths:
            current_values = []
            hybrid_values = []
            for image in rows:
                row = next(r for r in image["rows"] if r["tail_width"] == tail_width)
                current_values.append(row["current_ratio"])
                hybrid_values.append(row["hybrid_ratio"])
            print(
                f"  low{tail_width:02d}: current={geomean(current_values):.3f}x "
                f"hybrid={geomean(hybrid_values):.3f}x"
            )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_glob = args.glob.replace("*", "star").replace(".", "_")
    output = RESULTS_DIR / (
        f"tail_fixed_prior_{safe_glob}_tile{args.tile_size}"
        f"_prev{args.previous_bits}_crop{args.crop_size}"
        f"_tail{args.tail_widths.replace(',', '-')}"
        f"_ctx{args.context_bits.replace(',', '-')}"
        f"_pt{args.previous_tail_bits.replace(',', '-')}"
        f"_{args.train_policy}_{args.tail_source}.json"
    )
    output.write_text(json.dumps(rows, indent=2))
    print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
