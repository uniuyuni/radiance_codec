"""Probe exact nonnegative/signless routes for float32 images.

If all sign bits in a tile/channel are zero, a lossless codec can signal that
fact and omit the sign payload.  This does not save a raw 1 bit/sample when the
existing entropy coder already models constant-zero signs well, so this probe
measures the remaining compressed sign-stream cost.
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
from probe_grouped_tail_mlp import CONTEXT_FAMILIES, context_family_cost  # noqa: E402
from structural_delta_grouped_roundtrip import choose_group_payloads  # noqa: E402

RESULTS_DIR = ROOT / "results"


def best_sign_cost(sign_payload: np.ndarray) -> float:
    return min(context_family_cost(sign_payload, 31, family) for family in CONTEXT_FAMILIES)


def summarize_image(
    path: Path,
    crop_size: int,
    tile_size: int,
    previous_bits: int,
    flag_bits: float,
    scope: str,
) -> dict:
    pixels = read_exr(path)
    if crop_size:
        pixels = pixels[:crop_size, :crop_size]
    pixels = np.ascontiguousarray(pixels, dtype=np.float32)
    bits = float_to_bits(pixels).copy()
    sign_bits = ((bits >> np.uint32(31)) & np.uint32(1)).astype(np.uint8)
    _modes, payloads, _carries = choose_group_payloads(
        bits,
        pixels,
        tile_size,
        previous_bits,
        "body",
    )

    height, width, channels = bits.shape
    current_bits = 0.0
    routed_bits = 0.0
    raw_sign_bits = int(bits.size)
    selected = Counter()
    sign_class = Counter()

    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            sl = np.s_[y:y + tile_size, x:x + tile_size]
            tile_sign = sign_bits[sl]
            sign_payload = payloads["sign"][sl]
            current = best_sign_cost(sign_payload)
            current_bits += current

            if bool(np.all(tile_sign == 0)):
                sign_class["tile_all_positive_zero_sign"] += 1
            elif bool(np.all(tile_sign == 1)):
                sign_class["tile_all_negative_sign"] += 1
            else:
                sign_class["tile_mixed_sign"] += 1

            if scope == "tile":
                if bool(np.all(tile_sign == 0)):
                    routed = flag_bits
                    selected["tile_signless"] += 1
                else:
                    routed = flag_bits + current
                    selected["gdx"] += 1
            elif scope == "channel":
                routed = 0.0
                for channel in range(channels):
                    channel_sign = tile_sign[..., channel]
                    channel_payload = sign_payload[..., channel:channel + 1]
                    channel_current = best_sign_cost(channel_payload)
                    if bool(np.all(channel_sign == 0)):
                        routed += flag_bits
                        selected["channel_signless"] += 1
                    else:
                        routed += flag_bits + channel_current
                        selected["channel_gdx"] += 1
            else:
                raise ValueError(f"unknown scope: {scope}")
            routed_bits += routed

    return {
        "image": path.name,
        "shape": list(bits.shape),
        "crop_size": crop_size,
        "tile_size": tile_size,
        "previous_bits": previous_bits,
        "flag_bits": flag_bits,
        "scope": scope,
        "raw_sign_bits": raw_sign_bits,
        "current_sign_bits": current_bits,
        "routed_sign_bits": routed_bits,
        "current_sign_bps": current_bits / bits.size,
        "routed_sign_bps": routed_bits / bits.size,
        "gain": current_bits / routed_bits if routed_bits else 1.0,
        "selected": dict(selected),
        "sign_class": dict(sign_class),
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
    parser.add_argument("--flag-bits", type=float, default=1.0)
    parser.add_argument("--scope", choices=("tile", "channel"), default="tile")
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = sorted(DATA_DIR.glob(args.glob))
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        raise RuntimeError(f"no images matched {DATA_DIR / args.glob}")

    rows = []
    print(
        f"{'image':43s} {'current':>10s} {'routed':>10s} "
        f"{'gain':>8s} selected sign_class"
    )
    print("-" * 128)
    for path in paths:
        row = summarize_image(
            path,
            args.crop_size,
            args.tile_size,
            args.previous_bits,
            args.flag_bits,
            args.scope,
        )
        rows.append(row)
        print(
            f"{path.stem:43s} "
            f"{row['current_sign_bps']:9.5f} "
            f"{row['routed_sign_bps']:9.5f} "
            f"{row['gain']:7.4f} {row['selected']} {row['sign_class']}"
        )

    print("-" * 128)
    print(
        f"mean current={sum(row['current_sign_bps'] for row in rows) / len(rows):.5f} "
        f"routed={sum(row['routed_sign_bps'] for row in rows) / len(rows):.5f} "
        f"geomean_gain={geomean([row['gain'] for row in rows]):.4f}"
    )

    if not args.no_save:
        safe_glob = args.glob.replace("*", "star").replace(".", "_")
        output = RESULTS_DIR / (
            f"nonnegative_sign_route_{safe_glob}_tile{args.tile_size}"
            f"_prev{args.previous_bits}_{args.scope}_crop{args.crop_size}.json"
        )
        output.write_text(json.dumps(rows, indent=2))
        print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
