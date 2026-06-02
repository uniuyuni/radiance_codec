"""Round-trip check for grouped ordered-float delta payloads."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_structural_predictors import DATA_DIR, float_to_bits, read_exr  # noqa: E402
from evaluate_structural_delta_context import (  # noqa: E402
    bits_to_ordered,
    delta_predictors,
    ordered_delta,
    ordered_xor_predictors,
)
from evaluate_structural_delta_grouped import (  # noqa: E402
    GROUP_PRESETS,
    filter_group_predictors,
    group_bit_indices,
    group_candidates,
    grouped_delta,
)
from evaluate_structural_delta_sidecarry import (  # noqa: E402
    field_layout,
    incoming_carry_bits,
)
from structural_delta_field_roundtrip import predictor_scalar  # noqa: E402


def mode_payload(
    mode: str,
    ordered_tile: np.ndarray,
    xor_tiles: dict[str, np.ndarray],
    delta_tiles: dict[str, np.ndarray],
    bit_indices: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray | None]:
    if mode == "raw":
        return ordered_tile, None
    if mode.startswith("xor_"):
        return ordered_tile ^ xor_tiles[mode[4:]], None
    if mode.startswith("local_"):
        prediction = delta_tiles[mode[6:]]
        return grouped_delta(ordered_tile, prediction, bit_indices), None

    prediction = delta_tiles[mode]
    residual = ordered_delta(ordered_tile, prediction)
    low_bit = min(bit_indices)
    carry = None
    if low_bit > 0:
        carry = incoming_carry_bits(
            ordered_tile,
            prediction,
            residual,
            bit_indices,
        )
    return residual, carry


def choose_group_payloads(
    bits: np.ndarray,
    pixels: np.ndarray,
    tile_size: int,
    previous_bits: int,
    preset: str,
) -> tuple[dict[tuple[int, int, str], str], dict[str, np.ndarray], dict[str, np.ndarray]]:
    ordered = bits_to_ordered(bits)
    xor_predictor_bits = ordered_xor_predictors(bits, pixels, causal=True)
    delta_prediction_bits = delta_predictors(bits, pixels, causal=True)
    groups = GROUP_PRESETS[preset]
    modes: dict[tuple[int, int, str], str] = {}
    payloads = {
        group_name: np.zeros_like(ordered, dtype=np.uint32)
        for group_name, _fields in groups
    }
    carries = {
        group_name: np.zeros_like(ordered, dtype=np.uint8)
        for group_name, _fields in groups
    }
    height, width, _channels = bits.shape

    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            sl = np.s_[y:y + tile_size, x:x + tile_size]
            ordered_tile = ordered[sl]
            xor_tiles = {name: pred[sl] for name, pred in xor_predictor_bits.items()}
            delta_tiles = {
                name: pred[sl] for name, pred in delta_prediction_bits.items()
            }
            for group_name, fields in groups:
                group_xor_tiles, group_delta_tiles = filter_group_predictors(
                    group_name,
                    xor_tiles,
                    delta_tiles,
                )
                bit_indices = group_bit_indices(fields)
                candidates = group_candidates(
                    ordered_tile,
                    group_xor_tiles,
                    group_delta_tiles,
                    fields,
                    previous_bits,
                    carry_channel_context=False,
                    bitplane_coding="context",
                )
                mode = min(candidates, key=lambda name: candidates[name][0])
                modes[(y, x, group_name)] = mode
                payload, carry = mode_payload(
                    mode,
                    ordered_tile,
                    group_xor_tiles,
                    group_delta_tiles,
                    bit_indices,
                )
                low_bit, width_bits, mask = field_layout(bit_indices)
                group_mask = np.uint32(mask) << np.uint32(low_bit)
                payloads[group_name][sl] = payload & group_mask
                if carry is not None:
                    carries[group_name][sl] = carry

    return modes, payloads, carries


def decode_grouped(
    modes: dict[tuple[int, int, str], str],
    payloads: dict[str, np.ndarray],
    carries: dict[str, np.ndarray],
    shape: tuple[int, int, int],
    tile_size: int,
    preset: str,
) -> np.ndarray:
    groups = sorted(
        GROUP_PRESETS[preset],
        key=lambda group: min(group_bit_indices(group[1])),
    )
    height, width, channels = shape
    decoded = np.zeros(shape, dtype=np.uint32)

    for y in range(height):
        tile_y = (y // tile_size) * tile_size
        for x in range(width):
            tile_x = (x // tile_size) * tile_size
            for group_name, fields in groups:
                mode = modes[(tile_y, tile_x, group_name)]
                if mode.endswith("channel_green") and channels >= 3:
                    channel_order = (1, 0, 2, 3)[:channels]
                else:
                    channel_order = tuple(range(channels))
                for c in channel_order:
                    value = int(decoded[y, x, c])
                    bit_indices = group_bit_indices(fields)
                    low_bit, _width_bits, mask = field_layout(bit_indices)
                    payload_digit = (
                        int(payloads[group_name][y, x, c]) >> low_bit
                    ) & int(mask)

                    if mode == "raw":
                        digit = payload_digit
                    elif mode.startswith("xor_"):
                        pred = predictor_scalar(mode[4:], decoded, y, x, c)
                        digit = ((pred >> low_bit) & int(mask)) ^ payload_digit
                    elif mode.startswith("local_"):
                        pred = predictor_scalar(mode[6:], decoded, y, x, c)
                        pred_digit = (pred >> low_bit) & int(mask)
                        digit = (pred_digit + payload_digit) & int(mask)
                    else:
                        pred = predictor_scalar(mode, decoded, y, x, c)
                        pred_digit = (pred >> low_bit) & int(mask)
                        carry = 0 if low_bit == 0 else int(carries[group_name][y, x, c])
                        digit = (pred_digit + payload_digit + carry) & int(mask)

                    value |= digit << low_bit
                    decoded[y, x, c] = np.uint32(value)
    return decoded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="ph_studio_small_03_1k.exr")
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--previous-bits", type=int, default=3)
    parser.add_argument("--preset", choices=sorted(GROUP_PRESETS), default="body")
    parser.add_argument("--crop-size", type=int, default=128)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = sorted(DATA_DIR.glob(args.glob))
    if not paths:
        raise RuntimeError(f"no images matched {DATA_DIR / args.glob}")
    for path in paths:
        pixels = read_exr(path)
        if args.crop_size:
            pixels = pixels[: args.crop_size, : args.crop_size]
        bits = float_to_bits(np.ascontiguousarray(pixels, dtype=np.float32)).copy()
        ordered = bits_to_ordered(bits)
        modes, payloads, carries = choose_group_payloads(
            bits,
            pixels,
            args.tile_size,
            args.previous_bits,
            args.preset,
        )
        decoded = decode_grouped(
            modes,
            payloads,
            carries,
            ordered.shape,
            args.tile_size,
            args.preset,
        )
        if not np.array_equal(decoded, ordered):
            mismatch = np.argwhere(decoded != ordered)[0]
            y, x, c = (int(v) for v in mismatch)
            raise RuntimeError(
                f"{path.name}: mismatch at y={y} x={x} c={c}: "
                f"decoded=0x{int(decoded[y, x, c]):08x} "
                f"expected=0x{int(ordered[y, x, c]):08x}"
            )
        print(
            f"{path.name}: grouped roundtrip ok "
            f"preset={args.preset} shape={ordered.shape}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
