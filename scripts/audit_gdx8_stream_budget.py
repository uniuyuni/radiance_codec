"""Break down encoded GDX-family streams into coarse byte-budget sections."""
from __future__ import annotations

import argparse
import fnmatch
import json
import math
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import OpenImageIO as oiio

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "codec" / "python"))

import radiance_codec  # noqa: E402


DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

TOP_HEADER_SIZE = 4 + 1 + 4 + 4 + 1 + 1 + 4 + 1 + 1
GDX_HEADER_SIZE = 4 + 2 + 1 + 1 + 4 + 4 + 4
GDX_MAGICS = {b"GDX8", b"GDX9", b"GDXA", b"GDXB"}
TILE_SIZE = 128
GROUP_COUNT = 2
CHANNEL_MODE_SPLIT_FLAG = 0x80
CHANNEL_MODE_MASK = 0x1F
HALF_MODE_MIN = 25
HALF_MODE_MAX = 31
FAMILY_NAMES = {
    0: "base",
    1: "prev_only",
    2: "wn_prev",
    3: "spatial_prev_pc",
    4: "spatial_prev_channel",
    5: "spatial_hi_neighbors",
    6: "spatial_hi_channel",
    7: "spatial_hi_pc",
    8: "prev_xy_channel",
    9: "hash_all_xy",
    10: "prev4_channel",
    11: "wn_prev4_channel",
    12: "wn_prev4_pc",
    13: "spatial_xy",
    14: "constant_zero",
    15: "channel_split",
}


CORPUS_PATTERNS = {
    "custom": (),
    "all": ("*.exr",),
    "target-no-noise": ("*.exr",),
    "realistic-core": ("ph_*.exr", "oexr_ScanLines_*.exr"),
    "realistic-no-puresky": ("ph_*.exr", "oexr_ScanLines_*.exr"),
    "puresky-hard": ("ph_*puresky*.exr",),
    "easy": ("synth_gradient_1k.exr", "oexr_TestImages_GrayRampsHorizontal.exr"),
    "synthetic": ("synth_*.exr",),
    "noise-stress": ("synth_noise_1k.exr",),
}

CORPUS_EXCLUDES = {
    "target-no-noise": ("synth_noise_1k.exr",),
    "realistic-no-puresky": ("ph_*puresky*.exr",),
}


@dataclass
class Budget:
    image: str
    shape: tuple[int, int, int]
    raw_bytes: int
    encoded_bytes: int
    ratio: float
    top_header_bytes: int
    gdx_header_bytes: int
    record_mode_bytes: int
    channel_mode_bytes: int
    tail_selector_bytes: int
    family_bytes: int
    main_payload_bytes: int
    tail_payload_bytes: int
    tail_tiles: int
    body_tiles: int
    body_tail_tile_fraction: float
    body_half_tiles: int
    body_half_tile_fraction: float
    body_channel_split_tiles: int
    body_channel_split_fraction: float
    mode_histogram: dict[str, int]
    family_histogram: dict[str, int]

    @property
    def overhead_bytes(self) -> int:
        return (
            self.top_header_bytes
            + self.gdx_header_bytes
            + self.record_mode_bytes
            + self.channel_mode_bytes
            + self.tail_selector_bytes
            + self.family_bytes
        )


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="ph_studio_small_03_1k.exr")
    parser.add_argument(
        "--corpus",
        choices=sorted(CORPUS_PATTERNS),
        default="custom",
    )
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--crop-size", type=int, default=0)
    parser.add_argument("--effort", type=int, default=12)
    parser.add_argument("--save", action="store_true")
    return parser.parse_args()


def selected_paths(args: argparse.Namespace) -> list[Path]:
    if args.corpus == "custom":
        paths = sorted(DATA_DIR.glob(args.glob))
    else:
        paths = sorted(
            {
                path
                for pattern in CORPUS_PATTERNS[args.corpus]
                for path in DATA_DIR.glob(pattern)
            }
        )
    excludes = list(CORPUS_EXCLUDES.get(args.corpus, ())) + args.exclude
    return [
        path
        for path in paths
        if not any(
            fnmatch.fnmatch(path.name, pattern)
            or fnmatch.fnmatch(path.stem, pattern)
            for pattern in excludes
        )
    ]


def get_u16(data: bytes, offset: int) -> tuple[int, int]:
    return int.from_bytes(data[offset : offset + 2], "little"), offset + 2


def get_u32(data: bytes, offset: int) -> tuple[int, int]:
    return int.from_bytes(data[offset : offset + 4], "little"), offset + 4


def unpack_nibbles(data: bytes, count: int) -> list[int]:
    values = []
    bit_pos = 0
    for _ in range(count):
        value = 0
        for bit in range(4):
            if (data[bit_pos // 8] >> (bit_pos % 8)) & 1:
                value |= 1 << bit
            bit_pos += 1
        values.append(value)
    return values


def parse_gdx8(encoded: bytes, shape: tuple[int, int, int]) -> Budget:
    height, width, channels = shape
    if len(encoded) < TOP_HEADER_SIZE + GDX_HEADER_SIZE:
        raise RuntimeError("encoded stream is too small")
    gdx = encoded[TOP_HEADER_SIZE:]
    if gdx[:4] not in GDX_MAGICS:
        raise RuntimeError(f"expected GDX8/GDX9/GDXA/GDXB payload, got {gdx[:4]!r}")

    offset = 4
    tile_size, offset = get_u16(gdx, offset)
    previous_bits = gdx[offset]
    offset += 1
    reserved = gdx[offset]
    offset += 1
    if tile_size != TILE_SIZE or previous_bits != 4 or reserved != 0:
        raise RuntimeError(
            f"unexpected GDX8 params: tile={tile_size} prev={previous_bits} "
            f"reserved={reserved}"
        )
    record_count, offset = get_u32(gdx, offset)
    main_payload_size, offset = get_u32(gdx, offset)
    tail_payload_size, offset = get_u32(gdx, offset)
    if offset != GDX_HEADER_SIZE:
        raise AssertionError(offset)

    tiles_x = (width + TILE_SIZE - 1) // TILE_SIZE
    tiles_y = (height + TILE_SIZE - 1) // TILE_SIZE
    body_tile_count = tiles_x * tiles_y
    expected_record_count = body_tile_count * GROUP_COUNT
    if record_count != expected_record_count:
        raise RuntimeError(
            f"record count mismatch: stored={record_count} expected={expected_record_count}"
        )

    mode_start = offset
    mode_end = mode_start + record_count
    mode_selectors = list(gdx[mode_start:mode_end])
    offset = mode_end
    if len(mode_selectors) != record_count:
        raise RuntimeError("truncated mode selectors")

    channel_split_records = [
        i for i, selector in enumerate(mode_selectors) if selector & CHANNEL_MODE_SPLIT_FLAG
    ]
    channel_mode_size = len(channel_split_records) * (channels - 1)
    channel_mode_start = offset
    channel_mode_end = channel_mode_start + channel_mode_size
    channel_modes = list(gdx[channel_mode_start:channel_mode_end])
    if len(channel_modes) != channel_mode_size:
        raise RuntimeError("truncated channel modes")
    offset = channel_mode_end

    tail_selector_start = offset
    tail_selector_end = tail_selector_start + body_tile_count
    tail_selectors = list(gdx[tail_selector_start:tail_selector_end])
    if len(tail_selectors) != body_tile_count:
        raise RuntimeError("truncated tail selectors")
    offset = tail_selector_end
    if any(selector > 1 for selector in tail_selectors):
        raise RuntimeError("invalid tail selector")

    main_tail_total = main_payload_size + tail_payload_size
    family_bytes = len(gdx) - offset - main_tail_total
    if family_bytes < 0:
        raise RuntimeError("payload sizes exceed stream size")

    mode_histogram: Counter[str] = Counter()
    channel_mode_pos = 0
    body_channel_split_tiles = 0
    body_half_tiles = 0
    record_bit_counts: list[int] = []
    for i, selector in enumerate(mode_selectors):
        group = "body" if i % GROUP_COUNT == 0 else "sign"
        tile_index = i // GROUP_COUNT
        is_half_body = False
        if selector & CHANNEL_MODE_SPLIT_FLAG:
            first_mode = selector & CHANNEL_MODE_MASK
            modes = [first_mode] + channel_modes[
                channel_mode_pos : channel_mode_pos + channels - 1
            ]
            channel_mode_pos += channels - 1
            mode_histogram[f"{group}:split"] += 1
            if group == "body":
                body_channel_split_tiles += 1
            for mode in modes:
                mode_histogram[f"{group}:m{mode}"] += 1
            if (
                group == "body"
                and modes
                and all(HALF_MODE_MIN <= mode <= HALF_MODE_MAX for mode in modes)
            ):
                body_half_tiles += 1
                is_half_body = True
        else:
            mode = selector & CHANNEL_MODE_MASK
            mode_histogram[f"{group}:m{mode}"] += 1
            if group == "body" and HALF_MODE_MIN <= mode <= HALF_MODE_MAX:
                body_half_tiles += 1
                is_half_body = True
        if group == "body":
            record_bit_counts.append(16 if is_half_body or tail_selectors[tile_index] else 31)
        else:
            record_bit_counts.append(1)

    context_family_count = sum(record_bit_counts)
    packed_selector_size = (context_family_count * 4 + 7) // 8
    family_selector_start = offset
    family_selector_end = family_selector_start + packed_selector_size
    family_selectors = unpack_nibbles(
        bytes(gdx[family_selector_start:family_selector_end]),
        context_family_count,
    )
    split_family_count = sum(1 for value in family_selectors if value == 15)
    packed_channel_family_size = (split_family_count * channels * 4 + 7) // 8
    channel_family_values = unpack_nibbles(
        bytes(gdx[family_selector_end:family_selector_end + packed_channel_family_size]),
        split_family_count * channels,
    )
    family_histogram = Counter(
        FAMILY_NAMES.get(value, f"family_{value}") for value in family_selectors
    )
    family_histogram.update(
        f"split:{FAMILY_NAMES.get(value, f'family_{value}')}"
        for value in channel_family_values
    )

    tail_tiles = sum(1 for selector in tail_selectors if selector != 0)
    raw_bytes = height * width * channels * 4
    return Budget(
        image="",
        shape=shape,
        raw_bytes=raw_bytes,
        encoded_bytes=len(encoded),
        ratio=raw_bytes / len(encoded),
        top_header_bytes=TOP_HEADER_SIZE,
        gdx_header_bytes=GDX_HEADER_SIZE,
        record_mode_bytes=record_count,
        channel_mode_bytes=channel_mode_size,
        tail_selector_bytes=body_tile_count,
        family_bytes=family_bytes,
        main_payload_bytes=main_payload_size,
        tail_payload_bytes=tail_payload_size,
        tail_tiles=tail_tiles,
        body_tiles=body_tile_count,
        body_tail_tile_fraction=tail_tiles / body_tile_count,
        body_half_tiles=body_half_tiles,
        body_half_tile_fraction=body_half_tiles / body_tile_count,
        body_channel_split_tiles=body_channel_split_tiles,
        body_channel_split_fraction=body_channel_split_tiles / body_tile_count,
        mode_histogram=dict(sorted(mode_histogram.items())),
        family_histogram=dict(sorted(family_histogram.items())),
    )


def geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def pct(value: int, total: int) -> float:
    return 100.0 * value / total if total else 0.0


def main() -> int:
    args = parse_args()
    paths = selected_paths(args)
    if not paths:
        raise RuntimeError(
            f"no images matched corpus={args.corpus!r} glob={args.glob!r}"
        )

    budgets: list[Budget] = []
    for path in paths:
        pixels = read_exr(path)
        if args.crop_size:
            pixels = pixels[: args.crop_size, : args.crop_size]
        pixels = np.ascontiguousarray(pixels, dtype=np.float32)
        encoded = radiance_codec.encode(
            pixels,
            stages=radiance_codec.Stage.GROUPED_DELTA,
            effort=args.effort,
        )
        budget = parse_gdx8(bytes(encoded), tuple(pixels.shape))
        budget.image = path.stem
        budgets.append(budget)

    print(radiance_codec.version())
    selection = f"glob={args.glob}" if args.corpus == "custom" else "preset"
    print(
        f"corpus={args.corpus} {selection} crop={args.crop_size} "
        f"effort={args.effort}"
    )
    print(
        f"{'image':43s} {'ratio':>8s} {'main':>9s} {'tail':>9s} "
        f"{'families':>9s} {'overhead':>9s} {'tailT':>7s} "
        f"{'halfT':>7s} {'splitT':>7s}"
    )
    print("-" * 113)
    for budget in budgets:
        print(
            f"{budget.image:43s} {budget.ratio:7.2f}x "
            f"{pct(budget.main_payload_bytes, budget.encoded_bytes):8.1f}% "
            f"{pct(budget.tail_payload_bytes, budget.encoded_bytes):8.1f}% "
            f"{pct(budget.family_bytes, budget.encoded_bytes):8.1f}% "
            f"{pct(budget.overhead_bytes, budget.encoded_bytes):8.1f}% "
            f"{budget.tail_tiles:3d}/{budget.body_tiles:<3d} "
            f"{budget.body_half_tiles:3d}/{budget.body_tiles:<3d} "
            f"{budget.body_channel_split_tiles:3d}/{budget.body_tiles:<3d}"
        )
    print("-" * 113)
    print(f"geomean ratio: {geomean([budget.ratio for budget in budgets]):.3f}x")
    raw_total = sum(budget.raw_bytes for budget in budgets)
    encoded_total = sum(budget.encoded_bytes for budget in budgets)
    main_total = sum(budget.main_payload_bytes for budget in budgets)
    tail_total = sum(budget.tail_payload_bytes for budget in budgets)
    family_total = sum(budget.family_bytes for budget in budgets)
    overhead_total = sum(budget.overhead_bytes for budget in budgets)
    print(f"aggregate ratio: {raw_total / encoded_total:.3f}x")
    print(
        "aggregate sections: "
        f"main={pct(main_total, encoded_total):.1f}% "
        f"tail={pct(tail_total, encoded_total):.1f}% "
        f"families={pct(family_total, encoded_total):.1f}% "
        f"overhead={pct(overhead_total, encoded_total):.1f}%"
    )

    if args.save:
        RESULTS_DIR.mkdir(exist_ok=True)
        crop = args.crop_size
        out = (
            RESULTS_DIR
            / f"gdx8_stream_budget_{args.corpus}_effort{args.effort}_crop{crop}.json"
        )
        out.write_text(
            json.dumps([asdict(budget) for budget in budgets], indent=2),
            encoding="utf-8",
        )
        print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
