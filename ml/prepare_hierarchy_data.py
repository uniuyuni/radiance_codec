"""Prepare uint32 HDR tiles and checkerboard pass metadata for V2 teachers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import OpenImageIO as oiio

from hdr_hierarchy import float_to_bits, make_pass_map, nearest_context_residuals

ROOT = Path(__file__).resolve().parent.parent
DATASETS = {
    "test": ROOT / "data",
    "train": ROOT / "ml" / "training_data",
}
OUTPUT_ROOT = ROOT / "ml" / "hierarchy_data"


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
    parser.add_argument("--split", choices=sorted(DATASETS), default="test")
    parser.add_argument("--glob", default="*.exr")
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--max-step", type=int, default=16)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = sorted(DATASETS[args.split].glob(args.glob))
    if args.limit is not None:
        paths = paths[:args.limit]
    if not paths:
        raise RuntimeError("no EXR files matched")

    output_dir = OUTPUT_ROOT / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "split": args.split,
        "tile_size": args.tile_size,
        "max_step": args.max_step,
        "tiles": [],
        "skipped": [],
    }
    count = 0
    for path in paths:
        try:
            bits = float_to_bits(read_exr(path))
        except Exception as error:
            manifest["skipped"].append({"source": path.name, "error": str(error)})
            print(f"SKIP {path.name}: {error}")
            continue
        height, width, channels = bits.shape
        for y in range(0, height, args.tile_size):
            for x in range(0, width, args.tile_size):
                tile = bits[y:y + args.tile_size, x:x + args.tile_size]
                tile_height, tile_width, _ = tile.shape
                pass_map, pass_names = make_pass_map(
                    tile_height, tile_width, args.max_step
                )
                residuals, _ = nearest_context_residuals(tile, args.max_step)
                relative = Path(path.stem) / f"y{y:05d}_x{x:05d}.npz"
                destination = output_dir / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                np.savez(
                    destination,
                    bits=tile,
                    pass_map=pass_map,
                    residuals=residuals,
                )
                manifest["tiles"].append(
                    {
                        "path": str(relative),
                        "source": path.name,
                        "origin": [y, x],
                        "shape": [tile_height, tile_width, channels],
                        "pass_names": pass_names,
                    }
                )
                count += 1
        print(f"{path.name}: {height}x{width}x{channels}")

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Saved {count} tiles: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
