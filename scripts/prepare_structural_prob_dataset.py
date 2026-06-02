"""Create a tabular dataset for predicting structural residual bit probabilities."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_structural_predictors import (  # noqa: E402
    DATA_DIR,
    FIELD_GROUPS,
    float_to_bits,
    predictors,
    read_exr,
)

TRAIN_DIR = ROOT / "ml" / "training_data"
OUTPUT_DIR = ROOT / "ml" / "structural_prob_data"

FIELDS = list(FIELD_GROUPS)
MODES = ["zero", "west", "north", "northwest", "northeast", "float_med", "float_planar"]


def process_image(path: Path, tile_size: int, split: str) -> list[dict]:
    pixels = read_exr(path)
    bits = float_to_bits(pixels)
    pred_bits = predictors(pixels)
    height, width, channels = bits.shape
    rows = []
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            sl = np.s_[y:y + tile_size, x:x + tile_size]
            tile = bits[sl]
            tile_values = tile.size
            raw_mean_exp = float(((tile >> 23) & 0xFF).mean())
            raw_var_exp = float(((tile >> 23) & 0xFF).var())
            for field_index, (field, bit_indices) in enumerate(FIELD_GROUPS.items()):
                for mode_index, mode in enumerate(MODES):
                    residual = tile ^ pred_bits[mode][sl]
                    flat = residual.reshape(-1)
                    for bit_index in bit_indices:
                        symbols = (flat >> bit_index) & 1
                        ones = int(symbols.sum())
                        total = int(symbols.size)
                        rows.append(
                            {
                                "split": split,
                                "source": path.name,
                                "tile_origin": [y, x],
                                "tile_shape": list(tile.shape),
                                "channels": channels,
                                "field": field,
                                "field_index": field_index,
                                "mode": mode,
                                "mode_index": mode_index,
                                "bit_index": int(bit_index),
                                "ones": ones,
                                "total": total,
                                "p_one": ones / total,
                                "raw_mean_exp": raw_mean_exp,
                                "raw_var_exp": raw_var_exp,
                            }
                        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--train-limit", type=int, default=12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    specs = [
        ("train", sorted(TRAIN_DIR.glob("*.exr"))[:args.train_limit]),
        ("test", sorted(DATA_DIR.glob("*.exr"))),
    ]
    manifest = {"tile_size": args.tile_size, "files": []}
    for split, paths in specs:
        rows = []
        for index, path in enumerate(paths, start=1):
            try:
                image_rows = process_image(path, args.tile_size, split)
            except Exception as error:
                print(f"SKIP {split} {path.name}: {error}")
                continue
            rows.extend(image_rows)
            print(f"{split} {index}/{len(paths)} {path.name}: {len(image_rows)} rows")
        out = OUTPUT_DIR / f"{split}.jsonl"
        with out.open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        manifest["files"].append({"split": split, "path": str(out), "rows": len(rows)})
        print(f"Saved {len(rows)} rows: {out}")
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
