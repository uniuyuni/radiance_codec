"""Convert structural probability JSONL data into compact NumPy arrays."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "ml" / "structural_prob_data"


def convert(split: str) -> None:
    path = DATA_DIR / f"{split}.jsonl"
    features = []
    ones = []
    totals = []
    with path.open() as handle:
        for line in handle:
            if not line:
                continue
            row = json.loads(line)
            features.append(
                [
                    row["field_index"] / 4.0,
                    row["mode_index"] / 6.0,
                    row["bit_index"] / 31.0,
                    row["channels"] / 4.0,
                    math.log2(max(1.0, row["raw_mean_exp"] + 1.0)) / 8.0,
                    math.log2(max(1.0, row["raw_var_exp"] + 1.0)) / 16.0,
                ]
            )
            ones.append(row["ones"])
            totals.append(row["total"])
    out = DATA_DIR / f"{split}.npz"
    np.savez(
        out,
        features=np.asarray(features, dtype=np.float32),
        ones=np.asarray(ones, dtype=np.float32),
        totals=np.asarray(totals, dtype=np.float32),
    )
    print(f"Saved {split}: {len(ones)} rows -> {out}")


def main() -> int:
    convert("train")
    convert("test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
