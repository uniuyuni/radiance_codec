"""Combine per-field structural context CNN result files."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"


def geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile-size", type=int, default=64)
    parser.add_argument(
        "--fields",
        nargs="+",
        default=["mantissa_mid", "mantissa_lo"],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    per_field = {}
    for field in args.fields:
        path = RESULTS_DIR / f"structural_context_{field}_tile{args.tile_size}.json"
        per_field[field] = json.loads(path.read_text())["results"]

    base_rows = per_field[args.fields[0]]
    field_rows = {
        field: {row["image"]: row for row in rows}
        for field, rows in per_field.items()
    }
    combined = []
    print(
        f"{'image':43s} {'adaptive':>9s} {'context':>9s} "
        f"{'Blosc':>9s} {'hybrid':>9s}"
    )
    print("-" * 88)
    for base in base_rows:
        bits = base["adaptive_bits"]
        for field in args.fields:
            row = field_rows[field][base["image"]]
            bits -= row["target_adaptive_bits"]
            bits += row["target_hybrid_bits"]
        context_ratio = base["raw_bits"] / bits
        blosc_ratio = base.get("blosc_ratio")
        file_hybrid = max(context_ratio, blosc_ratio or 0.0)
        combined.append(
            {
                "image": base["image"],
                "adaptive_ratio": base["adaptive_ratio"],
                "context_ratio": context_ratio,
                "blosc_ratio": blosc_ratio,
                "file_hybrid_ratio": file_hybrid,
            }
        )
        blosc_text = f"{blosc_ratio:8.2f}x" if blosc_ratio is not None else "       -"
        print(
            f"{Path(base['image']).stem:43s} "
            f"{base['adaptive_ratio']:8.2f}x "
            f"{context_ratio:8.2f}x "
            f"{blosc_text} "
            f"{file_hybrid:8.2f}x"
        )
    print("-" * 88)
    for key in ["adaptive_ratio", "context_ratio", "blosc_ratio", "file_hybrid_ratio"]:
        values = [row[key] for row in combined if row.get(key) is not None]
        print(f"geomean {key:18s}: {geomean(values):.3f}x")

    output = RESULTS_DIR / (
        "structural_context_combined_"
        + "_".join(args.fields)
        + f"_tile{args.tile_size}.json"
    )
    output.write_text(json.dumps(combined, indent=2))
    print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
