"""Report file-level ML-or-Blosc selection from causal-model evaluation."""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ML_RESULTS = ROOT / "ml" / "eval_results_causal.json"
BASELINE_RESULTS = ROOT / "results" / "results.json"
BLOSC_METHOD = "blosc_zstd9_bitshuf"


def geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def main() -> int:
    ml_results = json.loads(ML_RESULTS.read_text())["model_best"]["per_file"]
    baseline_rows = json.loads(BASELINE_RESULTS.read_text())
    blosc = {
        Path(row["image"]).stem: row
        for row in baseline_rows
        if row["method"] == BLOSC_METHOD
    }

    ml_ratios = []
    blosc_ratios = []
    hybrid_ratios = []
    print(
        f"{'image':43s} {'ML est.':>9s} {'Blosc':>9s} "
        f"{'hybrid':>9s} {'pick':>7s}"
    )
    print("-" * 82)
    for image, result in sorted(ml_results.items()):
        raw_bytes = int(blosc[image]["raw_bytes"])
        ml_bytes = math.ceil(raw_bytes * float(result["bpb"]) / 8)
        blosc_bytes = int(blosc[image]["out_bytes"])
        hybrid_bytes = min(ml_bytes, blosc_bytes)
        pick = "ML" if ml_bytes < blosc_bytes else "Blosc"
        ml_ratio = raw_bytes / ml_bytes
        blosc_ratio = raw_bytes / blosc_bytes
        hybrid_ratio = raw_bytes / hybrid_bytes
        ml_ratios.append(ml_ratio)
        blosc_ratios.append(blosc_ratio)
        hybrid_ratios.append(hybrid_ratio)
        print(
            f"{image:43s} {ml_ratio:8.2f}x {blosc_ratio:8.2f}x "
            f"{hybrid_ratio:8.2f}x {pick:>7s}"
        )

    print("-" * 82)
    print(f"{'geomean':43s} {geomean(ml_ratios):8.2f}x "
          f"{geomean(blosc_ratios):8.2f}x {geomean(hybrid_ratios):8.2f}x")
    print("\nML sizes are cross-entropy estimates; verify with ml_rans_compress.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
