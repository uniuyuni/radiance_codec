"""
Aggregate benchmark results and produce a summary.

Reads results/results.json and produces:
  - results/summary.txt   : human-readable table (stdout + file)
  - results/summary.md    : markdown table for the project doc
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from tabulate import tabulate

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
RESULTS_JSON = RESULTS_DIR / "results.json"


def geomean(xs: list[float]) -> float:
    xs = [x for x in xs if x > 0]
    if not xs:
        return float("nan")
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def main() -> int:
    if not RESULTS_JSON.exists():
        print(f"Missing {RESULTS_JSON}. Run `pixi run bench` first.",
              file=sys.stderr)
        return 1

    data = json.loads(RESULTS_JSON.read_text())
    if not data:
        print("Empty results.", file=sys.stderr)
        return 1

    # Group by method
    by_method: dict[str, list[dict]] = defaultdict(list)
    for r in data:
        by_method[r["method"]].append(r)

    # Per-method aggregates
    rows = []
    n_images = len({r["image"] for r in data})
    for method, rs in by_method.items():
        ratios = [r["ratio"] for r in rs]
        bpps = [r["bpp"] for r in rs]
        enc_mbps = [r["encode_throughput_mbps"] for r in rs]
        dec_mbps = [r["decode_throughput_mbps"] for r in rs]
        ll_count = sum(1 for r in rs if r["lossless"])

        rows.append({
            "method": method,
            "ratio_geomean": geomean(ratios),
            "bpp_mean": statistics.mean(bpps),
            "bpp_min": min(bpps),
            "bpp_max": max(bpps),
            "enc_mbps_median": statistics.median(enc_mbps),
            "dec_mbps_median": statistics.median(dec_mbps),
            "lossless": f"{ll_count}/{len(rs)}",
        })

    # Sort by ratio desc, with lossless first
    def key(r):
        is_lossy = "/" in r["lossless"] and not r["lossless"].startswith(
            r["lossless"].split("/")[1])
        return (is_lossy, -r["ratio_geomean"])
    rows.sort(key=key)

    # Format as table
    headers = ["method", "ratio (geomean)", "bpp mean",
               "bpp range", "enc MB/s", "dec MB/s", "lossless"]
    table = []
    for r in rows:
        table.append([
            r["method"],
            f"{r['ratio_geomean']:.2f}x",
            f"{r['bpp_mean']:.2f}",
            f"{r['bpp_min']:.1f}-{r['bpp_max']:.1f}",
            f"{r['enc_mbps_median']:.1f}",
            f"{r['dec_mbps_median']:.1f}",
            r["lossless"],
        ])

    print(f"\n=== Float32 HDR EXR lossless compression — {n_images} images ===\n")
    txt = tabulate(table, headers=headers, tablefmt="github")
    print(txt)
    print(f"\n(raw size baseline: float32 RGBA = 96-128 bpp depending on channel count)")
    print(f"(geomean ratio shown; bpp/throughput per-method aggregates)")

    # Save text and markdown
    (RESULTS_DIR / "summary.txt").write_text(
        f"Float32 HDR EXR lossless compression - {n_images} images\n\n"
        + tabulate(table, headers=headers, tablefmt="plain") + "\n"
    )
    (RESULTS_DIR / "summary.md").write_text(
        f"# Float32 HDR EXR lossless compression\n\n"
        f"- Test set: {n_images} EXR images "
        f"(Poly Haven + OpenEXR samples)\n"
        f"- Median of 3 runs (1 warm-up) per measurement\n"
        f"- Apple Silicon, single thread per tool (default settings)\n\n"
        f"## Aggregate results\n\n"
        + tabulate(table, headers=headers, tablefmt="github")
        + "\n\nColumns:\n"
        "- **ratio (geomean)**: geometric mean of compression ratio "
        "across all test images (higher = better)\n"
        "- **bpp**: bits per pixel of the compressed output\n"
        "- **enc/dec MB/s**: throughput vs raw float32 data "
        "(higher = faster)\n"
        "- **lossless**: number of images verified byte-exact\n\n"
        "## Per-image results\n\n"
        "See `results.csv` / `results.json` for per-image breakdown.\n"
    )
    print(f"\nSaved: results/summary.txt, results/summary.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
