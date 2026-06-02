"""Summarize exact/near-lossless mantissa-quantization target reach.

This consumes:

* a budget audit JSON from ``audit_16x_budget.py`` for exact baseline sizes;
* one or more ``estimate_mantissa_quantization.py`` result JSON files.

It answers the product-shaped question: if exact GDX3 stays as the baseline and
we optionally zero low raw mantissa bits on measured images, how close does the
corpus get to a requested ratio target?
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"


def geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def default_budget_json() -> Path:
    candidates = [
        RESULTS_DIR / "budget_x16_star_exr_effort9_crop0.json",
        RESULTS_DIR / "budget_16x_star_exr_effort9_crop0.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_budget(path: Path) -> dict[str, dict[str, Any]]:
    rows = json.loads(path.read_text())
    return {row["image"]: row for row in rows}


def load_quant_rows(pattern: str) -> dict[str, dict[int, dict[str, Any]]]:
    by_image: dict[str, dict[int, dict[str, Any]]] = {}
    for path in sorted(RESULTS_DIR.glob(pattern)):
        rows = json.loads(path.read_text())
        for image_result in rows:
            image = image_result["image"]
            image_rows = by_image.setdefault(image, {})
            for row in image_result["rows"]:
                low_bits = int(row["low_bits_zeroed"])
                current = image_rows.get(low_bits)
                if current is None or row["ratio_vs_original"] >= current["ratio_vs_original"]:
                    merged = dict(row)
                    merged["source_json"] = path.name
                    image_rows[low_bits] = merged
    return by_image


def select_quant_row(
    exact_ratio: float,
    rows_by_low_bits: dict[int, dict[str, Any]],
    policy: str,
    target_ratio: float,
    fixed_low_bits: int,
) -> tuple[int, dict[str, Any] | None, str]:
    if exact_ratio >= target_ratio:
        return 0, rows_by_low_bits.get(0), "exact_reaches_target"
    if not rows_by_low_bits:
        return 0, None, "no_quant_data"
    if policy == "fixed-low-bits":
        row = rows_by_low_bits.get(fixed_low_bits)
        if row is None:
            return 0, None, f"missing_low{fixed_low_bits:02d}"
        status = "target_reached" if row["ratio_vs_original"] >= target_ratio else "below_target"
        return fixed_low_bits, row, status

    candidates = sorted(rows_by_low_bits.items())
    for low_bits, row in candidates:
        if row["ratio_vs_original"] >= target_ratio:
            return low_bits, row, "target_reached"
    low_bits, row = max(candidates, key=lambda item: item[1]["ratio_vs_original"])
    return low_bits, row, "below_target"


def fmt_ratio(value: float) -> str:
    return f"{value:7.2f}x"


def fmt_db(value: float) -> str:
    if math.isinf(value):
        return "inf"
    return f"{value:.1f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-ratio", type=float, default=12.0)
    parser.add_argument("--policy", choices=("reach-target", "fixed-low-bits"), default="reach-target")
    parser.add_argument("--low-bits", type=int, default=15)
    parser.add_argument("--budget-json", type=Path, default=default_budget_json())
    parser.add_argument(
        "--quant-glob",
        default="mantissa_quantization_*_effort9_crop0_low*.json",
        help="Glob under results/ for quantization result JSON files.",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    budget = load_budget(args.budget_json)
    quant = load_quant_rows(args.quant_glob)

    summary_rows = []
    for image, exact in budget.items():
        exact_ratio = float(exact["ratio"])
        exact_encoded_bits = float(exact["encoded_bits"])
        raw_bits = float(exact["raw_bits"])
        target_bits = raw_bits / args.target_ratio
        low_bits, selected_quant, status = select_quant_row(
            exact_ratio,
            quant.get(image, {}),
            args.policy,
            args.target_ratio,
            args.low_bits,
        )
        if selected_quant is None:
            selected_ratio = exact_ratio
            selected_encoded_bits = exact_encoded_bits
            max_rel = 0.0
            psnr = float("inf")
            changed = 0.0
            source_json = None
        else:
            selected_ratio = float(selected_quant["ratio_vs_original"])
            selected_encoded_bits = float(selected_quant["encoded_bytes"]) * 8.0
            max_rel = float(selected_quant["max_rel_nonzero"])
            psnr = float(selected_quant["psnr_peak_abs"])
            changed = float(selected_quant["changed_rate"])
            source_json = selected_quant["source_json"]

        summary_rows.append(
            {
                "image": image,
                "source_precision": exact.get("source_precision", {}).get("class", "unknown"),
                "exact_ratio": exact_ratio,
                "selected_low_bits": low_bits,
                "selected_ratio": selected_ratio,
                "selected_encoded_bits": selected_encoded_bits,
                "target_bits": target_bits,
                "bits_over_target": max(0.0, selected_encoded_bits - target_bits),
                "max_rel_nonzero": max_rel,
                "psnr_peak_abs": psnr,
                "changed_rate": changed,
                "status": status,
                "source_json": source_json,
                "raw_bits": raw_bits,
            }
        )

    exact_ratios = [row["exact_ratio"] for row in summary_rows]
    selected_ratios = [row["selected_ratio"] for row in summary_rows]
    exact_total_ratio = sum(row["raw_bits"] for row in summary_rows) / sum(
        budget[row["image"]]["encoded_bits"] for row in summary_rows
    )
    selected_total_ratio = sum(row["raw_bits"] for row in summary_rows) / sum(
        row["selected_encoded_bits"] for row in summary_rows
    )
    target_count = sum(row["selected_ratio"] >= args.target_ratio for row in summary_rows)
    exact_target_count = sum(row["exact_ratio"] >= args.target_ratio for row in summary_rows)
    total_gap_kib = sum(row["bits_over_target"] for row in summary_rows) / 8.0 / 1024.0
    max_rel = max(row["max_rel_nonzero"] for row in summary_rows)

    print(
        f"{'image':43s} {'exact':>8s} {'low':>5s} {'selected':>9s} "
        f"{f'gap{args.target_ratio:g}':>9s} {'max_rel':>10s} "
        f"{'psnr':>7s} {'status':>20s}"
    )
    print("-" * 122)
    for row in summary_rows:
        print(
            f"{Path(row['image']).stem:43s} "
            f"{fmt_ratio(row['exact_ratio'])} "
            f"{row['selected_low_bits']:5d} "
            f"{fmt_ratio(row['selected_ratio']):>9s} "
            f"{row['bits_over_target'] / 8.0 / 1024.0:8.1f}K "
            f"{row['max_rel_nonzero']:10.3e} "
            f"{fmt_db(row['psnr_peak_abs']):>7s} "
            f"{row['status']:>20s}"
        )

    print("-" * 122)
    print(f"exact geomean:      {geomean(exact_ratios):.3f}x")
    print(f"selected geomean:   {geomean(selected_ratios):.3f}x")
    print(f"exact total ratio:  {exact_total_ratio:.3f}x")
    print(f"selected total ratio: {selected_total_ratio:.3f}x")
    print(f"images >= {args.target_ratio:g}x: {target_count}/{len(summary_rows)} (exact {exact_target_count}/{len(summary_rows)})")
    print(f"total selected gap to {args.target_ratio:g}x: {total_gap_kib:.1f} KiB")
    print(f"max selected relative error: {max_rel:.3e}")

    output = args.output
    if output is None:
        output = RESULTS_DIR / (
            f"mantissa_target_summary_x{args.target_ratio:g}_{args.policy}"
            f"_low{args.low_bits}.json"
        )
    output.write_text(
        json.dumps(
            {
                "target_ratio": args.target_ratio,
                "policy": args.policy,
                "low_bits": args.low_bits,
                "budget_json": str(args.budget_json),
                "quant_glob": args.quant_glob,
                "exact_geomean": geomean(exact_ratios),
                "selected_geomean": geomean(selected_ratios),
                "exact_total_ratio": exact_total_ratio,
                "selected_total_ratio": selected_total_ratio,
                "target_count": target_count,
                "exact_target_count": exact_target_count,
                "total_gap_kib": total_gap_kib,
                "max_selected_relative_error": max_rel,
                "rows": summary_rows,
            },
            indent=2,
        )
    )
    print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
