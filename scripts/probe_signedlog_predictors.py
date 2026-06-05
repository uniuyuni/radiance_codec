"""Probe predictor choices for the signed-log escape stream only.

The route and quantized signed-log10 indices are fixed.  This is a pure
coding-only probe to check whether the current MED predictor should be replaced
by AVG or another decoder-visible predictor for masked signed-log samples.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
sys.path.insert(0, str(ROOT / "scripts"))

from audit_linear_index_payload import predict_for_channel, quantize_indices  # noqa: E402
from estimate_vst_signedlog_route import read_mask_png  # noqa: E402
from probe_adaptive_ycocg_router import build_mask, entropy_bits_from_counts  # noqa: E402
from probe_darkbits_router_payload import read_exr  # noqa: E402
from probe_vst_chroma_nr import safe_name  # noqa: E402


def residual_cost_for_predictor(
    indices: np.ndarray,
    mask: np.ndarray,
    bits: int,
    predictor: str,
) -> dict:
    alphabet = 1 << bits
    total_bits = 0.0
    rows = []
    for channel in range(indices.shape[2]):
        pred = predict_for_channel(indices, channel, predictor)
        residual = (
            indices[:, :, channel].astype(np.int32) + alphabet - pred
        ) & (alphabet - 1)
        selected = residual[mask]
        counts = np.bincount(selected.astype(np.int64), minlength=alphabet)
        bit_cost = entropy_bits_from_counts(counts)
        total_bits += bit_cost
        rows.append({
            "channel": channel,
            "samples": int(selected.size),
            "estimated_bits": bit_cost,
            "bits_per_sample": 0.0 if selected.size == 0 else bit_cost / float(selected.size),
            "top_symbols": [
                {"symbol": int(symbol), "count": int(counts[symbol])}
                for symbol in np.argsort(counts)[::-1][:8]
                if int(counts[symbol]) > 0
            ],
        })
    return {
        "predictor": predictor,
        "estimated_bits": total_bits,
        "estimated_bytes": int(math.ceil(total_bits / 8.0) + 32),
        "channels": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="sample_DSCF0009.EXR")
    parser.add_argument("--crop-size", type=int, default=0)
    parser.add_argument("--signedlog-bits", type=int, default=10)
    parser.add_argument("--mask-mode", default="dark-smooth")
    parser.add_argument("--dark-max", type=float, default=0.5)
    parser.add_argument("--mask-radius", type=int, default=2)
    parser.add_argument("--smooth-threshold", type=float, default=0.0025)
    parser.add_argument("--additional-mask-png", type=Path, default=None)
    parser.add_argument(
        "--predictors",
        default="med,avg,west,north,zero,prev-channel-med0,prev-channel-avg0,med-prev-avg",
    )
    parser.add_argument("--label", default="signedlog_predictors")
    args = parser.parse_args()

    path = DATA_DIR / args.input
    pixels = read_exr(path, args.crop_size)
    mask = build_mask(
        pixels,
        args.mask_mode,
        args.dark_max,
        args.mask_radius,
        args.smooth_threshold,
    )
    additional_summary = None
    if args.additional_mask_png is not None:
        additional = read_mask_png(args.additional_mask_png, mask.shape)
        additional_summary = {
            "path": str(args.additional_mask_png),
            "pixel_rate": float(np.mean(additional)),
            "new_pixel_rate": float(np.mean(additional & ~mask)),
        }
        mask |= additional

    indices = quantize_indices(pixels[:, :, :3], args.signedlog_bits, "signed-log")
    predictors = [item.strip() for item in args.predictors.split(",") if item.strip()]
    rows = [
        residual_cost_for_predictor(indices, mask, args.signedlog_bits, predictor)
        for predictor in predictors
    ]
    rows.sort(key=lambda row: row["estimated_bytes"])
    baseline = next(row for row in rows if row["predictor"] == "med")
    best = rows[0]
    summary = {
        "image": args.input,
        "shape": list(pixels.shape),
        "raw_bytes": int(pixels.nbytes),
        "signedlog_bits": args.signedlog_bits,
        "mask_mode": args.mask_mode,
        "dark_max": args.dark_max,
        "mask_radius": args.mask_radius,
        "smooth_threshold": args.smooth_threshold,
        "additional_mask": additional_summary,
        "mask_pixel_rate": float(np.mean(mask)),
        "baseline_predictor": "med",
        "baseline_estimated_bytes": baseline["estimated_bytes"],
        "best_predictor": best["predictor"],
        "best_estimated_bytes": best["estimated_bytes"],
        "gain_bytes": int(baseline["estimated_bytes"] - best["estimated_bytes"]),
        "rows": rows,
        "note": "Coding-only predictor probe for masked signed-log indices.",
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    crop_label = "full" if args.crop_size == 0 else f"crop{args.crop_size}"
    output = RESULTS_DIR / f"{safe_name(args.label)}_{safe_name(path.stem)}_{crop_label}.json"
    output.write_text(json.dumps(summary, indent=2))
    print(
        f"mask={summary['mask_pixel_rate']:.2%} "
        f"med={baseline['estimated_bytes']:,} "
        f"best={best['predictor']}:{best['estimated_bytes']:,} "
        f"gain={summary['gain_bytes']:,}"
    )
    for row in rows:
        print(f" {row['predictor']}: {row['estimated_bytes']:,}")
    print(f"summary={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
