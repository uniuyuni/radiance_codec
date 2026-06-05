"""Search dark-refinement router candidates with the proxy visual gate.

This is a research probe, not bitstream behavior.  It combines three things
that had been separate:

* base transform-index quantization
* dark-region refinement mask and correction payload estimate
* display-space proxy visual gate

The goal is to find candidates that deserve human review before spending time
on full-resolution previews or C++ implementation.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
sys.path.insert(0, str(ROOT / "codec" / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

import radiance_codec  # noqa: E402
from audit_linear_index_payload import residual_stats  # noqa: E402
from audit_visual_gate import (  # noqa: E402
    candidate_metrics,
    default_gate_args,
    evaluate_gate,
    summarize_key_metrics,
)
from probe_darkbits_router_payload import (  # noqa: E402
    ROUTER_HEADER_BYTES,
    reconstruct_mixed,
    summarize_binary_mask,
    summarize_refinement,
    quantize_signed_log,
    read_exr,
)
from probe_power_recon_table import table_reconstruct  # noqa: E402


BUILTIN_TRANSFORMS = {"linear", "signed-log", "gamma075", "asinh"}


def parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(",") if part.strip())


def parse_floats(text: str) -> tuple[float, ...]:
    return tuple(float(part) for part in text.split(",") if part.strip())


def parse_strings(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in text.split(",") if part.strip())


def safe_name(text: str) -> str:
    return (
        text.replace("*", "star")
        .replace("/", "_")
        .replace(",", "-")
        .replace(".", "_")
    )


def estimate_base_payload(
    pixels: np.ndarray,
    base_bits: int,
    transform: str,
    power_gamma: float,
    predictor: str,
    encode_base: bool,
    cache: dict[tuple, dict],
) -> dict:
    key = (base_bits, transform, power_gamma, predictor, encode_base)
    if key in cache:
        return cache[key]

    indices, decoded, _ = quantize_signed_log(
        pixels,
        base_bits,
        transform,
        power_gamma,
    )
    stats = residual_stats(indices, base_bits, predictor)
    estimated_bytes = int(
        math.ceil(
            stats["residual_order0_entropy_bits_per_sample"] * pixels.size / 8.0
        )
        + 128
    )
    row = {
        "base_bits": base_bits,
        "transform": transform,
        "power_gamma": power_gamma,
        "predictor": predictor,
        "estimated_bytes": estimated_bytes,
        "estimated_bps": stats["residual_order0_entropy_bits_per_sample"],
        "nonzero_rate": stats["nonzero_rate"],
        "encoded_bytes": None,
    }
    if encode_base and transform in BUILTIN_TRANSFORMS:
        encoded = radiance_codec.encode_linear_index_near_lossless(
            pixels,
            bits=base_bits,
            effort=9,
            transform=transform,
        )
        row["encoded_bytes"] = len(encoded)
        row["estimated_bytes"] = len(encoded)
    cache[key] = row
    return row


def summarize_candidate(
    pixels: np.ndarray,
    anchor_regions: dict,
    base_bits: int,
    dark_bits: int,
    dark_max: float,
    mask_mode: str,
    mask_radius: int,
    std_step_threshold: float,
    transform: str,
    power_gamma: float,
    recon_table: str,
    predictor: str,
    encode_base: bool,
    base_cache: dict[tuple, dict],
    white: float,
    gamma: float,
) -> dict:
    started = time.perf_counter()
    decoded, route = reconstruct_mixed(
        pixels,
        base_bits,
        dark_bits,
        dark_max,
        mask_mode,
        mask_radius,
        std_step_threshold,
        transform,
        power_gamma,
        recon_table,
    )
    if recon_table != "center":
        # The base payload estimate is still the index stream; table payload is
        # tiny for the tested 8-10 bit tables and not modeled here.
        base_indices, _, _ = quantize_signed_log(
            pixels,
            base_bits,
            transform,
            power_gamma,
        )
        route["base_indices"] = base_indices
        decoded = np.ascontiguousarray(decoded)

    base = estimate_base_payload(
        pixels,
        base_bits,
        transform,
        power_gamma,
        predictor,
        encode_base,
        base_cache,
    )
    mask_summary = summarize_binary_mask(route["mask"])
    refinement = summarize_refinement(
        route["base_indices"],
        route["dark_indices"],
        route["mask"],
        base_bits,
        dark_bits,
    )
    mask_best = min(
        mask_summary["order0"],
        mask_summary["west_north"],
        key=lambda row: row["total_bytes"],
    )
    estimated_bytes = (
        int(base["estimated_bytes"])
        + ROUTER_HEADER_BYTES
        + int(mask_best["total_bytes"])
        + int(refinement["total_bytes"])
    )
    regions = candidate_metrics(pixels, decoded, white, gamma)
    gate = evaluate_gate(regions, anchor_regions, default_gate_args())
    key_metrics = summarize_key_metrics(regions, anchor_regions)
    return {
        "label": (
            f"{transform}_g{power_gamma:g}_b{base_bits}"
            f"_d{dark_bits}_max{dark_max:g}_{mask_mode}"
            f"_std{std_step_threshold:g}_r{mask_radius}_{recon_table}"
        ),
        "base_bits": base_bits,
        "dark_bits": dark_bits,
        "dark_max": dark_max,
        "mask_mode": mask_mode,
        "mask_radius": mask_radius,
        "std_step_threshold": std_step_threshold,
        "transform": transform,
        "power_gamma": power_gamma,
        "recon_table": recon_table,
        "predictor": predictor,
        "raw_bytes": int(pixels.nbytes),
        "estimated_bytes": estimated_bytes,
        "estimated_ratio": pixels.nbytes / float(estimated_bytes),
        "base": base,
        "mask": mask_summary,
        "mask_best_total_bytes": int(mask_best["total_bytes"]),
        "refinement": refinement,
        "gate": gate,
        "key_metrics": key_metrics,
        "elapsed_sec": time.perf_counter() - started,
    }


def decision_rank(decision: str) -> int:
    return {"pass": 0, "maybe": 1, "reject": 2}.get(decision, 9)


def summarize_image(
    path: Path,
    args: argparse.Namespace,
) -> dict:
    pixels = read_exr(path, args.crop_size)
    anchor = radiance_codec.quantize_linear_index(
        pixels,
        bits=args.anchor_bits,
        transform=args.anchor_transform,
    )
    anchor_regions = candidate_metrics(pixels, anchor, args.white, args.gamma)
    rows = []
    base_cache: dict[tuple, dict] = {}
    transforms = parse_strings(args.transforms)
    power_gammas = parse_floats(args.power_gammas)
    for transform in transforms:
        gamma_values = power_gammas if transform == "power" else (1.0,)
        for power_gamma in gamma_values:
            for recon_table in parse_strings(args.recon_tables):
                for base_bits in parse_ints(args.base_bits):
                    for dark_bits in parse_ints(args.dark_bits):
                        if dark_bits <= base_bits:
                            continue
                        for dark_max in parse_floats(args.dark_max):
                            for std_step_threshold in parse_floats(args.std_step_thresholds):
                                row = summarize_candidate(
                                    pixels,
                                    anchor_regions,
                                    base_bits,
                                    dark_bits,
                                    dark_max,
                                    args.mask_mode,
                                    args.mask_radius,
                                    std_step_threshold,
                                    transform,
                                    power_gamma,
                                    recon_table,
                                    args.predictor,
                                    args.encode_base,
                                    base_cache,
                                    args.white,
                                    args.gamma,
                                )
                                rows.append(row)
                                if args.verbose:
                                    print_row(row, prefix="  ")
    rows.sort(
        key=lambda row: (
            decision_rank(row["gate"]["decision"]),
            row["estimated_bytes"],
        )
    )
    return {
        "image": path.name,
        "shape": list(pixels.shape),
        "crop_size": args.crop_size,
        "white": args.white,
        "gamma": args.gamma,
        "anchor": f"quant:{args.anchor_transform}:{args.anchor_bits}",
        "rows": rows,
    }


def print_row(row: dict, prefix: str = "") -> None:
    dark = row["key_metrics"]["dark_0_0.25"]
    high = row["key_metrics"]["highlight_1_4"]
    gate = row["gate"]["decision"].upper()
    reasons = "; ".join(
        f"{item['region']} {item['metric']}={item['value']:.2e}"
        for item in row["gate"]["failures"][:2]
    )
    if not reasons:
        reasons = "ok"
    print(
        f"{prefix}{gate:6s} {row['label']:48s} "
        f"est={row['estimated_bytes']:,} "
        f"ratio={row['estimated_ratio']:.2f}x "
        f"dark={row['mask']['dark_pixel_rate']:.2%} "
        f"mask={row['mask_best_total_bytes']:,} "
        f"ref={row['refinement']['total_bytes']:,} "
        f"darkLostD={dark['lost_detail_delta_vs_anchor']:.2%} "
        f"hiLostD={high['lost_detail_delta_vs_anchor']:.2%} "
        f"- {reasons}",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="sample_DSCF0009.EXR")
    parser.add_argument("--crop-size", type=int, default=1024)
    parser.add_argument("--transforms", default="signed-log,gamma075,asinh")
    parser.add_argument("--power-gammas", default="0.75,0.8,0.85")
    parser.add_argument("--base-bits", default="8")
    parser.add_argument("--dark-bits", default="9,10")
    parser.add_argument("--dark-max", default="0.03,0.05,0.08,0.12")
    parser.add_argument("--mask-mode", choices=("luma", "banding"), default="banding")
    parser.add_argument("--mask-radius", type=int, default=2)
    parser.add_argument("--std-step-thresholds", default="0.05,0.1,0.15,0.25")
    parser.add_argument(
        "--recon-tables",
        default="center",
        help="comma list: center,value-mean,signed-log-mean",
    )
    parser.add_argument("--predictor", default="med")
    parser.add_argument("--anchor-transform", default="signed-log")
    parser.add_argument("--anchor-bits", type=int, default=10)
    parser.add_argument("--white", type=float, default=4.0)
    parser.add_argument("--gamma", type=float, default=2.2)
    parser.add_argument("--encode-base", action="store_true")
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    summaries = []
    for path in sorted(DATA_DIR.glob(args.glob)):
        print(path.name, flush=True)
        summary = summarize_image(path, args)
        summaries.append(summary)
        for row in summary["rows"][:args.top]:
            print_row(row, prefix="  ")
        if args.limit and len(summaries) >= args.limit:
            break

    if not summaries:
        print(f"no files matched: {args.glob}", flush=True)
        return 1

    if not args.no_save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        output = RESULTS_DIR / (
            f"dark_router_visual_search_{safe_name(args.glob)}"
            f"_crop{args.crop_size}_base{args.base_bits.replace(',', '-')}"
            f"_dark{args.dark_bits.replace(',', '-')}"
            f"_max{args.dark_max.replace(',', '-')}"
            f"_std{args.std_step_thresholds.replace(',', '-')}.json"
        )
        output.write_text(json.dumps(summaries, indent=2))
        print(f"\nSaved: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
