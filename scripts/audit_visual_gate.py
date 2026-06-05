"""Proxy visual judge for near-lossless HDR candidates.

The human-facing failures so far were concentrated in two places:

* dark smooth gradients losing tonal steps
* highlight texture/detail getting flattened

This audit compares each candidate against the original in display space and
also against a high-quality anchor.  It is deliberately conservative: a
"reject" means "do not bother the human reviewer yet", while "maybe" means
"worth a small visual check".
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
sys.path.insert(0, str(ROOT / "codec" / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

import radiance_codec  # noqa: E402
from audit_display_quality_regions import (  # noqa: E402
    display_map,
    gradient_stats,
    luminance,
    read_exr,
    region_masks,
    scalar_error_stats,
)
from probe_darkbits_router_payload import reconstruct_mixed  # noqa: E402
from probe_channel_gamma_index import quantize_channel_gamma  # noqa: E402
from probe_parametric_transform_index import quantize_power  # noqa: E402
from probe_power_recon_table import table_reconstruct  # noqa: E402


REGION_ORDER = (
    "dark_0_0.25",
    "mid_0.25_1",
    "highlight_1_4",
    "extreme_gt4",
    "all",
)


def safe_name(text: str) -> str:
    return (
        text.replace("*", "star")
        .replace("/", "_")
        .replace(":", "_")
        .replace(",", "-")
        .replace(".", "p")
    )


def decode_candidate(pixels: np.ndarray, spec: str) -> tuple[str, np.ndarray, dict]:
    """Decode a candidate spec without requiring a full bitstream where possible."""
    parts = spec.split(":")
    started = time.perf_counter()
    if parts[0] == "quant":
        if len(parts) != 3:
            raise ValueError("quant candidate format: quant:transform:bits")
        transform = parts[1]
        bits = int(parts[2])
        decoded = radiance_codec.quantize_linear_index(
            pixels,
            bits=bits,
            transform=transform,
        )
        return (
            f"quant_{transform}_bits{bits}",
            np.ascontiguousarray(decoded),
            {
                "family": "quant",
                "transform": transform,
                "bits": bits,
                "decode_seconds": time.perf_counter() - started,
            },
        )

    if parts[0] == "codec":
        if len(parts) != 3:
            raise ValueError("codec candidate format: codec:transform:bits")
        transform = parts[1]
        bits = int(parts[2])
        t0 = time.perf_counter()
        encoded = radiance_codec.encode_linear_index_near_lossless(
            pixels,
            bits=bits,
            effort=9,
            transform=transform,
        )
        t1 = time.perf_counter()
        decoded = radiance_codec.decode(encoded, pixels.shape)
        return (
            f"codec_{transform}_bits{bits}",
            decoded,
            {
                "family": "codec",
                "transform": transform,
                "bits": bits,
                "encoded_bytes": len(encoded),
                "ratio_vs_original": pixels.nbytes / len(encoded),
                "encode_seconds": t1 - t0,
                "decode_seconds": time.perf_counter() - t1,
            },
        )

    if parts[0] == "power-table":
        if len(parts) != 4:
            raise ValueError("power-table candidate format: power-table:gamma:bits:table")
        power_gamma = float(parts[1])
        bits = int(parts[2])
        table_mode = parts[3]
        indices, decoded = quantize_power(pixels, bits, power_gamma)
        if table_mode != "center":
            decoded = table_reconstruct(pixels, indices, bits, table_mode)
        return (
            f"power{power_gamma:g}_bits{bits}_{table_mode}",
            decoded,
            {
                "family": "power-table",
                "power_gamma": power_gamma,
                "bits": bits,
                "recon_table": table_mode,
                "decode_seconds": time.perf_counter() - started,
            },
        )

    if parts[0] == "channel-power":
        if len(parts) != 6:
            raise ValueError(
                "channel-power candidate format: "
                "channel-power:bits:gamma0:gamma1:gamma2:table"
            )
        bits = int(parts[1])
        gammas = (float(parts[2]), float(parts[3]), float(parts[4]))
        table_mode = parts[5]
        indices, decoded = quantize_channel_gamma(pixels, bits, gammas)
        if table_mode != "center":
            decoded = table_reconstruct(pixels, indices, bits, table_mode)
        gamma_label = "-".join(f"{gamma:g}" for gamma in gammas)
        return (
            f"channelpower_bits{bits}_g{gamma_label}_{table_mode}",
            decoded,
            {
                "family": "channel-power",
                "bits": bits,
                "gammas": list(gammas),
                "recon_table": table_mode,
                "decode_seconds": time.perf_counter() - started,
            },
        )

    if parts[0] == "dark-router":
        if len(parts) != 8:
            raise ValueError(
                "dark-router candidate format: "
                "dark-router:transform:power_gamma:base_bits:dark_bits:dark_max:mask_mode:recon_table"
            )
        transform = parts[1]
        power_gamma = float(parts[2])
        base_bits = int(parts[3])
        dark_bits = int(parts[4])
        dark_max = float(parts[5])
        mask_mode = parts[6]
        recon_table = parts[7]
        decoded, route = reconstruct_mixed(
            pixels,
            base_bits,
            dark_bits,
            dark_max,
            mask_mode,
            mask_radius=2,
            std_step_threshold=0.25,
            transform=transform,
            power_gamma=power_gamma,
            recon_table=recon_table,
        )
        return (
            (
                f"darkrouter_{transform}_g{power_gamma:g}_bits{base_bits}"
                f"_dark{dark_bits}_max{dark_max:g}_{mask_mode}_{recon_table}"
            ),
            decoded,
            {
                "family": "dark-router",
                "transform": transform,
                "power_gamma": power_gamma,
                "base_bits": base_bits,
                "dark_bits": dark_bits,
                "dark_max": dark_max,
                "mask_mode": mask_mode,
                "recon_table": recon_table,
                "dark_pixel_rate": float(np.mean(route["mask"])),
                "decode_seconds": time.perf_counter() - started,
            },
        )

    raise ValueError(f"unknown candidate spec: {spec}")


def candidate_metrics(
    pixels: np.ndarray,
    decoded: np.ndarray,
    white: float,
    gamma: float,
) -> dict[str, dict]:
    original_display = display_map(pixels, white, gamma)
    decoded_display = display_map(decoded, white, gamma)
    original_luma = luminance(original_display)
    decoded_luma = luminance(decoded_display)
    masks = region_masks(pixels)
    regions = {}
    for name in REGION_ORDER:
        mask = masks[name]
        regions[name] = {
            "display_rgb": scalar_error_stats(
                original_display,
                decoded_display,
                np.repeat(mask[:, :, None], 3, axis=2),
            ),
            "display_luma": scalar_error_stats(original_luma, decoded_luma, mask),
            "display_luma_gradient": gradient_stats(original_luma, decoded_luma, mask),
        }
    return regions


def metric(regions: dict, region: str, group: str, key: str) -> float:
    value = regions[region][group][key]
    if isinstance(value, float) and not math.isfinite(value):
        return 0.0
    return float(value)


def add_rule(
    failures: list[dict],
    severity: str,
    region: str,
    metric_name: str,
    value: float,
    threshold: float,
    detail: str,
) -> None:
    failures.append({
        "severity": severity,
        "region": region,
        "metric": metric_name,
        "value": value,
        "threshold": threshold,
        "detail": detail,
    })


def evaluate_gate(regions: dict, anchor_regions: dict, args: argparse.Namespace) -> dict:
    failures: list[dict] = []

    dark_lost = metric(regions, "dark_0_0.25", "display_luma_gradient", "lost_detail_rate")
    anchor_dark_lost = metric(
        anchor_regions,
        "dark_0_0.25",
        "display_luma_gradient",
        "lost_detail_rate",
    )
    dark_lost_delta = dark_lost - anchor_dark_lost
    if dark_lost_delta > args.dark_lost_reject:
        add_rule(
            failures,
            "reject",
            "dark_0_0.25",
            "lost_detail_rate_delta",
            dark_lost_delta,
            args.dark_lost_reject,
            "dark gradients lose too much detail versus the anchor",
        )
    elif dark_lost_delta > args.dark_lost_warn:
        add_rule(
            failures,
            "maybe",
            "dark_0_0.25",
            "lost_detail_rate_delta",
            dark_lost_delta,
            args.dark_lost_warn,
            "dark detail loss is above the review band",
        )

    dark_grad = metric(regions, "dark_0_0.25", "display_luma_gradient", "grad_nrmse")
    anchor_dark_grad = metric(
        anchor_regions,
        "dark_0_0.25",
        "display_luma_gradient",
        "grad_nrmse",
    )
    dark_grad_limit = max(
        anchor_dark_grad * args.dark_grad_ratio_reject,
        anchor_dark_grad + args.dark_grad_abs_reject,
    )
    dark_grad_warn = max(
        anchor_dark_grad * args.dark_grad_ratio_warn,
        anchor_dark_grad + args.dark_grad_abs_warn,
    )
    if dark_grad > dark_grad_limit:
        add_rule(
            failures,
            "reject",
            "dark_0_0.25",
            "grad_nrmse",
            dark_grad,
            dark_grad_limit,
            "dark gradient error is beyond the anchor-relative limit",
        )
    elif dark_grad > dark_grad_warn:
        add_rule(
            failures,
            "maybe",
            "dark_0_0.25",
            "grad_nrmse",
            dark_grad,
            dark_grad_warn,
            "dark gradient error is close to the danger zone",
        )

    for region, warn, reject in (
        ("mid_0.25_1", args.mid_lost_warn, args.mid_lost_reject),
        ("highlight_1_4", args.highlight_lost_warn, args.highlight_lost_reject),
        ("extreme_gt4", args.extreme_lost_warn, args.extreme_lost_reject),
    ):
        lost = metric(regions, region, "display_luma_gradient", "lost_detail_rate")
        anchor_lost = metric(anchor_regions, region, "display_luma_gradient", "lost_detail_rate")
        lost_delta = lost - anchor_lost
        if lost_delta > reject:
            add_rule(
                failures,
                "reject",
                region,
                "lost_detail_rate_delta",
                lost_delta,
                reject,
                f"{region} loses too much local detail versus the anchor",
            )
        elif lost_delta > warn:
            add_rule(
                failures,
                "maybe",
                region,
                "lost_detail_rate_delta",
                lost_delta,
                warn,
                f"{region} detail loss needs a quick look",
            )

    for region in ("highlight_1_4", "extreme_gt4"):
        samples = int(regions[region]["display_luma_gradient"]["gradient_samples"])
        if samples < args.min_gradient_samples:
            continue
        energy = metric(regions, region, "display_luma_gradient", "grad_energy_ratio")
        if energy < args.highlight_energy_min or energy > args.highlight_energy_max:
            add_rule(
                failures,
                "reject",
                region,
                "grad_energy_ratio",
                energy,
                args.highlight_energy_min
                if energy < args.highlight_energy_min
                else args.highlight_energy_max,
                "highlight texture energy changed too much",
            )

    if any(item["severity"] == "reject" for item in failures):
        decision = "reject"
    elif failures:
        decision = "maybe"
    else:
        decision = "pass"
    return {
        "decision": decision,
        "failures": failures,
    }


def summarize_key_metrics(regions: dict, anchor_regions: dict) -> dict:
    summary = {}
    for region in ("dark_0_0.25", "mid_0.25_1", "highlight_1_4", "extreme_gt4"):
        grad = regions[region]["display_luma_gradient"]
        anchor_grad = anchor_regions[region]["display_luma_gradient"]
        luma = regions[region]["display_luma"]
        summary[region] = {
            "samples": regions[region]["display_luma"]["samples"],
            "luma_rmse": luma["rmse"],
            "luma_p99": luma["p99_abs"],
            "grad_nrmse": grad["grad_nrmse"],
            "grad_energy_ratio": grad["grad_energy_ratio"],
            "grad_correlation": grad["grad_correlation"],
            "lost_detail_rate": grad["lost_detail_rate"],
            "lost_detail_delta_vs_anchor": (
                grad["lost_detail_rate"] - anchor_grad["lost_detail_rate"]
            ),
        }
    return summary


def parse_candidates(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def default_gate_args() -> SimpleNamespace:
    return SimpleNamespace(
        dark_lost_warn=0.04,
        dark_lost_reject=0.10,
        dark_grad_ratio_warn=1.45,
        dark_grad_ratio_reject=2.1,
        dark_grad_abs_warn=0.035,
        dark_grad_abs_reject=0.08,
        mid_lost_warn=0.06,
        mid_lost_reject=0.12,
        highlight_lost_warn=0.035,
        highlight_lost_reject=0.075,
        extreme_lost_warn=0.06,
        extreme_lost_reject=0.14,
        highlight_energy_min=0.94,
        highlight_energy_max=1.06,
        min_gradient_samples=1024,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="sample_DSCF0009.EXR")
    parser.add_argument("--crop-size", type=int, default=1024)
    parser.add_argument("--anchor", default="quant:signed-log:10")
    parser.add_argument(
        "--candidates",
        default="power-table:0.8:8:signed-log-mean",
    )
    parser.add_argument("--white", type=float, default=4.0)
    parser.add_argument("--gamma", type=float, default=2.2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-save", action="store_true")

    defaults = default_gate_args()
    parser.add_argument("--dark-lost-warn", type=float, default=defaults.dark_lost_warn)
    parser.add_argument("--dark-lost-reject", type=float, default=defaults.dark_lost_reject)
    parser.add_argument("--dark-grad-ratio-warn", type=float, default=defaults.dark_grad_ratio_warn)
    parser.add_argument("--dark-grad-ratio-reject", type=float, default=defaults.dark_grad_ratio_reject)
    parser.add_argument("--dark-grad-abs-warn", type=float, default=defaults.dark_grad_abs_warn)
    parser.add_argument("--dark-grad-abs-reject", type=float, default=defaults.dark_grad_abs_reject)
    parser.add_argument("--mid-lost-warn", type=float, default=defaults.mid_lost_warn)
    parser.add_argument("--mid-lost-reject", type=float, default=defaults.mid_lost_reject)
    parser.add_argument("--highlight-lost-warn", type=float, default=defaults.highlight_lost_warn)
    parser.add_argument("--highlight-lost-reject", type=float, default=defaults.highlight_lost_reject)
    parser.add_argument("--extreme-lost-warn", type=float, default=defaults.extreme_lost_warn)
    parser.add_argument("--extreme-lost-reject", type=float, default=defaults.extreme_lost_reject)
    parser.add_argument("--highlight-energy-min", type=float, default=defaults.highlight_energy_min)
    parser.add_argument("--highlight-energy-max", type=float, default=defaults.highlight_energy_max)
    parser.add_argument("--min-gradient-samples", type=int, default=defaults.min_gradient_samples)
    args = parser.parse_args()

    candidate_specs = parse_candidates(args.candidates)
    rows = []
    processed = 0
    for path in sorted(DATA_DIR.glob(args.glob)):
        pixels = read_exr(path, args.crop_size)
        anchor_label, anchor_decoded, anchor_payload = decode_candidate(pixels, args.anchor)
        anchor_regions = candidate_metrics(pixels, anchor_decoded, args.white, args.gamma)
        image_rows = []
        print(path.name, flush=True)
        print(f"  anchor={anchor_label}", flush=True)
        for spec in candidate_specs:
            label, decoded, payload = decode_candidate(pixels, spec)
            regions = candidate_metrics(pixels, decoded, args.white, args.gamma)
            gate = evaluate_gate(regions, anchor_regions, args)
            key_metrics = summarize_key_metrics(regions, anchor_regions)
            row = {
                "candidate": spec,
                "label": label,
                **payload,
                "gate": gate,
                "key_metrics": key_metrics,
                "regions": regions,
            }
            image_rows.append(row)
            reasons = "; ".join(
                f"{item['region']} {item['metric']}={item['value']:.3e}>{item['threshold']:.3e}"
                for item in gate["failures"][:3]
            )
            if not reasons:
                reasons = "no visual-gate failures"
            dark = key_metrics["dark_0_0.25"]
            high = key_metrics["highlight_1_4"]
            print(
                f"  {gate['decision'].upper():6s} {label}: "
                f"dark_lost_delta={dark['lost_detail_delta_vs_anchor']:.3%} "
                f"dark_grad={dark['grad_nrmse']:.3e} "
                f"hi_lost_delta={high['lost_detail_delta_vs_anchor']:.3%} "
                f"hi_energy={high['grad_energy_ratio']:.3f} "
                f"- {reasons}",
                flush=True,
            )
        rows.append({
            "image": path.name,
            "shape": list(pixels.shape),
            "crop_size": args.crop_size,
            "white": args.white,
            "gamma": args.gamma,
            "anchor": {
                "candidate": args.anchor,
                "label": anchor_label,
                **anchor_payload,
                "key_metrics": summarize_key_metrics(anchor_regions, anchor_regions),
                "regions": anchor_regions,
            },
            "rows": image_rows,
        })
        processed += 1
        if args.limit and processed >= args.limit:
            break

    if not rows:
        print(f"no files matched: {args.glob}", flush=True)
        return 1

    if not args.no_save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        output = RESULTS_DIR / (
            f"visual_gate_{safe_name(args.glob)}"
            f"_crop{args.crop_size}_anchor{safe_name(args.anchor)}"
            f"_w{args.white:g}_g{args.gamma:g}.json"
        )
        output.write_text(json.dumps(rows, indent=2))
        print(f"\nSaved: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
