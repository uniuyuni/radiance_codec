"""Export small visual-review PNGs for adaptive YCoCg router candidates."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs" / "previews" / "adaptive_ycocg_router"
sys.path.insert(0, str(ROOT / "codec" / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

import radiance_codec  # noqa: E402
from audit_visual_gate import (  # noqa: E402
    candidate_metrics,
    default_gate_args,
    evaluate_gate,
    summarize_key_metrics,
)
from export_linear_index_preview_png import to_preview_u16, write_png  # noqa: E402
from probe_adaptive_ycocg_router import (  # noqa: E402
    build_mask,
    quantize_asymmetric_allow_zero,
    selected_residual_cost,
)
from probe_darkbits_router_payload import read_exr, summarize_binary_mask  # noqa: E402


def parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(",") if part.strip())


def safe_stem(text: str) -> str:
    return (
        text.replace(" ", "_")
        .replace("*", "star")
        .replace(".", "_")
        .replace(",", "-")
        .replace("/", "_")
    )


def route_payload_bytes(
    base_indices: np.ndarray,
    dark_indices: np.ndarray,
    mask: np.ndarray,
    base_bits: tuple[int, int, int],
    dark_bits: tuple[int, int, int],
) -> dict:
    non_mask = ~mask
    base_payload = selected_residual_cost(base_indices, base_bits, non_mask)
    dark_payload = selected_residual_cost(dark_indices, dark_bits, mask)
    mask_summary = summarize_binary_mask(mask)
    mask_payload = min(
        mask_summary["order0"],
        mask_summary["west_north"],
        key=lambda row: row["total_bytes"],
    )
    total_bytes = (
        int(base_payload["estimated_bytes"])
        + int(dark_payload["estimated_bytes"])
        + int(mask_payload["total_bytes"])
        + 128
    )
    return {
        "estimated_bytes": total_bytes,
        "base_payload": base_payload,
        "dark_payload": dark_payload,
        "mask_payload": mask_payload,
    }


def reconstruct_route(
    pixels: np.ndarray,
    mask: np.ndarray,
    base_bits: tuple[int, int, int],
    dark_bits: tuple[int, int, int],
    base_color: str,
    dark_color: str,
    base_transform: str,
    dark_transform: str,
    power_gamma: float,
) -> tuple[np.ndarray, dict]:
    base_transforms = (base_transform, base_transform, base_transform)
    dark_transforms = (dark_transform, dark_transform, dark_transform)
    base_indices, base_decoded = quantize_asymmetric_allow_zero(
        pixels,
        base_color,
        base_bits,
        base_transforms,
        power_gamma,
    )
    dark_indices, dark_decoded = quantize_asymmetric_allow_zero(
        pixels,
        dark_color,
        dark_bits,
        dark_transforms,
        power_gamma,
    )
    decoded = np.array(base_decoded, copy=True)
    decoded[mask, :] = dark_decoded[mask, :]
    payload = route_payload_bytes(base_indices, dark_indices, mask, base_bits, dark_bits)
    return decoded, payload


def export_case(
    path: Path,
    crop_size: int,
    mask_mode: str,
    dark_max: float,
    smooth_radius: int,
    smooth_threshold: float,
    base_bits: tuple[int, int, int],
    dark_y_bits: int,
    dark_chroma_bits: int,
    base_color: str,
    dark_color: str,
    base_transform: str,
    dark_transform: str,
    power_gamma: float,
    anchor_transform: str,
    anchor_bits: int,
    white: float,
    gamma: float,
    output_dir: Path,
    decoded_only: bool,
) -> dict:
    pixels = read_exr(path, crop_size)
    mask = build_mask(pixels, mask_mode, dark_max, smooth_radius, smooth_threshold)
    dark_bits = (dark_y_bits, dark_chroma_bits, dark_chroma_bits)
    decoded, payload = reconstruct_route(
        pixels,
        mask,
        base_bits,
        dark_bits,
        base_color,
        dark_color,
        base_transform,
        dark_transform,
        power_gamma,
    )

    anchor = radiance_codec.quantize_linear_index(
        pixels,
        bits=anchor_bits,
        transform=anchor_transform,
    )
    anchor_regions = candidate_metrics(pixels, anchor, white, gamma)
    regions = candidate_metrics(pixels, decoded, white, gamma)
    gate = evaluate_gate(regions, anchor_regions, default_gate_args())
    key_metrics = summarize_key_metrics(regions, anchor_regions)

    stem = safe_stem(path.stem)
    crop_label = "full" if crop_size == 0 else f"crop{crop_size}"
    route_label = (
        f"baseY{base_bits[0]}C{base_bits[1]}"
        f"_darkY{dark_y_bits}C{dark_chroma_bits}"
        f"_{mask_mode}_dark{dark_max:g}_smooth{smooth_threshold:g}"
    )
    suffix = f"{stem}_{crop_label}_adaptive-ycocg_{route_label}_w{white:g}_g{gamma:g}"

    original_png = output_dir / f"{stem}_{crop_label}_original_w{white:g}_g{gamma:g}.png"
    decoded_png = output_dir / f"{suffix}_decoded.png"
    if not decoded_only and not original_png.exists():
        write_png(original_png, to_preview_u16(pixels, white, gamma))
    write_png(decoded_png, to_preview_u16(decoded, white, gamma))

    estimated_bytes = int(payload["estimated_bytes"])
    row = {
        "image": path.name,
        "shape": list(pixels.shape),
        "crop_size": crop_size,
        "candidate": "adaptive-ycocg",
        "base_color": base_color,
        "dark_color": dark_color,
        "base_bits": list(base_bits),
        "dark_bits": list(dark_bits),
        "base_transform": base_transform,
        "dark_transform": dark_transform,
        "power_gamma": power_gamma,
        "mask_mode": mask_mode,
        "dark_max": dark_max,
        "smooth_radius": smooth_radius,
        "smooth_threshold": smooth_threshold,
        "mask_pixel_rate": float(np.mean(mask)),
        "raw_bytes": int(pixels.nbytes),
        "estimated_bytes": estimated_bytes,
        "estimated_ratio": pixels.nbytes / float(estimated_bytes),
        "white": white,
        "gamma": gamma,
        "anchor": f"quant:{anchor_transform}:{anchor_bits}",
        "gate": gate,
        "key_metrics": key_metrics,
        "original_png": None if decoded_only else str(original_png),
        "decoded_png": str(decoded_png),
        **payload,
    }
    manifest = output_dir / f"{suffix}_manifest.json"
    manifest.write_text(json.dumps(row, indent=2))
    row["manifest"] = str(manifest)
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="sample_DSCF0009.EXR")
    parser.add_argument("--crop-size", type=int, default=1024)
    parser.add_argument("--mask-mode", choices=("dark", "dark-smooth", "dark-or-smooth"), default="dark-smooth")
    parser.add_argument("--dark-max", type=float, default=0.05)
    parser.add_argument("--smooth-radius", type=int, default=2)
    parser.add_argument("--smooth-threshold", type=float, default=0.002)
    parser.add_argument("--base-color", default="ycocg")
    parser.add_argument("--dark-color", default="ycocg")
    parser.add_argument("--base-y-bits", type=int, default=10)
    parser.add_argument("--base-chroma-bits", type=int, default=6)
    parser.add_argument("--dark-y-bits", type=int, default=10)
    parser.add_argument("--dark-chroma-bits", default="6,0")
    parser.add_argument("--base-transform", default="gamma075")
    parser.add_argument("--dark-transform", default="gamma075")
    parser.add_argument("--power-gamma", type=float, default=0.75)
    parser.add_argument("--anchor-transform", default="signed-log")
    parser.add_argument("--anchor-bits", type=int, default=10)
    parser.add_argument("--white", type=float, default=4.0)
    parser.add_argument("--gamma", type=float, default=2.2)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--decoded-only", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    base_bits = (args.base_y_bits, args.base_chroma_bits, args.base_chroma_bits)
    rows = []
    for path in sorted(DATA_DIR.glob(args.glob)):
        print(path.name, flush=True)
        for dark_chroma_bits in parse_ints(args.dark_chroma_bits):
            row = export_case(
                path=path,
                crop_size=args.crop_size,
                mask_mode=args.mask_mode,
                dark_max=args.dark_max,
                smooth_radius=args.smooth_radius,
                smooth_threshold=args.smooth_threshold,
                base_bits=base_bits,
                dark_y_bits=args.dark_y_bits,
                dark_chroma_bits=dark_chroma_bits,
                base_color=args.base_color,
                dark_color=args.dark_color,
                base_transform=args.base_transform,
                dark_transform=args.dark_transform,
                power_gamma=args.power_gamma,
                anchor_transform=args.anchor_transform,
                anchor_bits=args.anchor_bits,
                white=args.white,
                gamma=args.gamma,
                output_dir=args.output_dir,
                decoded_only=args.decoded_only,
            )
            rows.append(row)
            dark = row["key_metrics"]["dark_0_0.25"]
            high = row["key_metrics"]["highlight_1_4"]
            reasons = "; ".join(
                f"{item['region']} {item['metric']}={item['value']:.2e}"
                for item in row["gate"]["failures"][:2]
            )
            if not reasons:
                reasons = "ok"
            print(
                f"  {row['gate']['decision'].upper():6s} "
                f"darkC={dark_chroma_bits} "
                f"est={row['estimated_bytes']:,} ratio={row['estimated_ratio']:.2f}x "
                f"mask={row['mask_pixel_rate']:.2%} "
                f"darkD={dark['lost_detail_delta_vs_anchor']:.2%} "
                f"hiD={high['lost_detail_delta_vs_anchor']:.2%} - {reasons}",
                flush=True,
            )
            print(f"    decoded_png={row['decoded_png']}", flush=True)
        if args.limit and len({item["image"] for item in rows}) >= args.limit:
            break
    if not rows:
        print(f"no files matched: {args.glob}", flush=True)
        return 1

    summary_path = args.output_dir / (
        f"adaptive_ycocg_preview_{safe_stem(args.glob)}"
        f"_crop{args.crop_size}_{args.mask_mode}_dark{args.dark_max:g}"
        f"_baseY{args.base_y_bits}C{args.base_chroma_bits}"
        f"_darkY{args.dark_y_bits}C{safe_stem(args.dark_chroma_bits)}.json"
    )
    summary_path.write_text(json.dumps(rows, indent=2))
    print(f"Saved summary: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
