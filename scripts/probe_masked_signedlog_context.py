"""Probe context coding for masked signed-log escape samples.

This targets the current best visual route:

* VST-chroma base outside a risk mask;
* faithful signed-log indices only inside the mask.

The main question is whether the signed-log mask payload can be compressed
better without changing the protected value range or visual reconstruction.
The probe uses deterministic decoder-available contexts: channel, image phase,
and mask-neighborhood state.  It also tests a small-residual + escape split.
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
from probe_adaptive_ycocg_router import build_mask  # noqa: E402
from probe_darkbits_router_payload import read_exr  # noqa: E402
from probe_vst_chroma_nr import safe_name  # noqa: E402


PROB_BITS = 14
PROB_SCALE = 1 << PROB_BITS
STREAM_FLUSH_BYTES = 4


def parse_strings(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in text.split(",") if part.strip())


def build_quantized_freq(counts: np.ndarray) -> np.ndarray:
    total = int(counts.sum())
    if total <= 0:
        return np.zeros(counts.shape, dtype=np.uint16)
    seen = counts > 0
    freqs = np.zeros(counts.shape, dtype=np.int64)
    raw = (counts[seen] * PROB_SCALE + total // 2) // total
    raw = np.clip(raw, 1, PROB_SCALE - 1)
    freqs[seen] = raw
    assigned = int(freqs.sum())
    while assigned > PROB_SCALE:
        best = int(np.argmax(freqs))
        if freqs[best] <= 1:
            break
        freqs[best] -= 1
        assigned -= 1
    while assigned < PROB_SCALE:
        best = int(np.argmax(freqs))
        freqs[best] += 1
        assigned += 1
    return freqs.astype(np.uint16)


def ideal_bits(counts: np.ndarray) -> float:
    total = float(counts.sum())
    if total <= 0.0:
        return 0.0
    nz = counts[counts > 0].astype(np.float64)
    return float(-(nz * np.log2(nz / total)).sum())


def summarize_joint(joint: np.ndarray, alphabet: int, label: str) -> dict:
    context_totals = joint.sum(axis=1)
    nonempty = np.flatnonzero(context_totals)
    samples = int(context_totals.sum())
    if samples == 0:
        return {
            "label": label,
            "sample_count": 0,
            "ideal_bits": 0.0,
            "finite_bits": 0.0,
            "payload_bytes": 0,
            "model_sparse_pairs_bytes": 0,
            "total_sparse_pairs_bytes": 0,
            "bits_per_sample": 0.0,
            "context_count_seen": 0,
            "symbol_pairs_seen": 0,
        }

    total_ideal = 0.0
    total_finite = 0.0
    symbol_pairs_seen = 0
    for cid in nonempty:
        counts = joint[cid]
        total_ideal += ideal_bits(counts)
        freqs = build_quantized_freq(counts)
        nz = counts > 0
        symbol_pairs_seen += int(np.count_nonzero(nz))
        total_finite += float(
            (
                counts[nz].astype(np.float64)
                * (PROB_BITS - np.log2(freqs[nz].astype(np.float64)))
            ).sum()
        )

    context_id_bytes = 2 if joint.shape[0] <= 65536 else 4
    symbol_id_bytes = 1 if alphabet <= 256 else 2
    model_sparse_pairs = int(
        nonempty.size * (context_id_bytes + 2)
        + symbol_pairs_seen * (symbol_id_bytes + 2)
    )
    payload_bytes = int(math.ceil(total_finite / 8.0)) + STREAM_FLUSH_BYTES
    return {
        "label": label,
        "sample_count": samples,
        "ideal_bits": total_ideal,
        "finite_bits": total_finite,
        "ideal_bytes": int(math.ceil(total_ideal / 8.0)),
        "payload_bytes": payload_bytes,
        "model_sparse_pairs_bytes": model_sparse_pairs,
        "total_sparse_pairs_bytes": payload_bytes + model_sparse_pairs,
        "bits_per_sample": total_finite / samples,
        "context_count_possible": int(joint.shape[0]),
        "context_count_seen": int(nonempty.size),
        "symbol_pairs_seen": symbol_pairs_seen,
    }


def context_count(mode: str) -> int:
    if mode == "order0":
        return 1
    if mode == "channel":
        return 3
    if mode == "phase2_channel":
        return 4 * 3
    if mode == "xtrans6_channel":
        return 36 * 3
    if mode == "maskwn_channel":
        return 4 * 3
    if mode == "phase2_maskwn_channel":
        return 4 * 4 * 3
    if mode == "xtrans6_maskwn_channel":
        return 36 * 4 * 3
    raise ValueError(f"unknown context mode: {mode}")


def context_map(mask: np.ndarray, channel: int, mode: str) -> np.ndarray:
    height, width = mask.shape
    if mode == "order0":
        return np.zeros(mask.shape, dtype=np.uint32)
    if mode == "channel":
        return np.full(mask.shape, channel, dtype=np.uint32)
    yy, xx = np.indices(mask.shape, dtype=np.uint32)
    phase2 = ((yy & 1) << 1) | (xx & 1)
    xtrans6 = (yy % 6) * 6 + (xx % 6)
    mask_u8 = mask.astype(np.uint32, copy=False)
    west = np.zeros(mask.shape, dtype=np.uint32)
    north = np.zeros(mask.shape, dtype=np.uint32)
    west[:, 1:] = mask_u8[:, :-1]
    north[1:, :] = mask_u8[:-1, :]
    maskwn = west + 2 * north
    if mode == "phase2_channel":
        return phase2 + 4 * channel
    if mode == "xtrans6_channel":
        return xtrans6 + 36 * channel
    if mode == "maskwn_channel":
        return maskwn + 4 * channel
    if mode == "phase2_maskwn_channel":
        return phase2 + 4 * maskwn + 16 * channel
    if mode == "xtrans6_maskwn_channel":
        return xtrans6 + 36 * maskwn + 144 * channel
    raise ValueError(f"unknown context mode: {mode}")


def residual_symbols(indices: np.ndarray, bits: int, channel: int) -> tuple[np.ndarray, np.ndarray]:
    alphabet = 1 << bits
    pred = predict_for_channel(indices, channel, "med")
    residual = (
        indices[:, :, channel].astype(np.int32) + alphabet - pred
    ) & (alphabet - 1)
    signed = residual.astype(np.int16)
    signed[signed >= (alphabet // 2)] -= alphabet
    return residual.astype(np.uint16), signed


def add_selected(
    joint: np.ndarray,
    symbols: np.ndarray,
    contexts: np.ndarray,
    mask: np.ndarray,
    alphabet: int,
) -> None:
    if not np.any(mask):
        return
    flat = contexts[mask].astype(np.int64) * alphabet + symbols[mask].astype(np.int64)
    counts = np.bincount(flat, minlength=joint.size).reshape(joint.shape)
    joint += counts


def split_small(signed: np.ndarray, small: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    category = np.zeros(signed.shape, dtype=np.uint16)
    abs_signed = np.abs(signed).astype(np.uint16)
    pos_small = (signed > 0) & (signed <= small)
    neg_small = (signed < 0) & (signed >= -small)
    category[pos_small] = abs_signed[pos_small]
    category[neg_small] = small + abs_signed[neg_small]
    pos_escape = signed > small
    neg_escape = signed < -small
    category[pos_escape] = 2 * small + 1
    category[neg_escape] = 2 * small + 2
    pos_detail = (signed.astype(np.int32) - small - 1).clip(min=0).astype(np.uint16)
    neg_detail = ((-signed.astype(np.int32)) - small - 1).clip(min=0).astype(np.uint16)
    return category, pos_escape, neg_escape, pos_detail, neg_detail


def summarize_context_mode(
    indices: np.ndarray,
    bits: int,
    mask: np.ndarray,
    mode: str,
    small: int,
) -> dict:
    residual_alphabet = 1 << bits
    category_alphabet = 2 * small + 3
    detail_alphabet = max(1, (1 << (bits - 1)) - small)
    contexts = context_count(mode)
    residual_joint = np.zeros((contexts, residual_alphabet), dtype=np.int64)
    category_joint = np.zeros((contexts, category_alphabet), dtype=np.int64)
    pos_joint = np.zeros((contexts, detail_alphabet), dtype=np.int64)
    neg_joint = np.zeros((contexts, detail_alphabet), dtype=np.int64)

    for channel in range(indices.shape[2]):
        residual, signed = residual_symbols(indices, bits, channel)
        ctx = context_map(mask, channel, mode)
        add_selected(residual_joint, residual, ctx, mask, residual_alphabet)
        category, pos_escape, neg_escape, pos_detail, neg_detail = split_small(signed, small)
        add_selected(category_joint, category, ctx, mask, category_alphabet)
        add_selected(pos_joint, pos_detail, ctx, mask & pos_escape, detail_alphabet)
        add_selected(neg_joint, neg_detail, ctx, mask & neg_escape, detail_alphabet)

    residual_summary = summarize_joint(residual_joint, residual_alphabet, f"residual:{mode}")
    streams = [
        summarize_joint(category_joint, category_alphabet, f"category:{mode}"),
        summarize_joint(pos_joint, detail_alphabet, f"pos_escape:{mode}"),
        summarize_joint(neg_joint, detail_alphabet, f"neg_escape:{mode}"),
    ]
    split_payload = sum(int(row["payload_bytes"]) for row in streams)
    split_model = sum(int(row["model_sparse_pairs_bytes"]) for row in streams)
    split_total = split_payload + split_model
    split_ideal = sum(float(row["ideal_bits"]) for row in streams)
    return {
        "context": mode,
        "direct_residual": residual_summary,
        "small_escape": {
            "small": small,
            "streams": streams,
            "ideal_bytes": int(math.ceil(split_ideal / 8.0)),
            "payload_bytes": split_payload,
            "model_sparse_pairs_bytes": split_model,
            "total_sparse_pairs_bytes": split_total,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="sample_DSCF0009.EXR")
    parser.add_argument("--crop-size", type=int, default=0)
    parser.add_argument("--bits", type=int, default=10)
    parser.add_argument("--small", type=int, default=7)
    parser.add_argument("--mask-mode", default="dark-smooth")
    parser.add_argument("--dark-max", type=float, default=0.5)
    parser.add_argument("--mask-radius", type=int, default=2)
    parser.add_argument("--smooth-threshold", type=float, default=0.0025)
    parser.add_argument(
        "--contexts",
        default="order0,channel,phase2_channel,xtrans6_channel,maskwn_channel,phase2_maskwn_channel,xtrans6_maskwn_channel",
    )
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
    indices = quantize_indices(pixels[:, :, :3], args.bits, "signed-log")
    rows = [
        summarize_context_mode(indices, args.bits, mask, mode, args.small)
        for mode in parse_strings(args.contexts)
    ]
    rows.sort(
        key=lambda row: (
            row["direct_residual"]["ideal_bytes"],
            row["direct_residual"]["total_sparse_pairs_bytes"],
        )
    )
    summary = {
        "image": args.input,
        "shape": list(pixels.shape),
        "bits": args.bits,
        "small": args.small,
        "mask_mode": args.mask_mode,
        "dark_max": args.dark_max,
        "mask_radius": args.mask_radius,
        "smooth_threshold": args.smooth_threshold,
        "mask_pixel_rate": float(np.mean(mask)),
        "mask_samples": int(np.count_nonzero(mask)),
        "rows": rows,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    crop_label = "full" if args.crop_size == 0 else f"crop{args.crop_size}"
    output = RESULTS_DIR / (
        f"masked_signedlog_context_{safe_name(path.stem)}_{crop_label}"
        f"_bits{args.bits}_{args.mask_mode}{args.dark_max:g}_st{args.smooth_threshold:g}.json"
    )
    output.write_text(json.dumps(summary, indent=2))
    print(
        f"mask={summary['mask_pixel_rate']:.2%} samples={summary['mask_samples']:,}"
    )
    for row in rows:
        direct = row["direct_residual"]
        split = row["small_escape"]
        print(
            f"  {row['context']:28s} "
            f"direct ideal={direct['ideal_bytes']:,} "
            f"payload={direct['payload_bytes']:,} "
            f"model={direct['model_sparse_pairs_bytes']:,} "
            f"total={direct['total_sparse_pairs_bytes']:,} "
            f"split ideal={split['ideal_bytes']:,} "
            f"split total={split['total_sparse_pairs_bytes']:,}"
        )
    print(f"summary={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
