"""ML + rANS proof of concept for causal residual-byte compression.

This prototype intentionally runs on CPU. A neural entropy codec must produce
the same quantized probability model during encode and decode; cross-device
determinism needs a separate deployment design before using GPU inference.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import constriction
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "codec" / "python"))
import radiance_codec  # type: ignore  # noqa: E402

from model import ByteLM, RF  # noqa: E402

CHECKPOINT_DIR = ROOT / "ml" / "checkpoints_causal"
RESIDUAL_DIR = ROOT / "ml" / "residuals" / "test"
FRAMING_BYTES = 21 + 6
PROB_BITS = 14
PROB_SCALE = 1 << PROB_BITS
UNIFORM = np.full(256, 1.0 / 256, dtype=np.float32)


def quantize_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """Map float PMFs to deterministic fixed-point PMFs."""
    original_shape = probabilities.shape
    rows = np.asarray(probabilities, dtype=np.float32).reshape(-1, 256)
    quantized = np.maximum(
        np.floor(rows * PROB_SCALE).astype(np.int32), 1
    )
    differences = PROB_SCALE - quantized.sum(axis=1)
    for row_idx, difference in enumerate(differences):
        if difference > 0:
            quantized[row_idx, int(np.argmax(rows[row_idx]))] += difference
            continue
        remaining = int(-difference)
        if remaining == 0:
            continue
        for symbol in np.argsort(-quantized[row_idx]):
            removable = int(quantized[row_idx, symbol] - 1)
            removed = min(removable, remaining)
            quantized[row_idx, symbol] -= removed
            remaining -= removed
            if remaining == 0:
                break
        if remaining:
            raise RuntimeError("failed to normalize fixed-point PMF")
    if not np.all(quantized.sum(axis=1) == PROB_SCALE):
        raise RuntimeError("invalid fixed-point PMF total")
    return (quantized.astype(np.float32) / PROB_SCALE).reshape(original_shape)


def infer_probabilities(
    model: ByteLM,
    residuals: np.ndarray,
    start: int,
    end: int,
) -> np.ndarray:
    """Return P(residuals[i] | residuals[:i]) for i in [start, end)."""
    if start < 0 or end < start or end > len(residuals):
        raise ValueError(f"invalid probability range [{start}, {end})")
    probs = np.empty((end - start, 256), dtype=np.float32)
    predict_start = start
    if start == 0 and end > 0:
        probs[0] = UNIFORM
        predict_start = 1
    if predict_start == end:
        return probs

    context_start = max(0, predict_start - RF)
    context_end = end - 1
    context = torch.from_numpy(
        residuals[context_start:context_end].copy()
    ).long().unsqueeze(0)
    with torch.no_grad():
        logits = model(context)
        block_probs = F.softmax(logits, dim=1).squeeze(0).T.cpu().numpy()

    offset = predict_start - 1 - context_start
    count = end - predict_start
    probs[predict_start - start:] = block_probs[offset:offset + count]
    return quantize_probabilities(probs)


def encode_residuals(
    model: ByteLM,
    residuals: np.ndarray,
    block_size: int,
) -> tuple[np.ndarray, int, float, float]:
    """Encode residual bytes in bounded-memory blocks."""
    model_family = constriction.stream.model.Categorical(perfect=False)
    coder = constriction.stream.stack.AnsCoder()
    ce_bits = 0.0
    t0 = time.perf_counter()
    starts = range(0, len(residuals), block_size)
    for start in reversed(list(starts)):
        end = min(start + block_size, len(residuals))
        probs = infer_probabilities(model, residuals, start, end)
        symbols = residuals[start:end].astype(np.int32)
        selected = probs[np.arange(len(symbols)), symbols]
        ce_bits -= float(np.log2(np.maximum(selected, 1e-30)).sum())
        coder.encode_reverse(symbols, model_family, probs)
    elapsed = time.perf_counter() - t0
    return coder.get_compressed(), coder.num_valid_bits(), ce_bits, elapsed


def decode_residuals(
    model: ByteLM,
    compressed_words: np.ndarray,
    n_bytes: int,
) -> tuple[np.ndarray, float]:
    """Decode sequentially. This is deliberately simple, not yet optimized."""
    coder = constriction.stream.stack.AnsCoder(compressed_words)
    decoded = np.empty(n_bytes, dtype=np.uint8)
    t0 = time.perf_counter()
    for i in range(n_bytes):
        if i == 0:
            probs = UNIFORM
        else:
            probs = infer_probabilities(model, decoded, i, i + 1)[0]
        categorical = constriction.stream.model.Categorical(
            probs, perfect=False
        )
        decoded[i] = coder.decode(categorical)
    elapsed = time.perf_counter() - t0
    if not coder.is_empty():
        raise RuntimeError("rANS decoder has trailing state")
    return decoded, elapsed


def residuals_from_image(arr: np.ndarray) -> tuple[np.ndarray, bytes]:
    stages = radiance_codec.Stage.SPATIAL_PREDICT | radiance_codec.Stage.BITSHUFFLE
    framed = radiance_codec.encode(arr, stages=stages)
    if len(framed) != FRAMING_BYTES + arr.nbytes:
        raise RuntimeError("unexpected predict+bitshuffle framing size")
    return np.frombuffer(framed[FRAMING_BYTES:], dtype=np.uint8), framed


def image_from_residuals(
    residuals: np.ndarray,
    framed_template: bytes,
    shape: tuple[int, ...],
) -> np.ndarray:
    framed = framed_template[:FRAMING_BYTES] + residuals.tobytes()
    return radiance_codec.decode(framed, shape)


def load_model(checkpoint: Path) -> ByteLM:
    model = ByteLM()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()
    return model


def run_causality_check(model: ByteLM) -> None:
    rng = np.random.default_rng(0)
    prefix = torch.from_numpy(rng.integers(0, 256, 64, dtype=np.uint8))\
        .long().unsqueeze(0)
    suffix = torch.from_numpy(rng.integers(0, 256, 64, dtype=np.uint8))\
        .long().unsqueeze(0)
    with torch.no_grad():
        prefix_logits = model(prefix)
        extended_logits = model(torch.cat([prefix, suffix], dim=1))[:, :, :64]
    max_diff = float((prefix_logits - extended_logits).abs().max())
    if max_diff != 0.0:
        raise RuntimeError(f"model is not causal: max prefix diff {max_diff}")
    print(f"causality: OK (max prefix diff {max_diff:.1f})")


def run_stream_roundtrip(
    model: ByteLM,
    residual_file: Path,
    n_bytes: int,
    block_size: int,
) -> None:
    residuals = np.frombuffer(residual_file.read_bytes(), dtype=np.uint8)
    residuals = residuals[:min(n_bytes, len(residuals))]
    words, valid_bits, ce_bits, encode_sec = encode_residuals(
        model, residuals, block_size
    )
    decoded, decode_sec = decode_residuals(model, words, len(residuals))
    if not np.array_equal(residuals, decoded):
        raise RuntimeError("residual stream round-trip mismatch")
    disk_bytes = words.nbytes
    ratio = len(residuals) / disk_bytes if disk_bytes else float("inf")
    print(
        f"stream round-trip: OK ({len(residuals)} -> {disk_bytes} bytes, "
        f"{ratio:.3f}x, {valid_bits} valid bits, "
        f"CE {ce_bits / len(residuals):.3f} bpb, "
        f"encode {encode_sec:.2f}s, decode {decode_sec:.2f}s)"
    )


def run_image_roundtrip(
    model: ByteLM,
    block_size: int,
) -> None:
    rng = np.random.default_rng(1)
    arr = rng.standard_normal((4, 8, 3), dtype=np.float32)
    residuals, framed = residuals_from_image(arr)
    words, _, _, _ = encode_residuals(model, residuals, block_size)
    decoded, _ = decode_residuals(model, words, len(residuals))
    reconstructed = image_from_residuals(decoded, framed, arr.shape)
    if arr.tobytes() != reconstructed.tobytes():
        raise RuntimeError("float32 image round-trip mismatch")
    print(f"float32 image round-trip: OK ({arr.nbytes} bytes)")


def measure_file(
    model: ByteLM,
    residual_file: Path,
    block_size: int,
    max_bytes: int | None,
) -> None:
    residuals = np.frombuffer(residual_file.read_bytes(), dtype=np.uint8)
    if max_bytes is not None:
        residuals = residuals[:max_bytes]
    words, valid_bits, ce_bits, elapsed = encode_residuals(
        model, residuals, block_size
    )
    disk_bytes = words.nbytes
    framed_bytes = FRAMING_BYTES + disk_bytes
    ratio = len(residuals) / framed_bytes
    ce_ratio = len(residuals) * 8 / ce_bits
    print(
        f"measure: {residual_file.name}: {len(residuals)} -> {disk_bytes} "
        f"payload bytes, {framed_bytes} with frame ({ratio:.3f}x), "
        f"valid={valid_bits} bits, "
        f"CE={ce_ratio:.3f}x, delta={ratio - ce_ratio:+.3f}x, "
        f"encode={elapsed:.2f}s"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINT_DIR / "model_best.pt",
    )
    parser.add_argument(
        "--residual-file",
        type=Path,
        default=RESIDUAL_DIR / "synth_mixed_1k.bin",
    )
    parser.add_argument("--block-size", type=int, default=8192)
    parser.add_argument("--roundtrip-bytes", type=int, default=1024)
    parser.add_argument("--measure-bytes", type=int)
    parser.add_argument("--skip-measure", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.checkpoint.exists():
        print(
            f"missing causal checkpoint: {args.checkpoint}\n"
            "Run `pixi run ml-train` before measuring ML+rANS.",
            file=sys.stderr,
        )
        return 2
    model = load_model(args.checkpoint)
    print(f"checkpoint: {args.checkpoint}")
    run_causality_check(model)
    run_stream_roundtrip(
        model, args.residual_file, args.roundtrip_bytes, args.block_size
    )
    run_image_roundtrip(model, args.block_size)
    if not args.skip_measure:
        measure_file(model, args.residual_file, args.block_size, args.measure_bytes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
