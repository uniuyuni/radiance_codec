"""
Train a small causal byte-level LM on residual bytes of the v4d pipeline
(predict + bitshuf) and report bits/byte → bits/pixel → equivalent ratio.

Data layout (created by prepare_training_data.py):
  ml/residuals/train/ph_<name>.bin   ← 54 Poly Haven HDRIs
  ml/residuals/test/*.bin            ← 13 bench-corpus images (held out)

Each .bin is the raw byte stream that would enter the entropy coder
in our v4d pipeline. The framing header has already been stripped, so
every byte the LM sees is one we'd actually encode.

Conversion to the metrics we care about:
  bits/byte (LM cross-entropy)
    × 4 bytes/pixel (float32, regardless of channel count)
    = bits/pixel of the residual stream
  ratio = raw_bpp / residual_bpp = (channels * 32) / (bpb * 4 * channels)
        = 8 / bpb
  → bpb = 1.33 corresponds to 6.0x  (PLAN_ML.md Quick-Win threshold)
  → bpb = 2.53 corresponds to 3.16x (our_v4a classical baseline)

Model: small causal dilated CNN with gated activations.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from model import ByteLM, CHANNELS, DILATIONS, EMBED_DIM, RF

ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR  = ROOT / "ml" / "residuals" / "train"
TEST_DIR   = ROOT / "ml" / "residuals" / "test"
CHECKPOINT_DIR = ROOT / "ml" / "checkpoints_causal"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = ROOT / "ml" / "train.log"

# Tee stdout to both console and log file so progress is visible
# regardless of how the process is launched.
class _Tee:
    def __init__(self, *targets):
        self._targets = targets
    def write(self, data):
        for t in self._targets:
            t.write(data)
            t.flush()
    def flush(self):
        for t in self._targets:
            t.flush()

_log_fh = open(LOG_FILE, "w", buffering=1)
sys.stdout = _Tee(sys.__stdout__, _log_fh)
sys.stderr = _Tee(sys.__stderr__, _log_fh)

# ─── hyperparameters ──────────────────────────────────────────────
CONTEXT_LEN     = 1024     # window length used for both train & eval
BATCH_SIZE      = 16
LR              = 1e-3
EPOCHS          = 40
STEPS_PER_EPOCH = 400
SEED            = 0


def pick_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ─── helpers ──────────────────────────────────────────────────────

def bpb_to_ratio(bpb: float) -> float:
    """Convert bits/byte to equivalent compression ratio against raw
    float32 (= 8 bits per source byte)."""
    if bpb <= 0:
        return float("inf")
    return 8.0 / bpb


# ─── dataset ──────────────────────────────────────────────────────

class ResidualDataset(Dataset):
    """Random crops of CONTEXT_LEN + 1 bytes from the residual files."""
    def __init__(self, file_paths: list[Path], context_len: int,
                 virtual_len: int):
        self.files: list[np.ndarray] = []
        self.weights: list[int] = []
        for p in sorted(file_paths):
            data = np.frombuffer(p.read_bytes(), dtype=np.uint8)
            if len(data) <= context_len + 1:
                continue
            self.files.append(data)
            self.weights.append(len(data) - context_len - 1)
        if not self.files:
            raise RuntimeError("no usable training files")
        self.context_len = context_len
        total = float(sum(self.weights))
        self.cum = np.cumsum(np.asarray(self.weights, dtype=np.float64)) / total
        self.virtual_len = virtual_len

    def __len__(self):
        return self.virtual_len

    def __getitem__(self, _idx):
        r = random.random()
        file_idx = int(np.searchsorted(self.cum, r))
        file_idx = min(file_idx, len(self.files) - 1)
        data = self.files[file_idx]
        start = random.randint(0, len(data) - self.context_len - 1)
        window = data[start : start + self.context_len + 1]
        ctx = torch.from_numpy(window[:-1].copy()).long()
        tgt = torch.from_numpy(window[1:].copy()).long()
        return ctx, tgt


# ─── evaluation ───────────────────────────────────────────────────

EVAL_WINDOWS_PER_FILE = 20   # sample this many windows per file for speed

def eval_file(model, data: np.ndarray, ctx_len: int, device) -> float:
    """Cross-entropy in bits/byte, estimated from random windows.

    Uses EVAL_WINDOWS_PER_FILE random windows for speed on CPU.
    Returns NaN if file is too short.
    """
    model.eval()
    if len(data) <= ctx_len + 1:
        return float("nan")
    n_avail = len(data) - ctx_len - 1
    offsets = np.random.randint(0, n_avail, size=EVAL_WINDOWS_PER_FILE)
    ctxs = []
    tgts = []
    for start in offsets:
        window = data[int(start) : int(start) + ctx_len + 1]
        ctxs.append(torch.from_numpy(window[:-1].copy()).long())
        tgts.append(torch.from_numpy(window[1:].copy()).long())
    with torch.no_grad():
        ctx = torch.stack(ctxs).to(device)
        tgt = torch.stack(tgts).to(device)
        logits = model(ctx)
        log_probs = F.log_softmax(logits, dim=1)
        lp = torch.gather(log_probs, 1, tgt.unsqueeze(1)).squeeze(1)
        total_bits = -lp.sum().item() / math.log(2)
        total_bytes = tgt.numel()
    return total_bits / max(1, total_bytes)


def eval_all_test(model, test_files: list[Path],
                  ctx_len: int, device) -> dict[str, float]:
    out = {}
    for p in test_files:
        data = np.frombuffer(p.read_bytes(), dtype=np.uint8)
        out[p.stem] = eval_file(model, data, ctx_len, device)
    return out


def order0_entropy(data: np.ndarray) -> float:
    """Reference baseline: order-0 byte entropy of the data."""
    counts = np.bincount(data, minlength=256).astype(np.float64)
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


# ─── main ─────────────────────────────────────────────────────────

def main() -> int:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Prefer MPS on Apple Silicon. Use RADIANCE_CODEC_FORCE_CPU=1 only when
    # diagnosing backend-specific failures or sharing the GPU.
    force_cpu = os.environ.get("RADIANCE_CODEC_FORCE_CPU", "0") == "1"
    device = pick_device(force_cpu=force_cpu)
    print(f"device: {device}")

    train_files = sorted(TRAIN_DIR.glob("*.bin"))
    test_files  = sorted(TEST_DIR.glob("*.bin"))
    if not train_files:
        print(f"no training files in {TRAIN_DIR}", file=sys.stderr)
        return 1
    if not test_files:
        print(f"no test files in {TEST_DIR}", file=sys.stderr)
        return 1
    print(f"train: {len(train_files)} files, "
          f"test: {len(test_files)} files")

    # ── baselines (no learning, for sanity) ────────────────────────
    # order-0 entropy across all training bytes (concatenated) lower-
    # bounds what a static rANS frequency model could achieve.
    h0_chunks = []
    train_byte_total = 0
    for p in train_files:
        d = np.frombuffer(p.read_bytes(), dtype=np.uint8)
        h0_chunks.append(np.bincount(d, minlength=256))
        train_byte_total += len(d)
    train_hist = np.sum(h0_chunks, axis=0)
    h0 = -float(np.sum(
        np.where(train_hist > 0,
                 train_hist * np.log2(np.maximum(train_hist, 1)
                                       / train_byte_total),
                 0))) / train_byte_total
    print(f"\nbaseline order-0 entropy of training set: "
          f"{h0:.3f} bpb (≈ {bpb_to_ratio(h0):.2f}x ratio)")
    print(f"our classical v4a target to beat:         "
          f"~2.53 bpb (≈ 3.16x ratio)")
    print(f"PLAN_ML Quick-Win 1 threshold:           "
          f"~1.33 bpb (≈ 6.00x ratio)")

    # ── dataset/loader ────────────────────────────────────────────
    train_set = ResidualDataset(train_files, CONTEXT_LEN,
                                  STEPS_PER_EPOCH * BATCH_SIZE)
    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, drop_last=True)

    # ── model ────────────────────────────────────────────────────
    model = ByteLM().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nmodel: {n_params/1e3:.1f}K params, "
          f"receptive field = {RF} bytes, context window = {CONTEXT_LEN}")
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)

    # ── training loop ────────────────────────────────────────────
    print(f"\nTraining for {EPOCHS} epochs, "
          f"{STEPS_PER_EPOCH} steps × batch {BATCH_SIZE} "
          f"({STEPS_PER_EPOCH*BATCH_SIZE} samples/epoch)\n")
    print(f"{'epoch':>5s} {'train_bpb':>10s} {'train_ratio':>12s}  "
          f"{'eval_bpb':>9s} {'eval_ratio':>11s}  {'time':>6s}")
    print("─" * 72)

    # ── resume from checkpoint if available ─────────────────────
    resume_path = CHECKPOINT_DIR / "resume.pt"
    history: list[dict] = []
    start_epoch = 0
    best_eval = float("inf")
    if resume_path.exists():
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        if ckpt.get("opt") is not None:
            opt.load_state_dict(ckpt["opt"])
        history = ckpt.get("history", [])
        start_epoch = ckpt.get("next_epoch", 0)
        best_eval = ckpt.get("best_eval", float("inf"))
        print(f"Resumed from epoch {start_epoch} "
              f"(best_eval={best_eval:.3f} bpb = {bpb_to_ratio(best_eval):.2f}x)")

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        t0 = time.perf_counter()
        loss_sum = 0.0
        n_loss = 0
        for step, (ctx, tgt) in enumerate(train_loader):
            if step >= STEPS_PER_EPOCH:
                break
            ctx = ctx.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
            logits = model(ctx)
            loss = F.cross_entropy(logits, tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.item()
            n_loss += 1
        train_bpb = (loss_sum / n_loss) / math.log(2)

        # Eval on full test set
        eval_results = eval_all_test(model, test_files, CONTEXT_LEN, device)
        eval_bpbs = [v for v in eval_results.values()
                     if not math.isnan(v)]
        mean_eval_bpb = float(np.mean(eval_bpbs)) if eval_bpbs else float("nan")
        epoch_time = time.perf_counter() - t0

        history.append({
            "epoch": epoch,
            "train_bpb": train_bpb,
            "train_ratio": bpb_to_ratio(train_bpb),
            "eval_bpb_mean": mean_eval_bpb,
            "eval_ratio_mean": bpb_to_ratio(mean_eval_bpb),
            "eval_bpb_per_file": eval_results,
            "time_sec": epoch_time,
        })

        is_best = mean_eval_bpb < best_eval
        marker = " ★" if is_best else ""
        if is_best:
            best_eval = mean_eval_bpb
            torch.save(model.state_dict(),
                        CHECKPOINT_DIR / "model_best.pt")

        print(f"{epoch:5d} "
              f"{train_bpb:10.3f} {bpb_to_ratio(train_bpb):11.2f}x  "
              f"{mean_eval_bpb:9.3f} {bpb_to_ratio(mean_eval_bpb):10.2f}x  "
              f"{epoch_time:6.1f}s{marker}")

        # ── save resume checkpoint every epoch ───────────────────
        torch.save({
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "history": history,
            "next_epoch": epoch + 1,
            "best_eval": best_eval,
        }, resume_path)
        # ── save metrics every epoch so progress survives crashes ─
        (CHECKPOINT_DIR / "metrics.json").write_text(json.dumps({
            "history": history,
            "h0_train": h0,
            "h0_train_ratio": bpb_to_ratio(h0),
            "n_params": n_params,
            "receptive_field": RF,
            "context_len": CONTEXT_LEN,
            "device": str(device),
        }, indent=2))

    # ── final per-file report (best model) ───────────────────────
    print("\n" + "═" * 72)
    print("Per-file eval (best model checkpoint):")
    model.load_state_dict(torch.load(CHECKPOINT_DIR / "model_best.pt"))
    final_eval = eval_all_test(model, test_files, CONTEXT_LEN, device)
    print(f"{'image':45s} {'bpb':>8s} {'ratio':>8s}")
    for name, bpb in sorted(final_eval.items()):
        ratio_str = (f"{bpb_to_ratio(bpb):.2f}x"
                     if not math.isnan(bpb) else "—")
        print(f"  {name:43s} {bpb:8.3f} {ratio_str:>8s}")

    valid_bpbs = [v for v in final_eval.values() if not math.isnan(v)]
    if valid_bpbs:
        geomean_ratio = math.exp(
            sum(math.log(bpb_to_ratio(v)) for v in valid_bpbs)
            / len(valid_bpbs))
        print(f"\n  geomean ratio over test: {geomean_ratio:.2f}x")

    # ── save ────────────────────────────────────────────────────
    torch.save(model.state_dict(), CHECKPOINT_DIR / "model_final.pt")
    (CHECKPOINT_DIR / "metrics.json").write_text(json.dumps({
        "history": history,
        "h0_train": h0,
        "h0_train_ratio": bpb_to_ratio(h0),
        "n_params": n_params,
        "receptive_field": RF,
        "context_len": CONTEXT_LEN,
        "device": str(device),
        "final_per_file": final_eval,
    }, indent=2))
    print(f"\nSaved best+final model to {CHECKPOINT_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
