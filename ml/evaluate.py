"""
Precise evaluation of model_best.pt on all 13 test images.

Unlike the 20-window sampling used during training, this evaluates
EVERY non-overlapping window in each file for an accurate bpb estimate.
Also computes geomean ratio and compares to classical baselines.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model import ByteLM

ROOT = Path(__file__).resolve().parent.parent
TEST_DIR   = ROOT / "ml" / "residuals" / "test"
CKPT_DIR   = ROOT / "ml" / "checkpoints_causal"
LOG_FILE   = ROOT / "ml" / "eval.log"

# Mirror train.py constants
CONTEXT_LEN = 1024

# Tee stdout to eval.log
class _Tee:
    def __init__(self, *targets):
        self._targets = targets
    def write(self, data):
        for t in self._targets:
            t.write(data); t.flush()
    def flush(self):
        for t in self._targets: t.flush()

_log_fh = open(LOG_FILE, "w", buffering=1)
sys.stdout = _Tee(sys.__stdout__, _log_fh)
sys.stderr = _Tee(sys.__stderr__, _log_fh)

import os
sys.path.insert(0, str(Path(__file__).parent))

# ─── full evaluation ──────────────────────────────────────────────

def eval_file_full(model, data: np.ndarray, ctx_len: int,
                   device, batch_size: int = 8) -> float:
    """Exact cross-entropy over ALL non-overlapping windows, batched."""
    if len(data) <= ctx_len + 1:
        return float("nan")
    model.eval()
    offsets = list(range(0, len(data) - ctx_len - 1, ctx_len))
    total_bits  = 0.0
    total_bytes = 0
    with torch.no_grad():
        for i in range(0, len(offsets), batch_size):
            batch_offsets = offsets[i : i + batch_size]
            ctxs, tgts = [], []
            for start in batch_offsets:
                w = data[start : start + ctx_len + 1]
                ctxs.append(torch.from_numpy(w[:-1].copy()).long())
                tgts.append(torch.from_numpy(w[1:].copy()).long())
            ctx = torch.stack(ctxs).to(device)   # (B, T)
            tgt = torch.stack(tgts).to(device)   # (B, T)
            logits   = model(ctx)                  # (B, 256, T)
            log_prob = F.log_softmax(logits, dim=1)
            lp = torch.gather(log_prob, 1, tgt.unsqueeze(1)).squeeze(1)
            total_bits  += -lp.sum().item() / math.log(2)
            total_bytes += tgt.numel()
    return total_bits / max(1, total_bytes)

def bpb_to_ratio(bpb: float) -> float:
    return 8.0 / bpb if bpb > 0 else float("inf")

def geomean(xs):
    xs = [x for x in xs if x > 0 and not math.isnan(x)]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")

# ─── main ─────────────────────────────────────────────────────────

def main():
    force_cpu = os.environ.get("RADIANCE_CODEC_FORCE_CPU", "0") == "1"
    if not force_cpu and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"device: {device}")

    # Load models
    models_to_eval = []
    if (CKPT_DIR / "model_best.pt").exists():
        m = ByteLM().to(device)
        m.load_state_dict(torch.load(CKPT_DIR / "model_best.pt", map_location=device))
        models_to_eval.append(("model_best", m))
    if (CKPT_DIR / "model_final.pt").exists():
        m2 = ByteLM().to(device)
        m2.load_state_dict(torch.load(CKPT_DIR / "model_final.pt", map_location=device))
        models_to_eval.append(("model_final", m2))

    test_files = sorted(TEST_DIR.glob("*.bin"))
    print(f"Evaluating {len(test_files)} test files (FULL, all windows)\n")

    # Classical baselines for comparison
    baselines = {
        "our_v4a (classical best)": 3.16,
        "blosc_zstd9_bitshuf":      4.63,
        "xz_9":                     4.31,
    }

    results = {}
    for model_name, model in models_to_eval:
        print(f"{'='*65}")
        print(f"Model: {model_name}")
        print(f"{'='*65}")
        print(f"{'image':45s} {'bpb':>8s} {'ratio':>8s}")
        print("-" * 65)

        per_file = {}
        for p in test_files:
            data = np.frombuffer(p.read_bytes(), dtype=np.uint8)
            bpb  = eval_file_full(model, data, CONTEXT_LEN, device)
            ratio = bpb_to_ratio(bpb)
            per_file[p.stem] = {"bpb": bpb, "ratio": ratio}
            print(f"  {p.stem:43s} {bpb:8.3f}  {ratio:7.2f}x")

        ratios = [v["ratio"] for v in per_file.values() if not math.isnan(v["ratio"])]
        gm = geomean(ratios)
        print(f"\n  {'geomean ratio':43s} {'':8s}  {gm:7.2f}x")

        print(f"\n  --- vs baselines ---")
        for name, baseline in baselines.items():
            delta = gm - baseline
            sign  = "+" if delta >= 0 else ""
            print(f"  vs {name:35s}: {sign}{delta:.2f}x  "
                  f"({'BEATS' if delta >= 0 else 'below'})")

        results[model_name] = {"per_file": per_file, "geomean": gm}

    # Save
    out = ROOT / "ml" / "eval_results_causal.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
