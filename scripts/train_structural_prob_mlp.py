"""Train a tiny MLP to predict residual bit probabilities from tile metadata."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "ml" / "structural_prob_data"
OUT_DIR = ROOT / "ml" / "checkpoints_structural_prob"
OUT_DIR.mkdir(exist_ok=True)


def load_npz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as data:
        return data["features"], data["ones"], data["totals"]


class ProbMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def bit_cost(ones: np.ndarray, totals: np.ndarray, probs: np.ndarray) -> float:
    probs = np.clip(probs, 1e-6, 1 - 1e-6)
    return float(
        -(
            ones * np.log2(probs)
            + (totals - ones) * np.log2(1 - probs)
        ).sum()
    )


def predict_numpy(
    model: nn.Module,
    features: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    chunks = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            xb = torch.from_numpy(features[start:start + batch_size]).to(device)
            chunks.append(torch.sigmoid(model(xb)).cpu().numpy())
    model.train()
    return np.concatenate(chunks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=262_144)
    parser.add_argument("--eval-batch-size", type=int, default=524_288)
    parser.add_argument("--steps-per-epoch", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    x_train, ones_train, totals_train = load_npz(DATA_DIR / "train.npz")
    x_test, ones_test, totals_test = load_npz(DATA_DIR / "test.npz")
    print(f"rows: train={len(ones_train)}, test={len(ones_test)}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    model = ProbMLP().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    target_train = (ones_train / totals_train).astype(np.float32, copy=False)
    mean_total = float(totals_train.mean())
    best = float("inf")
    history = []
    for epoch in range(args.epochs):
        losses = []
        for _ in range(args.steps_per_epoch):
            index = torch.randint(
                len(x_train),
                (args.batch_size,),
                generator=generator,
            ).numpy()
            xb = torch.from_numpy(x_train[index]).to(device)
            target = torch.from_numpy(target_train[index]).to(device)
            weight_np = (totals_train[index] / mean_total).astype(np.float32)
            weight = torch.from_numpy(weight_np).to(device)
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, target, weight=weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if epoch % args.eval_every == 0 or epoch == args.epochs - 1:
            probs = predict_numpy(model, x_test, device, args.eval_batch_size)
            cost = bit_cost(ones_test, totals_test, probs)
            raw_bits = float(totals_test.sum())
            ratio = raw_bits / cost
            history.append({"epoch": epoch, "test_bits": cost, "ratio": ratio})
            if cost < best:
                best = cost
                torch.save(model.state_dict(), OUT_DIR / "model_best.pt")
            print(
                f"epoch={epoch:3d} loss={np.mean(losses):.4f} "
                f"test_ratio={ratio:.3f}x"
            )
    (OUT_DIR / "metrics.json").write_text(json.dumps(history, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
