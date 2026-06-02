"""Evaluate V2 hierarchy teacher bit cost against Blosc on all test tiles."""
from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from hierarchy_teacher import (
    HierarchyTeacher,
    bits_to_planes,
    make_teacher_sample_from_planes,
)
from hdr_hierarchy import make_pass_map

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "ml" / "hierarchy_data" / "test"
CHECKPOINT = ROOT / "ml" / "checkpoints_hierarchy" / "model_best.pt"
BASELINE_RESULTS = ROOT / "results" / "results.json"
OUTPUT = ROOT / "ml" / "hierarchy_teacher_eval.json"
BATCH_SIZE = 16


def pick_device() -> torch.device:
    if os.environ.get("RADIANCE_CODEC_FORCE_CPU", "0") == "1":
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@lru_cache(maxsize=32)
def load_tile(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    with np.load(path) as tile:
        bits = tile["bits"]
        return (
            bits_to_planes(bits),
            bits_to_planes(tile["residuals"]),
            tile["pass_map"],
            bits.shape[2],
        )


class FullEvaluationDataset(Dataset):
    def __init__(self):
        manifest = json.loads((DATA_ROOT / "manifest.json").read_text())
        self.items = []
        self.shapes = []
        self.tile_costs = []
        for item in manifest["tiles"]:
            path = str(DATA_ROOT / item["path"])
            height, width, channels = item["shape"]
            pass_map, _ = make_pass_map(height, width, int(manifest["max_step"]))
            self.tile_costs.append(
                {
                    "source": item["source"],
                    "raw_bits": int(height * width * channels * 32),
                    "anchor_bits": int(np.count_nonzero(pass_map == 0) * channels * 32),
                }
            )
            for pass_index in range(1, int(pass_map.max()) + 1):
                self.items.append((path, item["source"], pass_index))
                self.shapes.append(tuple(item["shape"][:2]))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        path, source, pass_index = self.items[index]
        bit_planes, residual_planes, pass_map, channels = load_tile(path)
        context, target, loss_mask = make_teacher_sample_from_planes(
            bit_planes, residual_planes, pass_map, pass_index, channels
        )
        return (
            torch.from_numpy(context),
            torch.from_numpy(target),
            torch.from_numpy(loss_mask),
            source,
        )


class ShapeGroupedBatchSampler:
    def __init__(self, shapes: list[tuple[int, int]], batch_size: int):
        grouped = defaultdict(list)
        for index, shape in enumerate(shapes):
            grouped[shape].append(index)
        self.batches = []
        for indices in grouped.values():
            for start in range(0, len(indices), batch_size):
                self.batches.append(indices[start:start + batch_size])

    def __iter__(self):
        return iter(self.batches)

    def __len__(self) -> int:
        return len(self.batches)


def geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def main() -> int:
    device = pick_device()
    print(f"device: {device}")
    dataset = FullEvaluationDataset()
    loader = DataLoader(
        dataset,
        batch_sampler=ShapeGroupedBatchSampler(dataset.shapes, BATCH_SIZE),
        num_workers=0,
    )
    print(f"evaluation samples: {len(dataset)}")

    model = HierarchyTeacher().to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model.eval()

    totals = defaultdict(lambda: {"raw_bits": 0, "teacher_bits": 0.0})
    for tile_cost in dataset.tile_costs:
        result = totals[tile_cost["source"]]
        result["raw_bits"] += tile_cost["raw_bits"]
        result["teacher_bits"] += tile_cost["anchor_bits"]

    with torch.no_grad():
        for batch_index, (context, target, loss_mask, sources) in enumerate(loader):
            logits = model(context.to(device, dtype=torch.float32))
            losses = F.binary_cross_entropy_with_logits(
                logits,
                target.to(device, dtype=torch.float32),
                reduction="none",
            ) / math.log(2)
            selected = losses * loss_mask.to(device)
            bits_per_sample = selected.flatten(1).sum(dim=1).cpu().numpy()
            for source, bit_cost in zip(sources, bits_per_sample):
                totals[source]["teacher_bits"] += float(bit_cost)
            if batch_index % 100 == 0:
                print(f"  batch {batch_index}/{len(loader)}")

    baseline_rows = json.loads(BASELINE_RESULTS.read_text())
    blosc = {
        row["image"]: row
        for row in baseline_rows
        if row["method"] == "blosc_zstd9_bitshuf"
    }
    output = {}
    ratios = []
    hybrid_ratios = []
    print(
        f"\n{'image':43s} {'teacher':>9s} {'Blosc':>9s} "
        f"{'vs Blosc':>10s} {'hybrid':>9s}"
    )
    print("-" * 88)
    for source, result in sorted(totals.items()):
        raw_bytes = result["raw_bits"] / 8
        teacher_bytes = result["teacher_bits"] / 8
        blosc_bytes = int(blosc[source]["out_bytes"])
        ratio = raw_bytes / teacher_bytes
        blosc_ratio = raw_bytes / blosc_bytes
        versus = teacher_bytes / blosc_bytes
        hybrid_ratio = raw_bytes / min(teacher_bytes, blosc_bytes)
        ratios.append(ratio)
        hybrid_ratios.append(hybrid_ratio)
        output[source] = {
            "raw_bytes": raw_bytes,
            "teacher_estimated_bytes": teacher_bytes,
            "blosc_bytes": blosc_bytes,
            "teacher_ratio": ratio,
            "blosc_ratio": blosc_ratio,
            "teacher_bytes_vs_blosc": versus,
            "hybrid_ratio": hybrid_ratio,
        }
        print(
            f"{Path(source).stem:43s} {ratio:8.2f}x {blosc_ratio:8.2f}x "
            f"{versus:9.1%} {hybrid_ratio:8.2f}x"
        )
    print("-" * 88)
    print(f"{'geomean':43s} {geomean(ratios):8.2f}x {'':9s} {'':10s} "
          f"{geomean(hybrid_ratios):8.2f}x")
    OUTPUT.write_text(json.dumps(output, indent=2))
    print(f"\nSaved: {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
