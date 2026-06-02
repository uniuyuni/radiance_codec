"""
Download a diverse Poly Haven HDRI corpus for ML training.

Selection strategy:
  - Stratify by category (indoor / outdoor / sky / studio / etc.) so
    the trained model sees variety
  - Pick up to N_PER_CATEGORY from each major category
  - Skip anything already in bench/data/ (those are our held-out test set)
  - 1k resolution keeps each file small (~1-5 MB)

Output:
  ml/training_data/*.exr  ← training corpus, NEVER touched by bench
"""
from __future__ import annotations

import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT / "ml" / "training_data"
TRAIN_DIR.mkdir(parents=True, exist_ok=True)
TEST_DIR = ROOT / "data"

API_BASE = "https://api.polyhaven.com"
CDN_URL  = "https://dl.polyhaven.org/file/ph-assets/HDRIs/exr/{res}/{name}_{res}.exr"
RES      = "1k"          # 1024x512 = ~1-5 MB per file

# How many to pick from each category. The Poly Haven taxonomy is
# multi-label so a single file may match several categories; we count
# usage per file to avoid duplicates.
N_PER_CATEGORY = 8
TARGET_CATEGORIES = [
    "indoor",
    "outdoor",
    "natural light",
    "artificial light",
    "studio",
    "nature",
    "urban",
    "skies",
    "sunrise-sunset",
    "morning-afternoon",
    "night",
    "high contrast",
    "low contrast",
]


def existing_test_names() -> set[str]:
    """Names already in our test set, to avoid leakage."""
    out = set()
    for p in TEST_DIR.glob("ph_*.exr"):
        # ph_<name>_1k.exr → strip prefix + suffix
        stem = p.stem
        if stem.startswith("ph_") and stem.endswith(f"_{RES}"):
            out.add(stem[len("ph_"):-len(f"_{RES}")])
    return out


def fetch_catalog() -> dict:
    print("Fetching Poly Haven HDRI catalog...")
    r = requests.get(f"{API_BASE}/assets?t=hdris", timeout=30)
    r.raise_for_status()
    return r.json()


def stratify(catalog: dict, skip: set[str]) -> list[str]:
    """Return a deduplicated list of file names covering target categories."""
    by_cat: dict[str, list[str]] = defaultdict(list)
    for name, meta in catalog.items():
        if name in skip:
            continue
        for cat in meta.get("categories", []):
            if cat in TARGET_CATEGORIES:
                by_cat[cat].append(name)

    selected: dict[str, None] = {}  # ordered set
    for cat in TARGET_CATEGORIES:
        items = sorted(by_cat.get(cat, []))    # deterministic
        for n in items[:N_PER_CATEGORY]:
            selected.setdefault(n)
    return list(selected.keys())


def download(name: str, dest: Path) -> bool:
    url = CDN_URL.format(res=RES, name=name)
    if dest.exists() and dest.stat().st_size > 0:
        return True
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(dest, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True,
                desc=dest.name, leave=False,
            ) as pbar:
                for chunk in r.iter_content(chunk_size=64 * 1024):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        return True
    except Exception as e:
        print(f"  FAIL {name}: {e}", file=sys.stderr)
        if dest.exists():
            dest.unlink()
        return False


def main() -> int:
    skip = existing_test_names()
    print(f"Held-out test names (skipped): {len(skip)}: {sorted(skip)}")

    catalog = fetch_catalog()
    print(f"Catalog size: {len(catalog)}")

    selected = stratify(catalog, skip)
    print(f"\nSelected {len(selected)} files for training "
          f"(targeting {N_PER_CATEGORY}/category over "
          f"{len(TARGET_CATEGORIES)} categories):")
    for n in selected:
        cats = catalog[n].get("categories", [])
        print(f"  {n:40s}  {cats}")

    # Show estimated total size (rough; 1k EXRs average ~2-4 MB)
    print(f"\nEstimated total: ~{len(selected) * 3} MB")

    print("\nDownloading...")
    ok = 0
    for name in selected:
        dest = TRAIN_DIR / f"ph_{name}_{RES}.exr"
        if download(name, dest):
            ok += 1
        # Be polite to the CDN
        time.sleep(0.1)

    total_bytes = sum(p.stat().st_size for p in TRAIN_DIR.glob("*.exr"))
    print(f"\nDownloaded {ok}/{len(selected)} files")
    print(f"Total size: {total_bytes / 1e6:.1f} MB")
    print(f"Saved to {TRAIN_DIR}/")
    return 0 if ok > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
