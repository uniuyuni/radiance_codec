"""
Download float32 HDR test images from public CC0 / open sources.

Sources:
  - Poly Haven (HDRI Haven) — CC0, 1k resolution EXRs (~3-7 MB each)
  - OpenEXR official sample images — BSD-3, technical test EXRs
"""
from __future__ import annotations

import sys
from pathlib import Path
from urllib.parse import urlparse

import requests
from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Mix of natural HDR environments (Poly Haven, CC0) and synthetic/technical
# scenes (OpenEXR project). 1k resolution keeps each file small enough
# to download in seconds while still being representative.
POLYHAVEN_HDRIS = [
    "studio_small_03",        # Studio lighting, controlled HDR
    "kloppenheim_06_puresky", # Sunset sky, smooth gradient
    "abandoned_tiled_room",   # Indoor texture, moderate HDR
    "belfast_sunset_puresky", # Bright sun + sky
    "spruit_sunrise",         # Outdoor with high dynamic range
]
POLYHAVEN_RES = "1k"  # 1024 x 512

OPENEXR_SAMPLES = [
    # path within AcademySoftwareFoundation/openexr-images
    "ScanLines/Tree.exr",
    "ScanLines/CandleGlass.exr",
    "ScanLines/Cannon.exr",
    "TestImages/GrayRampsHorizontal.exr",
    "StillLife/StillLife.exr",
]
OPENEXR_BASE = (
    "https://raw.githubusercontent.com/AcademySoftwareFoundation/"
    "openexr-images/main/"
)


def download(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  skip (exists): {dest.name}")
        return True
    try:
        with requests.get(url, stream=True, timeout=30) as r:
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
        print(f"  FAIL {dest.name}: {e}", file=sys.stderr)
        if dest.exists():
            dest.unlink()
        return False


def fetch_polyhaven() -> int:
    print(f"\n[Poly Haven] {POLYHAVEN_RES} HDRIs:")
    ok = 0
    for name in POLYHAVEN_HDRIS:
        url = (
            f"https://dl.polyhaven.org/file/ph-assets/HDRIs/exr/"
            f"{POLYHAVEN_RES}/{name}_{POLYHAVEN_RES}.exr"
        )
        dest = DATA_DIR / f"ph_{name}_{POLYHAVEN_RES}.exr"
        if download(url, dest):
            ok += 1
    return ok


def fetch_openexr_samples() -> int:
    print(f"\n[OpenEXR samples]:")
    ok = 0
    for relpath in OPENEXR_SAMPLES:
        url = OPENEXR_BASE + relpath
        flat = relpath.replace("/", "_")
        dest = DATA_DIR / f"oexr_{flat}"
        if download(url, dest):
            ok += 1
    return ok


def main() -> int:
    print(f"Downloading test data to: {DATA_DIR}")
    n1 = fetch_polyhaven()
    n2 = fetch_openexr_samples()
    print(f"\nTotal: {n1 + n2} files in {DATA_DIR}")
    # Show summary
    total_bytes = sum(p.stat().st_size for p in DATA_DIR.glob("*.exr"))
    print(f"Total size: {total_bytes / 1e6:.1f} MB")
    if n1 + n2 == 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
