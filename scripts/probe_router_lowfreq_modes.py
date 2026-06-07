"""Compare near-lossless router payload streams across low-frequency modes."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

STREAM_LABELS = (
    "route_mask",
    "high_mask",
    "y",
    "co_low",
    "cg_low",
    "co_high",
    "cg_high",
    "signed_r",
    "signed_g",
    "signed_b",
    "dark_refine_mask",
    "dark_refine_g",
)

STREAM_METHODS = {
    0: "raw",
    1: "rans0",
    2: "rans1",
    3: "zstd",
    4: "index_symbol_rans",
    5: "mask_binary",
    6: "mask_tiled",
}

MODES = {
    "cpu_scale1": {
        "RADIANCE_CODEC_NO_METAL_GUIDED": "1",
        "RADIANCE_CODEC_NO_METAL_DOWNSAMPLE": "1",
        "RADIANCE_CODEC_NO_METAL_HIGHPASS": "1",
        "RADIANCE_CODEC_NO_METAL_VISUAL_GUARD": "1",
        "RADIANCE_CODEC_ROUTER_CPU_GUIDED_SCALE": "1",
    },
    "cpu_scale2": {
        "RADIANCE_CODEC_NO_METAL_GUIDED": "1",
        "RADIANCE_CODEC_NO_METAL_DOWNSAMPLE": "1",
        "RADIANCE_CODEC_NO_METAL_HIGHPASS": "1",
        "RADIANCE_CODEC_NO_METAL_VISUAL_GUARD": "1",
        "RADIANCE_CODEC_ROUTER_CPU_GUIDED_SCALE": "2",
    },
    "cpu_scale2_no_order1": {
        "RADIANCE_CODEC_NO_METAL_GUIDED": "1",
        "RADIANCE_CODEC_NO_METAL_DOWNSAMPLE": "1",
        "RADIANCE_CODEC_NO_METAL_HIGHPASS": "1",
        "RADIANCE_CODEC_NO_METAL_VISUAL_GUARD": "1",
        "RADIANCE_CODEC_ROUTER_CPU_GUIDED_SCALE": "2",
        "RADIANCE_CODEC_ROUTER_NO_ORDER1_STREAMS": "1",
    },
    "metal_guided": {
        "RADIANCE_CODEC_USE_METAL_GUIDED": "1",
        "RADIANCE_CODEC_NO_METAL_DOWNSAMPLE": "1",
        "RADIANCE_CODEC_NO_METAL_HIGHPASS": "1",
        "RADIANCE_CODEC_NO_METAL_VISUAL_GUARD": "1",
    },
    "metal_guided_no_order1": {
        "RADIANCE_CODEC_USE_METAL_GUIDED": "1",
        "RADIANCE_CODEC_NO_METAL_DOWNSAMPLE": "1",
        "RADIANCE_CODEC_NO_METAL_HIGHPASS": "1",
        "RADIANCE_CODEC_NO_METAL_VISUAL_GUARD": "1",
        "RADIANCE_CODEC_ROUTER_NO_ORDER1_STREAMS": "1",
    },
    "metal_guided_rans0_byte": {
        "RADIANCE_CODEC_USE_METAL_GUIDED": "1",
        "RADIANCE_CODEC_NO_METAL_DOWNSAMPLE": "1",
        "RADIANCE_CODEC_NO_METAL_HIGHPASS": "1",
        "RADIANCE_CODEC_NO_METAL_VISUAL_GUARD": "1",
        "RADIANCE_CODEC_ROUTER_RANS0_BYTE_STREAMS": "1",
    },
    "metal_all": {
        "RADIANCE_CODEC_USE_METAL_GUIDED": "1",
        "RADIANCE_CODEC_USE_METAL_DOWNSAMPLE": "1",
        "RADIANCE_CODEC_USE_METAL_HIGHPASS": "1",
        "RADIANCE_CODEC_USE_METAL_VISUAL_GUARD": "1",
    },
}


def read_u32(data: bytes, offset: int) -> tuple[int, int]:
    if offset + 4 > len(data):
        raise ValueError("unexpected end while reading u32")
    return int.from_bytes(data[offset : offset + 4], "little"), offset + 4


def skip_f32(offset: int, count: int = 1) -> int:
    return offset + 4 * count


def parse_router_payload(encoded: bytes, shape: tuple[int, int, int]) -> dict:
    inner_start = encoded.find(b"NLR1")
    if inner_start < 0:
        raise ValueError("not a near-lossless router payload")
    offset = inner_start + 4
    version = encoded[offset]
    offset += 1
    channels = encoded[offset]
    offset += 1
    y_bits = encoded[offset]
    chroma_low_bits = encoded[offset + 1]
    high_bits = encoded[offset + 2]
    anchor_bits = encoded[offset + 3]
    low_scale = encoded[offset + 4]
    offset += 5
    low_w, offset = read_u32(encoded, offset)
    low_h, offset = read_u32(encoded, offset)
    offset = skip_f32(offset, 5 * 2)
    offset = skip_f32(offset, 3 * 2)
    dark_refine_bits = encoded[offset]
    offset += 1
    offset = skip_f32(offset, 2)

    streams = []
    for label in STREAM_LABELS:
        method = encoded[offset]
        offset += 1
        payload_size, offset = read_u32(encoded, offset)
        stream_start = offset - 5
        offset += payload_size
        streams.append(
            {
                "label": label,
                "method": STREAM_METHODS.get(method, f"unknown_{method}"),
                "payload_bytes": payload_size,
                "framed_bytes": offset - stream_start,
            }
        )

    extras = []
    for c in range(3, channels):
        extra_kind = encoded[offset]
        offset += 1
        if extra_kind == 1:
            offset = skip_f32(offset, 1)
            extras.append({"channel": c, "kind": "constant", "framed_bytes": 5})
        elif extra_kind == 2:
            method = encoded[offset]
            offset += 1
            payload_size, offset = read_u32(encoded, offset)
            offset += payload_size
            extras.append(
                {
                    "channel": c,
                    "kind": "raw_stream",
                    "method": STREAM_METHODS.get(method, f"unknown_{method}"),
                    "payload_bytes": payload_size,
                    "framed_bytes": 1 + 5 + payload_size,
                }
            )
        else:
            raise ValueError(f"unknown extra channel kind: {extra_kind}")
    if offset != len(encoded):
        raise ValueError(f"payload parse ended at {offset}, expected {len(encoded)}")

    stream_total = sum(item["framed_bytes"] for item in streams)
    header_bytes = len(encoded) - stream_total - sum(item["framed_bytes"] for item in extras)
    return {
        "outer_header_bytes": inner_start,
        "version": version,
        "channels": channels,
        "shape": list(shape),
        "params": {
            "y_bits": y_bits,
            "chroma_low_bits": chroma_low_bits,
            "high_bits": high_bits,
            "anchor_bits": anchor_bits,
            "low_scale": low_scale,
            "low_w": low_w,
            "low_h": low_h,
            "dark_refine_bits": dark_refine_bits,
        },
        "header_bytes": header_bytes,
        "streams": streams,
        "extras": extras,
    }


def worker(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(ROOT / "codec" / "python"))
    sys.path.insert(0, str(ROOT / "scripts"))
    import radiance_codec  # noqa: WPS433
    from audit_display_quality_regions import read_exr  # noqa: WPS433

    pixels = read_exr(DATA_DIR / args.image, 0)
    started = time.perf_counter()
    encoded = radiance_codec.encode_near_lossless_router_v1(pixels, effort=args.effort)
    encode_s = time.perf_counter() - started
    parsed = parse_router_payload(encoded, pixels.shape)
    parsed.update(
        {
            "image": args.image,
            "mode": args.mode,
            "encoded_bytes": len(encoded),
            "encoded_mib": len(encoded) / 1048576.0,
            "ratio": pixels.nbytes / float(len(encoded)),
            "encode_s": encode_s,
        }
    )
    json.dump(parsed, sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


def run_mode(args: argparse.Namespace, mode: str, env_updates: dict[str, str]) -> dict:
    env = os.environ.copy()
    for key in [
        "RADIANCE_CODEC_USE_METAL_GUIDED",
        "RADIANCE_CODEC_USE_METAL_DOWNSAMPLE",
        "RADIANCE_CODEC_USE_METAL_HIGHPASS",
        "RADIANCE_CODEC_USE_METAL_VISUAL_GUARD",
        "RADIANCE_CODEC_NO_METAL_GUIDED",
        "RADIANCE_CODEC_NO_METAL_DOWNSAMPLE",
        "RADIANCE_CODEC_NO_METAL_HIGHPASS",
        "RADIANCE_CODEC_NO_METAL_VISUAL_GUARD",
        "RADIANCE_CODEC_ROUTER_CPU_GUIDED_SCALE",
        "RADIANCE_CODEC_ROUTER_NO_ORDER1_STREAMS",
        "RADIANCE_CODEC_ROUTER_RANS0_BYTE_STREAMS",
        "RADIANCE_CODEC_ROUTER_DARK_SMOOTH_BYPASS",
        "RADIANCE_CODEC_ROUTER_NO_DARK_SMOOTH_BYPASS",
        "RADIANCE_CODEC_ROUTER_DARK_NOISE_THRESHOLD",
    ]:
        env.pop(key, None)
    env.update(env_updates)
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--image",
        args.image,
        "--mode",
        mode,
        "--effort",
        str(args.effort),
    ]
    completed = subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def add_deltas(rows: list[dict]) -> None:
    by_mode = {row["mode"]: row for row in rows}
    base = by_mode.get("cpu_scale2") or rows[0]
    base_streams = {item["label"]: item["framed_bytes"] for item in base["streams"]}
    for row in rows:
        row["delta_vs_cpu_scale2_bytes"] = row["encoded_bytes"] - base["encoded_bytes"]
        for item in row["streams"]:
            item["delta_vs_cpu_scale2_bytes"] = (
                item["framed_bytes"] - base_streams[item["label"]]
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="sample_light_snow.EXR")
    parser.add_argument("--modes", default="cpu_scale1,cpu_scale2,metal_guided,metal_all")
    parser.add_argument("--effort", type=int, default=11)
    parser.add_argument("--output", default="")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--mode", default="")
    args = parser.parse_args()

    if args.worker:
        return worker(args)

    rows = []
    for mode in [part.strip() for part in args.modes.split(",") if part.strip()]:
        if mode not in MODES:
            raise SystemExit(f"unknown mode: {mode}")
        print(f"running {mode} on {args.image}", file=sys.stderr, flush=True)
        rows.append(run_mode(args, mode, MODES[mode]))
    add_deltas(rows)
    result = {"image": args.image, "rows": rows}

    if args.output:
        out = Path(args.output)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    json.dump(result, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
