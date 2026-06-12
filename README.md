# radiance_codec

`radiance_codec` is a research codec for float32 HDR image data. Its primary
input format is interleaved IEEE 754 binary32 pixels, usually shaped as
`(height, width, channels)`.

The repository currently contains:

- exact lossless compression via `GroupedDelta`
- fast exact lossless compression via `ByteplaneRans`
- mantissa-based near-lossless compression
- a visual near-lossless router tuned for HDR photographs
- C++ API, C ABI, and Python ctypes bindings
- a macOS CLI package that can read/write EXR files

The file format and APIs are still research-stage and may change.

## Documents

| Document | Purpose |
|---|---|
| `README.md` | English quick guide. |
| `README_JA.md` | Japanese quick guide. |
| `docs/DETAILED_DESIGN_JA.md` | Detailed design, public APIs, frame format, stage layout. |
| `docs/CPP_NEAR_LOSSLESS_OPTIMIZATION_SUMMARY_JA.md` | C++ near-lossless router optimization notes and current defaults. |
| `docs/NEAR_LOSSLESS_QUALITY_CRITERIA_JA.md` | Quality criteria and audit policy for near-lossless modes. |
| `docs/LOSSLESS_RESEARCH_REBOOT.md` | Lossless research direction and historical notes. |

## What It Does

General-purpose compressors often treat float32 images as raw byte streams.
This codec tries to use the structure of float32 HDR data:

- ordered float representations
- tile-level prediction choices
- bitplane entropy coding
- source precision hints such as half-like or bfloat-like data
- optional low mantissa bit quantization for near-lossless use

Current practical recommendations:

| Use case | Recommendation |
|---|---|
| Fast lossless | `encode_lossless(..., preset="fast")` / `StageByteplaneRans`, `effort=5` |
| Compact fast lossless | `encode_lossless(..., preset="compact")` / `StageByteplaneRans`, `effort=6` |
| Practical lossless default | `encode_lossless(..., preset="quality")` / `StageGroupedDelta`, `effort=11` |
| Higher-compression lossless | `encode_lossless(..., preset="max")` / `StageGroupedDelta`, `effort=12` |
| Quality-focused near-lossless | `low_bits=12`, `effort=11` |
| Ratio-focused near-lossless | `low_bits=15`, `effort=11` |
| Visual HDR-photo near-lossless | `StageNearLosslessRouter`, `effort=11` |

Near-lossless modes are not bit-exact. Decoding returns the quantized or
router-reconstructed image, not the original source image.

Codec route selection is done through API options, not environment variables.
Environment variables are reserved for runtime tuning such as OpenMP thread
behavior and experimental internal feature flags.

## Requirements

The recommended development environment is pixi. `pixi.toml` defines CMake,
Ninja, Python, OpenImageIO, NumPy, OpenEXR, and other dependencies.

```bash
pixi install
```

Manual builds require at least:

- CMake 3.20+
- Ninja
- a C++20 compiler
- Python 3.11+ and NumPy for the Python binding

## Build

```bash
pixi run build
```

This runs:

```bash
cmake -G Ninja -S codec -B codec/build -DCMAKE_BUILD_TYPE=Release
cmake --build codec/build
```

On macOS the main library is:

```text
codec/build/libradiance_codec.dylib
```

On Linux it is usually:

```text
codec/build/libradiance_codec.so
```

## Test

```bash
pixi run test-codec
```

Individual C++ tests are built under `codec/build/`, for example:

```bash
codec/build/test_codec
codec/build/test_grouped_delta
codec/build/test_near_lossless_router
codec/build/test_rans
```

## Install

For C/C++ consumers:

```bash
pixi run build
pixi run cmake --install codec/build --prefix ./dist/radiance_codec
```

Installed layout:

```text
dist/radiance_codec/
  include/radiance_codec/codec.hpp
  include/radiance_codec/c_api.h
  lib/libradiance_codec.dylib
```

For Python:

```bash
pixi run install-python
```

Build a wheel:

```bash
pixi run wheel
```

`setup.py` builds `libradiance_codec` with CMake/Ninja and copies the shared
library next to `radiance_codec.py`. To use an existing library:

```bash
RADIANCE_CODEC_SKIP_CMAKE_BUILD=1 pixi run python -m pip install .
RADIANCE_CODEC_LIBRARY=/path/to/libradiance_codec.dylib python your_script.py
```

The Python binding searches for the native library in this order:

1. `RADIANCE_CODEC_LIBRARY`
2. the bundled library next to `radiance_codec.py`
3. the development-tree `codec/build` directory

Editable installs should be used after `pixi run build`:

```bash
pixi run build
pixi run install-python-editable
```

## File CLI

For simple file-level encode/decode through Python and OpenImageIO:

```bash
pixi run python scripts/radiance_codec_file.py encode input.exr output.rcodec
pixi run python scripts/radiance_codec_file.py decode output.rcodec decoded.exr
```

Modes:

```bash
pixi run python scripts/radiance_codec_file.py encode input.exr output.rcodec --mode lossless --preset quality
pixi run python scripts/radiance_codec_file.py encode input.exr output.rcodec --mode near --low-bits 12
pixi run python scripts/radiance_codec_file.py encode input.exr output.rcodec --mode router
```

## macOS CLI Package

`app/` contains a C++ macOS CLI package with EXR support and bundled dylibs.

Build the package:

```bash
pixi run bash app/package_macos.sh
```

Generated package:

```text
app/radiance-codec-macos/
  bin/radiance-codec
  lib/*.dylib
  README.md
  THIRD_PARTY_NOTICES.txt
```

Usage:

```bash
app/radiance-codec-macos/bin/radiance-codec encode input.exr output.rcodec
app/radiance-codec-macos/bin/radiance-codec decode output.rcodec decoded.exr
app/radiance-codec-macos/bin/radiance-codec info output.rcodec
```

## Python API

### Exact Lossless

```python
import numpy as np
import radiance_codec

pixels = np.random.default_rng(0).standard_normal((128, 128, 3)).astype(np.float32)

encoded = radiance_codec.encode_lossless(pixels, preset="quality")
decoded = radiance_codec.decode(encoded)

assert decoded.dtype == np.float32
assert decoded.tobytes() == pixels.tobytes()
```

Lossless presets:

| preset | stage | effort | Notes |
|---|---|---:|---|
| `fast` | `StageByteplaneRans` | 5 | Fast full-image lossless route. |
| `compact` | `StageByteplaneRans` | 6 | Slightly more compact than `fast`. |
| `balanced` | `StageGroupedDelta` | 10 | Middle ground. |
| `quality` | `StageGroupedDelta` | 11 | Practical default. |
| `max` | `StageGroupedDelta` | 12 | Heavier search. |

Input arrays may be shaped as `(height, width, channels)` or `(height, width)`.
Supported channel counts are `1..4`.

### Mantissa Near-Lossless

```python
import numpy as np
import radiance_codec

pixels = np.random.default_rng(1).standard_normal((128, 128, 3)).astype(np.float32)

low_bits = 12
encoded = radiance_codec.encode_near_lossless(pixels, low_bits=low_bits, effort=11)
decoded = radiance_codec.decode(encoded)
expected = radiance_codec.quantize_mantissa(pixels, low_bits)

assert decoded.tobytes() == expected.tobytes()
```

`low_bits` must be in `0..23`. Larger values usually improve compression ratio
but increase numerical error.

### Visual Near-Lossless Router

```python
import numpy as np
import radiance_codec

pixels = np.random.default_rng(2).standard_normal((128, 128, 3)).astype(np.float32)

encoded = radiance_codec.encode_near_lossless_router_v1(pixels, effort=11)
decoded = radiance_codec.decode(encoded)

assert decoded.shape == pixels.shape
assert decoded.dtype == np.float32
```

On Apple/Metal systems, guided filtering, downsampling, high-pass, and visual
guard paths are enabled by default. CPU fallback is used when Metal is not
available. `dark smooth bypass` is disabled by default because it can affect
shadow gradation and color stability.

Useful experimental flags:

```bash
RADIANCE_CODEC_NO_METAL_GUIDED=1
RADIANCE_CODEC_NO_METAL_DOWNSAMPLE=1
RADIANCE_CODEC_NO_METAL_HIGHPASS=1
RADIANCE_CODEC_NO_METAL_VISUAL_GUARD=1
RADIANCE_CODEC_ROUTER_DARK_SMOOTH_BYPASS=1
```

The Python binding sets `OMP_WAIT_POLICY=PASSIVE` and `KMP_BLOCKTIME=0` when
they are not already set, reducing OpenMP worker spin and encode-time jitter.

## License

`radiance_codec` is distributed under the BSD 3-Clause License. See
`LICENSE`.

The macOS CLI package bundles third-party dynamic libraries. See
`app/THIRD_PARTY_NOTICES.txt` and the packaged `THIRD_PARTY_NOTICES.txt` for
their licenses and source package references.

## C++ API

Include:

```cpp
#include <radiance_codec/codec.hpp>
```

Minimal example:

```cpp
#include <radiance_codec/codec.hpp>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

int main() {
    constexpr std::uint32_t W = 128;
    constexpr std::uint32_t H = 128;
    constexpr std::uint8_t C = 3;

    radiance_codec::ImageMeta meta{
        .width = W,
        .height = H,
        .channels = C,
        .format = radiance_codec::PixelFormat::Float32,
    };

    std::vector<float> pixels(W * H * C, 0.5f);
    std::vector<std::uint8_t> raw(meta.raw_size());
    std::memcpy(raw.data(), pixels.data(), raw.size());

    radiance_codec::PipelineConfig cfg{
        .stages = radiance_codec::StageGroupedDelta,
        .effort = 11,
        .rans_mode = 1,
    };

    std::vector<std::uint8_t> compressed;
    if (radiance_codec::encode(raw, meta, cfg, compressed)
        != radiance_codec::Status::Ok) {
        return 1;
    }

    std::vector<std::uint8_t> decoded;
    radiance_codec::ImageMeta decoded_meta;
    if (radiance_codec::decode(compressed, decoded, &decoded_meta)
        != radiance_codec::Status::Ok) {
        return 1;
    }

    if (decoded != raw) return 1;

    std::cout << raw.size() << " -> " << compressed.size() << " bytes\n";
    return 0;
}
```

Decode uses the stage and image metadata stored in the frame header. A legacy
overload that cross-checks caller-provided metadata is still available.

For CMake consumers:

```cmake
find_package(radiance_codec CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE radiance_codec::radiance_codec)
```

## C ABI

```c
#include <radiance_codec/c_api.h>
```

Main functions:

```c
radiance_codec_encode(...)
radiance_codec_decode(...)
radiance_codec_decode_auto(...)
radiance_codec_near_lossless_router_v1_reconstruct(...)
radiance_codec_buffer_free(...)
radiance_codec_version()
```

Buffers returned by `radiance_codec_encode`, `radiance_codec_decode`, and
`radiance_codec_decode_auto` are allocated by the library and must be released
with `radiance_codec_buffer_free`.

New C ABI callers should generally use `radiance_codec_decode_auto`, which reads
image metadata from the frame header and can optionally return it via `meta_out`.

## Stages

| Stage | Purpose |
|---|---|
| `StageNone` | Passthrough / smoke tests. |
| `StageRans` | Byte-stream rANS. |
| `StageBitshuffle | StageRans` | Bitshuffle followed by rANS. |
| `StageByteplaneRans` | Fast chunked float32 byteplane lossless codec. |
| `StageGroupedDelta` | Current primary exact-lossless codec. |
| `StageMantissaQuantize` | Near-lossless pre-quantization stage. |
| `StageLinearIndex` | Transform-index near-lossless experiment. |
| `StageNearLosslessRouter` | Visual near-lossless router for HDR photos. |
| `StageStructuralContext` | Older structural-context experiment. |

Mantissa near-lossless mode uses `StageMantissaQuantize | StageGroupedDelta`.
The visual near-lossless router uses `StageNearLosslessRouter` by itself.

## Notes

- This is a research codec. The file format and stage IDs may change.
- Decode reads width, height, channels, and stage config from the frame header.
- Near-lossless modes are not bit-exact.
- The Python wheel bundles the CMake-built shared library.
- CMake install provides `find_package(radiance_codec CONFIG)`.
- Public PyPI-style packaging, dependency auditing, and ABI compatibility policy
  are not finalized yet.
