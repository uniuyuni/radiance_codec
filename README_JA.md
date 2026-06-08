# radiance_codec 日本語クイックガイド

`radiance_codec` は、float32 HDR 画像を対象にした研究用の画像圧縮ライブラリです。
現在の主な入力は IEEE 754 binary32 の interleaved float32 画像データです。

このリポジトリでは、完全 lossless の `GroupedDelta` と、低 mantissa bit を
制御して圧縮率を上げる near-lossless mode、HDR写真向けの visual near-lossless
router を実装しています。

## 概要

- C++20 の shared library として `libradiance_codec` をビルドする。
- C++ API、C ABI、Python ctypes binding から利用できる。
- 入力は主に `(height, width, channels)` の interleaved float32 HDR画像。
- 完全lossless、mantissa near-lossless、visual near-lossless router を切り替えて使う。
- Apple/Metal 環境では router の guided/downsample/high-pass/visual-guard が
  デフォルトで有効になる。非Metal環境ではCPU fallbackを使う。

## ドキュメント案内

| 文書 | 内容 |
|---|---|
| `README_JA.md` | 導入、ビルド、APIの最短ガイド。 |
| `docs/DETAILED_DESIGN_JA.md` | 全体設計、公開API、frame format、stage構成、運用上の注意。 |
| `docs/CPP_NEAR_LOSSLESS_OPTIMIZATION_SUMMARY_JA.md` | C++ near-lossless router の最適化ログと現行default。 |
| `docs/NEAR_LOSSLESS_QUALITY_CRITERIA_JA.md` | near-lossless の品質基準と監査方針。 |
| `docs/LOSSLESS_RESEARCH_REBOOT.md` | lossless 系研究の大きな流れ。 |

## 何をするライブラリか

通常の圧縮器は float32 画像を byte 列として扱いがちですが、この codec は
float32 の構造を使います。

- float32 を数値順に近い整数表現へ変換する。
- tile ごとに予測方式を選ぶ。
- residual を bitplane として entropy coding する。
- half-like / bfloat-like なソース精度も利用する。
- near-lossless では、有限 float32 の低 mantissa bit を 0 にしてから圧縮する。

現在の実用上のおすすめは以下です。

| 用途 | 推奨 |
|---|---|
| 高速 lossless | `encode_lossless(..., preset="fast")` / `StageByteplaneRans`, `effort=5` |
| 高速・圧縮率寄り lossless | `encode_lossless(..., preset="compact")` / `StageByteplaneRans`, `effort=6` |
| 実用 lossless | `encode_lossless(..., preset="quality")` / `StageGroupedDelta`, `effort=11` |
| 最大寄り lossless | `encode_lossless(..., preset="max")` / `StageGroupedDelta`, `effort=12` |
| 品質重視 near-lossless | `low_bits=12`, `effort=11` |
| 圧縮率重視 near-lossless | `low_bits=15`, `effort=11` |
| 視覚品質重視のHDR写真 near-lossless | `StageNearLosslessRouter`, `effort=11` |

near-lossless は完全復元ではありません。decode 結果は元画像ではなく、
低 mantissa bit を 0 にした量子化後画像になります。

codec の方式選択は環境変数ではなく API オプションで行います。Python では
`radiance_codec.encode(..., stages=..., effort=..., rans_mode=...)`、または
lossless 用の `encode_lossless(..., preset=...)` を使います。C++ では
`PipelineConfig` の `stages` / `effort` / `rans_mode` を指定します。
環境変数は `OMP_NUM_THREADS` などのスレッド数調整や、実験的な内部挙動の
切り替えに使います。

## 必要なもの

推奨環境は pixi です。`pixi.toml` に CMake、Ninja、Python、OpenImageIO、
NumPy などの依存関係が定義されています。

```bash
pixi install
```

手動でビルドする場合は、少なくとも以下が必要です。

- CMake 3.20 以上
- Ninja
- C++20 対応コンパイラ
- Python 3.11 以上、NumPy

## ビルド方法

通常はこれで十分です。

```bash
pixi run build
```

これは内部で以下を実行します。

```bash
cmake -G Ninja -S codec -B codec/build -DCMAKE_BUILD_TYPE=Release
cmake --build codec/build
```

ビルド後、macOS では次の共有ライブラリができます。

```text
codec/build/libradiance_codec.dylib
```

Linux では通常、次の名前になります。

```text
codec/build/libradiance_codec.so
```

## テスト方法

最小テスト:

```bash
pixi run test-codec
```

主要な C++ テストを個別に実行する場合:

```bash
codec/build/test_codec
codec/build/test_grouped_delta
codec/build/test_structural_context
codec/build/test_bitshuffle
codec/build/test_predictor
codec/build/test_rans
codec/build/test_rans_binary
```

near-lossless のベンチ:

```bash
pixi run bench-near-lossless-cpp
```

## インストール方法

C/C++ から使う場合は、CMake install を使えます。

```bash
pixi run build
pixi run cmake --install codec/build --prefix ./dist/radiance_codec
```

インストール後の構成はおおむね以下です。

```text
dist/radiance_codec/
  include/radiance_codec/codec.hpp
  include/radiance_codec/c_api.h
  lib/libradiance_codec.dylib
```

システム領域へ入れる場合は prefix を変更します。

```bash
pixi run cmake --install codec/build --prefix /usr/local
```

Python binding は今のところ開発ツリー用です。`codec/python/radiance_codec.py` は
`codec/build/libradiance_codec.dylib` または `codec/build/libradiance_codec.so` を探します。
Python から使う場合は、まず `pixi run build` してから、このリポジトリ内で
`codec/python` を import path に追加してください。

スクリプトから使う場合は、例えば次のように実行できます。

```bash
PYTHONPATH=codec/python pixi run python your_script.py
```

Python package としてインストールする場合は、リポジトリrootで次を実行します。

```bash
pixi run install-python
```

wheel を作る場合:

```bash
pixi run wheel
```

`setup.py` は CMake/Ninja で `libradiance_codec` をビルドし、Python module と同じ場所へ
共有ライブラリを同梱します。既にビルド済みのライブラリを使いたい場合は以下を使えます。

```bash
RADIANCE_CODEC_SKIP_CMAKE_BUILD=1 pixi run python -m pip install .
RADIANCE_CODEC_LIBRARY=/path/to/libradiance_codec.dylib python your_script.py
```

開発中の editable install は、共有ライブラリ探索の都合上、先に `pixi run build` してから
使ってください。

```bash
pixi run build
pixi run install-python-editable
```

## Python での使い方

### 完全 lossless

```python
import sys
import numpy as np

sys.path.insert(0, "codec/python")
import radiance_codec

pixels = np.random.default_rng(0).standard_normal((128, 128, 3)).astype(np.float32)

encoded = radiance_codec.encode_lossless(pixels, preset="quality")

decoded = radiance_codec.decode(encoded, pixels.shape)
assert decoded.dtype == np.float32
assert decoded.tobytes() == pixels.tobytes()

print(f"{pixels.nbytes} -> {len(encoded)} bytes")
```

lossless preset は以下の API 設定に対応します。
`encode_lossless(pixels)` の既定は `preset="quality"` です。full画像の反応速度を
優先する場合は `preset="fast"`、少し圧縮率寄りにする場合は `preset="compact"` を明示します。

| preset | stage | effort | 目安 |
|---|---|---:|---|
| `fast` | `StageByteplaneRans` | 5 | full画像の高速 lossless。byteplane chunk + entropy gate + spatial filter + rANS/Zstd |
| `compact` | `StageByteplaneRans` | 6 | fastより少し圧縮率寄り。high byteplane filterを実圧縮で探索 |
| `balanced` | `StageGroupedDelta` | 10 | 速度と容量の中間 |
| `quality` | `StageGroupedDelta` | 11 | 実用default |
| `max` | `StageGroupedDelta` | 12 | 最大寄り探索、かなり重い |

入力 shape は `(height, width, channels)` または `(height, width)` です。
channels は `1..4` に対応しています。

### Near-lossless

```python
import sys
import numpy as np

sys.path.insert(0, "codec/python")
import radiance_codec

pixels = np.random.default_rng(1).standard_normal((128, 128, 3)).astype(np.float32)

low_bits = 12
encoded = radiance_codec.encode_near_lossless(
    pixels,
    low_bits=low_bits,
    effort=11,
)

decoded = radiance_codec.decode(encoded, pixels.shape)
expected = radiance_codec.quantize_mantissa(pixels, low_bits)

# near-lossless では元画像ではなく、量子化後画像と一致する。
assert decoded.tobytes() == expected.tobytes()
```

`low_bits` は `0..23` です。値を大きくすると圧縮率は上がりやすくなりますが、
数値誤差も大きくなります。

### Visual near-lossless router

HDR写真向けの現行主力 near-lossless route は `encode_near_lossless_router_v1` です。
decode 結果は元画像ではなく、router が再構成した候補画像になります。

```python
import sys
import numpy as np

sys.path.insert(0, "codec/python")
import radiance_codec

pixels = np.random.default_rng(2).standard_normal((128, 128, 3)).astype(np.float32)

encoded = radiance_codec.encode_near_lossless_router_v1(pixels, effort=11)
decoded = radiance_codec.decode(encoded, pixels.shape)

assert decoded.shape == pixels.shape
assert decoded.dtype == np.float32
```

現行の Metal guided / downsample / high-pass / visual-guard はデフォルトで有効です。
dark smooth bypass は暗部階調に影響しやすいためデフォルトでは無効で、比較実験時だけ
明示的に有効化します。

```bash
RADIANCE_CODEC_NO_METAL_GUIDED=1
RADIANCE_CODEC_NO_METAL_DOWNSAMPLE=1
RADIANCE_CODEC_NO_METAL_HIGHPASS=1
RADIANCE_CODEC_NO_METAL_VISUAL_GUARD=1
RADIANCE_CODEC_ROUTER_DARK_SMOOTH_BYPASS=1
```

Python binding は未指定時に `OMP_WAIT_POLICY=PASSIVE` / `KMP_BLOCKTIME=0` を設定し、
OpenMP worker の待機スピンによる encode 時間の揺れを抑えます。

圧縮せずに router の再構成候補と report だけを見るには
`reconstruct_near_lossless_router_v1(...)` を使います。

## C++ での使い方

`codec.hpp` を include して、raw float32 bytes を渡します。

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
    if (radiance_codec::encode(raw, meta, cfg, compressed) != radiance_codec::Status::Ok) {
        return 1;
    }

    std::vector<std::uint8_t> decoded;
    if (radiance_codec::decode(compressed, meta, cfg, decoded) != radiance_codec::Status::Ok) {
        return 1;
    }

    if (decoded != raw) {
        return 1;
    }

    std::cout << raw.size() << " -> " << compressed.size() << " bytes\n";
    return 0;
}
```

visual near-lossless router を直接使う場合は、stage を
`StageNearLosslessRouter` にします。

```cpp
radiance_codec::PipelineConfig cfg{
    .stages = radiance_codec::StageNearLosslessRouter,
    .effort = 11,
    .rans_mode = 1,
};
```

現在の frame header には stage/meta が入りますが、decode API には caller 側でも
`ImageMeta` を渡します。ヘッダと caller meta が一致しない場合は decode に失敗します。

インストール先が `./dist/radiance_codec` の場合、macOS では例えば次のように
ビルドできます。

```bash
c++ -std=c++20 example.cpp \
  -I./dist/radiance_codec/include \
  -L./dist/radiance_codec/lib \
  -lradiance_codec \
  -Wl,-rpath,$PWD/dist/radiance_codec/lib \
  -o example
```

実行ファイルの場所によって `rpath` は調整してください。開発中は
`codec/build` を直接指定しても構いません。

CMake project からは install prefix を `CMAKE_PREFIX_PATH` に渡すと
`find_package` できます。

```cmake
find_package(radiance_codec CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE radiance_codec::radiance_codec)
```

```bash
c++ -std=c++20 example.cpp \
  -Icodec/include \
  -Lcodec/build \
  -lradiance_codec \
  -Wl,-rpath,$PWD/codec/build \
  -o example
```

## C API について

Swift や他言語 FFI 用に C ABI もあります。

```c
#include <radiance_codec/c_api.h>
```

主な関数:

```c
radiance_codec_encode(...)
radiance_codec_decode(...)
radiance_codec_near_lossless_router_v1_reconstruct(...)
radiance_codec_buffer_free(...)
radiance_codec_version()
```

`radiance_codec_encode` / `radiance_codec_decode` が返す buffer はライブラリ側で確保されます。
使い終わったら必ず `radiance_codec_buffer_free` を呼んでください。

## Stage の目安

| Stage | 用途 |
|---|---|
| `StageNone` | passthrough / 動作確認 |
| `StageRans` | byte stream rANS。`rans_mode` で order0/order1 を選ぶ |
| `StageBitshuffle | StageRans` | bitshuffle 後に rANS |
| `StageByteplaneRans` | float32 byteplane を chunk 並列で entropy gate + spatial filter + rANS/Zstd 符号化する高速 lossless codec |
| `StageGroupedDelta` | 現在の主力 lossless codec |
| `StageMantissaQuantize` | near-lossless 用の前段量子化 |
| `StageLinearIndex` | transform-index near-lossless 実験 |
| `StageNearLosslessRouter` | HDR写真向け visual near-lossless router |
| `StageStructuralContext` | 旧系統の structural context 実験 |

near-lossless では `StageMantissaQuantize | StageGroupedDelta` の組み合わせを使います。
Python では `encode_near_lossless` がこの組み合わせを内部で指定します。
visual near-lossless router は `StageNearLosslessRouter` 単独で使います。

## 注意点

- 現在は研究用 codec です。ファイル形式や stage id は将来変わる可能性があります。
- decode 時も画像の width / height / channels を API に渡す必要があります。
  フレームヘッダにも meta は入っていますが、現在の API は caller 側の meta と
  cross-check します。
- near-lossless は bit-exact ではありません。用途に応じて `low_bits` を選んでください。
- Python wheel は CMake build で作った共有ライブラリを同梱します。
- CMake install は `find_package(radiance_codec CONFIG)` 用の package config も入れます。

## 関連ドキュメント

- 再開用ハンドオフ: `docs/HANDOFF_JA.md`
- 結果の日本語要約: `docs/RESULTS_JA.md`
- 詳細な研究ログ: `docs/PLAN_ML_V2.md`
- 研究方針の整理: `docs/LOSSLESS_RESEARCH_REBOOT.md`
