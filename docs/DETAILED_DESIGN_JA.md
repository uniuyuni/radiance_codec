# radiance_codec 詳細設計書

最終更新: 2026-06-08

## 位置づけ

`radiance_codec` は float32 HDR 画像を対象にした研究用 codec である。
現在の公開境界は C++ API、C ABI、Python ctypes binding の3層で、実装本体は
`codec/src` 配下の C++20 stage pipeline として構成されている。

この文書は、コードベース全体の構造、公開API、内部データ形式、主要stage、
ビルド/インストール方法、運用上のデフォルトをまとめる。
研究ログや個別実験の詳細は既存の `docs/*` を参照する。

## ディレクトリ構成

| path | 役割 |
|---|---|
| `codec/include/radiance_codec/codec.hpp` | C++公開API。`ImageMeta`, `PipelineConfig`, `encode`, `decode` を定義する。 |
| `codec/include/radiance_codec/c_api.h` | C ABI。Python/Swift/他言語FFI向けの安定境界。 |
| `codec/src` | codec本体。pipeline、stage実装、C ABI wrapper、Metal実装を含む。 |
| `codec/python/radiance_codec.py` | Python ctypes binding。開発ツリー用。 |
| `codec/tests` | C++単体/回帰テスト。 |
| `scripts` | ベンチ、品質監査、実験プローブ、preview生成。 |
| `docs` | 研究ログ、品質基準、最適化記録、設計文書。 |
| `data` | ローカル評価用EXR。git管理外/大容量を含む。 |
| `results`, `outputs` | 実験結果とpreview。多くは再生成可能。 |

## 公開API

### C++ API

ヘッダ:

```cpp
#include <radiance_codec/codec.hpp>
```

主要型:

- `radiance_codec::ImageMeta`
  - `width`, `height`, `channels`, `format`
  - 現在の実装対象は `PixelFormat::Float32`
  - raw layout は little-endian float32 interleaved
- `radiance_codec::PipelineConfig`
  - `stages`: `Stage*` bitmask
  - `effort`: codec固有の探索/品質レベル
  - `rans_mode`: classic RANS stage用
  - `near_lossless_bits`, `near_lossless_policy`
- `radiance_codec::Status`
  - `Ok`, `InvalidArg`, `UnsupportedFormat`, `DecompressFailed`, `SizeMismatch` など

主要関数:

```cpp
Status encode(std::span<const std::uint8_t> raw,
              const ImageMeta& meta,
              const PipelineConfig& config,
              std::vector<std::uint8_t>& out) noexcept;

Status decode(std::span<const std::uint8_t> compressed,
              const ImageMeta& meta,
              const PipelineConfig& config,
              std::vector<std::uint8_t>& out) noexcept;
```

`decode` は frame header 内の stage/meta を読むが、caller も `ImageMeta` を渡す。
header と caller meta が一致しない場合は `SizeMismatch` になる。

推奨例:

- fast lossless: `StageByteplaneRans`, `effort=5`
- compact lossless: `StageByteplaneRans`, `effort=6`
- balanced lossless: `StageGroupedDelta`, `effort=10`
- quality lossless: `StageGroupedDelta`, `effort=11`
- max lossless: `StageGroupedDelta`, `effort=12`
- lightweight lossless smoke: `StageRans`、または `StageBitshuffle | StageRans`
  と `rans_mode`
- mantissa near-lossless: `StageMantissaQuantize | StageGroupedDelta`
- visual near-lossless: `StageNearLosslessRouter`
- transform-index near-lossless: `StageLinearIndex`

codec の方式選択は API オプションで行う。環境変数は `OMP_NUM_THREADS` などの
OpenMP runtime調整や、研究用の内部feature flagに限定する。
Python binding では `encode_lossless(..., preset="fast" | "compact" | "balanced" | "quality" | "max")`
が上記 API 設定への薄い wrapper になっている。
`encode_lossless(pixels)` の既定は `preset="quality"` で、full画像の反応速度を
優先する場合は `preset="fast"`、少し圧縮率寄りにする場合は `preset="compact"` を明示する。

### C ABI

ヘッダ:

```c
#include <radiance_codec/c_api.h>
```

主要関数:

- `radiance_codec_encode`
- `radiance_codec_decode`
- `radiance_codec_near_lossless_router_v1_reconstruct`
- `radiance_codec_buffer_free`
- `radiance_codec_version`

C ABI は POD struct のみを境界に使う。`radiance_codec_encode` /
`radiance_codec_decode` / `radiance_codec_near_lossless_router_v1_reconstruct`
が返す `radiance_codec_buffer_t` は library 側が `malloc` し、caller が
`radiance_codec_buffer_free` で解放する。

`radiance_codec_near_lossless_router_v1_reconstruct` は圧縮 frame を作らず、
router の再構成候補と report だけを返す品質監査/preview向けAPIである。

### Python API

モジュール:

```python
import radiance_codec
```

Python package は root の `pyproject.toml` / `setup.py` で配布できる。
build 時に CMake/Ninja で `libradiance_codec` をビルドし、`radiance_codec.py` と
同じ wheel 内ディレクトリへ共有ライブラリをコピーする。

開発ツリーで直接使う場合は、従来通り `codec/python` を `PYTHONPATH` に追加し、
`codec/build/libradiance_codec.dylib` または `.so` を参照する。

主要関数:

- `encode(pixels, stages, effort=5, ...) -> bytes`
- `decode(compressed, shape, ...) -> np.ndarray`
- `encode_near_lossless(pixels, low_bits, effort=11, ...) -> bytes`
- `encode_near_lossless_router_v1(pixels, effort=11) -> bytes`
- `reconstruct_near_lossless_router_v1(pixels, **params) -> (decoded, report)`
- `encode_linear_index_near_lossless(pixels, bits=7, transform="linear") -> bytes`
- `encode_linear_index_preset(...)`
- `quantize_mantissa(...)`
- `quantize_linear_index(...)`
- `inspect_header(...)`

Python binding は import 時に未指定なら以下を設定する。

```bash
OMP_WAIT_POLICY=PASSIVE
KMP_BLOCKTIME=0
```

これは OpenMP worker の待機スピンが `std::async` payload 生成と競合し、
encode時間が揺れる問題を抑えるためである。

## Top-level frame format

実装: `codec/src/codec.cpp`

外側 frame は `HDR0` magic を持つ。現行 version は `3`。

概念的な layout:

```text
magic: "HDR0"
version: u8
width: u32 LE
height: u32 LE
channels: u8
format: u8
stages: u32 LE
rans_mode: u8
effort: u8
near_lossless_bits: u8
near_lossless_policy: u8
sign_class: u8
payload: bytes
```

encode 時は `PipelineConfig` から stage list を構築し、front-to-back に stage を実行する。
decode 時は header 内の stage/config を読み、同じ pipeline を構築して reverse order で
decode する。

`config` 引数は decode API に残っているが、実際の stage 復元は frame header に従う。

## Pipeline設計

実装:

- `codec/src/pipeline.hpp`
- `codec/src/pipeline.cpp`

すべての stage は `IStage` を実装する。

```cpp
class IStage {
public:
    virtual Status encode(std::span<const std::uint8_t> in,
                          const ImageMeta& meta,
                          std::vector<std::uint8_t>& out) noexcept = 0;
    virtual Status decode(std::span<const std::uint8_t> in,
                          const ImageMeta& meta,
                          std::vector<std::uint8_t>& out) noexcept = 0;
};
```

`build_pipeline` の優先順:

1. `StageLinearIndex` があれば専用stage単独で返す。
2. `StageNearLosslessRouter` があれば router stage 単独で返す。
3. `StageMantissaQuantize` があれば前段として追加する。
4. `StageGroupedDelta` があれば grouped-delta stage を追加し、そこで打ち切る。
5. `StageByteplaneRans` があれば byteplane-rANS stage を追加し、そこで打ち切る。
6. `StageStructuralContext` があれば structural-context stage を追加し、そこで打ち切る。
7. classic stack: color transform / predictor / bitshuffle / RANS。
8. stageなしなら passthrough。

このため `StageNearLosslessRouter` や `StageLinearIndex` は、他stageと組み合わせる
通常のfilterではなく、自己完結 codec として扱う。

## Stage一覧

### Passthrough

実装: `codec/src/passthrough.cpp`

入力をそのままコピーする。動作確認と `StageNone` 用。

### ColorTransform

実装: `codec/src/color_transform.cpp`

RGB(A) 向け reversible color transform。encode/decode は同じ変換で戻る。
classic stack 用。

### Predict

実装: `codec/src/predictor.cpp`

float32 byte stream を画像構造として見て、MED/LOCO-I系の空間予測を行う。
classic stack 用。

### Bitshuffle

実装: `codec/src/bitshuffle.cpp`

bit-level transposition。float32の同一bit位置をまとめ、後段entropy codingを助ける。

### Rans

実装:

- `codec/src/rans.cpp`
- `codec/src/rans_binary.cpp`
- `codec/src/rans_internal.hpp`
- `codec/src/rans_models.cpp`

Order0/Order1 byte rANS と adaptive binary rANS を提供する。
`rans_internal.hpp` は low-level state machine、`rans_models.*` は byte model を持つ。

### MantissaQuantize

実装: `codec/src/mantissa_quantize.cpp`

finite float32 の低mantissa bitを落とす near-lossless 前段。
decode は passthrough で、decode結果は「量子化後画像」になる。

policy:

- fixed
- tile
- exponent
- tile+exponent
- linear/log/sqrt/gamma/asinh transform-index 系

### GroupedDelta

実装: `codec/src/grouped_delta.cpp`

現在の主力 lossless codec。float32 bits を order-preserving domain に写し、
body bitplane と sign payload を decoder-safe adaptive binary rANS で符号化する。

内部 frame magic は `GDXB`。

特徴:

- 完全 lossless
- `effort` で探索量/方式を調整
- half-like / source precision / grouped tail などの研究成果を多く含む
- public API からは `StageGroupedDelta` として使う

### ByteplaneRans

実装: `codec/src/byteplane_rans.cpp`

full画像の高速 exact lossless preset 用 codec。raw float32 を value chunk に分け、
各 chunk の 4 byteplane を独立streamとして扱う。低 byteplane は raw のまま、高
byteplane は west/north spatial delta filter も候補に入れる。`effort=5` の候補選択は histogram
entropy 推定で行い、実際に rANS order0 / Zstd を走らせるのは選ばれたfilterだけ。
entropy gate で縮まないstreamは圧縮器を試さず raw fallback に逃がす。`effort>=6` では
high byteplane filter を実圧縮で比較する。encode は chunk 単位で4 byteplaneを1回の
raw scanから gather し、chunk 単位で OpenMP 並列化する。encode側は decode 用
`slot_to_sym` を作らない encode専用 rANS model を使い、rANS scratch buffer と Zstd
CCtx/scratch buffer は thread-local に再利用する。`StageGroupedDelta` より圧縮率は
浅いが、decode を含めた full画像の反応速度を優先する。

内部 frame magic は `BPR1`。

特徴:

- 完全 lossless
- `encode_lossless(..., preset="fast" | "compact")` の backend
- Zstd が build にある場合は byteplane stream の候補として使う

### StructuralContext

実装: `codec/src/structural_context.cpp`

旧系統の structural context 実験stage。内部 frame magic は `SCX1`。
現在は主力ではないが、比較用/研究継続用に残っている。

### LinearIndex

実装:

- `codec/src/linear_index.cpp`
- `codec/src/linear_index_transform.cpp`

専用 near-lossless transform-index codec。finite float32 を channel別 range に基づく
N-bit index に量子化し、predictor residual を adaptive/rANS 系 payload で符号化する。
内部 frame magic は `LIDX`。

decode結果は元画像ではなく、indexから再構成した量子化画像である。

### NearLosslessRouter

実装:

- `codec/src/near_lossless_router.cpp`
- `codec/src/near_lossless_router.hpp`
- `codec/src/near_lossless_router_metal.mm`
- `codec/src/near_lossless_router_metal.hpp`

HDR写真向け visual near-lossless router。内部 frame magic は `NLR1`。

大まかな encode flow:

1. float32 RGB(A) を読む。
2. VST/sign-log/YCoCg 系の特徴量を作る。
3. dark/outlier/visual guard に基づき route mask を決める。
4. low chroma guide、high-pass chroma、Y index、signed-log escape、dark refine を作る。
5. mask/index/value stream を rANS/専用 symbol codec で符号化する。
6. decode は stream を復元し、同じ量子化値から float32 候補を再構成する。

主要な default:

- Metal guided: ON
- Metal downsample: ON
- Metal high-pass: ON
- Metal visual-guard: ON
- dark smooth bypass: OFF。暗部階調と色ムラの品質回帰を避けるため、実験時のみ
  `RADIANCE_CODEC_ROUTER_DARK_SMOOTH_BYPASS=1` で有効化する。
- dark noise threshold: `0.003`
- tiled masks: ON
- order1 byte streams: ON
- OpenMP wait policy: passive/blocktime 0 if unset
- OpenMP thread count: runtime側の `OMP_NUM_THREADS` で調整する

主な環境変数:

```bash
RADIANCE_CODEC_NO_METAL_GUIDED=1
RADIANCE_CODEC_NO_METAL_DOWNSAMPLE=1
RADIANCE_CODEC_NO_METAL_HIGHPASS=1
RADIANCE_CODEC_NO_METAL_VISUAL_GUARD=1
RADIANCE_CODEC_ROUTER_DARK_SMOOTH_BYPASS=1
RADIANCE_CODEC_ROUTER_NO_TILED_MASKS=1
RADIANCE_CODEC_ROUTER_NO_ORDER1_STREAMS=1
OMP_NUM_THREADS=4
```

Metal 実装は Apple/Metal framework が見つかった時のみ `RADIANCE_CODEC_HAS_METAL`
付きでビルドされる。非Apple/非Metal環境では CPU fallback が使われる。

## Stream codec設計

router 内部では複数の小さな stream frame を使う。
基本形は以下。

```text
method: u8
payload_size: u32 LE
payload: bytes
```

主な method:

- raw
- rANS order0
- rANS order1
- symbol rANS
- symbol context rANS
- symbol parity context rANS
- binary/tiled mask
- zstd実験枠

stream decode は `read_stream`, `read_stream_slice`, `decode_stream_slice` 周辺にまとまっている。

## Metal設計

実装: `codec/src/near_lossless_router_metal.mm`

Metal は near-lossless router の重い全画素処理を補助する。

担当:

- guided low chroma
- downsample
- high-pass threshold / mask
- visual guard
- intermediate cache の再利用

C++側は `#ifdef RADIANCE_CODEC_HAS_METAL` で Metal API を呼ぶ。
Metal関数が失敗した場合は原則CPU fallbackへ戻る。

## 並列化と性能

並列化の軸:

- OpenMP parallel for: 大きな全画素走査
- `std::async`: router payload stream の並列生成/復号
- Metal command: GPU/Apple Silicon向け処理

観測された問題:

- OpenMP worker が parallel region 後にスピン待機すると、直後の `std::async`
  payload 生成と競合し encode 時間が大きく揺れる。

対策:

- C++ router entry と Python binding で、未指定時に以下を既定設定する。

```bash
OMP_WAIT_POLICY=PASSIVE
KMP_BLOCKTIME=0
```

これにより、DSCF系のベンチでウォーム後の encode wall が 2.3s 台まで安定することを確認した。

## ビルドとインストール

推奨は pixi。

```bash
pixi install
pixi run build
```

内部では CMake/Ninja を使う。

```bash
cmake -G Ninja -S codec -B codec/build -DCMAKE_BUILD_TYPE=Release
cmake --build codec/build
```

C/C++/C ABI の install:

```bash
pixi run cmake --install codec/build --prefix ./dist/radiance_codec
```

install されるもの:

```text
include/radiance_codec/codec.hpp
include/radiance_codec/c_api.h
lib/libradiance_codec.dylib or .so
```

現状の制約:

- Python wheel は `setup.py` の custom `build_py` で共有ライブラリを同梱する。
- editable install は先に `pixi run build` して、開発ツリーの `codec/build` を参照する。
- `RADIANCE_CODEC_LIBRARY=/path/to/lib` で共有ライブラリ探索を明示上書きできる。
- CMake install は package config を入れるため、consumer は
  `find_package(radiance_codec CONFIG REQUIRED)` を使える。

## テスト

主要テスト:

- `codec/build/test_codec`
- `codec/build/test_rans`
- `codec/build/test_rans_binary`
- `codec/build/test_bitshuffle`
- `codec/build/test_predictor`
- `codec/build/test_structural_context`
- `codec/build/test_grouped_delta`
- `codec/build/test_byteplane_rans`
- `codec/build/test_near_lossless_router`

よく使う検証:

```bash
cmake --build codec/build --target test_near_lossless_router test_codec
codec/build/test_near_lossless_router
codec/build/test_codec
python3 -m py_compile codec/python/radiance_codec.py
git diff --check
```

実画像router計測:

```bash
RADIANCE_CODEC_ROUTER_TRACE=1 \
pixi run python scripts/probe_router_lowfreq_modes.py \
  --worker --image sample_DSCF0009.EXR --mode default --effort 11
```

## 運用上の注意

- この codec は研究用であり、file format と stage id は将来変更されうる。
- near-lossless 系は bit-exact ではない。
- `StageNearLosslessRouter` は HDR写真向けに強く調整されており、汎用科学データには
  そのまま適さない可能性がある。
- Python API は開発ツリー前提なので、アプリ組み込み時は共有ライブラリ探索と
  ABI/version の一致を別途設計する必要がある。
- Metal default は Apple環境での current best に合わせたもの。比較実験では
  `RADIANCE_CODEC_NO_METAL_*` を明示する。

## 今後の設計課題

1. Python packaging hardening
   - cibuildwheel 等による platform wheel 自動生成
   - macOS/Linux/Windows の shared library 名と依存ライブラリ監査
   - editable install 時の CMake build hook 改善
2. API安定化
   - frame format versioning policy
   - stage id compatibility policy
   - router params/report のABI version
3. Router payload scheduling
   - `std::async` 乱立ではなく、固定幅thread poolまたは段階的payload生成へ移行
4. 品質監査の標準化
   - sample set
   - preview/contact sheet
   - numeric/display-space criteria
