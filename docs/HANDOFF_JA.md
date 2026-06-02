# radiance_codec 再開用ハンドオフ

この文書は、Codex / アプリを閉じて再開したあとに、現在地へ素早く戻るためのメモです。
既存ドキュメントを全部読み返す前に、まずここを読めば「何が実装済みで、何が詰まりどころで、次に何を見るべきか」が分かるようにしてあります。

## 最初に読む順番

1. `README_JA.md`
   - ビルド方法、インストール方法、Python からの使い方。
   - ライブラリ名、Python モジュール名、C API 名の現在形を確認する。

2. `docs/RESULTS_JA.md`
   - 現時点の圧縮率と、exact / near-lossless の結果サマリ。
   - 「どのケースが勝っていて、どのケースが壁か」を確認する。

3. `docs/PLAN_ML_V2.md`
   - 長い研究ログ。全部を一気に読む必要はない。
   - 再開時は、後半の GDXB / puresky / near-lossless 周辺を重点的に読む。

4. `docs/LOSSLESS_RESEARCH_REBOOT.md`
   - 一度落ち着いて再整理した研究方針。
   - exact lossless の限界、near-lossless を別オプション化する判断、今後の仮説がまとまっている。

## 現在の場所と名前

プロジェクトフォルダは次です。

```text
/Users/uniuyuni/LibraryProjects/radiance_codec
```

以前の作業名やパスとして `SwuftArchives` や `hdrcodec` がありましたが、現在の表向きの名前は `radiance_codec` です。
アプリ再開後に「現在の作業ディレクトリがありません」と出た場合は、古い `SwuftArchives` や `SwiftProjects` を見に行っている可能性が高いです。

主な公開名は次の通りです。

- C++ namespace: `radiance_codec`
- CMake project / target: `radiance_codec`
- 生成ライブラリ: `libradiance_codec.dylib`
- include: `codec/include/radiance_codec`
- Python module: `codec/python/radiance_codec.py`
- C API prefix: `radiance_codec_*`
- C macro prefix: `RADIANCE_CODEC_*`

## フォルダ構成

```text
README_JA.md        日本語の導入、ビルド、使い方
codec/              C++ 本体、C API、Python バインディング
data/               評価用 EXR サンプル
docs/               設計、研究ログ、結果サマリ
ml/                 ML 実験用コードと学習データ
results/            探索・評価結果 JSON / CSV / log
scripts/            評価、監査、プローブ用スクリプト
pixi.toml           開発環境
```

`bench/` は整理済みで、現在は通常の作業対象ではありません。

## 実装済みの核

exact lossless の主力は `StageGroupedDelta` です。

- stage bit: `StageGroupedDelta = 0x0040`
- C++ 主実装: `codec/src/grouped_delta.cpp`
- C++ 公開ヘッダ: `codec/include/radiance_codec/codec.hpp`
- Python wrapper: `codec/python/radiance_codec.py`

near-lossless は `StageMantissaQuantize` として実装済みです。

- stage bit: `StageMantissaQuantize = 0x0080`
- C++ 実装: `codec/src/mantissa_quantize.cpp`
- Python API:
  - `radiance_codec.encode_near_lossless(pixels, low_bits, effort=11)`
  - `radiance_codec.quantize_mantissa(pixels, low_bits)`

near-lossless は有限の float32 について low mantissa bits を 0 にする方式です。
NaN / Inf は保持されます。
デコード側は、保存済みの量子化済み float32 をそのまま復元します。

## 最終確認できているビルドとテスト

2026-06-02 に、現在の `LibraryProjects` パスで再確認済みです。
旧 `SwiftProjects` パス由来の `codec/build/CMakeCache.txt` が残っていると
`pixi run build` が失敗します。その場合は CMake の生成キャッシュを作り直してください。

作業ディレクトリ:

```bash
cd /Users/uniuyuni/LibraryProjects/radiance_codec
```

ビルド:

```bash
pixi run build
```

確認済みの C++ テスト:

```bash
codec/build/test_codec
codec/build/test_grouped_delta
codec/build/test_structural_context
codec/build/test_bitshuffle
codec/build/test_predictor
codec/build/test_rans
codec/build/test_rans_binary
```

Python の確認:

```bash
pixi run python -m py_compile \
  codec/python/radiance_codec.py \
  scripts/estimate_mantissa_quantization.py \
  scripts/benchmark_structural_context_cpp.py \
  scripts/audit_gdx8_stream_budget.py \
  scripts/audit_16x_budget.py \
  ml/ml_rans_compress.py \
  ml/prepare_training_data.py
```

Python smoke test では、`import radiance_codec`、GroupedDelta の byte-exact encode/decode、
near-lossless low12 の量子化後 byte-exact decode まで確認済みです。

## 重要な結果

exact lossless:

- 全 13 画像、effort9 系の現行 GDX 系: geomean 約 `7.317x`
- realistic-no-puresky crop128 effort11: 約 `8.015x`
- realistic-no-puresky crop128 effort12: 約 `8.043x`
- puresky-hard crop128 effort11: 約 `2.345x`
- puresky-hard crop128 effort12: 約 `2.347x`

near-lossless:

- ph_*_1k crop128 effort11
  - low00 exact: 約 `6.076x`
  - low08: 約 `7.634x`
  - low12: 約 `9.314x`
  - low15: 約 `12.071x`
- puresky-hard crop128 effort11
  - low00 exact: 約 `2.345x`
  - low08: 約 `4.151x`
  - low12: 約 `6.825x`
  - low15: 約 `13.051x`

puresky 個別:

- `ph_belfast_sunset_puresky_1k`
  - exact 約 `2.40x`
  - low12 約 `7.33x`
  - low15 約 `14.88x`
  - low15 max relative error 約 `0.00348`
  - PSNR 約 `60.96 dB`
- `ph_kloppenheim_06_puresky_1k`
  - exact 約 `2.29x`
  - low12 約 `6.35x`
  - low15 約 `11.44x`
  - low15 max relative error 約 `0.00389`
  - PSNR 約 `64.44 dB`

low15 は float32 mantissa 23 bit のうち下位 15 bit を落とし、上位 8 bit を残します。
正規化数では相対誤差の上限はおおむね `2^-8 = 0.00390625`、つまり約 0.39% です。

## これまでに見えたこと

1. AI を最初から主役にするより、float32 のビット構造を扱う方が効いた。
   - 画像モデルや byte-LM だけでは、現時点の exact compression を大きく超える根拠が弱かった。
   - AI は「難しい残差の context mixer」として後段に置く方が筋が良い。

2. puresky は見た目に smooth でも、exact lossless では low mantissa tail が重い。
   - 連続グラデーションに見えても、下位 mantissa には測定・生成・丸め由来の細かい情報が大量にある。
   - exact で 12x / 16x を狙うには、この tail に新しい条件付き構造を見つける必要がある。

3. 「near-lossless base + exact correction stream」は exact では大きく勝てなかった。
   - low bits を正確に戻す correction が結局大きい。
   - ただし near-lossless を別オプションとして提供する価値は高い。

4. FPC / Gorilla 的な XOR leading/trailing zero 方向は、単体では現行 GDXB より弱めだった。
   - ただしタイル分類や fallback route として再利用できる可能性はある。

5. context tree / tile classifier は exact 側でまだ一番可能性が残る。
   - ただし side information が増えるとすぐ負ける。
   - 小さく安い signaled context tree が鍵。

## 次に進めるなら

最短で価値が出るのは、near-lossless の policy / selector layer です。

候補:

- exact: すでに half-like / bfloat-like / low-tail が小さいタイル
- near_low12: 画質劣化をかなり抑えたい通常 HDR
- near_low15: puresky や smooth gradient で大きく圧縮率を取りに行く
- fallback: noise / random stress data / 下位 mantissa が本当に情報を持つケース

最初にやること:

1. `scripts/` に policy probe を作る。
2. タイル単位で次を測る。
   - exponent 分布
   - low mantissa zero 率
   - low bits entropy
   - horizontal / vertical delta entropy
   - puresky-like smoothness
3. exact / low12 / low15 のどれを選ぶべきかを推定する。
4. JSON 結果で「実際に selector が fixed low bits より勝つか」を見る。
5. 勝ち筋が見えたら C++ frame metadata に mode signal を入れる。

再開直後の確認ポイント:

- このフォルダ単体には `.git` が無い状態だったため、履歴確認や commit 前提の作業をするなら、まず管理元を確認する。
- `pixi run build` と `codec/build/test_*` はこのパス基準で実行する。
- `test_codec` のソースは `codec/tests/test_passthrough.cpp`。near-lossless の基本 roundtrip もここに入っている。
- `results/` は履歴が多いので、最新方針は `docs/RESULTS_JA.md`、`docs/PLAN_ML_V2.md` 後半、`docs/LOSSLESS_RESEARCH_REBOOT.md` 後半を優先する。

exact lossless を続ける場合の次手:

- puresky 専用の low-tail 条件付きモデルをもう一段だけ試す。
- per-tile の cheap context tree を、side information 予算込みで検証する。
- AI は tail bit の確率推定器として小さく使う。完全な画像再構成モデルには戻らない。

## 注意点

- `results/` は歴史的な探索ログが大量にある。古い結果を現在の性能と混同しない。
- `ml/` の生成データや log / manifest には、古い絶対パスが残っている可能性がある。
- source / docs / public API では `hdrcodec` ではなく `radiance_codec` を使う。
- sandbox から MPS が見えない場合がある。GPU 学習を本気で回すなら、sandbox 外または MPS が見える環境で実行する。
- exact 12x / 16x は夢として追う価値があるが、puresky の low mantissa tail が現在の最大の壁。
- near-lossless は逃げではなく、HDR 用ライブラリとしてかなり実用的な別モード。

## 再出発の合言葉

このプロジェクトの勝ち筋は、一般画像コーデックの模倣ではなく、float32 HDR の数値表現そのものを利用すること。

exact は「情報を消せない」ので、構造を見つけた分だけ勝つ。
near-lossless は「人間やレンダリング上ほぼ意味の薄い low mantissa を整理する」ことで、一気に 12x 以上へ寄せられる。
AI はそのあとで、残った hard tail の確率を読むために使う。
