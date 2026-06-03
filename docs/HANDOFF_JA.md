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

5. `docs/LOSSLESS_12X_ROADMAP_JA.md`
   - 2026-06-02 時点の true/exact lossless 12x 再整理。
   - near-lossless ではなく、float32 ビット列完全一致だけを対象に、文献・ボトルネック・次の実験順をまとめている。

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
  - 可変bit/値域量子化版: `fixed`, `tile`, `exponent`, `tile_exponent`,
    `linear_range`, `log_range` 相当の `NearLosslessPolicy` を指定可能

2026-06-02 追加:

- 外側フレームは version 3。version 1/2 decode 互換は維持。
- v3 header に `near_lossless_policy` と `sign_class` を追加。
  - `sign_class`: mixed / all sign bits 0 / all sign bits 1。
  - 現時点では header metadata。sign payload 省略routeは未実装。
- `StageMantissaQuantize` は固定bitに加えて、タイル低位bitのランダム性、
  exponent、またはその合成で画素ごとに clear bits を変える MVP を持つ。
- 32x/64x near-lossless 目標向けに `linear_range` と `log_range` を追加。
  `linear_range` は finite values を channelごとの global min/max で N-bit
  線形量子化し、その量子化済み float32 を既存codecで完全復元する。
- `sample_DSCF0009.EXR` crop512 / effort9:
  - fixed low15: `3.53x`
  - exponent low15: `7.66x`
  - tile_exponent low15: `7.66x`
  - linear_range low7: `28.19x` (`effort11`: `28.95x`)
  - linear_range low6: `54.49x`, PSNR `56.95dB`, signed-log RMSE
    `7.56e-3`, histogram KS256 `1.53e-2`
  - 方針更新: 品質優先で `bits7` を本命にし、`bits7` のまま `64x` を狙う。
- `scripts/probe_linear_index_route.py` を追加。
  - `sample_DSCF0009.EXR` crop512 / bits7:
    - quantized float32 path: `28.95x`
    - avg predictor residual entropy: `0.7203 bps` (`44.43x`)
    - tile oracle entropy:
      - tile16: `0.4560 bps` (`70.18x`)
      - tile32: `0.4674 bps` (`68.47x`)
      - tile64: `0.4797 bps` (`66.70x`)
      - tile128: `0.5046 bps` (`63.42x`)
    - zstd split baseline: `0.5190 bps` (`61.66x`)
    - context zero-mask + zstd nonzero-values:
      - tile32: `0.4921 bps` (`65.03x`)
      - tile64: `0.4869 bps` (`65.73x`)
      - tile128: `0.4897 bps` (`65.35x`)
  - 解釈: bits7/64x は専用 index codec + context-coded zero mask +
    nonzero value stream なら狙える。次は mask range coder と bitstream format の実装。
  - ただし 9箇所 crop 追試で、中央/端のディテール領域は bits7 index entropy が高く、
    tile oracle でも `7x-18x` 程度まで落ちる領域があった。bits7/64x は
    smooth/noise-floor tile では成立するが、本番写真全体に一律適用する目標としては
    現時点で無理寄り。
- 全画像フル解像度 `linear_range bits7 + GroupedDelta effort9` 実測:
  - `*.exr` 15画像 geomean: `59.813x`
  - `sample_DSCF0009.EXR`: `17.355x`
  - 本番写真 `sample_DSCF0009.EXR` が厳しい。`*.exr` glob は大文字 `.EXR` を拾わないので、
    本番ファイルは別コマンドで測定した。
- `scripts/probe_tile_router_near_lossless.py` を追加。
  - tileごとに local min/max linear index を作り、候補bitsの中から品質閾値を満たす
    最小bitsを選ぶ quality-first router probe。
  - 選択条件は signed-log RMSE / signed-log p99 / gradient signed-log NRMSE。
  - `sample_DSCF0009.EXR` crop1024 / tile128 / context64:
    - 候補 `7,8,10,12`, gradient条件なし相当: `7.36x`, selected `{7:61, 8:3}`。
    - 候補 `3,4,5,6,7,8,10,12`, gradient<=`0.8`: `19.02x`,
      selected `{3:23, 4:28, 5:4, 6:2, 7:4, 8:3}`。
    - 候補 `3,4,5,6,7,8,10,12`, gradient<=`0.5`: `14.79x`,
      selected `{4:35, 5:19, 6:3, 7:4, 8:3}`。
  - 解釈: local range が狭い高エントロピータイルは、bits7より低いbitsでも
    ログ誤差は小さい。ただしbits3/4は勾配差が壊れやすいので、画質優先では
    gradient guard が必須。現時点の本番写真向け現実レンジは `15x-20x` 級。
- `scripts/probe_noise_synthesis_quality.py` を追加。
  - local linear quantization 後の bin 内 residual に、カメラ/デモザイク由来の
    phase 構造や finite index table で拾える傾向が残るかを検証する。
  - これは codec 実装ではなく、実装判断用の held-out crop probe。
  - `sample_DSCF0009.EXR` crop512 / grid9 / tile128:
    - bits3: finite table の best log RMSE 改善 `1.81%`
    - bits4: `0.92%`
    - bits5: `0.28%`
    - bits6: `0.06%`
    - bits7: `0.02%`
    - hash_uniform は全 bits で log または gradient を悪化。
  - `*.exr` crop256 / grid5 の横展開:
    - realish 11画像 median 改善は bits4 `2.54%`, bits5 `0.32%`,
      bits6 `0.10%`, bits7 `0.00%`。
    - photo/env 7画像 median 改善は bits4 `5.03%`, bits5 `0.75%`,
      bits6 `0.16%`, bits7 `0.00%`。
  - 判断: 白ノイズ合成は現時点で採用しない。finite residual table は
    bits3/4 など攻めた tile の補助としては候補だが、bits7 品質重視routeを
    変えるほどの効果はない。優先度は tile router / index codec の後。
- `StageLinearIndex` / `scripts/benchmark_linear_index_codec.py` を追加。
  - `linear_range bits7` の quantized float32 を GroupedDelta に渡すのではなく、
    global per-channel min/max + N-bit index plane を直接bitstream化するMVP。
  - index residual は avg predictor、zero mask は left/up/up-left context の
    adaptive binary rANS。nonzero residual value は byte rANS / bitplane rANS /
    symbol rANS のうち最小payloadを選ぶ。
  - decode は元画像ではなく、per-channel linear quantized float32 を復元する。
  - 現MVPは finite float32 のみ対応。NaN/Inf exception stream は未実装。
  - local tile min/max版も試したが、`sample_DSCF0009.EXR` top-left crop512
    bits7 が `6.92x` と悪く、固定bits7 route の初手としては不採用。
    以前の65x級推定は global per-channel min/max 前提だった。
  - `sample_DSCF0009.EXR` top-left crop512:
    - bits6: `125.12x`, signed-log RMSE `7.555e-3`, gradient NRMSE `0.769`,
      PSNR `56.95dB`
    - bits7: `55.88x`, signed-log RMSE `5.167e-3`, gradient NRMSE `0.452`,
      PSNR `60.77dB`
  - `sample_DSCF0009.EXR` top-left crop1024:
    - bits6: `109.94x`, signed-log RMSE `1.366e-2`, gradient NRMSE `1.393`,
      PSNR `52.86dB`
    - bits7: `58.72x`, signed-log RMSE `7.369e-3`, gradient NRMSE `0.758`,
      PSNR `58.43dB`
  - 判断: 専用index bitstreamは実装値でも `32x` を大きく超え、bits7で
    `56x-59x` まで来た。ただし crop1024 の gradient は品質重視には強めなので、
    次は tile router / gradient guard / local-or-global range selection が必要。
- `scripts/probe_quality_router_modes.py` を追加。
  - global per-channel index と local tile index を同じ品質閾値で比較し、
    選択済みtileを貼り合わせた復元画像全体で gradient/log を再評価するprobe。
  - 初期の品質重視しきい値:
    - signed-log RMSE <= `0.004`
    - signed-log p99 <= `0.018`
    - gradient signed-log NRMSE <= `0.5`
  - `sample_DSCF0009.EXR` crop512 / tile128:
    - router: `14.72x`, selected `{local4:11, local5:2, local7:2, local8:1}`,
      full gradient `0.184`, PSNR `72.83dB`
    - baseline global10 estimate: `15.48x`, full gradient `0.202`,
      PSNR `74.53dB`
  - `sample_DSCF0009.EXR` crop1024 / tile128:
    - router: `14.74x`, selected `{global10:3, global8:4, local4:35,
      local5:17, local7:5}`, full gradient `0.222`, PSNR `71.80dB`
    - baseline global10 estimate: `14.53x`, full gradient `0.213`,
      PSNR `74.05dB`
  - 判断: このsampleでは複雑なtile routerは global10 付近の単純routeに
    明確には勝っていない。品質優先なら、まず global linear-index の
    bits自動選択を採用する方が安全。
- `scripts/benchmark_linear_index_codec.py` に品質閾値による selected bits 出力を追加。
  - `sample_DSCF0009.EXR` crop512, thresholds上記:
    - bits7 `56.14x`, grad `0.452`, log RMSE `5.167e-3`
    - bits8 `34.42x`, grad `0.499`, log RMSE `4.329e-3`
    - bits9 `20.21x`, grad `0.399`, log RMSE `2.549e-3`
    - bits10 `14.51x`, grad `0.202`, log RMSE `1.110e-3`
    - selected: bits9
  - `sample_DSCF0009.EXR` crop1024, thresholds上記:
    - bits7 `58.79x`, grad `0.758`, log RMSE `7.369e-3`
    - bits8 `31.62x`, grad `0.632`, log RMSE `5.212e-3`
    - bits9 `19.94x`, grad `0.407`, log RMSE `2.745e-3`
    - bits10 `14.09x`, grad `0.213`, log RMSE `1.259e-3`
    - selected: bits9
  - 品質優先プリセット候補:
    - ratio重視: bits7/8。ただし crop1024 では gradient/log がしきい値超過。
    - quality: bits9。現sample cropでは `20x` 級でしきい値内。
    - quality+: bits10。`14x` 級で PSNR/log/gradient がかなり安全。
  - Python helper:
    - `radiance_codec.encode_linear_index_preset(pixels, "ratio")` -> bits7
    - `"balanced"` -> bits8
    - `"quality"` -> bits9
    - `"quality_plus"` -> bits10
  - 追加実装: bits9/10 改善のため、nonzero residual value に symbol rANS を追加。
    crop1024 bits9 は `15.30x` -> `19.94x`、bits10 は `10.28x` -> `14.09x`。
  - 追加実装: residual predictor を AVG と MED から自動選択するようにした。
    mask phase は 4x4 も試したが少し悪化したため、2x2 phase を維持。
    crop1024 bits9 は `19.94x` -> `20.17x`、bits10 は `14.09x` -> `14.33x`。
    full bits10 は `11.19x` -> `11.92x` (`42,714,477` -> `40,098,490` bytes)。
    品質値は変わらない。現実装は AVG/MED を両方本エンコードして選ぶため、
    encode 時間は増える。後で軽量な predictor preselect に置き換える余地あり。
  - 追加実装: mask context を `west/north/northwest` から
    `west/north/northwest/northeast/previous-channel` に拡張。
    bitstream version は `5`。
    crop1024 bits9 は `20.17x` -> `20.46x`、bits10 は `14.33x` -> `14.45x`。
    full bits9 は `15.63x` -> `16.89x` (`30,571,227` -> `28,292,387` bytes)、
    full bits10 は `11.92x` -> `12.03x` (`40,098,490` -> `39,715,096` bytes)。
    bits9 の strict quality はまだ `log RMSE 4.086e-3` で僅かに超過。
  - 次の実装優先度: bits10 の mask/value payload 分解を進める。
    value stream は symbol rANS でエントロピー近傍なので、残る主戦場は
    residual zero mask と predictor/context。tile router本体は、global bits9/10 に
    明確に勝つ条件が出てからでよい。
  - bits9 strict化 probe:
    `scripts/probe_quality_router_modes.py` の mask見積もりを新contextに更新。
    crop1024 の global9/10 tile切替は `17.46x` 見込み、
    mode内訳は `global9:14`, `global10:50`。full品質は通るが、
    tile局所基準では fallback tile が残る。local10 を入れる案は `5.52x` まで
    落ちたため、現時点では不採用。
- 21MB級のRAW圧縮に勝つための transform-index 実験:
  - `StageLinearIndex` に `signed-log`, `sqrt`, `gamma075`, `gamma025`,
    `asinh` transform mode を追加。Python API:
    `encode_linear_index_near_lossless(..., transform="gamma075")`。
  - `scripts/probe_transform_index_quantization.py` を追加。
  - full `sample_DSCF0009.EXR` 実測:
    - linear bits8: `18.85MB`, `25.35x`, log RMSE `8.071e-3`。サイズは
      21MB未満だが品質不足。
    - signed-log bits7: `20.07MB`, `23.81x`, log RMSE `6.195e-3`。
      サイズは21MB未満だが品質不足。
    - gamma075 bits7: `18.41MB`, `25.95x`, log RMSE `7.009e-3`。
    - asinh bits7: `18.68MB`, `25.58x`, log RMSE `7.298e-3`。
    - gamma075 bits8: `27.46MB`, `17.40x`, log RMSE `3.487e-3`。strict品質通過。
    - asinh bits8: `28.02MB`, `17.05x`, log RMSE `3.646e-3`。strict品質通過。
    - signed-log bits8: `29.67MB`, `16.10x`, log RMSE `3.065e-3`。strict品質通過。
  - 結論: transform単体では「21MB未満」と「strict品質」の両立はまだ未達。
    最小のstrict通過は gamma075 bits8 の `27.46MB`。
  - sparse refinement 下限:
    crop1024 の signed-log bits7 -> bits8 で strict log RMSE へ戻すには
    約 `8.3%` サンプル補正が必要。mask+1bit の理想下限だけでも full換算
    7MiB級で、21MBまでの残り約0.93MBには収まらない見込み。
  - 次の候補:
    1. demosaic後ではなくRAW/CFA面に近い表現で試す。
    2. transform-index前の色/チャンネル decorrelation。
    3. global transformではなくtileごとの transform/gamma 選択。ただしlocal range
       metadataは重いので、global range + tile transform selector を優先。
    4. learned/AI predictor は次フェーズ候補。
- color/channel decorrelation と tile transform selector の初回probe:
  - `scripts/probe_color_tile_transform_index.py` を追加。
    color mode は `rgb`, `g-diff` (`G, R-G, B-G`), `ycocg`。
    品質は必ずRGBへ戻して測る。
  - `scripts/benchmark_color_transform_index_codec.py` を追加。
    Pythonでcolor変換し、実装済み `StageLinearIndex` で圧縮、decode後にRGB復元して評価。
  - crop512:
    - `g-diff + gamma075 bits7`: `27.85x`, strict品質通過。
    - tile-selector: `20.28x`。品質安全だがbits8選択が多く重い。
  - crop1024:
    - `rgb + asinh bits8`: `24.96x`, strict品質通過。
    - `ycocg + asinh bits8`: `22.80x`, strict品質通過。
    - `rgb + gamma075 bits7`: `28.56x` だが gradient `0.538` で失敗。
    - `g-diff + gamma075 bits7`: `27.60x` だが gradient `0.615` で失敗。
    - tile-selector: `19.49x`。単純なtile選択は現時点で重い。
  - full実測 `g-diff + gamma075`:
    - bits7: `17,822,169` bytes (`26.81x`), log RMSE `8.535e-3` で品質不足。
    - bits8: `26,720,804` bytes (`17.88x`), log RMSE `4.240e-3` でstrictに少し届かず。
  - 判断:
    - 単純なglobal `g-diff` はRGB/gamma075 bits8より少し小さくなるが、
      品質が悪化してstrict未達。完全に捨てるほどではないが、21MB突破の本命ではない。
    - 単純tile-selectorは局所品質を守るためbits8へ寄り、サイズが重い。
    - 次にやるなら、`g-diff` の係数探索 (`R-aG`, `B-bG`) か、
      RGB/gamma075 bits7 のgradientだけ救う軽量補正。
- 21MB突破の追加探索:
  - `scripts/probe_codebook_index_quantization.py` を追加。
    - density-based codebook は失敗。index面が荒れ、品質も大きく悪化。
    - uniform indexを保ったまま復元代表値だけ最適化する `recon-table` は
      cropでは強い。crop1024:
        - signed-log bits7: log RMSE `4.732e-3` -> `3.048e-3`,
          gradient `0.635` -> `0.471`
        - gamma075 bits7: log RMSE `3.433e-3` -> `3.180e-3`,
          gradient `0.538` -> `0.489`
      ただし full ではまだ不足:
        - signed-log bits7 + recon-table: `20,070,729` bytes,
          log RMSE `5.804e-3`
        - gamma075 bits7 + recon-table: `18,414,068` bytes,
          log RMSE `6.958e-3`
  - `scripts/probe_predictive_dequantization.py` を追加。
    - decoder側でbin中心を近傍予測へ寄せる方式。
    - crop1024では gamma075 bits7 alpha=0.5 が strict通過。
    - fullでは未達:
      gamma075 bits7 alpha=0.5 log RMSE `6.246e-3`,
      signed-log bits7 alpha=0.5 log RMSE `5.605e-3`。
  - region allocation:
    - crop1024 tile512 の `rgb/gamma075 bits7/8` は `22.91x` でstrict通過。
    - full tile512 では全176 tileがbits8を選び、`27,772,178` bytes相当。
      bits7を残せず不採用。
  - `scripts/probe_color_coefficient_search.py` を追加。
    - crop1024では `a=0.5,b=0.0 gamma075 bits7` が `27.51x` でstrict通過。
    - full実測では失敗:
      `17,725,469` bytes, log RMSE `7.296e-3`。
  - `scripts/probe_hard_region_quality_map.py` を追加。
    - 512 tileごとのfull品質を見ると、bits7系は全176 tileがlog RMSE閾値を超過。
      hard regionが局所的にあるのではなく、bits7全体の精度が足りない。
  - 判断更新:
    - bits7を小手先の復元改善・局所補正でstrict品質へ持ち上げる方針は弱い。
    - 次は bits8 strict候補 (`gamma075/asinh/signed-log`) の index payload を
      小さくする方向、またはRAW/CFAに近い入力へ戻る方向が本命。
- 整理メモ:
  - transform-index の変換関数を `codec/src/linear_index_transform.cpp`
    / `.hpp` に分離。`linear_index.cpp` は index生成、mask/value rANS、
    decode復元に集中。
  - Python側も transform名のalias、policy変換、順変換/逆変換を
    helperに集約。C++ decode と Python期待値の一致確認は
    `scripts/benchmark_linear_index_codec.py --no-save` で行う。
  - 失敗probeや結果JSONは削除していない。研究ログとして保持。
- `sample_DSCF0009.EXR` full resolution (`7728x5152`) linear-index audit:
  - command:
    `pixi run python scripts/benchmark_linear_index_codec.py --glob sample_DSCF0009.EXR --crop-size 0 --bits 7,8,9,10 --log-rmse-threshold 0.004 --log-p99-threshold 0.018 --gradient-threshold 0.5`
  - bits7: `36.67x`, bytes `13,029,521`, enc `16.55s`, dec `6.02s`,
    log RMSE `1.579e-2`, p99 `3.225e-2`, grad `0.444`, PSNR `54.07dB`
  - bits8: `23.42x`, bytes `20,395,989`, enc `16.86s`, dec `5.90s`,
    log RMSE `8.071e-3`, p99 `1.615e-2`, grad `0.263`, PSNR `59.95dB`
  - bits9: `15.63x`, bytes `30,571,227`, enc `25.12s`, dec `4.01s`,
    log RMSE `4.086e-3`, p99 `8.172e-3`, grad `0.149`, PSNR `65.89dB`
  - bits10: `11.19x`, bytes `42,714,477`, enc `30.00s`, dec `7.80s`,
    log RMSE `2.009e-3`, p99 `4.066e-3`, grad `0.0778`, PSNR `72.03dB`
  - predictor auto-select後の bits10 再測定:
    `11.92x`, bytes `40,098,490`, enc `67.92s`, dec `7.17s`,
    log RMSE `2.009e-3`, p99 `4.066e-3`, grad `0.0778`, PSNR `72.03dB`
  - mask context拡張後の再測定:
    - bits9: `16.89x`, bytes `28,292,387`, enc `48.44s`, dec `7.24s`,
      log RMSE `4.086e-3`, p99 `8.172e-3`, grad `0.149`, PSNR `65.89dB`
    - bits10: `12.03x`, bytes `39,715,096`, enc `75.84s`, dec `12.06s`,
      log RMSE `2.009e-3`, p99 `4.066e-3`, grad `0.0778`, PSNR `72.03dB`
  - 21MB目標の再測定:
    - signed-log bits7: `20,069,217` bytes, log RMSE `6.195e-3`
    - gamma075 bits8: `27,456,725` bytes, log RMSE `3.487e-3`
    - asinh bits8: `28,022,177` bytes, log RMSE `3.646e-3`
  - threshold結果: strict quality (`log RMSE <= 0.004`) では bits10 が選択。
    bits9 は p99/gradient は十分良いが log RMSE が `0.004085` でわずかに超過。
  - 判断更新:
    - ratio: bits7 (`36.7x`) だが log/p99 は品質基準外。
    - balanced: bits8 (`23.4x`) だが log RMSE は品質基準外。
    - quality: bits9 (`15.6x`) は実用候補。strict閾値からは僅差で外れる。
    - quality_plus / strict: bits10 (`11.2x`) が現時点の安全プリセット。

near-lossless の fixed/tile/exponent 系は、有限の float32 について low mantissa
bits を 0 にする方式です。linear/log range 系は値そのものを量子化します。
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

true / exact lossless を続ける場合の次手:

- まず `docs/LOSSLESS_12X_ROADMAP_JA.md` を読む。
- 最優先は side information 込みの MDL-coded signaled context tree。入口は `scripts/probe_signaled_context_tree_mdl.py`。
- 次に reversible ordered-body block transform と source-precision aware route を試す。
- puresky low-tail は、12x 主戦場というより certificate と低 support feature 探索として扱う。
- AI は最後の entropy context mixer としてだけ戻す。完全な画像再構成モデルには戻らない。

## 最新 near-lossless / 21MB target メモ

`sample_DSCF0009.EXR` は X-Trans 由来データ。疑似RAW/CFA分解を試す場合は
Bayer 4 pattern ではなく X-Trans 6x6 周期を扱う必要があるため、当面は低優先。

`scripts/audit_linear_index_payload.py` を追加。codec本体は変更せず、実装済み
`StageLinearIndex` の `HDR0` / `LIDX` header を読み、mask/value payload と
transform-index residual 統計を監査する。

full `sample_DSCF0009.EXR`, `gamma075 bits8`:

- encoded: `27,456,725` bytes, `17.40x`
- payload split:
  - mask: `13,762,148` bytes, `0.922 bits/sample`
  - value: `13,694,504` bytes, `0.917 bits/sample`
  - fixed overhead: `73` bytes
- predictor: `med`
- value mode: `symbol-rans`
- nonzero residual rate: `42.75%`
- value entropy: `2.132 bits/nonzero`
- dominant residuals are `0`, `-1`, `+1`, then `-2`, `+2`.

21MB target は約 `1.406 bits/sample`。現行 `gamma075 bits8` は
`1.839 bits/sample` なので、約 `0.433 bits/sample`、全体で約 `6.45MB`
削る必要がある。mask と value がほぼ半々なので、片側だけの小改善では届きにくい。

同じ index 面での predictor 監査:

- `med`: residual entropy `1.896 bits/sample`, nonzero `42.75%`
- `avg`: residual entropy `2.050 bits/sample`, nonzero `45.66%`
- `north`: residual entropy `2.178 bits/sample`, nonzero `46.50%`
- `west`: residual entropy `2.306 bits/sample`, nonzero `47.67%`
- simple previous-channel predictors are worse.

判断: 現行 index 面では `med` が妥当。21MB突破には、単純な空間/前チャンネル
predictor差し替えではなく、予測残差index化、量子化空間の再設計、または
RAW/CFA寄り表現への大きめの変更が必要。

追加probe:

- `scripts/probe_predictive_residual_index.py`
  - causal DPCM-like に transform residual を量子化する案。
  - cropでは品質は良くなるが index 面が荒れ、baselineより重くなる。
  - 現状は本命ではない。
- `scripts/probe_parametric_transform_index.py`
  - signed-power `sign(x) * abs(x)^gamma` を連続探索。
  - full `gamma=0.795 bits8`: strict品質通過。
    - estimated order0 bytes: `26,911,194`
    - log RMSE `3.956e-3`, p99 `8.081e-3`, gradient `1.526e-1`
  - `gamma=0.800 bits8` は log RMSE `4.014e-3` で僅差fail。
- `scripts/probe_tail_escape_range_index.py`
  - bright tail をcap外escapeに分離する案。
  - sampleは全プラス、max約 `6.20`、`>4.0` は約 `0.401%`。
  - full `main7 tail6 gamma0.75 cap3.5`: `23.84MB` 見込みだが
    log RMSE `5.020e-3` でfail。
  - full `main8 tail5 gamma0.90 cap4.0`: `27.50MB` 見込み、
    log RMSE `4.213e-3` でfail。単独では突破口にならない。
- `scripts/probe_index_residual_context_entropy.py`
  - mask/value分離ではなく、transform-index residual を文脈付き多値symbolとして
    符号化した場合の conditional entropy を監査。
  - full `gamma=0.75 bits8`:
    - order0 `28.31MB` 相当
    - best `west_north_prev_channel`: `24.24MB` 相当
  - full `gamma=0.795 bits8`:
    - order0 `26.91MB` 相当
    - best `west_north_prev_channel`: `22.73MB` 相当
  - 判断: 21MBにはまだ約 `1.7MB` 足りないが、現時点で最も現実的な前進。
    次はこの context residual coder を production寄りestimate/実装候補として詰める。
- `scripts/probe_small_residual_escape_entropy.py`
  - residual を `0, +/-1, ..., +/-small` と sparse escape に分ける案。
  - pred index の coarse bin (`pred-bin`) を context に追加すると大きく効く。
  - full `gamma=0.795 bits8`, `small=7`,
    category context `west_north_prev_phase_predbin`,
    detail context `west_north_prev_predbin`:
    - strict品質通過
    - estimated `21,862,541` bytes, `21.85x`
- `scripts/probe_channel_gamma_index.py`
  - channel別gamma + small residual escape + pred-bin context を探索。
  - full `(R,G,B)=(0.795,0.800,0.800)`:
    - strict品質通過
    - estimated `21,754,052` bytes
  - `scripts/probe_power_recon_table.py` の signed-log mean recon-table を使うと
    all `gamma=0.800` が僅差failからstrict通過へ戻る。
  - full all `gamma=0.800` + signed-log mean recon-table:
    - strict品質通過
    - estimated `21,706,344` bytes (`20.70 MiB`)
    - log RMSE `3.997e-3`, p99 `8.203e-3`, gradient `1.535e-1`
  - decimal `21MB` にはまだ約 `0.71MB` 足りない。実bitstreamではmodel/table overhead
    も必要なので、次は実payload化と追加context/selectorで詰める。

## bits10 quality-first pivot

見た目確認で `bits8` は暗部階調に目に見える劣化があり、画質優先モードとしては
不採用。`signed-log bits10` のトーンマップPNGは許容されたため、以降は
`bits10相当品質` を主軸にする。

SmallEscapeRans実装前の実codec `signed-log bits10`:

- encoded: `56,021,152` bytes, `8.53x`
- log RMSE `7.574e-4`, p99 `1.350e-3`, gradient `2.986e-2`

`scripts/audit_linear_index_payload.py` full `signed-log bits10`:

- mask: `11,021,878` bytes (`0.738 bits/sample`)
- value: `44,999,201` bytes (`3.014 bits/sample`)
- nonzero residual rate: `76.71%`
- value entropy: `3.856 bits/nonzero`
- 判断: bits10では mask ではなく value stream が支配的。

bits10向け探索:

- `scripts/probe_index_residual_context_entropy.py`
  - `signed-log bits10` full-symbol context:
    - best `west_north_prev_channel`: `50.07MB` 相当
- `scripts/probe_small_residual_escape_entropy.py`
  - `signed-log bits10`, `small=15`, simple context:
    - `48.51MB` 相当
  - 改善はあるが、signed-log bits10のままでは重い。
- `scripts/probe_small_escape_payload_budget.py`
  - `signed-log bits10` の復元値を固定したまま、small residual + escape を
    production寄りに見積もるprobe。rANSの `PROB_SCALE=16384` への頻度丸め、
    stream flush、context model table の保存コストを含む。
  - full `sample_DSCF0009.EXR`, category context `west_north_prev_channel`,
    detail context `order0`:
    - `small=15`: `51,768,823` bytes, `9.23x`, escape `2.53%`
    - `small=7`: `49,561,883` bytes, `9.64x`, escape `7.18%`
    - `small=6`: `49,620,305` bytes, `9.63x`
    - `small=8`: `49,588,647` bytes, `9.63x`
  - 比較: `small=7`, category context `west_north` では
    `51,431,288` bytes, `9.29x`。`prev_channel` は model cost を払っても効く。
  - 判断: 現行実codec `56,021,152` bytes から、画質を一切変えずに
    約 `49.6MB` へ落とす最初のproduction候補。detail側の空間contextは
    理想payloadを少し縮めるが、model tableが増えて総量では負ける。
- 実装更新:
  - `codec/src/linear_index.cpp` に `ValueMode::SmallEscapeRans` を追加し、
    LIDX version を `7` に更新。version `6` streamも読める。
  - 現時点では `bits10` 専用候補。復元値は `signed-log bits10` と同一で、
    残差payloadだけを `small=7`, category `west_north_prev_channel`,
    detail `order0` に置き換える。
  - encoder は従来の mask/value 方式と比較し、小さい場合だけ
    `SmallEscapeRans` を選ぶ。1024 cropでは固定費のため従来 `symbol-rans`
    が選ばれ、fullでは新modeが選ばれる。
  - full `sample_DSCF0009.EXR`, `signed-log bits10` 実測:
    - encoded `49,554,947` bytes, `9.64x`
    - encode `90.165s`, decode `14.612s`
    - log RMSE `7.574e-4`, gradient `2.986e-2`, PSNR `78.78dB`
  - 判断: 画質アンカーを崩さず `56.0MB -> 49.6MB`。改善幅は
    約 `6.47MB` で、見積もりとの差も数KBレベル。
- 追加context軸の確認:
  - `scripts/probe_small_escape_payload_budget.py` に category context の
    `current_channel`, `2x2 phase`, `6x6 phase` 系を追加。
  - full `small=7`, detail `order0`:
    - baseline `west_north_prev_channel`: `49,561,883` bytes
    - `west_north_prev_current_channel`: `49,620,217` bytes
      - payloadは `49.29MB -> 49.06MB` に下がるが、model table増加で総量は悪化。
    - `west_north_prev_xtrans6`: `58,061,492` bytes
      - payloadは `49.03MB` まで下がるが、model tableが `9.04MB` まで膨らむ。
  - 判断: 高次元情報はpayload entropyを少し下げるが、単純な直積contextでは
    表コストに負ける。X-Trans phase はこのbitstreamの直接contextとしては
    現時点で不採用。次にやるなら、直積ではなく context mixing / fallback tree /
    tile限定適用が必要。
- fallback tree 実装:
  - `scripts/probe_small_escape_context_split.py` を追加。base context
    `west_north_prev_channel` を親にし、追加軸ごとに「親modelのまま」か
    「子modelへ分割」を親contextごとに選ぶ。
  - full `small=7`:
    - axis `channel`: `49,382,197` bytes見込み、baseline比 `-172,795` bytes
    - axis `xtrans6`: `49,554,994` bytes見込み、split親 `0`
  - `ValueMode::SmallEscapeChannelSplitRans` を追加し、LIDX version を `8` に更新。
    v6/v7 stream decode互換は維持。
  - full `sample_DSCF0009.EXR`, `signed-log bits10` 実測:
    - encoded `49,382,153` bytes, `9.68x`
    - encode `76.919s`, decode `18.113s`
    - log RMSE `7.574e-4`, gradient `2.986e-2`, PSNR `78.78dB`
  - 判断: channel fallback split は小さいが実装値でも約 `173KB` 改善。
    X-Trans phase は fallback してもこのカテゴリ残差には刺さらない。
- `sunny-crunching-sky.md` / dither突破口メモ:
  - 新仮説: `bits10` が必要だった主因は全画素のノイズ詳細保存ではなく、
    暗部banding回避かもしれない。
  - 助言メモ上の観察:
    - bits深度+1で残差entropyがほぼ `+1.0 bit/sample`
    - signed-log median5平滑後のbits10量子化で残差entropyが大きく下がる
    - banding-prone画素率は bits8 `3.5%`, bits9 `0.1%`, bits10 `0%` という見立て
  - 次の本命検証:
    - `signed-log bits9 + adaptive dither`
    - 暗部banding-prone領域だけ、FS / ordered / block-average 方式で
      ノイズ注入を比較する。
    - decode側は不変。dither済みindexを今のcodecで符号化する。
  - 期待:
    - bits9+dither: `~30MB` 級
    - bits8+dither: `~21MB` 級だったが、2026-06-03 の3 stops持ち上げ目視で
      不採用。通常表示では真っ暗で差が見えにくいが、画質優先の検査条件では
      暗部ノイズが馴染まない。
  - 注意:
    - `bits8` は現時点で本線から外す。再評価するなら、まず `bits9` で
      ノイズ注入方法を確立してから。
    - FS強度 `s0.35` / `s0.5` では浮き方は大きく変わらなかったため、
      問題は強度より「ノイズの入れ方」。
    - その後の目視で、問題のノイズはdither由来ではなく `bits9 none` の時点で
      既にあるデコード後ノイズだと判明。見え方としては「浮く」より
      「周囲より少し暗い」。
    - 次の本命は dither pattern ではなく、暗部限定の reconstruction bias /
      recon table。probeには `--recon-bias` と `--recon-bias-dark-max` を追加済み。
    - `+0.125/+0.25/+0.375/+0.5 LSB` の単純biasは目視で大差なし。
      一律持ち上げではなく、binごとの代表値を変える `--recon-table`
      (`signed-log-mean` / `value-mean`) を次に確認する。
    - 方針更新: 暗部だけ `bits10` に逃がすなら、非暗部の `bits8` は
      軽量routeとして復活する。全画素bits8ではなく、`bits8 + darkbits10`
      routerとして見る。
    - crop1024 / 3 stops持ち上げprobe:
      - `bits9 + darkbits10 darkmax0.18`: log `6.158e-4`, grad `1.094e-1`
      - `bits9 + darkbits10 darkmax0.25`: log `6.140e-4`, grad `1.091e-1`
      - `bits8 + darkbits10 darkmax0.18`: log `8.317e-4`, grad `1.480e-1`
      - `bits8 + darkbits10 darkmax0.25`: log `8.250e-4`, grad `1.468e-1`
      - いずれも素の `bits9 none` より数値は良い。次は目視確認とpayload見積もり。
    - ユーザー目視:
      - `bits9 none` は最悪。
      - `bits9+dark10 max0.25` / `bits8+dark10 max0.18` /
        `bits8+dark10 max0.25` はほぼ見分けがつかない。
      - 微かに `bits9+dark10 max0.25` が良い気もするが、判断差はごく小さい。
    - payload方向:
      - `scripts/probe_darkbits_router_payload.py` を追加。
      - crop1024は暗部が広く、`darkmax0.05` でもdark対象 `82.43%`、
        `darkmax0.18/0.25` は `91%` 超。サイズ判定には厳しいcrop。
      - full画像のluma率は `0.05:42.84%`, `0.08:55.21%`,
        `0.12:70.13%`, `0.18:79.30%`, `0.25:83.90%`。
      - crop1024 payload見積もり:
        - `bits8+dark10 max0.05`: `1,303,770 bytes`, `9.65x`
        - `bits8+dark10 max0.08`: `1,327,093 bytes`, `9.48x`
        - `bits8+dark10 max0.12`: `1,379,861 bytes`, `9.12x`
        - `bits9+dark10 max0.05`: `1,301,103 bytes`, `9.67x`
        - `bits9+dark10 max0.08`: `1,317,172 bytes`, `9.55x`
        - `bits9+dark10 max0.12`: `1,351,407 bytes`, `9.31x`
      - このcropではdark refinementが支配的で、bits8/9差は小さい。
        次は `dark && banding-risk` mask でdark10対象を絞る。
      - ユーザー要望: 比較サンプルは少なくする。以後は原則2候補まで。
      - `bits8+dark10 max0.08` mask比較:
        - `luma`: dark対象 `84.88%`, `1,327,093 bytes`, `9.48x`,
          log `9.960e-4`, grad `1.771e-1`
        - `dark && banding-risk`: dark対象 `65.24%`, `1,163,255 bytes`, `10.82x`,
          log `1.373e-3`, grad `2.424e-1`
        - 比較preview:
          `outputs/previews/dither_breakthrough/sample_DSCF0009_crop1024_bits8_dark10_luma_vs_banding_w0.5_g2.2_sheet.png`
      - ユーザー目視:
        - `luma mask` が良いが、`banding-risk mask` でも比較しないと分からない程度。
        - 差が出るのは「真っ暗ではないが暗い」shadow transition、右下の木付近。
      - full `bits8+dark10 max0.08 banding-risk`, base encodeなしprobe:
        - dark10対象 `20.11%`
        - mask `596,217 bytes`
        - refinement `6,826,917 bytes`
        - quality log `2.738e-3`, p99 `5.374e-3`, grad `1.074e-1`
        - 既存 `signed-log bits8` base `29,674,477 bytes` を足す概算は
          `37,097,671 bytes`, `12.88x`
        - 品質寄りrouteとしては `bits10` より少し軽いが、21MB級にはまだ遠い。
          次はbase stream改善またはrefinement対象の再選別。
        - 通常表示 `white=4.0, gamma=2.2` のfull previewは、ユーザー目視で
          オリジナルと区別がつかない。
        - previewは `outputs/previews/highlight_guard/` に分離済み。
    - 2026-06-03 追加サンプル:
      - `sample_bright_park.EXR`: shape `5152x7728x3`, float,
        luma `<=0.08` は `32.01%`。
        `bits8+dark10 max0.08 banding-risk` はdark10対象 `0.85%`,
        mask `124,034 bytes`, refinement `283,327 bytes`,
        log `2.195e-3`, grad `9.126e-2`。
        `signed-log bits8` base実測は `43,114,889 bytes`, `11.08x`。
        router概算は `43,522,310 bytes`, `10.98x`。
      - `sample_middle_flower.EXR`: shape `7728x5152x3`, float,
        luma `<=0.08` は `2.94%`。
        `bits8+dark10 max0.08 banding-risk` はdark10対象ほぼ `0%`,
        mask `578 bytes`, refinement `140 bytes`,
        log `1.795e-3`, grad `1.323e-1`。
        `signed-log bits8` base実測は `50,512,430 bytes`, `9.46x`。
      - 判断: 新2枚は評価セットに入れる価値あり。明るい画像ではdark10追加は
        ほぼ無料に近いが、base bits8 streamが重い。次の主戦場は
        `dark10` 対象選別ではなく base index stream の圧縮改善。
    - detail消失指標:
      - `scripts/benchmark_linear_index_codec.py` の `error_stats` に、
        表示空間 `white=4.0, gamma=2.2` の領域別detail指標を追加済み。
      - JSONには `display_w4_g22_*_lost_detail_rate`,
        `display_w4_g22_*_grad_energy_ratio`, `display_w4_g22_*_grad_correlation`,
        `display_w4_g22_*_visible_detail_rate` が入る。
      - benchmark出力には `hi_lost` / `ex_lost` が出る。
      - ハイライトdetail消失の監視は、今後この指標を標準で見る。
    - 20M route再検証:
      - `gamma0.8 bits8 + signed-log-mean recon + pred-bin context` は
        estimated `21,706,344 bytes`, `22.01x` で20M目標に最接近。
      - しかし通常表示previewで暗部グラデーションの階調飛びが確認され、
        単独採用不可。
      - 暗部救済込み見積もり:
        - `darkbits10 max0.08 banding-risk`: `30,665,604 bytes`, `15.58x`
        - `darkbits9 max0.08 banding-risk`: `27,997,428 bytes`, `17.06x`
        - `darkbits9 max0.05 banding-risk`: `27,612,179 bytes`, `17.30x`
      - 判断: 20Mに近いbaseへ暗部救済を足すと `27-31MB` 圏へ戻る。
        20M達成には、base stream / entropy coderの改善か、別base表現が必要。
    - ハイライトは基本ditherしない。暗部とハイライトの視覚確認が必須。
    - `scripts/probe_dithered_linear_index.py` には `adaptive-block` と
      `--block-size` を追加済み。ブロック内のfloor/ceil数を合わせ、
      局所平均を保てるかを見るためのprobe。
  - 専用計画: `docs/DITHER_BREAKTHROUGH_PLAN_JA.md`
- signed-power gamma再探索:
  - full `power gamma=1.15 bits10`: strict品質通過、order0 `34.73MB` 相当
  - full `power gamma=1.16/1.17/1.18 bits10`: center復元ではfail
  - `signed-log mean recon-table` で `1.16`, `1.17` がstrict通過へ戻る。
- `scripts/probe_channel_gamma_index.py`
  - full `power gamma=1.16 bits10` + signed-log mean recon-table +
    `small=15` + pred-bin context:
    - strict品質通過
    - estimated `25,201,632` bytes
    - log RMSE `3.514e-3`, p99 `8.416e-3`, gradient `1.275e-1`
  - full `power gamma=1.17 bits10` + same:
    - strict品質通過
    - estimated `24,821,075` bytes
    - log RMSE `3.638e-3`, p99 `8.789e-3`, gradient `1.313e-1`

判断更新: 数値上は `power gamma=1.17 bits10 + signed-log mean recon-table` で
`56.0MB -> 24.8MB` まで縮む道が見えたが、後述の目視確認で暗部階調と
ハイライトdetail lossが出たため、画質優先候補から外す。次のproduction候補は、
復元を実codec `signed-log bits10` と完全一致させたまま値streamだけを置き換える
`small=7 + category west_north_prev_channel + detail order0`。

品質判断更新:

- `power gamma=1.17 bits10` のトーンマップPNGは、暗部階調とハイライト
  ディテールに目視劣化が出たため、画質優先候補から外す。
- `sample_DSCF0009_crop0_signed-log_bits10_w4_g2.2_decoded.png` は目視で問題なし。
  今後の品質アンカーは実codec `signed-log bits10` とする。
- `scripts/audit_display_quality_regions.py` を追加。`white=4.0`, `gamma=2.2`
  表示空間で dark/mid/highlight/extreme を分けて誤差と勾配を測る。
- full comparison:
  - `signed-log bits10`
    - dark luma RMSE `1.239e-3`, dark grad NRMSE `2.483e-1`,
      dark lost-detail `10.75%`
    - highlight luma RMSE `1.840e-4`, highlight grad NRMSE `7.367e-3`
  - `power1.17 bits10 + signed-log mean`
    - dark luma RMSE `4.385e-3`, dark grad NRMSE `6.687e-1`,
      dark lost-detail `41.59%`
    - highlight luma RMSE `2.597e-4`, highlight grad NRMSE `1.028e-2`
- 判断: 既存の HDR log RMSE / p99 / gradient だけでは暗部階調や局所的な
  ハイライトdetail lossを拾い切れない。以後は表示空間・領域別指標を必須にする。

## 注意点

- `results/` は歴史的な探索ログが大量にある。古い結果を現在の性能と混同しない。
- `ml/` の生成データや log / manifest には、古い絶対パスが残っている可能性がある。
- source / docs / public API では `hdrcodec` ではなく `radiance_codec` を使う。
- sandbox から MPS が見えない場合がある。GPU 学習を本気で回すなら、sandbox 外または MPS が見える環境で実行する。
- exact 12x / 16x は夢として追う価値があるが、puresky の low mantissa tail と no-puresky の main payload は別の壁。
- near-lossless は逃げではなく、HDR 用ライブラリとしてかなり実用的な別モード。

## 再出発の合言葉

このプロジェクトの勝ち筋は、一般画像コーデックの模倣ではなく、float32 HDR の数値表現そのものを利用すること。

exact は「情報を消せない」ので、構造を見つけた分だけ勝つ。
near-lossless は「人間やレンダリング上ほぼ意味の薄い low mantissa を整理する」ことで、一気に 12x 以上へ寄せられる。
AI はそのあとで、残った hard tail の確率を読むために使う。
