# 真のロスレス 12x ロードマップ

作成日: 2026-06-02

このメモは near-lossless ではなく、float32 のビット列を完全一致で復元する
exact / true lossless だけを対象にする。`StageMantissaQuantize` や low mantissa
zeroing は別の製品モードとして有望だが、ここでは 12x 達成の根拠には使わない。

## 0. 現在地

現行 exact の主力は `StageGroupedDelta` / GDXB 系。既存ログから見ると、単一の
改善で全体を押し上げる局面ではなく、画像タイプごとに別の壁に当たっている。

- full 13 画像、effort9 系: geomean 約 `7.317x`
- full 13 画像、effort11 既存監査からの再計算: geomean `7.493x`
- full 13 画像、effort11 の byte-weighted total ratio: `3.373x`
- full 13 画像、effort11 で 12x 以上: `2/13`
- full 13 画像、effort11 で 12x までに必要な削減量: 約 `20.9 MiB`
- realistic-no-puresky crop128, effort12: geomean `8.043x`
- puresky-hard crop128, effort12: geomean `2.347x`

12x は float32 sample あたり `32 / 12 = 2.667 bits` しか使えない。puresky の
低位 tail だけで約 `11.25 bits/sample` を消費しているため、puresky を exact
12x に乗せるには tail を捨てるのではなく、decoder-visible な情報から予測できる
形へ変える必要がある。

## 1. ボトルネックの分解

### puresky-hard

`results/gdx8_stream_budget_puresky-hard_effort12_crop128.json`:

- geomean `2.347x`
- encoded stream の `tail_payload` が約 `82.1%`
- `main_payload` は約 `17.9%`

`results/budget_x8_star_exr_effort11_crop0.json` を 12x 目標で再計算:

- `ph_belfast_sunset_puresky_1k`: `2.47x`, 12x gap 約 `2633 KiB`,
  `body_low_0_14 = 11.25 bits/sample`
- `ph_kloppenheim_06_puresky_1k`: `2.40x`, 12x gap 約 `2727 KiB`,
  `body_low_0_14 = 11.27 bits/sample`

見た目の滑らかな空は、exact では滑らかではない。低位 mantissa tail は RGB で
ほぼ高エントロピー、alpha だけが定数に近い。普通の spatial predictor や汎用
backend では届かない。

### realistic-no-puresky

`results/gdx8_stream_budget_realistic-no-puresky_effort12_crop128.json`:

- geomean `8.043x`
- byte-weighted total ratio `6.780x`
- encoded stream の `main_payload` が約 `99.8%`
- `tail_payload` はほぼゼロ

ここは puresky tail 問題ではない。half/bfloat-like なタイルでは低位 mantissa は
既にゼロまたは小さいため、残りは `body_mid_15_20` と `body_high_21_30` の
表現効率、mode signaling、bitplane context の問題になる。

### synthetic / random stress

`synth_noise`, `synth_mixed`, `synth_rgba_hdr` は低位 body が `11-15 bits/sample`
級で、12x どころか 8x も exact では厳しい。ここは「写真 HDR 目標から除外する」
か、「incompressible certificate を出す」か、「入力生成過程に由来する別の構造を
見つける」かを明示する必要がある。

## 2. 文献からの再整理

### FPZIP / floating-point array compression

一次情報:

- [LLNL fpzip](https://computing.llnl.gov/projects/fpzip)
- [LLNL Floating Point Compression](https://computing.llnl.gov/projects/floating-point-compression)

FPZIP は lossless / lossy の多次元浮動小数点配列用 compressor で、空間相関の
ある regular arrays を前提にする。LLNL の概要では exact lossless の fpzip は
データ精度と滑らかさに依存しておおむね `1.5x-4x` 程度とされている。

示唆:

- 現行 GDXB はこの範囲を大きく超えているので、単純に fpzip 型へ戻る価値は低い。
- ただし「浮動小数点をそのまま LZ に投げない」「予測、整数化、bitplane coding」
  という設計思想は現行路線と一致する。
- puresky tail の `11+ bits/sample` は、fpzip 系でも自然には消えないはず。

### zfp reversible mode

一次情報:

- [zfp compression modes](https://zfp.readthedocs.io/en/release1.0.0/modes.html)
- [LLNL zfp](https://computing.llnl.gov/projects/zfp)

zfp は 4^d ブロック単位の数値配列 compressor で、reversible mode では
floating-point data を bit-for-bit に復元できる。

示唆:

- exact で試す価値があるのは zfp そのものではなく、ブロックごとの可逆変換思想。
- `ordered-float body` に対して 4x4 / 8x8 の lifting, Lorenzo, Haar-like 変換を
  可逆整数演算で入れ、変換後に既存 bitplane rANS を当てるルートが次候補。
- puresky 低位 tail にも試すが、まずは non-puresky main payload の削減候補。

### FPC / Gorilla / Chimp 系 XOR coding

一次情報:

- [FPC author page](https://userweb.cs.txstate.edu/~mb92/research/FPC/)
- [FPC paper PDF](https://userweb.cs.txstate.edu/~mb92/papers/tc09.pdf)
- [Gorilla paper PDF](https://www.vldb.org/pvldb/vol8/p1816-teller.pdf)
- [Chimp paper PDF](https://www.vldb.org/pvldb/vol15/p3058-liakos.pdf)

FPC は predictor と XOR residual、leading-zero 方向の符号化で線形 double stream を
高速に扱う。Gorilla / Chimp は time-series 向けに XOR 差分と zero run / leading
zero 情報を強く使う。

示唆:

- 既存ログでは FPC / Gorilla 的 route は単体で GDXB より弱い。
- ただし time-series 的な EXR channel、deep sample、scanline 由来データには
  per-tile fallback として残す価値がある。
- exact 12x の主戦場ではなく、classifier が選ぶ局所 route として扱う。

### FLIF / JPEG XL modular / MANIAC / MA tree

一次情報:

- [FLIF specification](https://flif.info/spec.html)
- [FLIF ICIP 2016 paper](https://flif.info/papers/FLIF_ICIP16.pdf)
- [JPEG XL ISO/IEC 18181-1](https://www.iso.org/standard/85066.html)
- [JPEG XL white paper](https://ds.jpeg.org/whitepapers/jpeg-xl-whitepaper.pdf)

FLIF の MANIAC は、確率だけでなく context model 自体を画像ごとに適応させる
tree-based な entropy coding。JPEG XL modular も lossless image coding で MA tree,
predictor, reversible transforms を使う系譜にある。

示唆:

- ローカルの positive signal と一致する。`grouped_context_tree` は小さいながら
  exact で生きている数少ない方向。
- ただし puresky low-tail の large dictionary は side information で負けた。
  次は raw dictionary ではなく、MDL で costed split のみを送る小さな tree。
- tree の対象は puresky tail だけではなく、non-puresky main payload の高コスト
  bitplane に広げる。そこで side-info が amortize される可能性が高い。

### CCSDS-123 hyperspectral lossless

一次情報:

- [CCSDS all publications entry](https://ccsds.org/publications/allpubs/entry/3211/)
- [NASA NTRS: Issue 2 overview](https://ntrs.nasa.gov/citations/20180006784)

CCSDS-123 は multispectral / hyperspectral imagery 向けの低複雑度 predictive
coding 標準。Issue 2 は near-lossless も持つが、lossless path の核は spectral /
spatial prediction と residual entropy coding。

示唆:

- EXR RGBA は hyperspectral ほど channel 数は多くないが、channel 間 predictor,
  local sum, adaptive residual mapping の考え方は使える。
- 特に half-like / bfloat-like なタイルでは、float bitplane ではなく source-precision
  aware な residual value model を再試験する価値がある。

### ALP / decimal-origin floating point

一次情報:

- [ALP paper PDF](https://ir.cwi.nl/pub/33334/33334.pdf)
- [ALP repository](https://github.com/cwida/ALP)

ALP は double の lossless compression で、decimal-origin な値を整数として扱う
PseudoDecimals 系と、front-bit compression を適応的に使う。

示唆:

- EXR でも half/bfloat/decimal/fixed-grid 由来タイルは多い。
- 既存の `source_precision` classifier を codec decision にもっと強く使う。
- exact 12x では「値が浮動小数点だから float として圧縮する」より、「実際の発生源
  が half, bfloat, decimal, fixed grid のどれか」を先に判定するべき。

### OpenEXR compression

一次情報:

- [OpenEXR Technical Introduction](https://openexr.com/en/latest/TechnicalIntroduction.html)

OpenEXR の ZIP/PIZ などは lossless で、PIZ は wavelet + Huffman、ZIP は隣接差分 +
deflate の系統。写真や film grain では典型的に 35-55% 程度に縮むという説明がある。

示唆:

- 現行 GDXB は OpenEXR 標準 compression より強い領域にいる。
- PIZ の wavelet 思想は、float 全体ではなく ordered integer body への可逆局所変換
  として再解釈すると試す価値がある。

## 3. 12x へ向けた候補ルート

### Route A: MDL-coded signaled context tree

優先度: 最高

目的:

- `body_mid_15_20` / `body_high_21_30` の main payload を削る。
- puresky tail にも適用するが、puresky 単独突破より main payload 側の gain を
  先に狙う。

仕様案:

- decoder-visible features:
  - channel
  - bit index
  - exponent bucket
  - higher mantissa/body bits
  - previous higher bits
  - W/N/NW current-bit contexts
  - x/y parity and tile position
  - source_precision class
- split operation だけを bitstream に送る。
- leaf は binary probability table または small family selector。
- training cost, split metadata, leaf stats をすべて含めた MDL score で採択する。
- per-tile では side-info が重すぎる場合、group-of-tiles または image-local tree に
  する。

成功条件:

- realistic-no-puresky crop128 で `8.04x -> 8.8x` 以上の局所改善。
- full realistic-no-puresky で main payload `5%` 以上削減。
- puresky crop128 で既存 hybrid leaves32 を明確に超える。

停止条件:

- side-info 込みで main payload `2%` 未満の改善なら C++ 実装しない。
- puresky tail が `11.3 -> 9 bits/sample` 程度にも落ちないなら、puresky 12x route
  ではなく hard certificate 側へ回す。

次の script:

```text
scripts/probe_signaled_context_tree_mdl.py
```

軽量スモーク:

```text
pixi run python scripts/probe_signaled_context_tree_mdl.py --glob 'ph_abandoned_tiled_room_1k.exr' --crop-size 64 --bit-min 15 --bit-max 20 --max-leaves 8 --no-save
```

初回スモーク結果:

- `ph_abandoned_tiled_room_1k`, crop64, bits15-20, leaves8:
  existing `9.837x`, hybrid `10.043x`, gain `1.0210`
- `ph_belfast_sunset_puresky_1k`, crop64, bits0-14, leaves8:
  existing `2.802x`, hybrid `2.838x`, gain `1.0130`
- 同 puresky tail に `--feature-set with_ordered_high` を足しても hybrid `2.840x`

解釈: これは小さな構文・動作確認で、性能結論ではない。ただし現時点の兆候は
ロードマップ通りで、puresky tail 単独突破より non-puresky main payload 側を優先する。

### Route B: reversible ordered-body block transform

優先度: 高

目的:

- non-puresky の main payload を、予測 residual bitplane ではなく可逆 transform
  coefficient として再表現する。

仕様案:

- ordered float body を channel-wise 16-31bit 程度の整数 field として扱う。
- 4x4 / 8x8 block に reversible lifting を適用する。
- candidate:
  - reversible Haar 2D
  - Lorenzo predictor residual
  - gradient / plane predictor residual
  - channel decorrelation after spatial lifting
- block ごとに transform route と current GDXB route を exact cost で選ぶ。
- transform metadata は bitplane family selector より安くする。

成功条件:

- `oexr_ScanLines_Tree`, `Cannon`, `CandleGlass` で `body_mid/high` を 10%以上削減。
- ph non-puresky の half/bfloat-like tiles で悪化しない fallback を作る。

停止条件:

- crop128 で variable tile split 程度の小幅 gain しか出ないなら C++ 実装しない。

次の script:

```text
scripts/probe_ordered_body_block_transform.py
```

### Route C: source-precision aware exact routes

優先度: 高

目的:

- half_like / bfloat_or_coarser / decimal_or_fixed_grid を早期分類し、それぞれ専用の
  exact 表現へ送る。

仕様案:

- classifier を tile だけでなく channel/tile に細分化する。
- exact binary16 route:
  - half-convertible の body/sign を half index stream として符号化。
  - exponent/mantissa を float32 ordered body ではなく half value domain で扱う。
- bfloat/coarse route:
  - low body zero を前提に high/mid だけを residual value coding。
- decimal/fixed-grid route:
  - ALP 的に scale を推定し、integer residual と outlier map を送る。

成功条件:

- `ph_abandoned`, `ph_spruit`, `ph_studio` の full で geomean `+0.5x` 以上。
- OpenEXR scanline half_like で gap の一部を削る。

停止条件:

- route signal と outlier map を含めて GDXB と同等以下なら、classifier は speed
  pruning のみに使う。

次の script:

```text
scripts/probe_source_precision_routes.py
```

### Route D: puresky exact-tail certificate

優先度: 中だが重要

目的:

- puresky exact 12x が可能かどうかを、感覚ではなく条件付きエントロピーと
  side-info lower bound で判定する。

仕様案:

- decoder-visible feature set を固定する。
- low15 tail の conditional entropy を測る。
- 同時に model/table/tree を送るための最小 side-info を見積もる。
- impossible ではなく「この feature set と side-info budget では不可」と書ける
  certificate にする。

成功条件:

- low15 tail を side-info 込みで `11.3 -> 6 bits/sample` 未満へ落とす route が見つかる。
  それでも 12x には遠いが、研究継続の根拠になる。

停止条件:

- 追加 feature でも `9 bits/sample` を切れない、または table support が sample 数に
  近い場合は、puresky exact 12x を primary claim から外す。

次の script:

```text
scripts/probe_puresky_tail_certificate.py
```

### Route E: AI context mixer

優先度: 後回し

目的:

- 生成復元ではなく、exact entropy model の probability predictor としてだけ使う。

条件:

- Route A-D で hard bitplanes と feature set を確定してから。
- model weights の送信はしない。固定モデルまたは小さな signaled calibration のみ。
- decode speed と deterministic reproducibility を最優先。

現時点では、画像 autoencoder や lossy base + correction stream は exact 12x の本命ではない。
correction が大きくなり、結局 low tail を保存する問題に戻る。

## 4. すぐやる実験順

1. 12x dashboard を確定する。

```text
pixi run python scripts/audit_16x_budget.py --target-ratio 12 --effort 11 --crop-size 0
pixi run python scripts/audit_16x_budget.py --target-ratio 12 --effort 12 --crop-size 128
pixi run python scripts/audit_gdx8_stream_budget.py --corpus realistic-no-puresky --effort 12 --crop-size 128 --save
pixi run python scripts/audit_gdx8_stream_budget.py --corpus puresky-hard --effort 12 --crop-size 128 --save
```

2. `probe_signaled_context_tree_mdl.py` を書く。

- まず Python estimate でよい。
- metadata cost を必ず入れる。
- puresky tail と realistic main payload の両方に同じ scorer を当てる。

3. `probe_ordered_body_block_transform.py` を書く。

- GDXB payload before/after を bitplane cost で比較する。
- transform route は必ず per-block fallback ありにする。

4. source precision route を codec decision へ近づける。

- `audit_source_precision.py` の class を mode search pruning と route selection に使う。
- low-tail zero / half-convertible / bfloat-like / decimal-grid を channel-tile 単位で出す。

5. puresky certificate を作る。

- exact 12x を諦めるためではなく、どの情報が足りないかを特定するため。
- ここで見つかった低 support feature だけを Route A に戻す。

## 5. 判断基準

12x には 2 つの定義がある。

- geomean 12x: 個別の大きいファイルをある程度無視できる。研究目標としては到達しやすい。
- byte-weighted total 12x: 保存容量の実利に近い。puresky / synthetic の重さが直接効く。

exact で製品主張にするなら、少なくとも次を分けて表示する。

- photographic HDR exact geomean
- photographic HDR exact byte-weighted
- puresky-hard exact
- synthetic/random stress exact
- near-lossless optional mode

今の結論は悲観ではなく、分岐である。

- puresky は tail を予測可能にしない限り 12x は遠い。
- non-puresky は main payload を削れば 8x 台から伸びる余地がある。
- synthetic noise は codec 改善対象ではなく entropy certificate 対象に寄せる。
- AI は復元器ではなく、最後の entropy context mixer としてだけ戻す。
