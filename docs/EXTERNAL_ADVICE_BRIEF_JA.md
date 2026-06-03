# Radiance Codec 外部相談用ブリーフ

作成日: 2026-06-03

この文書は、`radiance_codec` の HDR/EXR float32 画像圧縮研究について、
外部の詳しい人に助言を求めるための要約です。実装履歴を全部読む必要がないよう、
現状、過去に失敗した方向、現在のボトルネック、相談したい論点をまとめます。

## 相談したいこと

float32 HDR/EXR 画像を、画質優先の near-lossless で圧縮しています。
現在の品質アンカーは `signed-log bits10` 相当で、これは目視上許容されたものです。

現状の最良実装は、主サンプル `sample_DSCF0009.EXR` に対して:

- 入力: `7728 x 5152 x 3`, float32, raw `477,775,872 bytes`
- 出力: `49,382,153 bytes`
- 圧縮率: `9.68x`
- 品質指標:
  - signed-log RMSE `7.574e-4`
  - gradient signed-log NRMSE `2.986e-2`
  - PSNR `78.78dB`
- 目視:
  - `signed-log bits10` は許容
  - `bits8` 系や一部の power/gamma 系は暗部階調やハイライトdetail lossで却下

目標は、画質を崩さずにここからさらに縮めることです。
理想的には既存の同思想RAW圧縮の `21MB` 級に勝ちたいですが、
現状の画質基準ではすぐ届く見込みはありません。

最新の検証仮説として、`signed-log bits8/9 + adaptive dither` を再検討します。
過去に却下したのは「素のbits8」であり、暗部bandingだけを誤差拡散ディザで潰せるなら、
bits10画質を `21-30MB` 級で実現できる可能性があります。実験計画は
`docs/DITHER_BREAKTHROUGH_PLAN_JA.md` に分離しています。

## 現在の方式

現在の主力は `StageLinearIndex` です。

1. 各channelのfloat32値を signed-log transform する
2. channelごとに global min/max を保存する
3. transform後の値を `bits10` の整数indexへ量子化する
4. decoderでは同じindexからfloat32を復元する
5. つまり、元のfloat32とはbit exactではなく、量子化済みfloat32を完全復元する

符号化対象は、元のfloat32 bit列ではなく `bits10 index plane` です。

現在の残差符号化:

- 予測器: MED predictor
- 残差: `(index - pred) mod 1024`
- signed residualへ変換
- small residual + escape:
  - category:
    - `0`
    - `+1..+7`
    - `-1..-7`
    - positive escape
    - negative escape
  - escape detail:
    - `abs(residual) - 8`
- category context:
  - `west category`
  - `north category`
  - `previous channel category`
- escape detail:
  - order-0 rANS
- 追加:
  - `channel fallback split`
  - 親contextごとに、channelで分割した方が得な場合だけ子modelを使う
  - X-Trans 6x6 phaseも試したが、このカテゴリ残差には効かなかった

実装上のbitstream mode:

- `ValueMode::SmallEscapeChannelSplitRans`
- LIDX version `8`
- v6/v7 stream decode互換は維持

## サイズ推移

同じ `signed-log bits10` 復元値を保ったまま、値streamだけ改善した推移です。

| 段階 | encoded bytes | ratio | 備考 |
|---|---:|---:|---|
| 初期 `signed-log bits10` | `56,021,152` | `8.53x` | mask/value分離、value stream支配 |
| small residual + escape 推定 | `49,561,883` | `9.64x` | 表コスト込み |
| `SmallEscapeRans` 実装 | `49,554,947` | `9.64x` | 推定とほぼ一致 |
| channel fallback split 推定 | `49,382,197` | `9.68x` | 親contextごとに分岐 |
| 現在実装 | `49,382,153` | `9.68x` | 品質値は同じ |

この方式の理論下限にかなり近づいています。
`small=7` 方式の理想エントロピーはおよそ `49.27MB` で、
現在実装との差は約 `0.1MB` から `0.3MB` 程度です。

したがって、今のentropy coder/context tableを磨くだけで大きく縮む余地は少ないと見ています。

## 現在のボトルネック

### 1. bits10 index residual がまだ重い

`signed-log bits10` の初期実装をauditした結果:

- mask: `11,021,878 bytes`
- value: `44,999,201 bytes`
- nonzero residual rate: `76.71%`
- value entropy: `3.856 bits/nonzero`

bits10では、非ゼロ残差が多く、値streamが支配的でした。
small residual + escapeでかなり整理できましたが、まだ全体で約 `3.3 bits/sample`
程度の情報量が残ります。

### 2. 追加contextは表コストに負けやすい

直積contextの例:

- baseline `west_north_prev_channel`: `49,561,883 bytes`
- `current channel` 追加:
  - payloadは少し下がる
  - model table増加で `49,620,217 bytes` に悪化
- `X-Trans 6x6 phase` 追加:
  - payloadは `49.03MB` 程度まで下がる
  - model tableが約 `9.04MB` まで膨らみ、総量 `58.06MB` に悪化

fallback splitでは:

- `channel`: 約 `173KB` 改善
- `X-Trans 6x6 phase`: split親 `0`

つまり、単純な高次元contextは厳しいです。
context mixing、fallback tree、低rank表現、tile限定適用などが必要そうです。

### 3. 見た目評価が数値評価より厳しい

過去には、数値上は良いが目視でNGの候補がありました。

特に:

- `bits8` 系:
  - HDR log RMSE等では通りそうに見える候補もあった
  - しかし暗部階調が乱れ、目視で却下
- `power gamma=1.17 bits10 + signed-log mean recon-table`:
  - 推定 `24.8MB` まで縮む
  - 数値指標ではかなり良い
  - しかし暗部階調とハイライトdetail lossが目視で却下

以後は、単純なRMSE/PSNRだけでなく、表示空間・領域別評価を必須にしています。

## 過去に試したが弱かった方向

### exact lossless

元々は真のlosslessで高圧縮を狙っていました。
しかし float32 mantissa tail の高エントロピー部分が強く、
汎用的に4x/8x以上を安定して狙うのは厳しいと判断しています。

exact losslessは完全には捨てていませんが、現在の主軸は画質優先 near-lossless です。

### bits7/bits8の攻めた量子化

サイズだけならかなり縮みます。
例:

- linear bits8:
  - `20,395,989 bytes`, `23.42x`
  - ただし log RMSEや目視が不足
- gamma075 bits8:
  - `27,456,725 bytes`, `17.40x`
  - 数値上は良いが、最終的にbits8系は暗部で不安

ユーザー評価では、素の `bits8` は暗部階調が厳しく、`bits10` が必要という判断でした。
ただし現在は、`bits8/9 + adaptive dither` により暗部bandingだけを抑える仮説を
再検証対象にしています。

### power/gamma transform + recon table

最も派手に縮んだ方向です。

- `power gamma=1.17 bits10 + signed-log mean recon-table`
- estimated `24,821,075 bytes`
- 数値上は良い
- 目視で暗部階調とハイライトdetail lossがあり却下

この方向は、品質評価を作り直すか、暗部/ハイライト保護を入れない限り本命にはしません。

### X-Trans phase context

サンプルは X-Trans由来データです。
そのため 6x6 phase が効く可能性を疑いました。

結果:

- 直積contextでは表コストが爆発
- fallback treeでもsplit親が `0`

少なくとも現在の `signed-log bits10 small residual category` には直接効いていません。
ただし、別表現、RAW/CFAに近い空間、tile分類では再検討余地があります。

### tile router / local range

tileごとのlocal rangeやmode selectorも試しました。
一部cropでは良く見えますが、本番sample全体では:

- 局所品質を守るため高bits側に寄る
- metadataやmode選択コストが重い
- global bits10の安全性に明確には勝ちにくい

tile route自体は有望ですが、現時点では単純なlocal range selectorは本命ではありません。

## 現在の実装・検証コマンド

主な実装:

- `codec/src/linear_index.cpp`
- `codec/src/linear_index_transform.cpp`
- Python API: `codec/python/radiance_codec.py`

主なprobe:

- `scripts/benchmark_linear_index_codec.py`
- `scripts/audit_linear_index_payload.py`
- `scripts/probe_small_escape_payload_budget.py`
- `scripts/probe_small_escape_context_split.py`
- `scripts/audit_display_quality_regions.py`

代表的な実行:

```bash
pixi run build
pixi run test-codec
pixi run python scripts/benchmark_linear_index_codec.py \
  --glob sample_DSCF0009.EXR \
  --crop-size 0 \
  --bits 10 \
  --transform signed-log \
  --limit 1
```

現行結果:

```text
bits10: signed-log 9.68x bytes=49382153
log_rmse=7.574e-04 grad=2.986e-02 psnr=78.78dB
```

## 外部の方に特に聞きたいこと

### A. 予測器

現在はMED predictorです。
次に効きそうな予測器は何でしょうか。

候補:

- CALIC/JPEG-LS系のcontext adaptive predictor
- channel間線形予測
- local gradient direction predictor
- edge-aware predictor
- low-rank/tile-adaptive predictor
- reversible/causalな小型線形モデル

制約:

- decoderが同じ予測を再現できること
- model metadataが重すぎないこと
- 画質は `signed-log bits10` 復元を維持したい

### B. context modeling

今の直積contextはすぐ表コストに負けます。
軽いまま効くcontext modeling案はありますか。

興味がある方向:

- context mixing
- fallback tree
- PPM/CTW的な文脈木
- hashed context with collision tolerance
- adaptive model without large static tables
- bitplane別context
- escape/tail限定の高次元context

### C. transform/index表現

`signed-log bits10` は見た目がよく、現在の品質アンカーです。
しかし圧縮率は `9.68x` 程度で頭打ちです。

別の値表現で、暗部階調とハイライトdetailを守りながら縮める案はありますか。

過去に失敗した注意点:

- global power/gammaは数値上良くても目視で破綻した
- bits8は暗部階調が厳しい
- recon tableは数値改善するが局所階調を壊す場合がある

### D. 画質指標

HDR/EXR float32に対して、目視劣化を拾いやすい指標を探しています。

現在使っているもの:

- signed-log RMSE
- signed-log p99
- gradient signed-log NRMSE
- display-space region metrics
  - dark
  - mid
  - highlight
  - extreme

欲しいもの:

- 暗部階調飛びを検出する指標
- ハイライトdetail lossを検出する指標
- トーンマップ依存を減らした評価
- HDR画像向けの実用的なperceptual metric

### E. RAW/CFA寄り表現

sampleはX-Trans由来です。
デモザイク後のRGB float32ではなく、疑似RAW/CFA的な分解へ戻すことで、
分散したノイズ構造を再集中できる可能性はありますか。

懸念:

- 入力はすでにEXR RGBであり、元RAWではない
- X-Trans 6x6 phaseを単純contextに入れても効かなかった
- 疑似CFA分解は画質リスクが高い

### F. 現実的な圧縮率目標

画質アンカー `signed-log bits10` を守る前提で、
`49.4MB` から `30MB` や `21MB` へ行く現実的な道があるか。

現在の感触:

- entropy coderの改善だけでは無理
- 予測器か値表現を変える必要がある
- ただし値表現を変えると暗部/ハイライトで破綻しやすい

この認識が正しいか、別の攻め方があるかを知りたいです。

## 注意点

- このプロジェクトは研究段階で、コードや結果JSONが大量にあります。
- 現在のworktreeには未コミットの実験変更が多いです。
- 速度最適化はほぼ未着手です。
  - 現行full encodeは数十秒から100秒程度
  - 背景で重い学習タスクが走っていることもあります
- 現在の主目的は速度ではなく、画質を守った圧縮率の改善です。

## 一言で言うと

現在は「画質上安全な `signed-log bits10` index表現」を、
MED residual + small/escape + context rANSで `49.4MB` まで圧縮できています。

この方式自体のentropy下限にはかなり近く、次の大きな改善には、
より良い予測器、より良い表現、または表コストを抑えたcontext modelingが必要です。
