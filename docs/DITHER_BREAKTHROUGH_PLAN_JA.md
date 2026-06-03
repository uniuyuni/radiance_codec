# Adaptive Dither 突破口検証計画

作成日: 2026-06-03

## 要約

現在の画質アンカーは `signed-log bits10` で、現行実装は
`sample_DSCF0009.EXR` を `49,382,153 bytes` (`9.68x`) まで圧縮できている。
しかし、この表現のentropy下限にはかなり近く、contextやentropy coderを磨くだけでは
`21-30MB` 級へ行く見込みが薄い。

新しい仮説:

> `bits10` が必要だった主因は、ノイズ詳細の保存ではなく、暗部banding回避である。
> ならば `signed-log bits8/9 + adaptive dither` で、暗部bandingだけを潰し、
> サイズは bits8/9 級に落とせる可能性がある。

この計画は、この仮説を低リスクに検証し、採用可否を判断するためのもの。

## 背景

`sunny-crunching-sky.md` の助言メモによる観察:

- bits深度を1増やすと、残差エントロピーがほぼ `+1.0 bit/sample` 増える。
  - bits8/9/10 の residual entropy が `2.93/3.86/4.83` に近い。
  - 追加LSBがほぼセンサーノイズである可能性が高い。
- signed-log値を median5 平滑化してからbits10量子化すると、残差entropyが
  `4.83 -> 3.31` へ下がる。
  - 残差の相当部分がノイズ由来であることを示す。
- banding-prone画素率は、おおよそ:
  - bits8: `3.5%`
  - bits9: `0.1%`
  - bits10: `0%`
- bits8へのディザコストは order-0見積もりで `+0.07 bit/sample` 程度。

解釈:

- bits10は全画素のノイズ詳細を保存するためではなく、
  主に暗部の少数画素で量子化半歩が自然ノイズを上回ることを避けている可能性がある。
- その少数画素だけにadaptive ditherを入れれば、暗部階調を守りつつ
  bits8/9相当のサイズへ落とせるかもしれない。

## A-F への現時点回答

### A. 予測器

現状ではMED predictorを維持する。

理由:

- `signed-log bits10` の残差はノイズ支配で、この表現内ではentropy下限に近い。
- 予測器改善は数%の余地はあるが、`49MB -> 21-30MB` の主戦場ではない。

ただし、dither適用後に残差分布が変わる可能性があるため、
bits8/9+dither routeでは再auditする。

### B. context modeling

静的な直積contextは表コストに負けやすい。

現状:

- `channel fallback split` は約 `173KB` 改善。
- `X-Trans 6x6 phase` はfallbackしてもsplit親が `0`。

次にやるなら:

- adaptive/online binary model
- logistic context mixing
- fallback tree
- bitplane別context
- hard/tail tile限定context

ただし、dither仮説の検証中は優先度を下げる。

### C. transform/index表現

signed-log transformは維持する。

変更するのは:

- `bits10 -> bits8/9`
- quantization stageへadaptive ditherを追加

避けるもの:

- global power/gamma + recon-table route
  - 数値上は良かったが、暗部階調とハイライトdetail lossで目視却下済み。

### D. 画質指標

追加すべき指標:

- banding-risk map
  - 局所ノイズ標準偏差と量子化半歩を比較する。
  - `local_std < quant_step / 2` 付近を危険領域とみなす。
- 暗部領域での階調連続性指標
- ハイライトdetail retention
  - ただしノイズ保持に引っ張られる可能性があるため、目視も必須。

既存指標:

- signed-log RMSE
- signed-log p99
- gradient signed-log NRMSE
- display-space dark/mid/highlight/extreme metrics

### E. RAW/CFA寄り表現

現時点では低優先。

理由:

- 入力は既にEXR RGBであり、元RAWではない。
- X-Trans 6x6 phase contextは現在のカテゴリ残差には直接効かなかった。
- 疑似CFA分解は画質リスクが高い。

ただし、dither routeが失敗した場合には、暗部ノイズ構造を再集中する方向として再検討する。

### F. 現実的な圧縮率目標

現行bits10 route:

- `49.4MB`
- entropy coder改善だけで大幅改善する見込みは薄い。

dither仮説が成立した場合:

- `signed-log bits9 + adaptive dither`: `~30MB` 級が現実候補
- `signed-log bits8 + adaptive dither`: `~21MB` 級だが、2026-06-03 の
  3段持ち上げ目視では不採用寄り

成立条件:

- 暗部bandingがbits10並みに消えること
- ハイライトdetail lossが発生しないこと
- ディザ粒状感が許容範囲であること
- 暗部を3 stops持ち上げた検査表示でも、ノイズの輝度が周囲から浮かないこと
- codec実測でbits8/9級のサイズを維持できること

## 推奨実装順

### Step 1. Python probeでdither indexを作る

まずC++本体は触らない。

追加済み:

- `scripts/probe_dithered_linear_index.py`

実装内容:

- 入力EXRを読む
- signed-log transform
- channelごとのglobal min/maxを取る
- bits8/9へ量子化する
- quantization errorをFloyd-Steinbergで周辺画素へ拡散する
- adaptive maskにより、banding-prone領域だけditherを適用するモードも作る
- dither済みindexを使って:
  - 品質指標
  - PNG preview
  - manifest JSON
  を出す
- 出力先は `outputs/previews/dither_breakthrough/`。
  既存preview直下が混ざらないよう、現在フェーズ専用サブディレクトリに分ける。

未実装:

- residual entropy
- small/escape payload budget
  - 視覚確認後に、採用候補だけ測る。

最初の候補:

- `bits8`, no dither
- `bits8`, full FS dither
- `bits8`, adaptive FS dither
- `bits9`, no dither
- `bits9`, adaptive FS dither
- `bits10`, current anchor

### Step 2. banding-risk metricを追加

追加先:

- `scripts/audit_display_quality_regions.py`
  または専用probe

初期案:

- signed-log空間で局所標準偏差を測る
- channelごとの量子化stepを計算する
- `local_std < k * quant_step` をbanding-prone候補にする
- dark/mid/highlight別に危険率を出す

出したい値:

- banding-prone pixel rate
- dark banding-prone rate
- dither適用率
- bits8/9/10比較

### Step 3. 視覚確認PNGを書き出す

必須。

出力候補:

- `outputs/previews/sample_DSCF0009_crop0_signed-log_bits8_no-dither_w4_g2.2_decoded.png`
- `outputs/previews/sample_DSCF0009_crop0_signed-log_bits8_adaptive-dither_w4_g2.2_decoded.png`
- `outputs/previews/sample_DSCF0009_crop0_signed-log_bits9_adaptive-dither_w4_g2.2_decoded.png`
- `outputs/previews/sample_DSCF0009_crop0_signed-log_bits10_current_w4_g2.2_decoded.png`

見る場所:

- 暗部平坦部
- 暗部グラデーション
- ハイライトdetail
- エッジ周辺

判定:

- bits9+ditherで暗部bandingが消えるか
- ディザ粒状感が許容か
- ハイライトに余計なノイズが乗らないか
- bits9+ditherが安全寄り候補になるか

### Step 4. codec payload見積もり

dither済みindexに対して、現在のsmall/escape/context modelを流用して見積もる。

重要:

- order-0で `+0.07 bit/sample` でも、現行context coderで同じとは限らない。
- FS ditherは残差構造を荒らす可能性がある。
- そのため、必ず現行のpayload budgetで測る。

見る値:

- residual entropy
- nonzero residual rate
- small/escape rate
- estimated bytes
- model table cost

### Step 5. C++実装

Python probeと目視で成立した場合だけ進む。

実装場所:

- `codec/src/linear_index.cpp`

方針:

- quantization stageにdither optionを追加
- decode側は不変
  - indexから復元するだけ
- bitstreamにはdither modeだけ保存
  - full/adaptive/off
  - threshold parametersが必要なら保存

注意:

- deterministicであること
- platform差が出ないよう、float演算の丸めに注意
- adaptive ditherの判定もdecoderには不要
  - dither済みindexを符号化するため

## 採否基準

### 採用

以下を満たす場合:

- bits9+ditherが目視でbits10とほぼ同等
- かつ `30MB` 前後まで落ちる

または:

- bits8+ditherは現時点では採用条件から外す。再浮上するなら、bits9で
  ノイズ注入方式が確立してから再評価する。

### 保留

- 暗部bandingは消えるが、粒状感が強い
- ハイライトにノイズが乗る
- sizeがbits10 routeから十分改善しない
- metricsは良いが目視で違和感がある

### 却下

- bits9+ditherでも暗部階調が不自然
- 暗部bandingがノイズに置き換わっただけで画質優先に合わない
- residualが荒れて圧縮サイズが期待より大きくなる

## 最初に実行するコマンド案

probe作成後:

```bash
pixi run python scripts/probe_dithered_linear_index.py \
  --glob sample_DSCF0009.EXR \
  --crop-size 1024 \
  --bits 8,9,10 \
  --dither none,adaptive-fs \
  --output-dir outputs/previews/dither_breakthrough
```

フル検証:

```bash
pixi run python scripts/probe_dithered_linear_index.py \
  --glob sample_DSCF0009.EXR \
  --crop-size 0 \
  --bits 8,9,10 \
  --dither none,adaptive-fs \
  --output-dir outputs/previews/dither_breakthrough
```

## 未決事項

- adaptive ditherの判定をどう作るか
  - local std
  - local gradient
  - dark region限定
  - quantization step比
- Floyd-Steinbergが最適か
  - blue-noise ordered dither
  - error diffusion with clipping
  - serpentine scan
- channelごとのdither制御
- ハイライトへのdither禁止条件
- preview cropをどこにするか

## 2026-06-03 視覚確認メモ

crop1024 previewでのユーザー評価:

- 最も良い: `bits9 adaptive-fs std0.25 dark0.18 r2`
- `bits8` 版も許容範囲
- ただし両方とも、ノイズの入り方が強く、輝度が周囲と合わず少し浮いて見える
- オリジナルに近いのは `bits9`
- `bits8` は、この浮きが修正できれば採用候補になり得る
- 追加評価:
  - `s0.35` / `s0.5` へFS強度を変えても、浮き方は大きく変わらない。
  - 問題は強度より「ノイズの入れ方」そのものの可能性が高い。
  - 現時点では `bits9` が断然良い。`bits8` は保留。
- さらに追加評価:
  - 目視はすべて暗部を約3 stops持ち上げた検査条件。
  - 通常表示では真っ暗に近く、差はほぼ見えない。
  - それでも画質優先の検査条件では `bits8` は不採用。
  - `bits9` の課題は量子化深度そのものより、暗部へのノイズ注入が周囲の
    輝度・質感に馴染まないこと。
- 原因の修正:
  - 浮いて見えたノイズは、追加したditherノイズではなく `bits9 none` の時点で
    既に存在するデコード後ノイズだった。
  - 「浮く」というより、ノイズが周囲より少し暗く見える。
  - したがって次の本命は dither strength / pattern ではなく、
    reconstruction table または暗部限定のdequant bias。
  - 暗部限定 `+0.125/+0.25/+0.375/+0.5 LSB` の単純biasは、目視で大差なし。
    一律持ち上げでは原因を解決しない可能性が高い。
  - 次はbinごとの代表値を変える `signed-log-mean` / `value-mean`
    reconstruction table を確認する。
  - 追加判断:
    - 暗部だけ `bits10` に逃がすなら、非暗部の `bits8` は再び候補になる。
    - `bits8` は全画素に一律適用する候補ではなく、routerの軽量routeとして扱う。
    - crop1024 probeでは `bits8 + darkbits10` が、素の `bits9 none` より
      log/gradient数値で良い。
    - ただし `darkmax0.18/0.25` は広すぎる可能性が高い。
      full `sample_DSCF0009.EXR` のluma率は:
      - `<=0.05`: `42.8421%`
      - `<=0.08`: `55.2056%`
      - `<=0.10`: `63.4057%`
      - `<=0.12`: `70.1342%`
      - `<=0.18`: `79.3038%`
      - `<=0.25`: `83.9008%`
    - crop1024では `darkmax0.05` でも `82.43%` がdark対象で、暗い検査cropとしては
      サイズ判定に厳しすぎる。
    - crop1024 payload probe:
      - `bits8+dark10 max0.05`: `1,303,770 bytes`, `9.65x`,
        log `1.050e-3`, grad `1.868e-1`
      - `bits8+dark10 max0.08`: `1,327,093 bytes`, `9.48x`,
        log `9.960e-4`, grad `1.771e-1`
      - `bits8+dark10 max0.12`: `1,379,861 bytes`, `9.12x`,
        log `8.388e-4`, grad `1.493e-1`
      - `bits9+dark10 max0.05`: `1,301,103 bytes`, `9.67x`,
        log `6.789e-4`, grad `1.207e-1`
      - `bits9+dark10 max0.08`: `1,317,172 bytes`, `9.55x`,
        log `6.624e-4`, grad `1.177e-1`
      - `bits9+dark10 max0.12`: `1,351,407 bytes`, `9.31x`,
        log `6.177e-4`, grad `1.098e-1`
    - このcropでは `bits8` と `bits9` のrouterサイズ差は小さい。暗部refinementが
      支配的なので、次はdark対象をluma閾値だけでなくbanding-riskで絞る。
    - 比較サンプルが増えすぎると目視評価が疲れるため、以後は原則2候補までに絞る。
    - `bits8+dark10 max0.08` のmask比較:
      - `luma`: dark対象 `84.88%`, `1,327,093 bytes`, `9.48x`,
        log `9.960e-4`, grad `1.771e-1`
      - `dark && banding-risk`: dark対象 `65.24%`, `1,163,255 bytes`, `10.82x`,
        log `1.373e-3`, grad `2.424e-1`
      - banding-risk maskはサイズを大きく改善するが、数値品質は落ちる。
        まずこの見た目が許容か確認する。
    - ユーザー目視:
      - `luma mask` の方が良い。
      - ただし `banding-risk mask` でも比較しないと分からない程度。
      - 差が出るのは「真っ暗ではないが暗い」shadow transition、画像では右下の木付近。
      - `bits9 none` 以外の差は小さく、dark10 escape自体は有効。
    - full `sample_DSCF0009.EXR`, `bits8+dark10 max0.08 banding-risk`:
      - dark10対象 `20.11%`
      - mask `596,217 bytes`
      - refinement `6,826,917 bytes`
      - quality: log `2.738e-3`, p99 `5.374e-3`, grad `1.074e-1`
      - 既存 `signed-log bits8` base `29,674,477 bytes` を使う概算:
        `37,097,671 bytes`, `12.88x`
      - `bits10` より少し軽い品質寄りrouteにはなるが、21MB級にはまだ遠い。
        次はbaseを `signed-log bits8` 以外へ逃がすか、refinement対象をさらに絞る。
      - 通常表示 `white=4.0, gamma=2.2` のfull previewでは、ユーザー目視で
        オリジナルと区別がつかない。
      - preview:
        `outputs/previews/highlight_guard/sample_DSCF0009_crop0_signed-log_bits8_none_darkbits10_darkmax0.08_darkmaskbanding_w4_g2.2_decoded.png`
    - detail消失指標:
      - `scripts/benchmark_linear_index_codec.py` の `error_stats` に表示空間
        `white=4.0, gamma=2.2` の領域別detail指標を追加。
      - 追加キー例:
        - `display_w4_g22_highlight_1_4_lost_detail_rate`
        - `display_w4_g22_highlight_1_4_grad_energy_ratio`
        - `display_w4_g22_extreme_gt4_lost_detail_rate`
        - `display_w4_g22_extreme_gt4_grad_energy_ratio`
      - benchmark標準出力にも `hi_lost` / `ex_lost` を表示。
      - `lost_detail_rate` は、元画像の表示輝度勾配が見える水準なのに、
        復元側で半分未満へ弱くなった勾配の割合。
    - 20M最接近候補:
      - `gamma0.8 bits8 + signed-log-mean recon + pred-bin context`
      - estimated `21,706,344 bytes`, `22.01x`
      - quality: log `3.997e-3`, p99 `8.203e-3`, grad `1.535e-1`
      - detail指標:
        - dark lost `40.07%`
        - mid lost `15.67%`
        - highlight lost `5.61%`
        - extreme lost `11.27%`
      - 通常表示preview:
        `outputs/previews/20mb_route/sample_DSCF0009_crop0_power0.8_bits8_signed-log-mean_w4_g2.2_decoded.png`
      - 判断:
        - sizeは20M目標に最も近い。
        - ただし暗部detail lossが大きく、ユーザー目視でも暗部グラデーションの
          階調飛びが確認された。単独採用は不可。
        - 暗部救済込みのfull見積もり:
          - `darkbits10 max0.08 banding-risk`: `30,665,604 bytes`, `15.58x`,
            dark対象 `23.85%`, mask `940,138 bytes`, refine `8,019,058 bytes`,
            log `3.719e-3`, grad `1.453e-1`
          - `darkbits9 max0.08 banding-risk`: `27,997,428 bytes`, `17.06x`,
            refine `5,350,882 bytes`, log `3.778e-3`, grad `1.475e-1`
          - `darkbits9 max0.05 banding-risk`: `27,612,179 bytes`, `17.30x`,
            dark対象 `22.72%`, mask `811,141 bytes`, refine `5,094,630 bytes`,
            log `3.799e-3`, grad `1.482e-1`
        - 判断:
          - `gamma0.8 bits8` は20Mに近いが、画質を救うと `27-31MB` 圏へ戻る。
          - 20M達成には、暗部補正を足す発想だけでは不足。
          - 次の主戦場は entropy coder / base stream の実装改善、または
            visually safe な別base表現の探索。
    - 2026-06-03 追加サンプル:
      - `sample_bright_park.EXR`
        - shape `5152x7728x3`, float, raw `477,775,872 bytes`
        - luma `<=0.08`: `32.01%`
        - `bits8+dark10 max0.08 banding-risk`: dark10対象 `0.85%`,
          mask `124,034 bytes`, refinement `283,327 bytes`,
          log `2.195e-3`, grad `9.126e-2`
        - `signed-log bits8` base実測 `43,114,889 bytes`, `11.08x`
        - router概算 `43,522,310 bytes`, `10.98x`
      - `sample_middle_flower.EXR`
        - shape `7728x5152x3`, float, raw `477,775,872 bytes`
        - luma `<=0.08`: `2.94%`
        - `bits8+dark10 max0.08 banding-risk`: dark10対象ほぼ `0%`,
          mask `578 bytes`, refinement `140 bytes`,
          log `1.795e-3`, grad `1.323e-1`
        - `signed-log bits8` base実測 `50,512,430 bytes`, `9.46x`
        - router概算 `50,513,212 bytes`, `9.46x`
      - 判断:
        - 明るい/低暗部サンプルでは dark10 refinement はほぼ問題ではない。
        - ただしbase bits8 streamが重く、目標サイズの主ボトルネックは
          `dark10` ではなく base index stream に戻る。
        - 新サンプル2枚は評価セットに組み込む価値がある。

対応:

- `scripts/probe_dithered_linear_index.py` に以下を追加:
  - `--diffusion-strength`
  - `--error-clip`
  - `adaptive-ordered`
  - `--ordered-amplitude`
  - `adaptive-block`
  - `--block-size`
  - `--recon-bias`
  - `--recon-bias-dark-max`
  - `--recon-table`
- 追加preview:
  - `s0.5_clip0.5`
  - `s0.35_clip0.35`
  - `s0.25_clip0.25`
  - `adaptive-ordered amp1.0`
  - `adaptive-ordered amp0.5`
  - `adaptive-block b4/b8/b16`
  - `bits9 none` の dark-only reconstruction bias `0.125/0.25/0.375/0.5`
  - `bits9 none` の `signed-log-mean` / `value-mean` reconstruction table
  - `bits9 + darkbits10`
  - `bits8 + darkbits10`
- 意図:
  - FS誤差拡散を弱め、局所輝度として浮く粒状ノイズを抑える。
  - 弱くしすぎるとbanding抑制力が落ちるため、目視で最適点を探る。
  - ordered系は誤差を周囲へ流さないので、FS由来の浮き方が変わるかを見る。
  - block系はブロック内のfloor/ceil数を合わせ、局所平均を保ったまま
    bandingを崩せるかを見る。
  - recon-bias系は、暗部デコードノイズが下側へ寄って見えるかを切り分ける。
  - recon-table系は、一律biasではなくbin内の実分布に合わせた代表値で
    黒い粒状感が減るかを見る。

次の確認優先度:

1. `bits8 + darkbits10 darkmax0.08`, luma mask と banding-risk mask の2候補だけを見る
2. banding-risk maskが目視許容なら、full画像payload見積もりへ進む
3. 許容できなければ `std_step_threshold` を少し広げる
4. 採用候補だけfull画像payload見積もり
5. codec実装は、mask/refinementのpayloadが十分小さいと確認してから

## 一言で

今の `49.4MB` は、センサーノイズをbits10で律儀に運んでいるコストが大きい。
もしbits10が必要だった理由が暗部banding回避だけなら、
`signed-log bits8/9 + adaptive dither` は最も期待値の高い次手になる。
