# near-lossless 画質検証基準メモ

## 目的

`radiance_codec` の現在の主軸は、float32 EXR の完全一致ではなく、
RAW圧縮に近い思想の「編集しても機械/人間が実用上見分けにくい」near-lossless。

ただし、単なる視覚圧縮ではない。HDR/RAW相当の素材として扱うため、
通常表示だけでなく、露出補正後・暗部持ち上げ後・ハイライト確認後にも破綻しないことを
品質基準にする。

## 用語の整理

- `exact lossless`
  - float32 bit列が完全一致する。
  - 研究上は別ライン。
- `near-lossless faithful`
  - bit完全一致ではないが、線形値・log値・表示値・編集後表示の差が十分小さい。
  - 現在の本命。
- `visually lossless`
  - 指定した表示条件では人間が区別できない。
  - RAW相当を名乗るにはこれだけでは弱い。
- `RAW-competitive near-lossless`
  - 通常表示だけでなく、少なくとも `+3EV` 相当の暗部持ち上げ、ハイライトdetail確認、
    軽いホワイトバランス変更に耐える。

## 参照データ

現時点では、元の `sample_*.EXR` / `sample_*.exr` をRAW proxyとして扱う。

本当のRAW比較を行う場合は、以下を固定したうえで比較する。

- demosaic engine
- camera profile
- white balance
- black level / white level
- exposure default
- color transform
- tone mappingなしの線形現像

つまり、比較対象は「カメラRAWそのもの」ではなく、
固定現像パイプラインから出た scene-linear RGB とする。

## 必須比較層

### 1. 線形/scene値の健全性

目的は、明らかな数値破壊を早期に弾くこと。

確認項目:

- NaN/Inf が増えない。
- 符号の反転が不自然に増えない。
- `0` 付近の値が潰れて広い定数領域にならない。
- original の min/max に対して decoded の範囲が不自然に広がらない。

指標:

- signed-log2 RMSE
- signed-log2 p99 / p99.9
- sign flip rate
- zero/near-zero collapse rate
- channelごとの min/max/range

ただし、線形値PSNRだけでは不十分。暗部bandingやハイライトdetail lossを見逃す。

### 2. 表示空間比較

現在の目視で使っている preview に近い比較。

基本表示:

```text
display = clamp(linear / white, 0, 1) ** (1 / gamma)
```

初期値:

- `white=4.0`
- `gamma=2.2`
- tone mappingなし

追加表示:

- `+3EV`: `linear * 8` を同じ表示式へ入れる。
- `+5EV`: 暗部stress用。自動gateでは warn 扱い、最終判断は人間。
- highlight view: `white=1.0` または `white=4.0` の通常表示でdetail確認。

指標:

- display RGB RMSE / p99 / p99.9
- display luma RMSE / p99 / p99.9
- max error は記録するが、単独rejectにはしない。
- luma error の低周波成分。これが暗部のマダラ/黒浮きに効く。

### 3. 暗部階調/banding

過去に最も危険だった領域。

対象:

- `maxRGB <= 0.25`
- `+3EV` 表示で見える暗部
- smooth mask: 局所分散/局所勾配が低い領域

指標:

- display-luma gradient NRMSE
- lost-detail rate
- local mean residual p99
- local low-frequency residual energy
- run/plateau rate

判定方針:

- 暗部smoothの低周波誤差は、画素単位誤差より重く扱う。
- ディザ/ノイズでbandingが消えても、黒浮きやマダラが出るならNG。
- `+3EV` で一目で分かる階調飛びは即reject。

### 4. ハイライトdetail

過去に「階調」ではなく「detail消失」として出た問題。

対象:

- `1.0 < maxRGB <= 4.0`
- `maxRGB > 4.0`
- 強いエッジ/テクスチャ領域

指標:

- gradient energy ratio
- gradient correlation
- lost-detail rate
- Laplacian/high-pass energy ratio
- edge近傍の表示差分

初期gate:

- highlight gradient energy ratio は `0.96 .. 1.04` を目標。
- `0.94 .. 1.06` を超えたら人間確認。
- lost-detail delta は signed-log10 anchor 比で見る。

### 5. 色味/chroma

YCoCg/VST-chroma routeでは重要。

対象:

- 中間調以上の色差
- 暗部での色ノイズ/色転び
- 明るい部分の黄色/緑/赤寄り変化

指標:

- display RGB error
- display luma error と chroma error を分離
- YCoCg の Co/Cg error p99
- hue error。ただし低彩度領域のhueは不安定なので重みを下げる。

判定方針:

- 暗部の微小chroma差は許容しやすい。
- 中間調/明部の一目で分かる色転びはreject。
- full画像でcropと違う色になる問題は必ずfull gateに入れる。

### 6. ノイズ/粒状感

RAW相当で重要。ノイズを完全一致させる必要はないが、質感が変わるとNG。

指標:

- local variance ratio
- residual autocorrelation
- low-frequency residual energy
- block/tile境界の残差集中

判定方針:

- 白色ノイズ的な微細差は許容しやすい。
- 低周波のマダラ、黒浮き、境界haloはreject寄り。
- ノイズ除去風に平滑化される場合は、faithful near-losslessではなく別モード扱い。

## region分け

最低限のregion:

- `dark_0_0.25`
- `mid_0.25_1`
- `highlight_1_4`
- `extreme_gt4`
- `smooth`
- `edge`
- `texture`
- `route_mask`
- `nonmask`

既存 `audit_display_quality_regions.py` の tonal region に、
smooth/edge/texture と route mask/nonmask を追加する。

## anchorの扱い

候補を直接originalだけと比べると、どの程度が「許容される量子化差」なのか判断しづらい。
そのため anchor を置く。

現在のanchor/基準:

- faithful anchor: `signed-log bits10`
- reject anchor: `Y7` / rejected bits8 variants
- accepted/current anchor: `Y8 + visual-guard L=0.010`

ここでいう `visual-guard込み` は、少なくとも `sample_DSCF0009` では以下を指す。

- `Y8 CL8 signed-log10 avg`
- `dark-smooth st0.0025`
- `display-diff L=0.010` guard
- estimated `21,065,911 bytes`

品質検証では、basic routeではなくこの visual-guard込みを現行本命として扱う。

判定は以下の二段にする。

1. originalとの差が絶対閾値内か。
2. faithful anchorからの悪化量が小さいか。

この二段にすると、暗部やハイライトで「数値は悪くないが見た目がダメ」を拾いやすい。

## 初期PASS基準

これは固定値ではなく、現在の受理/却下サンプルで再校正する前提。

自動PASS:

- 全sampleで NaN/Inf/sign破壊なし。
- `EV0` と `+3EV` の display luma p99 が受理anchor近傍。
- dark smooth の lost-detail delta が `+4%` 未満。
- dark smooth の local low-frequency residual p99 が受理anchor近傍。
- highlight gradient energy ratio が `0.96 .. 1.04`。
- highlight lost-detail delta が `+3.5%` 未満。
- 色差/chroma p99 が受理anchor近傍。

自動REJECT:

- ユーザーが一目で分かる差を既知の指標が拾えていない場合。
- `+3EV` 暗部で階調飛び/マダラ/黒浮きが出る。
- ハイライトdetailが消える。
- full画像で色味が一目で変わる。
- Y7のように容量は良くても通常表示で明確に汚い。

MAYBE:

- 数値はPASSだが、edge/smooth境界に局所的な残差が出る。
- `+5EV` stressでのみ差が見える。
- ノイズ質感が変わるが、通常表示では区別困難。

## 人間確認の最小セット

人間確認は疲れるので、候補ごとに多くても以下に絞る。

1. full image normal view
2. full image `+3EV` view
3. dark smooth hotspot crop
4. highlight detail hotspot crop
5. edge boundary hotspot crop

候補が多い場合は、まず自動gateで `PASS/MAYBE/REJECT` に分け、
`MAYBE` だけ人間へ出す。

## 現行本命に対する扱い

現行本命:

- `Y8 CL8 signed-log10 avg`
- `dark-smooth st0.0025`
- `sample_DSCF0009` は `display-diff L=0.010` guard追加
- 以後の `sample_DSCF0009` 比較では、basic routeではなくこの visual-guard込みを使う。

現状:

- `sample_DSCF0009`: estimated `21,065,911 bytes`
- `sample_middle_flower`: estimated `19,363,179 bytes`
- `sample_bright_park`: estimated `22,770,606 bytes`

この visual-guard込み候補は、サイズだけでなく画質基準の再校正用にも使う。
Y7は「容量は良いが一目でNG」のreject anchorとして保存する。

## 次に作るべきもの

`scripts/audit_near_lossless_quality.py`

要件:

- original EXR と decoded EXR/PNG preview を比較できる。
- 最終bitstream実装後は decoded EXR を必須にする。
- 研究段階では表示PNG比較も許可する。
- EV0/+3EV/+5EV のdisplay metricsを出す。
- tonal/smooth/edge/texture/route mask別に集計する。
- accepted/rejected anchorsとの差分を出す。
- worst hotspot座標を出す。
- review用cropを最大5枚だけ出す。
- 結果は `PASS/MAYBE/REJECT` で返す。

これにより、ユーザー目視を最終確認に残しつつ、途中検証の流れを止めにくくする。

## 実装/初回検証メモ 2026-06-05

`scripts/audit_near_lossless_quality.py` を追加し、`sample_DSCF0009.EXR` の現行
visual-guard込み候補をフル解像度で検証した。

実行:

```text
pixi run python scripts/audit_near_lossless_quality.py
```

出力:

```text
results/near_lossless_quality_visual_guard_Y8_L001_sample_DSCF0009_full.json
```

検証ルート:

- `VST gamma075 Y8 CL8`
- signed-log10 escape
- `dark-smooth st0.0025`
- `display-diff L=0.010` visual guard
- route mask `22.80%`

判定:

- 自動判定は `reject`。
- ただし、最初の実装で出ていた大きな `lost-detail` はRAWノイズ勾配をディテール扱いしていたため、
  判定用には使わない。
- 修正版では、生の1px勾配は参考値として残し、軽い平滑化後の `structure` 勾配で判定する。

主な結果:

- ハイライト側は構造勾配では概ね良好。
  - `EV0 highlight_1_4 structure_grad_energy = 1.0034`
  - `EV3 highlight_1_4 structure_grad_energy = 1.0173`
  - `EV0 highlight_1_4 structure_lost_detail_delta = 0.0164`
  - `EV3 highlight_1_4 structure_lost_detail_delta = 0.0154`
- 暗部smoothはまだ signed-log10 anchor 比で悪化が大きい。
  - `EV0 dark_smooth structure_lost_detail_delta = 0.1497`
  - `EV3 dark_smooth structure_lost_detail_delta = 0.1714`
- 中間調chromaも `maybe`。
  - `EV0 mid_0.25_1 chroma_p99_abs = 0.04096`
  - `EV3 mid_0.25_1 chroma_p99_abs = 0.09766`

解釈:

- 目視で許容済みの21MB級候補でも、RAW proxy基準では暗部smoothがまだ厳しい。
- ハイライトdetail消失問題は、この候補では少なくとも構造勾配上は再発していない。
- 次の改善は、容量削減より先に暗部smoothの構造勾配を signed-log10 anchor に近づけるか、
  この指標が実際の見え方を過剰に罰していないかを受理/却下サンプルで再校正すること。

## sample横断検証メモ 2026-06-05

`scripts/audit_near_lossless_quality.py --export-png` で全 `sample_*` をフル解像度検証した。

PNG出力先:

```text
outputs/previews/near_lossless_quality_audit/visual_guard_quality_audit/
```

各sampleごとに以下を出力する。

- `*_original_w4_g2.2.png`
- `*_candidate_w4_g2.2.png`
- `*_signedlog10_anchor_w4_g2.2.png`
- `*_displaydiff_ev0_x4_neutralgray.png`
- `*_chromadiff_ev0_x4_rg.png`
- `*_route_mask.png`

`*_chromadiff_ev0_x4_rg.png` は、Rに `Co` 差分、Gに `Cg` 差分、Bに中立値を入れた診断画像。
中間調の色差maybeを視覚確認するための画像であり、通常鑑賞用ではない。

横断結果:

| image | decision | mask | EV0 mid Co p99 | EV3 mid Co p99 | dark smooth EV3 structure lost | highlight EV3 structure lost |
|---|---:|---:|---:|---:|---:|---:|
| `sample_1920×1280.exr` | maybe | 0.00% | 0.02947 | 0.06799 | 0.0000 | 0.0000 |
| `sample_DSCF0009.EXR` | reject | 22.80% | 0.04096 | 0.09766 | 0.1714 | 0.0154 |
| `sample_bright_park.EXR` | maybe | 5.77% | 0.04338 | 0.10814 | 0.0939 | 0.0035 |
| `sample_hilberts-mill-conference-room_2K.exr` | reject | 5.46% | 0.10245 | 0.25867 | 0.3058 | 0.0618 |
| `sample_middle_flower.EXR` | reject | 0.00% | 0.03287 | 0.08276 | 0.1579 | 0.1490 |

解釈:

- 中間調chroma maybeは `sample_DSCF0009` 固有ではない。
  ほぼ全sampleで `EV3 mid_0.25_1 Co p99` が signed-log10 anchor より大きく出る。
- 現在のY8/CL8 VST-chroma routeは、輝度より先に中間調色差が品質gate上の共通弱点になりつつある。
- `sample_hilberts` は圧縮率が非常に良かったが、品質proxyでは強いreject。
  圧縮率の良さだけでnear-lossless扱いしないための警告例として保存する。
- `sample_middle_flower` は中間調chromaより、ハイライト構造detail側がより危険。
- `sample_1920×1280` は暗部/ハイライトは通るが、中間調chromaでmaybe。

次の判断:

- `CL8` 固定が中間調chromaを削りすぎている可能性が高い。
- 次の候補は、暗部/低彩度だけ `CL8` を維持し、中間調以上または高彩度だけ `CL9/CL10`
  へ逃がすchroma guardが本命。
- ただし容量増を抑えるため、chroma guardは全画素ではなく `mid_0.25_1` かつ
  chroma差分/彩度/構造で限定する。

### 目視フィードバック

同じPNG群を目視確認した結果:

- `sample_DSCF0009` / `sample_bright_park` / `sample_1920×1280` は、
  original と candidate をほとんど区別できない。
- `sample_middle_flower` は、ごく僅かにdetailがボケるが、実用上はかなりOK寄り。
- `sample_hilberts-mill-conference-room_2K` は明確にNG。
  量子化誤差が目立ち、他sampleとは別系統の失敗に見える。

この結果から、自動gateは `sample_middle_flower` のような微細なdetail低下を
rejectに寄せすぎている可能性がある。一方で `sample_hilberts` のrejectは妥当。

次の校正:

- `middle_flower` は `MAYBE/pass寄り` の受理校正サンプルにする。
- `hilberts` は `reject anchor` として保存する。
- `hilberts` がfp16由来/低精度EXR由来なら、float32 RAW-like sampleと同じrouteで
  評価しないか、fp16/quantized source専用routeへ分岐する。

追加確認:

- EXR metadata上は `hilberts` も `float` / `piz`。
- `source_precision` の `half=0.250` は、RGBがhalfなのではなく、4ch目のAlphaが全画素 `1.0`
  でhalf exactになっている影響。
- RGBのhalf exact rateは各ch `0.0002` 程度なので、fp16画像ではない。
- `hilberts` RGBの最大値は `R=149.1`, `G=138.0`, `B=131.4` と非常に大きい。
  他sampleよりハイライト外れ値が桁違いに広く、グローバルrangeの `Y8/CL8` が
  中間域を粗く量子化している可能性が高い。

したがって `hilberts` の失敗原因は「fp16ルートに入った」ではなく、
「巨大HDR rangeに対して現在のglobal quantizationが弱い」と見る。

対策候補:

- global rangeではなく tile/local range quantization にする。
- `maxRGB > 4` または `> 8` の極端なハイライトを signed-log10 escape へ逃がし、
  本体VSTのrangeを中間域に合わせる。
- 画像単位でrangeが広すぎる場合は `Y8` を禁止し、`Y9/Y10` か局所range routeへ分岐する。
- alpha定数はpayloadから除外/定数チャンネル符号化する。品質とは別だがサイズ見積もりを正しくする。

追加の軸分解:

- `hilberts_Y10_CL8`
  - `EV0 dark_smooth lumaP99 = 0.02149`
  - `EV3 dark_smooth lumaP99 = 0.05529`
  - まだrejectだが、`Y8_CL8` より暗部誤差は大幅に減る。
- `hilberts_Y8_CL10`
  - `EV0 dark_smooth lumaP99 = 0.05936`
  - `EV3 dark_smooth lumaP99 = 0.15274`
  - `Y8_CL8` とほぼ同系統の失敗。

結論:

- `hilberts` の主因はchroma precision不足ではなく、Y側のglobal range量子化。
- `CL10` だけ上げても救えない。
- `Y10` でも完全には足りないため、単純な全体bits増加より、
  highlight outlier escape / tile-local range / range class router が必要。

## outlier range escape 初回検証 2026-06-05

固定しきい値ではなく、`maxRGB` の分位点とVST-Y量子化stepで外れ値escapeを選ぶ実験を追加。

実装:

```text
scripts/probe_outlier_range_escape.py
```

考え方:

- `maxRGB` が上位にある画素を signed-log10 escape へ逃がす。
- VST base のY rangeは、escapeしない画素だけで張る。
- これにより巨大HDR外れ値にY8の分解能を奪われないようにする。
- 分位点ごとに `route mask rate`, `Y step`, `estimated bytes`, quality gate を比較する。

`sample_hilberts-mill-conference-room_2K.exr` / `Y8 CL8` 結果:

| percentile | threshold maxRGB | route mask | Y step | estimated bytes | raw ratio | decision |
|---:|---:|---:|---:|---:|---:|---:|
| 95.0 | 0.63694 | 10.46% | 0.002708 | 1,530,403 | 21.93x | pass |
| 97.0 | 0.77190 | 8.46% | 0.003136 | 1,479,986 | 22.67x | pass |
| 98.0 | 0.96646 | 7.46% | 0.003695 | 1,370,763 | 24.48x | maybe |
| 98.5 | 1.84127 | 6.96% | 0.006020 | 1,171,546 | 28.64x | maybe |
| 99.0 | 6.85503 | 6.46% | 0.016113 | 804,849 | 41.69x | reject |
| 99.5 | 39.0255 | 5.96% | 0.060375 | 483,036 | 69.47x | reject |
| 99.9 | 65.6286 | 5.56% | 0.087880 | 432,064 | 77.66x | reject |
| 100.0 | 149.136 | 5.46% | 0.157838 | 377,715 | 88.84x | reject |

ベスト候補:

- `p97`
- threshold `maxRGB > 0.7719`
- route mask `8.46%`
- estimated `1,479,986 bytes`
- raw ratio `22.67x`
- quality gate `pass`

確認PNG:

```text
outputs/previews/outlier_range_escape/percentile_step_escape/
```

主な出力:

- `sample_hilberts-mill-conference-room_2K_full_outlier_p97_Y8_CL8_candidate_w4_g2.2.png`
- `sample_hilberts-mill-conference-room_2K_full_outlier_p97_Y8_CL8_original_w4_g2.2.png`
- `sample_hilberts-mill-conference-room_2K_full_outlier_p97_Y8_CL8_displaydiff_ev0_x4_neutralgray.png`
- `sample_hilberts-mill-conference-room_2K_full_outlier_p97_Y8_CL8_chromadiff_ev0_x4_rg.png`
- `sample_hilberts-mill-conference-room_2K_full_outlier_p97_Y8_CL8_route_mask.png`

解釈:

- 89x級の極端な圧縮率は失うが、`hilberts` の明確な量子化破綻はこの方向で救える可能性が高い。
- `p99` ではまだY stepが粗すぎ、暗部smoothでrejectが残る。
- このサンプルでは `Y step <= 約0.003〜0.004` がpassの目安に見える。
- 本命実装は固定percentileではなく、`Y step` が許容値を満たす最大percentileを選ぶ方式。
  つまり「逃がす画素を最小にしつつ、本体Y分解能を守る」。

目視フィードバック:

- `p97` candidate はハイライトが表示上クリップされているため、ハイライト差は分かりにくい。
- ただし、以前目立っていた量子化誤差は見受けられない。
- したがって `hilberts` については、image-level percentile/step outlier escape だけで
  まず十分に改善できている可能性が高い。

tile版について:

- 現時点では必須ではない。
- まず image-level percentile/step router を本命として進める。
- tile版は、明るい領域が局所的すぎてimage-level escapeが過剰になる場合、
  または大画像で局所range差が大きくサイズが悪化する場合の次段に回す。

## near-lossless router v1 組み込み 2026-06-05

`audit_near_lossless_quality.py` と `estimate_vst_signedlog_route.py` に
image-level percentile/step outlier routerを組み込んだ。

CLI:

```text
--enable-outlier-router
--outlier-percentiles 100,99.9,99.5,99,98.5,98,97,95
--target-y-step 0.0032
--outlier-activation-ratio 4.0
```

起動条件:

- `maxRGB p100 / p99`
- `maxRGB p99 / p97`

上記のどちらかが `4.0` 以上ならoutlier-heavy/wide HDRとみなし、routerをactiveにする。
通常sampleではinactiveなので、容量を無駄に増やさない。

全sample再検証:

| image | decision | mask | estimated bytes | raw ratio | outlier active | chosen percentile |
|---|---:|---:|---:|---:|---:|---:|
| `sample_1920×1280.exr` | maybe | 0.00% | 2,836,123 | 10.40x | no | - |
| `sample_DSCF0009.EXR` | reject | 22.80% | 21,082,479 | 22.66x | no | - |
| `sample_bright_park.EXR` | maybe | 5.77% | 22,770,606 | 20.98x | no | - |
| `sample_hilberts-mill-conference-room_2K.exr` | pass | 8.46% | 1,504,988 | 22.30x | yes | p97 |
| `sample_middle_flower.EXR` | reject | 0.00% | 19,363,179 | 24.67x | no | - |

PNG出力:

```text
outputs/previews/near_lossless_quality_audit/near_lossless_router_v1/
```

解釈:

- `hilberts` は従来 `89x` 近いが明確にNGだった。
  router v1では `22.30x` に落ちる代わりに、自動gate `pass` まで改善。
- 他sampleはoutlier-heavyではないためrouter inactive。
  従来サイズを維持し、余計なsigned-log escapeを増やさない。
- `DSCF` と `middle_flower` の自動rejectは残るが、これは既知のgate校正問題/暗部smooth・detail問題。
  outlier range問題とは別。

これにより、現行の主軸は以下のnear-lossless routeとして扱う。

1. 通常sample:
   `VST Y8/CL8 + dark/visual guard signed-log10`
2. wide HDR / outlier-heavy sample:
   上記に `image-level percentile/step outlier escape` を追加

速度メモ:

- `sample_hilberts` のrouter付き監査は復元約 `7.47s`。
- 大きい40MP級sampleは復元/監査に数分かかる。
- 次の速度UPは、品質監査ではなくcodec本体候補のC++/vectorized実装、
  またはPython監査のEV/region集計削減が対象。

## C++復元カーネル化 2026-06-05

near-lossless router v1 の復元候補生成をC++へ移植した。

追加:

- `codec/src/near_lossless_router.hpp`
- `codec/src/near_lossless_router.cpp`
- `radiance_codec_near_lossless_router_v1_reconstruct` C ABI
- `radiance_codec.reconstruct_near_lossless_router_v1(...)` Python binding
- `codec/tests/test_near_lossless_router.cpp`

現時点の位置づけ:

- 完成bitstream codecではない。
- VST/YCoCg + dark-smooth mask + percentile/step outlier router + signed-log10 escape の
  decoded candidateをC++で生成する高速カーネル。
- entropy stream packingは次段。

速度:

- `sample_hilberts-mill-conference-room_2K.exr` full:
  - Python復元: 約 `7.47s`
  - C++復元: 約 `1.67s`
  - 約 `4.5x` 速い。
- C++単体呼び出しでは同じfull imageで約 `0.83s`。
  `audit_near_lossless_quality.py --cpp-reconstruct` では周辺のPython処理込みで約 `1.67s`。

品質:

- `sample_hilberts` は C++復元でも `pass`。
- route mask `8.46%`
- outlier active `p97`

コマンド:

```text
pixi run build
codec/build/test_near_lossless_router
pixi run python scripts/audit_near_lossless_quality.py \
  --input sample_hilberts-mill-conference-room_2K.exr \
  --enable-outlier-router \
  --cpp-reconstruct \
  --label near_lossless_router_cpp_Y8_CL8
```

次:

- C++カーネルを最終bitstream stageへ拡張する。
- Y index / chroma low / chroma high / signed-log escape / mask を個別streamとして格納する。
- Python監査では `--cpp-reconstruct` を標準にし、外部visual guard mask対応を追加する。

## C++ compact bitstream stage 2026-06-05

`StageNearLosslessRouter = 0x0200` を追加し、near-lossless router v1 を
通常の `encode/decode` パイプラインから呼べるようにした。

追加/変更:

- `NearLosslessRouterStage`
- Python `Stage.NEAR_LOSSLESS_ROUTER`
- Python `encode_near_lossless_router_v1(...)`
- 内部payload magic: `NLR1`

payload構成:

- route mask
- high-pass mask
- VST/YCoCg `Y` index
- guided chroma low `Co/Cg` index
- sparse chroma high `Co/Cg` index
- route領域の signed-log10 `RGB` index
- 追加チャンネルはconstant検出、非constantならbyte stream保存

各streamは raw と rANS order0 を比較し、小さい方を採用する。
これにより、以前の「router候補float32をGroupedDeltaで包むだけ」の試作より
サイズ/速度が大幅に改善した。

注意:

- これは専用stream化された実codec stage。
- ただし、現時点のC++ stageは `dark-smooth + outlier router` まで。
- 研究時に `sample_DSCF0009` で使った外部PNG由来の
  `display-diff L=0.010 visual guard` はまだ自動内蔵していない。
- そのため、`visual-guard込み` を本命品質として扱う評価では、
  次に自動display-diff guard、または外部/sidecar guard maskの実装が必要。

compact実測:

```text
pixi run python - <<'PY'
from pathlib import Path
import sys, time, json
import numpy as np
ROOT=Path.cwd()
sys.path.insert(0, str(ROOT/'scripts'))
sys.path.insert(0, str(ROOT/'codec'/'python'))
from probe_darkbits_router_payload import read_exr
import radiance_codec as rc
...
PY
```

結果:

| image | raw MB | encoded MB | ratio | encode | decode |
|---|---:|---:|---:|---:|---:|
| `sample_1920×1280.exr` | 29.49 | 2.94 | 10.04x | 0.92s | 0.16s |
| `sample_DSCF0009.EXR` | 477.78 | 39.62 | 12.06x | 19.07s | 2.65s |
| `sample_bright_park.EXR` | 477.78 | 28.56 | 16.73x | 17.38s | 2.47s |
| `sample_middle_flower.EXR` | 477.78 | 20.21 | 23.64x | 16.68s | 2.20s |
| `sample_hilberts-mill-conference-room_2K.exr` | 33.55 | 1.70 | 19.69x | 0.85s | 0.14s |
| `sample_DSCF0009_denoise.exr` | 490.57 | 57.50 | 8.53x | 17.61s | 3.11s |

保存先:

```text
results/near_lossless_router_v1_cpp_compact_benchmark.json
```

一致確認:

`sample_DSCF0009.EXR` で compact decode と
`reconstruct_near_lossless_router_v1` の出力は `array_equal=True`。

```text
encoded=39,621,623
compact_roundtrip=20.16s
reconstruct=17.19s
max_abs=0
mean_abs=0
route_mask=22.14%
outlier_active=false
```

前段の失敗実測:

- `StageNearLosslessRouter | StageGroupedDelta` の包装試作は不採用。
- `sample_1920×1280.exr`: 19.97MB / encode 120.72s
- `sample_hilberts`: 14.20MB / encode 147.80s
- 原因はrouter候補をfloat32のまま汎用GroupedDeltaに渡していたこと。

次:

1. `display-diff L=0.010` 相当のvisual guardをC++ stageへ入れる。
2. signed-log route streamをMED residual + byte rANSより強いbitplane/context rANSへ更新する。
3. mask streamをwest/north contextで実符号化し、推定値との差を詰める。
4. C API/Python APIにrouter report取得付きencodeを追加する。

## C++ internal visual guard 2026-06-05

`StageNearLosslessRouter` に、外部PNGなしの内部visual guardを追加した。

内容:

- basic router候補をencoder内で一度生成する。
- 全画素signed-log10安全版を同じ `white=4, gamma=2.2` 表示空間へ変換する。
- 表示輝度差 `L >= 0.010` の画素をguard maskにする。
- dilationは現行accepted相当の `0`。
- guard maskをroute maskへORし、最終payloadはguard込みmaskで再生成する。

これは以前の診断:

```text
candidate_Y8_slog10_diffguard_L0_01_R0_d0_mask.png
```

をcodec内部で再現するための第一段。

compact visual-guard実測:

| image | raw MB | encoded MB | ratio | encode | decode |
|---|---:|---:|---:|---:|---:|
| `sample_1920×1280.exr` | 29.49 | 2.95 | 9.99x | 1.90s | 0.15s |
| `sample_DSCF0009.EXR` | 477.78 | 31.04 | 15.39x | 36.46s | 2.63s |
| `sample_bright_park.EXR` | 477.78 | 27.46 | 17.40x | 35.54s | 2.41s |
| `sample_middle_flower.EXR` | 477.78 | 20.35 | 23.48x | 33.99s | 2.19s |
| `sample_hilberts-mill-conference-room_2K.exr` | 33.55 | 1.71 | 19.68x | 1.77s | 0.14s |
| `sample_DSCF0009_denoise.exr` | 490.57 | 55.81 | 8.79x | 37.37s | 3.07s |

保存先:

```text
results/near_lossless_router_v1_cpp_compact_visual_guard_benchmark.json
outputs/previews/near_lossless_router_cpp_compact_visual_guard/
```

`sample_DSCF0009.EXR` は、visual guardなしcompactの `39.62MB` から
`31.04MB` へ改善した。

stream内訳の変化:

| stream | guardなし | guard込み |
|---|---:|---:|
| route mask | 1.26MB | 1.43MB |
| high mask | 2.15MB | 2.10MB |
| Y | 8.10MB | 8.07MB |
| chroma low | 2.93MB | 2.94MB |
| chroma high | 2.35MB | 2.14MB |
| signed-log RGB | 22.83MB | 14.35MB |

解釈:

- guardでroute maskは増えたが、通常側のrangeと高周波/escape分布が楽になり、
  結果的にDSCF0009では容量も改善した。
- encode時間は約2倍になった。原因はguard判定用にbasic候補をもう一度生成しているため。
- 次の高速化では、basic候補生成と最終payload生成の共通計算を共有し、
  `36s -> 20s未満` を狙う。

次:

1. visual guard込み候補のフルPNGを目視確認する。
2. C++ guard mask/reportを外へ出せるようにする。
3. signed-log RGB streamをbitplane/context化して、20MB台前半へ詰める。
4. guard判定用の再計算を消してencode速度を戻す。

## C++ stream compaction 2026-06-05

visual guard込みの復元結果を変えず、bitstreamだけを詰めた。

採用:

- generic stream: raw / rANS order0 / rANS order1 / zstd level9 の最小を選択
- index stream: byte stream候補に加え、量子化indexを直接symbol rANSで符号化
- mask stream: west/north adaptive binary rANS
- chroma high: MED residualではなくraw selected indexをsymbol rANSで符号化

不採用:

- bitpack index + rANS: DSCFで改善なし、遅い
- fixed bit-position adaptive binary: DSCFで改善なし、遅い
- signed-log raw index fallback: DSCFで選ばれず、遅い

DSCF最終実測:

```text
sample_DSCF0009.EXR
raw      = 477,775,872 bytes
encoded  = 23,204,463 bytes
decimal  = 23.20MB
binary   = 22.13MiB
ratio    = 20.59x
```

保存先:

```text
results/near_lossless_router_v1_cpp_dscf_final_22m.json
```

stream内訳:

| stream | method | payload |
|---|---:|---:|
| route mask | maskbin | 811,266 |
| high mask | maskbin | 1,738,206 |
| Y | rANS order1 | 7,649,107 |
| Co low | symbol rANS | 1,321,837 |
| Cg low | symbol rANS | 1,621,920 |
| Co high | symbol rANS | 563,723 |
| Cg high | symbol rANS | 667,889 |
| SLog R | symbol rANS | 3,268,885 |
| SLog G | symbol rANS | 2,623,972 |
| SLog B | symbol rANS | 2,937,501 |

重要:

- 10進の厳密な `22.00MB` 未満ではなく、`22.13MiB`。
- ユーザーの「DSCFは22MB」目標に対しては22MiB級まで到達。
- 10進22MB未満へさらに落とすには、Yのorder1 byte stream
  またはsigned-log RGB symbol streamを、より強い2D/context modelへ置き換える必要がある。

次に残る速度課題:

- encodeが重い。原因は以下。
  - visual guard判定用のbasic candidate再生成
  - streamごとの複数codec候補比較
  - order1 rANSとsymbol rANSの全量探索
- 品質固定後の速度UPでは、候補比較をDSCFで実際に選ばれた方式へ固定する。
