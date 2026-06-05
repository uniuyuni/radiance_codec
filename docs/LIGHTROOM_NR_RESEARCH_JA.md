# Lightroom風軽量NR調査メモ

2026-06-04 作成。目的は、Adobe Lightroom / Camera Raw のノイズ除去そのものを
再現することではなく、`radiance_codec` の near-lossless / visually-lossless 圧縮に
転用できる古典的な軽量NR構造を抽出すること。

## 結論

- Lightroomの現行 `Denoise` はAI系で、Adobe公式ブログではRawのデモザイクとノイズ除去を
  1ステップで行うMLモデルとして説明されている。これは軽量NR改造の直接対象ではない。
- ただし、Lightroom / Camera Rawの旧来手動NRと、AdobeのEric Chan氏による公開特許には、
  軽量NRとして使える骨格がかなり具体的に出ている。
- `radiance_codec` に効きそうな核は次の4点。
  - linear RGBのまま処理せず、ノイズの信号依存性をならす `flat noise space` に近い空間へ移す。
  - RGBを輝度 + 色差へ分離する。
  - 色差は強めに、ただし輝度/色差のエッジを見ながら平滑化する。
  - 輝度はdetail/non-detailを分け、平坦部だけを強めに、エッジやテクスチャは方向性を保って弱く処理する。
- このプロジェクトでは、NR済み画像をそのまま完成品にするより、
  `signal` / `residual` / `mask` を作るための分離器として使うのが安全。

## ソースの位置づけ

### 1. 現行AI Denoise

Adobe公式ブログでは、2023年のLightroom / Camera Raw `Denoise` はAI機能として説明されている。
Eric Chan氏の記事によると、モデルはRaw mosaic patternの補間とノイズ除去を同時に行うように
訓練され、学習には高ノイズ/低ノイズの画像patchペア、大規模なノイズシミュレーション、
shadow pattern noiseを学ぶためのdark frame集合が使われている。

重要な示唆:

- Raw段階の情報を使えるなら、デモザイク後よりもノイズ分離しやすい。
- shadowのpattern noiseは、通常画像だけでなくdark frame的な統計を持つと強い。
- ただしこの機能は深層学習・GPU寄りで、軽量codec部品としてそのまま入れる対象ではない。

Adobe公式ヘルプでは、Denoise対象はBayer/X-Trans mosaic raw、Linear DNG、HDR/Pano DNG、
Apple ProRAWなどとされ、JPEG/TIFF/HEIC/JXL/AVIF/PSD等は対象外とされている。
EXRを直接Lightroom Denoiseへ渡す設計は期待しない方がよい。

Sources:

- Adobe Blog, `Denoise demystified`:
  https://blog.adobe.com/en/publish/2023/04/18/denoise-demystified
- Adobe Help, `Easily enhance image quality in Lightroom`:
  https://helpx.adobe.com/lightroom-cc/using/enhance-details.html

### 2. 手動NRスライダー

Lightroom Classic公式ヘルプでは、ノイズを大きく
`luminance` と `chroma/color` に分けて扱う。

- `Luminance`: 輝度ノイズを減らす。
- `Luminance Detail`: 高いほどdetailを残すがノイズも残りやすい。低いほど滑らかだがdetailを消しやすい。
- `Luminance Contrast`: 高いほどcontrastを保つがmottlingを出しやすい。低いほど滑らかだがcontrastを落としやすい。
- `Color`: 色ノイズを減らす。
- `Color Detail`: 高いほど細い色エッジを守るが色speckleが残りやすい。低いほど色speckleを消すが色にじみが出やすい。

またLightroom 5.2系の資料では `Color Smoothness` が追加され、
shadowなどに出る低周波の色むら/色blotchを抑える用途として説明されている。
これは現在の我々の `full YCoCgが黄色い` / `暗部色差が暴れる` 問題に近い。

Sources:

- Adobe Help, `How to retouch photos in Lightroom Classic`:
  https://helpx.adobe.com/id_en/lightroom-classic/help/retouch-photos.html
- Lightroom 5.2 update PDF:
  https://ptgmedia.pearsoncmg.com/imprint_downloads/peachpit/peachpit/bookreg/9780321934406/lr5-2-update.pdf

### 3. Adobe特許 US7983511B1

Google Patents上の `Methods and apparatus for noise reduction in digital images` は、
Eric Chan氏が発明者、Adobeが譲受人として登録されている。これはLightroom現行実装そのものの
公開ではないが、Camera Raw / Lightroom系の古典NR思想を読むには非常に有用。

注意:

- 特許文書なので、ここに書かれた方法がLightroomの現在の手動NRそのものとは限らない。
- 実装時は、特許・ライセンスの扱いを別途確認すること。ここでは研究メモとして扱う。

Source:

- Google Patents, `US7983511B1`:
  https://patents.google.com/patent/US7983511

## 特許から読める古典NRパイプライン

### 1. Linear space to flat noise space

特許では、デジタルカメラのノイズ標準偏差を次のようにモデル化している。

```text
N(z) = sqrt(A * z + B)
```

- `A`: photon / shot noise に相当する係数。
- `B`: 光がない状態でも出るcamera system/read noiseに相当する係数。
- `z`: 0-1正規化された平均輝度/信号値。

この式は、明るさによってノイズ量が変わることを意味する。そこで特許では、
フィルタ前にlinear値を `flat noise space` へ変換し、ノイズ量が信号値に依存しにくい状態へ
持っていく。

実装上は、厳密なカメラ校正がなくても次の近似が候補になる。

- `log2(x + eps)`
- `asinh(x / scale)`
- `gamma075`
- Anscombe風の `sqrt(A*x+B)` 型変換

現在の `radiance_codec` では `signed-log`, `gamma075`, `asinh` をすでに試しており、
Lightroom風NRでもこの空間選択が最初の探索軸になる。

### 2. RGB to YCC

flat noise space化した後、RGBをYCCのような輝度 + 2色差へ変換する。
特許では、輝度と色の空間周波数感度が違うため、別々の方法を使えることが利点として説明されている。

`radiance_codec` では、既にYCoCg系があるため、最初は独自YCCではなく `YCoCg` でよい。
ただし、full画像でglobal chroma rangeが広がり黄色化した問題があるので、
次の点は必須。

- global chroma rangeをそのまま信用しない。
- low-frequency chromaとhigh-frequency chromaを分ける。
- chroma外れ値はrangeを広げるのではなく、escapeまたは別routeへ逃がす。

### 3. Color / chroma noise reduction

特許の高品質color NRは、色差チャンネルを2D近傍でweighted averageする。
重みは色差だけでなく、輝度Yと両方の色差C1/C2を見て決める。

この意味:

- 色差だけを見ると、輝度エッジをまたいで色がにじむ。
- Y, C1, C2全部の差を見れば、明るさエッジ・色エッジの両方で平滑化を止めやすい。
- 半径はノイズが少なければ小さく、高ノイズや低周波色むらには大きくできる。

特許のfast color NRはさらにcodec向き。

1. 2-level pyramidを作る。
2. high-pass chromaにrank filterをかけ、色speckleをclipする。
3. high-pass chromaへsoft thresholdをかけ、小さい高周波色差を消し、大きい色エッジを残す。
4. low-pass chromaへ重めのedge-aware smoothingをかける。
5. low-pass + high-passを戻す。

これは現在の圧縮方針とかなり噛み合う。

- low-pass chroma: 低解像度で精度高めに保存。
- high-pass chroma: 多くを捨てる/粗量子化/統計再合成。
- 強い色エッジ: escapeやC7/C8 routeで保持。

### 4. Luminance noise reduction

輝度Yは人間が敏感なので、色差より慎重に扱う。
特許ではdetail pixelとnon-detail pixelを分ける。

- detail pixel:
  - エッジやテクスチャ候補。
  - local window内の差が閾値を超えるかで判定する。
  - filterは1D。方向はlocal gradientに垂直、つまりエッジに沿う方向。
  - エッジをまたがず、エッジに沿ってノイズをならす。
- non-detail pixel:
  - 平坦/滑らか領域。
  - 2D filterで強めに平滑化する。

fast luminance NRではpyramidを使う。

- low-passだけに2D filterをかける。
- high-pass各階層のdetailには1D edge-aligned filterをかける。
- 小さいkernelでも低解像度階層では広い範囲に効く。

`radiance_codec` の暗部階調問題では、Yを強くNRしてしまうとbandingやグラデーション消失が出る。
したがって最初は `luma NR` を完成画像へ強く適用せず、
危険maskやresidual分類を作るために使う方がよい。

## radiance_codecへの落とし込み

### 使い方A: 完成画像をNRする

これは圧縮率は伸びやすいが、画質リスクが高い。

```text
original RGB
  -> flat/log space
  -> YCoCg
  -> Lightroom風NR
  -> quantize signal
  -> decode時はNR後signalを再構成
```

リスク:

- 暗部階調が少し消えるだけでユーザー目視に引っかかる。
- ハイライトdetailが落ちるとすぐ分かる。
- NRの見た目が良くても、オリジナルと違う絵になる。

この使い方は、`machine-not-obviously-different` の最終段階まで温存する。

### 使い方B: NRを分離器/mask生成器にする

現時点の本命はこちら。

```text
original RGB
  -> flat/log space
  -> YCoCg
  -> Lightroom風NRで signal estimate を作る
  -> residual = original - signal
  -> residualの性質で領域分類
      - Yの滑らかな階調: 保存/高bit route
      - Yの高周波detail: edge/detail route
      - chroma低周波: low-res map
      - chroma高周波speckle: 捨てる/粗量子化/共分散再合成
```

利点:

- NR出力そのものを信じすぎない。
- 「何を捨ててよいか」の判定に使える。
- 既存の `Y9C7`, `Y9C6/C7 router`, clipped range, dark bits route と組み合わせやすい。

### 使い方C: ChromaだけLightroom風にする

一番安全で、最初に試す価値が高い。

```text
flat/log RGB
  -> YCoCg
  -> Yは既存routeを維持
  -> Co/Cgだけ
      - low-pass chroma: guided/bilateral smoothing + 低解像度保存
      - high-pass chroma: rank clip + soft threshold + sparse escape
  -> RGBへ戻す
```

期待:

- full画像で出た黄色化/色味ズレを減らせる可能性。
- 暗部色差ノイズをかなり削れる。
- 輝度階調には手を出さないため、暗部bandingリスクが比較的小さい。

危険:

- 色差を削りすぎると、黒寄り/脱色/境界違和感が出る。
- low-frequency chromaを粗くしすぎると、full画像の明るい部分で一目で色が変わる。

## 軽量実装案

### Phase 1: Chroma-only Lightroom-style probe

新規スクリプト候補:

```text
scripts/probe_lightroom_chroma_nr.py
```

2026-06-04 実装時点では、特許名に寄せず、
`scripts/probe_vst_chroma_nr.py` として追加した。中身は
一般的なvariance-stabilized transform + YCoCg chroma-only low/high分離。

処理:

1. `sample_DSCF0009.EXR` をcrop512/crop1024/fullで読む。
2. `flat_transform` を選ぶ。
   - `gamma075`
   - `signed-log`
   - `asinh`
   - `sqrt_noise(A,B)` 近似
3. YCoCgへ変換。
4. Co/Cgを2-level pyramidへ分解。
5. high-pass Co/Cg:
   - 3x3 rank clip。
   - soft threshold:

```text
h_keep = h * (1 - exp(-(abs(h) / kT)^2))
```

6. low-pass Co/Cg:
   - guided filterまたはbilateral近似。
   - guideはY、またはY/Co/Cgのjoint guide。
7. Yは原則そのまま。
8. RGBへ戻し、previewと代理審査員で評価。
9. payload見積もり:
   - low-pass chroma entropy
   - high-pass sparse escape rate
   - thresholded residual entropy

最初のパラメータ:

```text
flat_transform: gamma075, signed-log
pyramid_scale: 2, 4
lowpass_radius: 4, 8, 16
lowpass_eps: 0.01, 0.05, 0.1
rank_r: 1, 2
kT: 0.5, 1.0, 1.5 * local_noise_sigma
chroma_bits_low: 6, 7, 8
chroma_escape_bits: 8, 10
```

判定:

- full通常表示で黄色化しない。
- crop暗部をユーザー側+3段表示して階調が飛ばない。
- highlight detailが落ちない。
- 推定サイズが `Y9C7 full 24.8MB` より下がる。

### Phase 2: Luma weak NR / risk mask

Chroma-onlyで色問題が改善した後に行う。

新規スクリプト候補:

```text
scripts/probe_lightroom_luma_mask.py
```

処理:

1. flat/log Yを作る。
2. local gradient / local variance / detail thresholdで分類。
3. non-detail dark smoothだけ弱くguided smoothingする。
4. `original Y - smoothed Y` を見て、階調を支えるresidualを検出。
5. そのresidualが必要な場所だけbits10/escape routeに送る。

重要:

- Yを直接消すのではなく、`どこを守るか` のmaskとして使う。
- 今までの `dark-smooth` maskを置き換える、または強化する役割。

### Phase 3: Combined codec route

候補:

```text
Y route:
  base Y9 or signed-log/gamma075 bits9
  risk mask only Y10/escape

Co/Cg route:
  low-pass chroma 低解像度 C7/C8
  high-pass chroma thresholded sparse escape
  dark chroma high-passは原則捨てる/共分散再合成
```

この構造なら、Lightroom風の品質思想を入れつつ、既存codecのindex entropy問題にも対処できる。

## これまでの失敗との接続

### log RGB guided covariance probe

`scripts/probe_dark_guided_covariance.py` では、log RGBを直接guided filterでsignal/noise分離した。

結果:

- 共分散ノイズmapは非常に安い。
- しかしサイズの大半はsignal indexが支配した。
- RGB直接処理なので、輝度階調・色差ノイズ・デモザイク由来構造が混ざった。

Lightroom風に直すなら:

- RGB直接ではなくYCoCg/YCCへ分ける。
- Yは守る。
- Co/Cgを強く処理する。
- residual分類はYとCo/Cgで別々に行う。

### YCoCg full黄色化

full画像で `Y9C7` やclipped range候補が黄色寄りになった。
原因候補は、global chroma rangeの広がりと、低周波chromaの扱いが粗いこと。

Lightroom風に直すなら:

- chromaをlow/high frequencyへ分ける。
- low-frequency chromaは小さくても丁寧に保持する。
- high-frequency chromaは人間に見えにくいので大胆に削る。
- 強い色エッジだけescapeする。

### 暗部banding

bits8/9やdark maskで暗部階調が飛んだ。
これはYの微細residualを雑に捨てていることが主因。

Lightroom風に直すなら:

- Yの平滑領域を強く圧縮する前に、local gradient continuityを見る。
- 暗部の滑らかな階調に必要なY residualだけを保存する。
- 色差ノイズ削減でサイズを稼ぎ、その分Y救済へbitを回す。

## 実装時の注意

- Lightroomそのものの再実装を目指さない。
  - 公開情報から一般化した `flat space + luma/chroma分離 + edge-aware filter` として実装する。
- 特許文書の式や手順をそのまま固定実装するのではなく、codec向けに変形する。
  - guided filter、box mean、rank clip、soft threshold、pyramid程度の一般的部品で構成する。
- 代理審査員を必ず使う。
  - 通常表示
  - 暗部持ち上げ相当
  - highlight detail
  - hue/chroma bias
- 目視サンプルは少なくする。
  - ユーザー確認は最大2-3枚。
  - full画像確認を早めに入れる。cropだけで決めない。

## 次にやるなら

1. `probe_lightroom_chroma_nr.py` を作る。
2. `sample_DSCF0009` crop512で、chroma-only NRの見た目とpayloadを測る。
3. crop1024で暗部階調を確認する。
4. fullで黄色化が戻るか確認する。
5. 効いた場合だけ、Y risk maskへ進む。

最初の成功条件:

- Yにほぼ手を入れず、fullの色味が `Y9C7 full` よりオリジナルに近い。
- 暗部+3段相当で階調が悪化しない。
- 推定サイズが `24.8MB` より下がる見込みがある。

この順番なら、Lightroom的な「人間が嫌うノイズだけ掃除する」知見を使いつつ、
今回のcodec目標である画質優先の20-25MB探索に自然につながる。

## 2026-06-04 初回probe結果

スクリプト:

```text
scripts/probe_vst_chroma_nr.py
```

設計:

- RGBへ一般的なVSTをかける。
  - `gamma075`
  - `sqrtvst`
  - `linear` control
- VST後RGBをYCoCgへ変換。
- YはNRしない。Y planeを量子化して保持する。
- Co/Cgだけ、Yをguideにしたguided low-passへ通す。
- `chroma = low + high` と分け、highはsoft thresholdしてsparse escapeする。

DSCF crop512 結果:

- 高品質control:
  - `vstchroma_gamma075_Y9_CL8_H8_s1_r2_ge0.1_tm0`
  - `PASS/PASS`
  - 推定 `423,994 bytes`, `7.42x`
  - high保持率 `100%`
  - payload内訳:
    - Y `65,054 bytes`
    - chroma low `69,434 bytes`
    - chroma high `289,242 bytes`
    - high mask `8 bytes`
  - dark detail delta `-0.89%`
  - lift dark detail delta `-1.10%`
  - highlight detail delta `+2.66%`
- 攻め候補:
  - `vstchroma_gamma075_Y10_CL8_H8_s2_r2_ge0.1_tm1`
  - `PASS/REJECT`
  - 推定 `265,216 bytes`, `11.86x`
  - high保持率 `50.98%`
  - payload内訳:
    - Y `90,859 bytes`
    - chroma low `27,419 bytes`
    - chroma high `117,289 bytes`
    - high mask `29,393 bytes`
  - dark detail delta `-2.89%`
  - lift dark detail delta `-2.63%`
  - highlight detail delta `+1.98%`

PNG:

- `outputs/previews/vst_chroma_nr/sample_DSCF0009_crop512_original_w4_g2.2.png`
- `outputs/previews/vst_chroma_nr/sample_DSCF0009_crop512_vstchroma_gamma075_Y9_CL8_H8_s1_r2_ge0_1_tm0_w4_g2.2_decoded.png`
- `outputs/previews/vst_chroma_nr/sample_DSCF0009_crop512_vstchroma_gamma075_Y10_CL8_H8_s2_r2_ge0_1_tm1_w4_g2.2_decoded.png`

判断:

- VST + chroma-only分離は、通常表示では破綻しにくい。
- しかし20-25MB級の主routeとしては、このままでは弱い。
  - 高品質controlは `7.42x` で重い。
  - highを約半分に削ると `11.86x` まで伸びるが、暗部lift評価でREJECT。
- サイズ支配はchroma high。
  - 高品質controlでは high payloadが約 `289KB` で最大。
  - highを削るとサイズは下がるが、暗部/質感に影響する。
- したがって次の使い道は「完成画像をNRする」ではなく、
  `Co/Cg highのどこが本当に必要か` を測るmask/selector作り。
- full確認はまだしていない。cropだけで決めない。

ユーザー目視:

- `vstchroma_gamma075_Y9_CL8_H8_s1_r2_ge0.1_tm0` はオリジナルとの差が分からない。
- `vstchroma_gamma075_Y10_CL8_H8_s2_r2_ge0.1_tm1` は少し平滑化していると分かる程度。
- 代理審査はlift側に厳しめで、実目視ではまだ攻められる可能性がある。

追加探索:

- `Y10 / CL8 / scale2 / radius2 / guide_eps0.1` 固定で、
  high chroma bits と threshold を振った。
- 20x級候補:
  - `vstchroma_gamma075_Y10_CL8_H6_s2_r2_ge0.1_tm2`
  - 推定 `156,420 bytes`, `20.11x`
  - high保持率 `14.64%`
  - dark detail delta `+0.94%`
  - lift dark detail delta `+1.55%`
  - highlight detail delta `+3.15%`
- 確認用PNGを2枚だけ追加:
  - `vstchroma_gamma075_Y10_CL8_H7_s2_r2_ge0.1_tm2`
    - 推定 `163,258 bytes`, `19.27x`, high保持率 `14.64%`
  - `vstchroma_gamma075_Y10_CL8_H6_s2_r2_ge0.1_tm1.75`
    - 推定 `166,775 bytes`, `18.86x`, high保持率 `20.03%`
- PNG:
  - `outputs/previews/vst_chroma_nr/sample_DSCF0009_crop512_vstchroma_gamma075_Y10_CL8_H7_s2_r2_ge0_1_tm2_w4_g2.2_decoded.png`
  - `outputs/previews/vst_chroma_nr/sample_DSCF0009_crop512_vstchroma_gamma075_Y10_CL8_H6_s2_r2_ge0_1_tm1_75_w4_g2.2_decoded.png`

更新判断:

- VST-chroma routeは「本命ではない」から「暗部/色差圧縮の有力な枝」へ少し昇格。
- ただしcrop512は暗部寄りなので、次はcrop1024/fullで黄色化・中間調・ハイライトを必ず見る。
- 目視負担を避けるため、次の人間確認は最大2枚にする。

## 2026-06-04 crop1024確認

ユーザー評価でcrop512の `18-20x` 級も「オリジナルから少し平滑化しただけ」に
見えたため、候補を増やさずcrop1024へ上げた。

- `vstchroma_gamma075_Y10_CL8_H6_s2_r2_ge0.1_tm2`
  - 推定 `618,656 bytes`, `20.34x`
  - high保持率 `16.83%`
  - dark detail delta `+3.92%`
  - lift dark detail delta `+4.68%`
  - highlight detail delta `+2.96%`
  - PNG:
    `outputs/previews/vst_chroma_nr/sample_DSCF0009_crop1024_vstchroma_gamma075_Y10_CL8_H6_s2_r2_ge0_1_tm2_w4_g2.2_decoded.png`
- `vstchroma_gamma075_Y10_CL8_H5_s2_r2_ge0.1_tm2.5`
  - 推定 `547,142 bytes`, `23.00x`
  - high保持率 `10.45%`
  - dark detail delta `+4.26%`
  - lift dark detail delta `+5.10%`
  - highlight detail delta `+3.41%`
  - PNG:
    `outputs/previews/vst_chroma_nr/sample_DSCF0009_crop1024_vstchroma_gamma075_Y10_CL8_H5_s2_r2_ge0_1_tm2_5_w4_g2.2_decoded.png`

判断:

- crop1024でも推定ratioは維持され、通常表示では大きな色崩れは見えにくい。
- 代理審査は `MAYBE/REJECT` または `REJECT/REJECT` だが、
  これまでの目視では代理審査がこの系統を厳しめに落としている。
- ユーザー目視:
  - crop1024の2枚はどちらも良い感じ。
  - ただしオリジナルとは違う。
  - これは `near-lossless faithful` ではなく、Lightroom風に少し平滑化された
    `visually-denoised` 方向へ寄っている。
- 次はこの方向を本線に混ぜず、別profileとして扱う。
  - faithful profile: オリジナル差分を極力残す。従来のY9C7/YCoCg/router系。
  - denoised profile: NR風の平滑化を許容し、20MB級/それ以下を狙うVST-chroma系。
- fullで黄色化・中間調・ハイライトを確認する場合も、`denoised profile` として評価する。

## 2026-06-04 10MB目標への再設定

ユーザー方針:

- VST-chroma路線はオリジナルから離れる。
- それなら20MB級で止めず、`denoised profile` として10MB級を目指したくなる。

サイズ目標:

- `sample_DSCF0009.EXR` は `7728 x 5152 x 3 x float32`。
- raw換算は `477,775,872 bytes`。
- 10,000,000 bytes目標なら約 `47.78x`。
- 10MiB目標なら約 `45.56x`。
- crop1024換算では、10MB級に相当するpayloadは約 `263KB`。

### Chroma-onlyから見えた壁

crop1024のVST-chroma候補:

- `Y10_CL8_H6_tm2`: `618,656 bytes`, `20.34x`
- `Y10_CL8_H5_tm2.5`: `547,142 bytes`, `23.00x`

`Y10_CL8_H5_tm2.5` の内訳:

- Y payload: `364,633 bytes`
- chroma low: `92,679 bytes`
- chroma high: `47,684 bytes`
- chroma high mask: `41,890 bytes`

10MB級にはcrop1024で約 `263KB` まで落とす必要がある。
つまりchromaだけ削っても届かない。Y payloadだけで目標を超える。

### 素朴なY低bit化のストレステスト

VST-chromaのままYを8/9bitへ落とし、chroma highをさらに削った。

- `vstchroma_gamma075_Y8_CL7_H4_s4_r2_ge0.1_tm3`
  - crop1024推定 `258,895 bytes`
  - `48.60x`
  - 10MB級には届く
  - ただし代理指標は `REJECT/REJECT`
  - dark detail delta `39.82%`
  - lift dark detail delta `41.59%`
  - highlight detail delta `10.30%`
  - PNG:
    `outputs/previews/vst_chroma_nr/sample_DSCF0009_crop1024_vstchroma_gamma075_Y8_CL7_H4_s4_r2_ge0_1_tm3_w4_g2.2_decoded.png`

判断:

- 数字だけなら10MB級は作れる。
- しかし単純なY8/Y9化は、NR風ではなく階調/detailを潰しすぎる。
- この方向をそのまま本命にしてはいけない。

### Y low/high分離probe

10MB級にはYを触る必要があるため、
`scripts/probe_vst_denoised_profile.py` を追加した。

設計:

- VST/YCoCg空間でYもCo/Cgもlow/highへ分離する。
- lowはguided-filter後にdownsample + quantize。
- highはsoft threshold後にsparse escape。
- 捨てたhighはdenoisingとして扱う。

初回結果:

- やりすぎ設定では `100x` 超も出るが、detailが壊滅。
- 保守寄り設定でも、`20-35x` 程度の範囲で
  highlight detail deltaが `24-29%` 程度まで悪化。
- 例:
  - `vstdenoise_gamma075_YL9s2_YH6t0.5_CL8s2_CH5t2.5_yr2_cr2`
  - crop512推定 `125,867 bytes`, `24.99x`
  - Y high保持率 `61.25%`
  - chroma high保持率 `8.50%`
  - dark detail delta `14.97%`
  - highlight detail delta `24.90%`

判断:

- 現在の軽いguided luma分離は、重要detailを削りすぎる。
- 10MB級の鍵はYだが、Yは雑にlow/high分離できない。
- VST-chromaの20MB級は有望だが、10MB級には別のluma NR/edge-detail保持モデルが必要。

次の候補:

- lumaは完成画像NRではなく、edge/detail maskとして使う。
- Y lowは大きく削らず、Y highのうちノイズらしい成分だけを削る。
- 方向性:
  - directional/edge-aligned luma high preservation
  - Laplacian pyramidでedge bandだけescape
  - local variance/JNDで平坦暗部だけY highを削る
  - AI denoise/latentは次フェーズ候補

## 2026-06-04 full 20MB級テスト

ユーザー指示で、denoised profileの20MB級候補をfull解像度で評価した。

候補:

- `vstchroma_gamma075_Y10_CL8_H5_s2_r2_ge0.1_tm2.5`

結果:

- full raw bytes: `477,775,872`
- estimated bytes: `23,285,958`
- estimated ratio: `20.52x`
- high chroma保持率: `15.88%`
- proxy gate: `REJECT/REJECT`
- dark detail delta: `11.90%`
- lift dark detail delta: `13.04%`
- highlight detail delta: `5.24%`
- mean RGB delta: `(-1.98e-04, -5.08e-04, +1.43e-03)`

PNG:

- original:
  `outputs/previews/vst_chroma_nr/sample_DSCF0009_full_original_w4_g2.2.png`
- decoded:
  `outputs/previews/vst_chroma_nr/sample_DSCF0009_full_vstchroma_gamma075_Y10_CL8_H5_s2_r2_ge0_1_tm2_5_w4_g2.2_decoded.png`

判断:

- crop1024では `23.00x` だったが、fullでは `20.52x` に低下した。
  それでも20MB級の入口には入っている。
- 以前のYCoCg fullで出た強い黄色化とは違い、通常表示では色崩れはかなり抑えられている。
- ただしproxyは暗部/detailを厳しく見てREJECT。
- これはfaithful profileではなく、denoised profileとしてユーザー目視で判断する候補。
- 20MB未満をさらに狙う場合、単純にYを落とすと破綻するため、次は
  VST-chromaを維持したまま、mask/escapeのentropy改善か、Y edge/detail selectorが必要。

## 2026-06-04 full dark-protect preview

ユーザー目視でfull 20MB級候補は明部detailが良い一方、暗部グラデーションがマダラに見えた。
そこで、baseを攻め候補のまま、暗部smooth領域だけ安全寄り候補へ差し替える
visual routing previewを追加した。

スクリプト:

```text
scripts/export_vst_chroma_dark_protect_preview.py
```

設定:

- base: `Y10 CL8 H5 tm2.5`
- safe: `Y10 CL8 H6 tm1.75`
- mask: `dark-smooth`, `dark_max=0.25`, `smooth_threshold=0.002`

crop1024:

- base estimate: `547,142 bytes`, `23.00x`
- safe estimate: `656,854 bytes`, `19.16x`
- mask rate: `67.23%`

full:

- base estimate: `23,285,958 bytes`, `20.52x`
- safe estimate: `25,945,773 bytes`, `18.41x`
- mask rate: `16.61%`
- PNG:
  `outputs/previews/vst_chroma_dark_protect/sample_DSCF0009_full_vstchroma_darkprotect_gamma075_Y10_CL8_baseH5_tm2_5_safeH6_tm1_75_dark-smooth0_25_w4_g2.2_decoded.png`

判断:

- 通常表示ではかなり自然。
- 明部はbaseのまま残るので、既に良かった明部detailを壊しにくい。
- dark-smooth maskはfullでは `16.61%` なので、局所保護として現実的。
- ただしこれはまだ正確なrouted bitstreamではなく、base/safe decoded画像をmaskで合成した
  visual preview。
- 次に必要なのは、dark-smooth領域だけsafe high residualを持つ場合の正確なpayload見積もり。

### 2026-06-04 まだら対策の切り分け

ユーザー目視:

- `Y10 CL8 H5 tm2.5` full 20MB級候補は、明部detailは良い。
- しかし暗部グラデーションがまだら。最初はブロックノイズに見える。
- `dark-smooth0.25` のsafe差し替えは、まだらがほぼ変わらない。

追加診断:

- `safe-source=original` を追加し、マスク内だけ完全に元画像へ戻すoracle previewを作れるようにした。
- `scripts/export_vst_chroma_dark_protect_preview.py` はmask PNGも出力するようにした。
- `dark-smooth0.25, smooth_threshold=0.002`:
  - mask `16.61%`
  - original oracleでも、問題領域を十分には救えていない可能性が高い。
- `dark-smooth0.5, smooth_threshold=0.002`:
  - mask `16.61%`
  - `dark_max` を広げてもmaskが増えないため、smooth判定が狭すぎる。
- `dark0.5`:
  - mask `91.56%`
  - 広すぎて本実装向きではないが、暗い/中間調のかなり広い領域が候補になることは分かった。

高速oracle合成:

- `scripts/export_mask_oracle_preview_png.py` を追加。
- VST-chromaを再計算せず、既存のfull original/decoded PNGとEXR由来maskだけで
  display-space oracleを作る。
- preview transformは画素ごとに独立なので、oracle診断としては
  linear合成後にpreviewする場合と同等。

`dark-smooth0.5` のmask率:

- `smooth_threshold=0.002`: `16.61%`
- `smooth_threshold=0.003`: `27.88%`
- `smooth_threshold=0.004`: `39.80%`
- `smooth_threshold=0.005`: `50.06%`
- `smooth_threshold=0.006`: `57.22%`
- `smooth_threshold=0.008`: `64.98%`
- `smooth_threshold=0.01`: `69.33%`

出力:

- `st0.004` oracle:
  `outputs/previews/vst_chroma_dark_protect/sample_DSCF0009_full_display_oracle_baseH5_original_dark-smooth0_5_r2_st0_004_decoded.png`
- `st0.004` mask:
  `outputs/previews/vst_chroma_dark_protect/sample_DSCF0009_full_display_oracle_baseH5_original_dark-smooth0_5_r2_st0_004_mask.png`
- `st0.01` oracle:
  `outputs/previews/vst_chroma_dark_protect/sample_DSCF0009_full_display_oracle_baseH5_original_dark-smooth0_5_r2_st0_01_decoded.png`

判断:

- `st0.002` は狭すぎる。
- `st0.01` は診断としては強いが、本実装には広すぎる。
- 次の目視確認は `st0.004` に絞る。これでまだらが消えるなら、
  20MB級routeの主問題は「baseの圧縮方式全体」ではなく、
  shadow/smooth領域を検出して安全経路に逃がすrouting問題として扱える。
- `st0.004` でもまだらが変わらない場合は、マスクの軸が間違っている。
  その場合は luma range だけでなく、表示空間gradient/局所低周波誤差から
  `shadow-gradient-risk` maskを作る。

ユーザー目視:

- `st0.004` oracleでまだらは消えた。
- ファイル名に `original` が入っているのは「mask内だけ元画像へ戻すoracle」の意味。
- したがって、主問題はbase表現の全面破綻ではなく、
  問題領域をsafe/original相当へ逃がすrouting漏れと見なせる。

実safe候補:

- base: `Y10 CL8 H5 tm2.5`
- safe: `Y10 CL8 H6 tm1.75`
- mask: `dark-smooth0.5 r2 st0.004`
- mask rate: `39.80%`
- base full estimate: `23,285,958 bytes`, `20.52x`
- safe full estimate: `25,945,773 bytes`, `18.41x`
- normal PNG:
  `outputs/previews/vst_chroma_dark_protect/sample_DSCF0009_full_vstchroma_darkprotect_gamma075_Y10_CL8_baseH5_tm2_5_safeH6_tm1_75_dark-smooth0_5_r2_st0_004_w4_g2.2_decoded.png`
- lift PNG:
  `outputs/previews/vst_chroma_dark_protect/sample_DSCF0009_full_vstchroma_darkprotect_gamma075_Y10_CL8_baseH5_tm2_5_safeH6_tm1_75_dark-smooth0_5_r2_st0_004_w0.5_g2.2_decoded.png`
- mask PNG:
  `outputs/previews/vst_chroma_dark_protect/sample_DSCF0009_full_vstchroma_darkprotect_gamma075_Y10_CL8_baseH5_tm2_5_safeH6_tm1_75_dark-smooth0_5_r2_st0_004_mask.png`

次の分岐:

- 実safe版でもまだらが消える:
  - 共通Y/low + region別chroma-highだけを保存するrouted payload見積もりを作る。
  - 20MB台前半を維持できるかを見る。
- 実safe版でまだらが残る:
  - safe側を `H7` または `tm1.0` 方向へ強める。
  - ただしmask `39.80%` なので、safeを強めすぎると20MB目標から離れる。

ユーザー目視:

- `safeH6_st004` はまだらが直っていない。
- oracleで消え、H6で残るので、maskは当たっているがsafe側が弱い。

追加候補:

- `safeH7 tm1.0` full単体:
  - estimated `32,441,648 bytes`, `14.73x`
  - Y `17,079,563 bytes`
  - chroma low `2,915,309 bytes`
  - high payload `8,560,456 bytes`
  - high mask `3,886,064 bytes`
- base H5:
  - estimated `23,285,958 bytes`, `20.52x`
  - Y `17,079,563 bytes`
  - chroma low `2,915,309 bytes`
  - high payload `1,516,358 bytes`
  - high mask `1,774,472 bytes`
- display-space合成候補:
  `outputs/previews/vst_chroma_dark_protect/candidate_safeH7_st004.png`
- mask:
  `outputs/previews/vst_chroma_dark_protect/candidate_safeH7_st004_mask.png`

判断:

- H7単独は重いが、Y/lowはbaseと共通なのでrouted実装では二重保存しない。
- 目視OKなら、非maskをH5 high、mask内をH7 highに分けてentropyを測る
  正確なpayload estimatorを作る。
- H7でもNGなら、safeをさらに強めるより、oracleとの差分を見て
  まだらの主因がchroma highではなくY/low側に残っていないか確認する。

ユーザー目視:

- `candidate_safeH7_st004.png` もまだら。
- `H6` -> `H7/tm1.0` でchroma highを強めても直らないため、
  主因はVST routeの共通Y/low側にある可能性が高い。

次の切り分け:

- mask内だけ既存のfaithful `signed-log bits10` decodedへ差し替える候補を作成。
- full signed-log bits10単独:
  - encoded `49,382,153 bytes`
  - ratio `9.68x`
  - 以前のユーザー目視で許容済みの品質アンカー。
- candidate:
  `outputs/previews/vst_chroma_dark_protect/candidate_signedlog10_st004.png`
- mask:
  `outputs/previews/vst_chroma_dark_protect/candidate_signedlog10_st004_mask.png`

判断:

- `signed-log10` 差し替えでまだらが消えれば、shadow-smooth領域には
  VST-chroma低周波を使わず、faithful/signed-log局所escapeを混ぜる必要がある。
- 目標20MBからは遠くなるが、mask `39.80%` の全域をbits10にする必要があるかは
  まだ未確定。次はmaskを細分化し、まだらが見えるサブ領域だけを
  signed-log10へ逃がす `gradient-risk` / `banding-risk` maskを作る。

ユーザー目視:

- `candidate_signedlog10_st004.png` はOK。まだら消失。
- つまり、品質上の逃げ先は `signed-log bits10` で足りる。

サイズ見積もり:

- `scripts/estimate_vst_signedlog_route.py` を追加。
- これは VST nonmask Y/low/high + signed-log10 mask を、選択領域entropyで測るprobe。
- `dark-smooth0.5 r2 st0.004`, mask `39.80%`:
  - estimated `32,304,535 bytes`
  - ratio `14.79x`
  - VST Y nonmask `11,899,075 bytes`
  - VST chroma low nonmask `1,993,358 bytes`
  - VST chroma high nonmask `1,497,594 + 1,693,109 bytes`
  - signed-log10 mask `14,295,624 bytes`
  - mask `925,263 bytes`

判断:

- 品質OK routeは20MBから約12MB超過。
- 20MBへ戻す鍵は、signed-log10へ逃がすmaskを `39.80%` からどこまで縮められるか。
- 次の探索は候補画像を増やしすぎない:
  - まず `st0.003` 付近のoracle/signed-log10 routingを1枚。
  - それでまだらが戻るなら、smooth thresholdではなく
    `gradient-risk` / `display low-frequency error` 系のmaskに切り替える。

### decoder-side dither

目的:

- `signed-log10` に逃がすと品質OKだが `32.30MB`。
- `signed-log9` に落とし、decode時にbin内でdeterministic ditherを入れて
  まだら/帯を粒状化できるか確認する。
- ditherは保存indexを変えないので、追加ビットなし。

実装:

- `scripts/export_masked_signedlog_decode_dither_preview.py` を追加。
- VST base PNGの上に、mask内だけ `signed-log9 + decode dither` のpreviewを合成する。
- ditherはsigned-log transform domainで `0.5 step` のhash jitter。
- endpoint binsはoffsetしない。

出力:

- `outputs/previews/vst_chroma_dark_protect/candidate_slog9_dither_st004.png`
- `outputs/previews/vst_chroma_dark_protect/candidate_slog9_dither_st004_mask.png`

サイズ:

- `scripts/estimate_vst_signedlog_route.py --signedlog-bits 9`
- mask: `dark-smooth0.5 r2 st0.004`, `39.80%`
- estimated `27,920,735 bytes`
- ratio `17.11x`
- signed-log9 mask `9,911,824 bytes`
- bits10 OK route `32,304,535 bytes` から約 `4.38MB` 減。

判断:

- 目視OKなら、bits10 escapeより明確に有望。
- ただし20MBにはまだ届かないため、次は `st0.003` などでmaskを縮めるか、
  `gradient-risk` maskへ切り替える必要がある。
- 目視NGなら、amplitudeを `0.75` / `1.0` に上げる前に、
  ditherがノイズとして浮いていないかを確認する。

ユーザー目視:

- まだらは消えた。
- しかし黒浮きが分散され、不自然な暗部になった。
- つまりdecode-side ditherは、低周波のまだらを高周波へ崩す効果はあるが、
  暗部の平均/質感を壊している。

判断:

- 画質優先ではdecode-side ditherは不採用。
- amplitudeを上げる方向は逆効果の可能性が高い。
- もし今後ditherを再検討するなら、decode-side random jitterではなく、
  encoder-sideの局所平均保存/stochastic index選択に限定する。

### signed-log10 mask shrink

ditherを捨て、OKだった `signed-log10` escapeのmask縮小へ戻る。

mask率:

- `st0.0025`: `22.14%`
- `st0.003`: `27.88%`
- `st0.0035`: `33.87%`
- `st0.004`: `39.80%`

生成:

- `outputs/previews/vst_chroma_dark_protect/candidate_signedlog10_st003.png`
- `outputs/previews/vst_chroma_dark_protect/candidate_signedlog10_st003_mask.png`

条件:

- `dark-smooth0.5 r2 st0.003`
- mask `27.88%`
- mask bytes `751,783`

判断:

- `st0.003` でまだらが消えれば、単純threshold縮小で進める。
- まだらが戻るなら、smooth thresholdでは問題領域を選び切れない。
  次は `gradient-risk` / 表示空間低周波誤差maskへ切り替える。

ユーザー目視:

- `candidate_signedlog10_st003.png` は黒浮きがあるが許容範囲。
- `st0.003` を新しい実用ラインとして扱う。

サイズ:

- `scripts/estimate_vst_signedlog_route.py --signedlog-bits 10 --smooth-threshold 0.003`
- estimated `28,798,615 bytes`
- ratio `16.59x`
- signed-log10 mask `8,908,985 bytes`
- mask `751,783 bytes`
- `st0.004` の `32,304,535 bytes` から約 `3.50MB` 減。

追加候補:

- `outputs/previews/vst_chroma_dark_protect/candidate_signedlog10_st0025.png`
- `outputs/previews/vst_chroma_dark_protect/candidate_signedlog10_st0025_mask.png`
- mask `22.14%`
- mask bytes `606,148`

次の判断:

- `st0.0025` も許容なら、さらにサイズ見積もりへ進む。
- NGなら `st0.003` を品質/サイズの現実ラインとして固定する。
- 20MBを狙うには、単純smooth thresholdだけではまだ不足。
  次は `gradient-risk` / local low-frequency error maskで、黒浮きが見える領域だけを
  signed-log10へ逃がす。

ユーザー目視:

- `st0.0025` もOKの予感。

サイズ:

- `scripts/estimate_vst_signedlog_route.py --signedlog-bits 10 --smooth-threshold 0.0025`
- mask `22.14%`
- estimated `27,256,787 bytes`
- ratio `17.53x`
- VST Y nonmask `14,361,930 bytes`
- VST chroma low nonmask `2,437,042 bytes`
- VST chroma high nonmask `1,513,475 + 1,761,051 bytes`
- signed-log10 mask `6,576,629 bytes`
- mask `606,148 bytes`

判断:

- `st0.003` から約 `1.54MB` 減。
- `st0.004` から約 `5.05MB` 減。
- 画質が正式OKなら `st0.0025` が現時点の最良ライン。
- ただし20MBにはまだ約7.3MB足りない。
- 次はsmooth thresholdの微調整だけではなく、VST Y/lowのpayload削減か、
  より賢い `gradient-risk` / local low-frequency error maskで、
  signed-log10 escape対象をさらに正確化する。

### signed-log10 mask context

目的:

- `st0.0025` のsigned-log10 mask payload `6.58MB` を、範囲を変えずに
  context codingで縮められるか確認する。

実装:

- `scripts/probe_masked_signedlog_context.py` を追加。
- 対象はmask内signed-log10 residual。
- contextはdecoderで再現しやすいものに限定:
  - `order0`
  - `channel`
  - `phase2_channel`
  - `xtrans6_channel`
  - `maskwn_channel`
  - `phase2_maskwn_channel`
  - `xtrans6_maskwn_channel`

結果:

- mask `22.14%`, samples `8,813,498`
- `order0` direct ideal `6,618,580 bytes`
- `channel` direct ideal `6,576,597 bytes`
- `maskwn_channel` direct ideal `6,529,923 bytes`
- `phase2_maskwn_channel` direct ideal `6,529,862 bytes`
- `xtrans6_maskwn_channel` direct ideal `6,529,308 bytes`
- model込みでは `maskwn_channel` が現実的bestで `6,532,691 bytes`

判断:

- 範囲を狭めないsigned-log10 maskのcontext改善は約 `45-50KB` 程度。
- ここはもう大きな改善余地がない。
- 20MBへ向けた本命は、signed-log10 escape自体ではなく、
  VST Y/low payload削減またはrisk maskのさらなる精密化。

### VST Y9 outside risk mask

signed-log10 mask contextはほぼ縮まなかったため、次に大きい
VST Y nonmask payloadを削る。

実装/見積もり:

- `scripts/estimate_vst_signedlog_route.py` の出力名に `Y/CL` を追加。
- 条件:
  - nonmask: `VST gamma075 Y9 CL8 H5 tm2.5`
  - mask: `signed-log10`
  - mask: `dark-smooth0.5 r2 st0.0025`, `22.14%`
- estimated `23,761,113 bytes`
- ratio `20.11x`
- VST Y nonmask `10,866,256 bytes`
- VST chroma low nonmask `2,437,042 bytes`
- VST chroma high nonmask `1,513,475 + 1,761,051 bytes`
- signed-log10 mask `6,576,629 bytes`
- mask `606,148 bytes`

比較:

- Y10版 `27,256,787 bytes`
- Y9化で約 `4.50MB` 減。
- Y9 VST base単体:
  - `18,975,157 bytes`
  - `25.18x`

preview:

- `outputs/previews/vst_chroma_dark_protect/candidate_Y9_slog10_st0025.png`
- `outputs/previews/vst_chroma_dark_protect/candidate_Y9_slog10_st0025_mask.png`

判断:

- 20MB目標に最も近い本命候補。
- 次はユーザー目視で、暗部階調だけでなく中間調とハイライトdetailも確認する。
- Y9で問題が出るなら、Y9/Y10 selectorをrisk maskとは別に作る。

ユーザー目視:

- `candidate_Y9_slog10_st0025.png` は違いがないように見える。
- Y9は画質OKとして、さらにY8を検証する。

### VST Y8 outside risk mask

条件:

- nonmask: `VST gamma075 Y8 CL8 H5 tm2.5`
- mask: `signed-log10`
- mask: `dark-smooth0.5 r2 st0.0025`, `22.14%`

見積もり:

- estimated `20,812,528 bytes`
- ratio `22.96x`
- VST Y nonmask `7,917,671 bytes`
- VST chroma low nonmask `2,437,042 bytes`
- VST chroma high nonmask `1,513,475 + 1,761,051 bytes`
- signed-log10 mask `6,576,629 bytes`
- mask `606,148 bytes`

比較:

- Y9版 `23,761,113 bytes`
- Y8化で約 `2.95MB` 減。
- Y8 VST base単体:
  - `15,387,258 bytes`
  - `31.05x`

preview:

- `outputs/previews/vst_chroma_dark_protect/candidate_Y8_slog10_st0025.png`
- `outputs/previews/vst_chroma_dark_protect/candidate_Y8_slog10_st0025_mask.png`

判断:

- 20MB目標目前の最重要候補。
- ユーザー目視で暗部/中間調/ハイライトを確認する。
- Y8で問題が見えなければ、次は20MB未満へ入れるために
  CL8 -> CL7 または high stream/maskの微削減を試す。

ユーザー目視:

- `Y8` でも全体差は見分けがつかない。
- ただしエッジ付近に違和感のある乱れを発見。
- 同じ違和感は `Y9` にもあったため、Y8化ではなくVST base共通のedge処理が原因。

### edge guard

目的:

- エッジ周辺だけfaithful `signed-log10` へ逃がし、VST baseのedge乱れを消せるか確認する。
- edge領域は暗部smooth maskより小さくできる可能性がある。

実装:

- `scripts/export_edge_guard_preview.py` を追加。
- 既存candidate PNGの上に、edge maskだけsigned-log10 previewを合成する診断。
- edgeは `log2(1+luma)` のgradient上位quantileで検出。

候補:

- base: `candidate_Y8_slog10_st0025.png`
- edge quantile: `0.98`
- dilate radius: `1`
- edge mask `4.63%`
- extra edge mask `4.63%`
- mask bytes `170,737`

preview:

- `outputs/previews/vst_chroma_dark_protect/candidate_Y8_slog10_st0025_edge98.png`
- `outputs/previews/vst_chroma_dark_protect/candidate_Y8_slog10_st0025_edge98_mask.png`

判断:

- まず目視確認。
- OKならedge追加分のsigned-log10 entropyを測り、
  `20.81MB + edge guard` の現実サイズを出す。
- NGならedge検出の軸を「画像edge」ではなく「VST baseとsigned-log10の差がedge周辺で大きい領域」
  に切り替える。

ユーザー目視:

- `edge98` では直っていない。
- 問題はハイライトとシャドウの境目のごく薄い乱れ。
- 細いedge線だけでは届いていない可能性が高い。

edge mask面積:

- `q0.98 d1`: extra `4.63%`
- `q0.975 d3`: extra `9.69%`
- `q0.97 d3`: extra `11.01%`
- `q0.95 d3`: extra `15.91%`

追加候補:

- `outputs/previews/vst_chroma_dark_protect/candidate_Y8_slog10_st0025_edge975d3.png`
- `outputs/previews/vst_chroma_dark_protect/candidate_Y8_slog10_st0025_edge975d3_mask.png`
- `edge_quantile=0.975`
- `dilate_radius=3`
- extra edge `9.69%`
- mask bytes `129,967`

判断:

- これで改善するなら、境界帯guardとして追加entropyを測る。
- これでもNGなら、単純luma edgeではなく、VST baseとsigned-log10 previewの
  display差分が境界付近で大きい領域を直接mask化する。

### point guard diagnostic

ユーザー指定:

- 問題箇所中心 `(x=2361, y=3811)`。
- 現在見ている箇所は既存edge maskに含まれていなかった。

既存mask確認:

- `candidate_Y8_slog10_st0025_mask`:
  - 中心は外。
  - 周辺33x33は `35.54%` 入っている。
- `edge98`:
  - 中心/周辺とも外。
- `edge975d3`:
  - 中心/周辺とも外。

実装:

- `scripts/export_point_guard_preview.py` を追加。
- 指定点周辺だけ手動でsigned-log10へ逃がす診断。

出力:

- 半径 `192px`
- mask `0.29%`
- mask bytes `954`
- full:
  `outputs/previews/vst_chroma_dark_protect/candidate_Y8_pointguard_2361_3811.png`
- crop:
  `outputs/previews/vst_chroma_dark_protect/candidate_Y8_pointguard_2361_3811_crop.png`
- base crop:
  `outputs/previews/vst_chroma_dark_protect/candidate_Y8_pointguard_2361_3811_base_crop.png`
- safe crop:
  `outputs/previews/vst_chroma_dark_protect/candidate_Y8_pointguard_2361_3811_safe_crop.png`

判断:

- これで直るなら、manual pointではなく、`VST base vs signed-log10` の
  表示差分/局所差分から自動risk maskを作る。
- 直らないなら、signed-log10との差し替えで直らない種類のartifactなので、
  base/safe cropを比較して発生源を見直す。
