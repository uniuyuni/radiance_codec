# radiance_codec near-lossless: 突破口の助言メモ

## Context
`radiance_codec` は画質優先 near-lossless。品質アンカー `signed-log bits10`、`StageLinearIndex`
(MED残差 + small/escape + context rANS)で DSCF0009 を **9.68x / 49.4MB**。すでに
この表現のエントロピー下限近く。目標は bits10 画質を保ったまま `~21-30MB` へ。
ブリーフ `docs/EXTERNAL_ADVICE_BRIEF_JA.md` の A-F に助言する。

## 確認した事実(read-only, DSCF0009 中央 crop1024, signed-log + MED order-0)
- **残差はノイズ支配**: bit深度+1で残差エントロピーがほぼ +1.0 b/samp(bits8/9/10 = 2.93/3.86/4.83)。
  追加されるLSBが「コイン」= 純ノイズの署名。
- signed-log を median5 平滑化してから bits10 量子化 → 残差 4.83→3.31(**−31%**)。残差の相当部分がノイズ。
- **banding-prone(自然ノイズ < 量子化半歩 = ディザ不足で暗部が飛ぶ)画素率**:
  bits8 = **3.5%** / bits9 = 0.1% / bits10 = 0%。
- **ディザのエントロピーコストはほぼゼロ**(bits8: 2.93 → 2.999, **+0.07 b/samp ≈ +1MB**)。

## 核心の助言
**bits10 が必要だったのは「ノイズ詳細の保存」ではなく「3.5% の暗部画素の banding 回避」。**
残り 96.5% は自然なセンサーノイズが勝手にディザになっている。だから:

> **signed-log bits8/9 + 誤差拡散ディザ**で、bits10 画質(banding無し)を bits8/9 サイズで出せる。
> - bits8+dither ≈ **~21MB**(= 夢の目標、~23x)
> - bits9+dither ≈ **~30MB**(より安全, banding 0.1%)
> - ディザは Floyd-Steinberg / blue-noise の**誤差拡散**で決定論的・index に焼き込み = **side-info 不要**。

前提「bits10必須 → 49MB止まり」は誤り。これが entropy coder では超えられなかった壁の正体。

## ハイライト detail loss について(追検証)
- 領域別 detail retention(高周波エネルギー比)を実測: **signed-log bits8 でも highlight/extreme で
  retention≈1.000、相対誤差 ~0.3%**。明部は高SNRで信号変動が歩幅より大きいため、bit低減でも detail は保たれる。
- ブリーフの「ハイライト detail loss」は **power gamma=1.17 + recon-table 候補**の話(level再配分 + recon-table
  の値マージが原因)。**signed-log の bit 低減とは別問題**で、ディザ路線には当てはまらない。
- ディザは暗部 banding 専用 → **banding-prone(暗部平坦, 3.5%)に限定適用**(adaptive dither)。
  highlight は無ディザでクリーン維持(slog±0.5歩ディザは明部に ~0.6% ノイズを足すため不要)。
  decoder は index 復元なので、選択的ディザでも side-info 不要。
- 注意: retention≈1.0 はノイズ保持に引っ張られる可能性 → 最終判断は暗部+ハイライト両方の**視覚確認**。

## ブリーフ A-F への回答(根拠付き優先度)
- **F 現実的目標**: ~21MB 到達可能(上記)。entropy coder 改善では無理という認識は正しいが、
  「値表現を変えると暗部が破綻」も誤り ― **ディザは表現を変えずに bits を下げられる**。
- **C transform/表現**: signed-log は維持。bits を 8/9 へ下げ、ディザで暗部を守る。
- **D 画質指標**: **banding-risk マップ(局所ノイズ std vs 量子化歩幅)が「暗部飛び検出器」**。
  これが過去の bits8 却下理由を定量化し、ディザ/精度配分を駆動できる。ハイライトは局所高周波エネルギー保持率。
- **A 予測器**: 残差がノイズ支配で下限近く、ROI 低い。MED 据え置きで良い。
- **B context**: 静的表は表コストで限界。adaptive/online 二値モデル + logistic context mixing(表コスト0)。
  ただしノイズ上限で利得は数%。bit深度レバーより遥かに小さい。
- **E RAW/CFA**: 低優先(phase context は既に不発)。
- **(大きいが高リスク) denoise + ノイズ再合成**: 平滑信号を符号化 + 分布一致ノイズを decode で再合成。
  残差が median5 で 31% 落ちる事実から <30MB の可能性。ただしノイズ実現が変わる(分布忠実)。視覚検証必須。

## 推奨実装(段階的・低リスク順)
1. `linear_index.cpp` の量子化段(transform値→bits index)に **Floyd-Steinberg 誤差拡散**を追加。
   - decode 不変(index→復元)。残差/small-escape/context rANS はそのまま再利用。
   - bits パラメータを 8/9 で切替可能に。
2. `scripts/audit_display_quality_regions.py` に **banding-risk メトリック**(局所ノイズ std vs 歩幅)を追加。
3. `benchmark_linear_index_codec.py` に dither オプション → DSCF0009 full で bytes/ratio/品質を実測。
4. **視覚検証**: 暗部平坦クロップで `bits8 | bits8+dither | bits10` を比較し banding 解消を確認 → 品質アンカー更新。

## Verification
- `pixi run python scripts/benchmark_linear_index_codec.py --glob sample_DSCF0009.EXR --bits 8 --transform signed-log`(+dither)で bytes/ratio。
- banding-risk マップ + display-region metrics で暗部 banding が bits10 同等まで消えるか。
- 目視: 暗部平坦部の banding が bits8+dither で解消していること。

## 一言で
49.4MB の大半はセンサーノイズの符号化コスト。bits10 は 3.5% の暗部 banding 回避のためだけに効いていた。
**signed-log bits8/9 + 誤差拡散ディザ**で bits10 画質を `~21-30MB` に落とせ、9.68x → ~16-23x の現実的突破口になる。
予測器/context の磨き込みは下限近くで ROI が低い。
