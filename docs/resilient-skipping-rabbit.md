# radiance_codec ― 突破口の再定義(アイデアメモ)

## Context

`radiance_codec` は float32 HDR の exact lossless を目指しているが、puresky など
で `2.3x` に詰まっている。ロードマップ(`docs/LOSSLESS_12X_ROADMAP_JA.md`,
`LOSSLESS_RESEARCH_REBOOT.md`)は MDL context tree / reversible block transform /
source precision / hash-LSH / prefix cascade / block fixed-point など膨大な exact
ルートを試し、ことごとく「条件付きエントロピー下限は低いのに side-info で全て失う」
で止まっている。

本メモは「次に何を試すか」ではなく、まず **なぜ全部失敗したのかをデータで確定**
させ、その上で勝てるゲームに資源を寄せる、という再定義を提案する。

ユーザ要望は「アイデアだけ。修正不要」。これは実装プランではなく研究方針メモ。

## 私が実施したこと(read-only)

- 全主要ドキュメント + `codec/src` 構成 + 最新 probe を確認。
- 外部調査: Aras Pranckevičius 2025(float画像lossless実測)、exponent/mantissa
  二画像分解、learned-lossless + shipped prior + 整数決定論的推論。
- **read-only の統計プローブを2回だけ実行**(EXRを読むのみ・保存なし):
  低位 mantissa の bitplane 毎 P(1)・空間相関・チャンネル間相関、および
  全画像の「ノイズ床(=ランダムな低位bit幅)」測定。

## 決定的な発見:コーパスは2つのレジームに割れる

read-only 測定(crop256)の結果:

| 画像 | random tail (bits/ch) | zero tail | レジーム |
|---|---|---|---|
| ph_belfast_puresky | ~16 | 0 | A: ノイズ床 |
| ph_kloppenheim_puresky | ~17 | 0 | A |
| sample_hilberts (true float) | ~17 | 0 | A |
| synth_mixed | 18 | 0 | A |
| oexr_Tree / CandleGlass | 0 | **13** (=half16) | B: 構造のみ |
| ph_abandoned / spruit / studio | 0 | **16** (=bfloat16) | B |

### Regime A(puresky / true-float / noise): exact 12x は情報理論的に不可能

puresky の mantissa bit 1〜22 は実測で:
- **P(bit=1) = 0.50**(公平なコイン)
- 水平/垂直 空間相関 **r ≈ 0.00**
- チャンネル間(west-delta後)相関 **r ≈ 0.00〜0.03**

これは「連続ノイズが標準偏差よりはるか下で量子化された」典型的な白色ノイズの署名。
**白色ノイズは Shannon 限界で圧縮不能**。予測器・context・shipped NN・可逆変換の
どれでも 1bit も縮まない。ロードマップの全 exact 失敗はこれ1つで説明できる。

ロードマップの「条件付きエントロピー 2.85 bps」という希望は **in-sample 過学習の
幻**(12bit hash で 4084 context / 65536 sample → context あたり16 sample で histogram
が偶然尖るだけ)。crop を 256 に広げると下限が 4.3〜4.6 bps へ上がったのが証拠で、
held-out なら 0.5 bit/coin = 上限に張り付くはず。

唯一の例外: **bit0 (LSB) だけ P(1)≈0.33** が全 puresky channel で一致 → 約 0.08
bits/sample の本物の exact 余地。ただし ratio への寄与は <1% で突破口ではない。

含意: puresky の低位 tail は「捨てると意味を失う計測情報」ではなく、おそらく
高解像度原版からの **float32 リサンプリング/処理の丸めノイズ**(値は白色だが
大きさは信号依存)。→ near-lossless で落としても視覚・物理的価値はほぼゼロ。

### Regime B(half/bfloat 起源): ここが勝てるゲーム

abandoned/spruit/studio は low **16** bit が常にゼロ = 実体は **bfloat16**。
Tree/CandleGlass は low **13** bit ゼロ = 実体は **half16**。ノイズ床ゼロ。
全コストは構造化された上位 bit にあり、**圧縮余地が実在する**。
exact 研究はここに集中すべき(現状 exact 努力は不可能な puresky に偏っている)。

## 本番データ DSCF0009 の判定(決定的・2026-06-02 追加)

ユーザが本番常用の `data/sample_DSCF0009.exr`(Fujifilm 7728x5152, 真のfloat32 RGB,
PIZ)を追加。read-only 解析の結果、**Regime A、しかも puresky より深刻**と確定:

- 9タイル空間スキャン(空/中間/前景)で **random tail = 18-21 bit/ch が一様**、
  zero tail = 0。mantissa 23bit のほぼ全域が白色ノイズ。
- held-out 条件付き CE = 10.01b(コイン=10b)= 予測力ゼロ。puresky と同一署名。
- **exact 天井 ≈ 1.24-1.30x**(random tail が raw なので空間予測は上位の僅かしか効かない)。
- near-lossless は単一ピクセル床でも nl15 ~3.0-3.3x、実コーデックの空間予測込みなら
  puresky 同様 8-15x 級に届く見込み。

結論: **実写真(本番)の exact-12x は物理的に不可能**(フォトンショットノイズ)。
「真のロスレス 12x」という現行目標は本番データに対して到達不能。製品の本体は
near-lossless 側であり、exact は Regime B + 証明書モードに退避させるべき。

## 提案する方向(優先度順・DSCF0009判定で改訂)

### 方向0 (新・最優先): ノイズ床適応 near-lossless = 「ノイズと戦わず、測って正確に捨てる」
- 固定 low_bits ではなく、**各ピクセルの局所センサーノイズ床まで** mantissa を落とす。
- フォトンショットノイズは信号依存(相対std ∝ 1/sqrt(信号))なので、暗部=多く落とし
  明部=少なく落とすのが、最も圧縮しつつ最も忠実(センサー精度に対して numerically
  lossless)。fixed-N より強く、かつ破綻しない。
- noise-floor 検出器(本メモ Direction 2)が drop 量を画素/タイル毎に決める。
- これが「exact が死んだ後に残る本物の前進」。現行 StageMantissaQuantize の自然な進化。

### 方向1 (脇モード・Regime B 限定): exact 努力は Regime B だけに残す = 「16bit画像の現代的可逆符号化」
- abandoned/spruit/studio は bfloat16、Tree/CandleGlass は half16 と確定。
  → ordered 16bit 整数画像として抽出し、**JPEG-XL modular / FLIF MANIAC 級の
  自己補正予測 + MA決定木**を当てる。side-info は画像全体で償却され、puresky tail
  辞書で負けた理由(per-context辞書)が起きない。
- 既存 `evaluate_grouped_context_tree.py` の tree が abandoned 5.88→6.12x を出した
  のは headroom の証拠。tree を弱い固定 family へ落とさず、画像ローカルの安価な
  signaled tree か shipped predictor として本気で実装する。
- 補助案: 外部ヒントの **exponent面 + 高位mantissa面の二画像分解**(どちらも滑らか)
  を 2D 画像予測器で個別符号化し、現行 grouped-delta と Regime B で比較。
- 成功条件: realistic-no-puresky で `8.04x → 9x+`、Regime B 個別で `+10%`。

### 方向2 (productize the reality): ノイズ床証明書 + 自動ルーター
- 私が走らせた診断(bitplane P(1)≈0.5 + 空間白色性)を **per-tile/channel の安価な
  検出器**にする(popcount≈half + lag1相関の近似で十分・decoder不要、encoderが信号)。
- tile を分類: (a) zero-tail大 → half/bfloat native route、(b) random-tail大 →
  exact では構造bitだけ精密符号化しノイズbitは raw 直送(モデル探索を浪費しない)、
  near-lossless許可時はノイズbitを破棄、(c) 中間 → 現行。
- 同時に各画像の **exact 圧縮率上限の証明書**(=32/(noise_bits+構造)) を出力。
  「これ以上は不可能」を感覚でなく数値で言える。研究の停止判断にも使える。
- 成功条件: noise tile を正しく検出し、探索時間削減と near-lossless 自動化を両立。

### 方向3 (小さい確実な exact 利得): shipped 固定 prior
- bit0 の P≈0.33 を **コーパス学習の固定 prior** として rANS に与える(side-info ゼロ、
  整数化で決定論的)。Regime A で約 0.08 bps、Regime B では無関係。
- 併せて「bit k | exponent bucket」の弱いコーパス bias も学習。突破ではないが無料。

### 方向4 (立ち位置確定): 外部ベースライン実測
- Aras 2025 によれば SOTA でも float 画像は `2.0〜2.3x`、JPEG-XL L7 で `2.186x`、
  mantissa が最難関。**現行の非puresky `8x` / puresky `2.3x` は既に競争力がある**。
- JPEG-XL(modular,lossless)/ zfp reversible / EXR ZIP・PIZ・HTJ2K / mesh-opt+zstd を
  同一13画像で実測し、レジーム別に勝敗表を作る。Regime B の勝ち、Regime A の
  「皆同じ床」を可視化して製品主張の土台にする。

## 既に死んでいる方向(再探索しないため・データ根拠あり)
- puresky tail の **チャンネル間ノイズ相関**: r≈0.00〜0.03 → 利得なし(実測済)。
- puresky の **低精度/単純変換起源**(per-ch scale 等): bit白色・41% unique で否定。
- puresky tail の予測器/辞書/hash/tree/wavelet/prefix: 白色なので原理的に不可。
- 「near-lossless base + exact correction」: correction = 白色ノイズなので元に戻る。

## Verification(次に実験する場合の入口)
- 方向1: `pixi run python scripts/evaluate_grouped_context_tree.py --glob 'ph_abandoned*.exr' --max-leaves 16`
  を Regime B 限定で sweep。exponent/mantissa 二画像分解は新規 probe が必要。
- 方向2: 上記 read-only 診断をスクリプト化し全コーパスで noise tile 率を出す。
- 方向4: `scripts/benchmark.py` / OpenEXR・JPEG-XL CLI で外部比較表を生成。

## ひとことで
本番データ(DSCF0009 = 実写真 float32)は mantissa の ~18-21bit がフォトンショット
ノイズで、exact 天井 ~1.3x。**exact-12x は本番では物理的に不可能**だとデータで証明
できた。だからヘッドライン目標を near-lossless へ転換する:
- 本命 = **ノイズ床適応 near-lossless**(各画素のノイズ床まで正確に捨てる)。
- exact = Regime B(half/bfloat 起源アセット)限定の脇モード + 「不可能」証明書。
- ルーター = ノイズ床検出で per-tile 自動振り分け。
near-lossless は逃げではなく、Regime A に対する情報理論的に正しい唯一の答え。
