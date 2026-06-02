# Float32 HDR 圧縮 実験結果サマリ

この文書は、現在の float32 HDR 専用圧縮ライブラリの結果を日本語で
見やすくまとめたものです。詳細な研究ログは `PLAN_ML_V2.md` と
`LOSSLESS_RESEARCH_REBOOT.md` にあります。

## 現在の結論

現時点で一番筋が良い lossless 本線は `GroupedDelta / GDXB` です。
float32 を単なる byte 列として扱うのではなく、順序保存 float 表現、
tile ごとの予測、bitplane rANS、context family を組み合わせています。

ただし puresky のような滑らかな空画像は、見た目のグラデーションが
簡単そうに見えても、float32 の低 mantissa tail がほぼランダムに近く、
完全 lossless ではそこを全部保存する必要があります。ここが `12x` 以上を
阻む最大要因です。

一方で near-lossless として低 mantissa bit を明示的に 0 にすると、
puresky は一気に `12x` 級へ入ります。これは「圧縮器が見落としていた」
というより、「保存すべき情報を near-lossless では捨てられる」ことが
効いています。

## 主要結果

### Exact lossless

| 条件 | 圧縮率 geomean | メモ |
|---|---:|---|
| GDX3/GDXB 系、13画像、effort9 | `7.317x` | 現在の exact lossless 本線 |
| realistic-no-puresky crop128、effort11 | `8.015x` | effort12 に近い高速寄り設定 |
| realistic-no-puresky crop128、effort12 | `8.043x` | 現在の高圧縮寄り設定 |
| puresky-hard crop128、effort11 | `2.345x` | 低 mantissa tail が支配的 |
| puresky-hard crop128、effort12 | `2.347x` | effort を上げてもほぼ改善しない |

解釈:

- puresky を除く現実寄りデータでは `8x` 近辺が見えています。
- puresky は完全 lossless だと `2.3x` 台で強く詰まっています。
- その理由は visible な空の滑らかさではなく、float32 の低 mantissa tail
  が exact payload の大半を占めるためです。

### Near-lossless

`StageMantissaQuantize` を追加し、圧縮前に有限 float32 の下位 mantissa bit
を 0 にできるようにしました。NaN / Inf はそのまま保持します。

`ph_*_1k.exr` 5枚、crop128、effort11:

| low mantissa bits | 圧縮率 geomean | 意味 |
|---:|---:|---|
| `0` | `6.076x` | exact 相当 |
| `8` | `7.634x` | かなり保守的 |
| `12` | `9.314x` | 品質重視 near-lossless 候補 |
| `15` | `12.071x` | 12x 級に到達 |

puresky-hard 2枚のみ、crop128、effort11:

| low mantissa bits | 圧縮率 geomean | メモ |
|---:|---:|---|
| `0` | `2.345x` | exact 相当 |
| `8` | `4.151x` | すでに大きく改善 |
| `12` | `6.825x` | max relative error 約 `4.9e-4` |
| `15` | `13.051x` | puresky では 12x 超え |

代表的な puresky 個別結果:

| 画像 | exact | low12 | low15 |
|---|---:|---:|---:|
| `ph_belfast_sunset_puresky_1k` | `2.40x` | `7.33x` | `14.88x` |
| `ph_kloppenheim_06_puresky_1k` | `2.29x` | `6.35x` | `11.44x` |

品質目安:

- low8: max relative error 約 `3.0e-5`、PSNR は puresky で `103dB` 以上。
- low12: max relative error 約 `4.9e-4`、PSNR は puresky で `79-83dB` 程度。
- low15: max relative error 約 `3.9e-3`、PSNR は puresky で `61-64dB` 程度。

## 重要な発見

### 1. AI より前に float32 の構造が重要だった

初期の learned / byte-LM 系は、既存圧縮に対して明確な優位を出せませんでした。
理由は、画像を単なる byte stream として扱うと、float32 の sign / exponent /
mantissa や、数値的な近さを十分に使えないためです。

現在の GDX 系は、float32 を ordered integer 的に扱い、数値的な近さを
bit payload に反映させる方向です。この方が AI モデルより先に大きく効きました。

### 2. puresky は「滑らかだから簡単」ではなかった

空のグラデーション自体は予測できます。しかし完全 lossless では、見た目に
ほぼ影響しない低 mantissa bit も保存しなければいけません。

puresky-hard crop128 では、payload の大半が tail 側に寄っています。
過去の audit では main payload 約 `17.9%`、tail payload 約 `82.1%` でした。

### 3. exact correction stream は勝ち筋ではなかった

「near-lossless base を作って、捨てた低 bit を別ストリームで exact に戻す」
方法も試しました。しかし puresky では、補正 bit を保存した瞬間に圧縮率が
ほぼ元へ戻りました。

つまり、near-lossless の利点は「低 bit を後から賢く保存できる」ことではなく、
「低 bit を保存しない選択ができる」ことです。

### 4. near-lossless は別オプションにするのが自然

lossless と near-lossless は目的が違います。

- exact lossless: float32 bit pattern を完全復元する研究本線。
- near-lossless: 低 mantissa tail を制御して、実用圧縮率を上げる製品モード。

したがって、near-lossless は lossless を置き換えるのではなく、別オプションとして
実装する方針が妥当です。

## 実装済みのもの

### C++ codec

- `StageMantissaQuantize` を追加。
- `near_lossless_bits` で 0 にする mantissa 下位 bit 数を指定。
- 有限 float32 のみ量子化し、NaN / Inf は保持。
- decode は passthrough。payload 自体が量子化後画像を保持します。
- 外側フレームヘッダを version `2` に更新し、`near_lossless_bits` を保存。
- version `1` フレームの decode 互換は維持。

### Python API

```python
radiance_codec.encode_near_lossless(pixels, low_bits, effort=11)
radiance_codec.quantize_mantissa(pixels, low_bits)
```

### テスト

- passthrough bit-exact roundtrip。
- near-lossless decode が「元画像」ではなく「量子化後画像」と一致すること。
- NaN / Inf を含む入力で finite 値のみ量子化すること。
- grouped-delta 既存テスト。
- rANS / bitshuffle / predictor / structural context 既存テスト。

## 再現コマンド

ビルド:

```bash
pixi run build
```

C++ テスト:

```bash
codec/build/test_codec
codec/build/test_grouped_delta
codec/build/test_structural_context
codec/build/test_bitshuffle
codec/build/test_predictor
codec/build/test_rans
codec/build/test_rans_binary
```

near-lossless ベンチ:

```bash
pixi run python scripts/estimate_mantissa_quantization.py \
  --glob 'ph_*_1k.exr' \
  --crop-size 128 \
  --effort 11 \
  --low-bits 0,8,12,15
```

pixi task:

```bash
pixi run bench-near-lossless-cpp
```

## 次の一手

### 短期

near-lossless の policy layer を作るのが一番効果が見えやすいです。

候補:

- half-like / bfloat-like tile は exact のまま。
- puresky-like true-float tail は `low12` または `low15` を選択。
- random/noise 的な tile は別扱いし、無理に 12x を主張しない。

### exact lossless 側

完全 lossless はまだ研究を続ける価値があります。ただし、低 tail へ
普通の context を追加するだけでは大きく動きにくいです。

次に狙うなら:

- signaled context tree の side information を安くする。
- upper-body / ordered-body の reversible block transform を試す。
- source precision classifier で half-like / true-float / random-tail を
  先に分ける。
- AI を使うなら full-image autoencoder ではなく、hard tile の context mixer
  として限定的に使う。

## ひとことで

完全 lossless では `8x` 付近までかなり来ていますが、puresky の低 mantissa tail が
`12x` の壁です。near-lossless ではその壁を明示的に扱えるようになり、
`low15` で puresky と `ph_*` crop128 geomean が `12x` 級に入りました。

次は「どの tile にどのモードを使うか」を決める selector が、最も実用的な前進です。
