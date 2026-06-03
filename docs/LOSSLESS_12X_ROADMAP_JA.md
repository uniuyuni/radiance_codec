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

追加サンプル `data/sample_1920×1280.exr`:

- OpenEXR scanline / uncompressed / `1920x1280` / RGB / `half`
- 値域は各 channel `0..1`、finite、float32 展開後の lower 13 bits は全ゼロ
- crop128, effort12, GDXB actual: `5.567x`
- crop512, effort12, GDXB actual: `7.177x`
- crop128 source-precision estimate: current `5.179x`, half16 routed `5.185x`

判断: true lossless 研究では「half由来の高解像度実データ」として使える。既存の
`realistic-no-puresky` full benchmark に無条件で混ぜると重くなるため、
`highres-sample` corpus として明示的に組み込む。

追加サンプル `data/sample_hilberts-mill-conference-room_2K.exr`:

- OpenEXR scanline / PIZ / `2048x1024` / RGBA / `float`
- RGB は finite、値域はおおむね `0.0028..149.1`、A は定数 `1.0`
- RGB だけで見ると lower 13 bits zero は約 `0.016%`、lower 16 bits zero は約
  `0.002%` で、half由来ではない float32 精度データと見てよい
- crop128, effort12, GDXB actual: `2.016x`
- crop128 source-precision estimate: current `1.991x`, routed `1.991x`,
  half/bf16 eligible tileなし
- crop128 actual stream は tail payload が約 `70.3%`

判断: true float32 の hard HDR sample として使える。`sample_*.exr` パターンに
入るため、`highres-sample` corpus の対象になる。12x 改善の確認では、half由来の
`sample_1920×1280.exr` と対になる「float32 tail hard」代表として扱う。

8x feasibility 初回判定:

- full, effort12, GDXB actual: `2.094x`
  - main `26.9%`, tail `73.0%`
  - tail split `128/128`, channel split `103/128`
- crop512, effort12, GDXB actual: `2.101x`
  - main `26.7%`, tail `73.3%`
- crop512, raw low mantissa conditional entropy:
  - low8: GDX low `5.999 bits/sample`, best conditional `2.706 bits/sample`
  - low10: GDX low `7.507 bits/sample`, best conditional `2.783 bits/sample`
  - low12: GDX low `9.014 bits/sample`, best conditional `2.805 bits/sample`
  - low15: GDX low `11.276 bits/sample`, best conditional `2.856 bits/sample`

8x は `4.0 bits/sample` が目標。crop512 の現行 total は `32 / 2.101 = 約15.23`
bits/sample で、tail 以外だけでも約 `4.07 bits/sample` ある。tail をかなり理想的な
条件付き entropy `2.856 bits/sample` に置き換えても、合計は約 `6.92 bits/sample`
で `4.6x` 程度に留まる。これは side-info なしの楽観下限なので、tail route 単独で
8x は現実的ではない。

判断更新: この sample で 8x を狙うには、tail を `~3 bps` まで落とすだけでなく、
main/high 側も `4.1 -> 1.1 bps` 級へ削る必要がある。次の探索は tail 専用route
だけではなく、PIZ-like な可逆 wavelet/byte-plane route や image-global transform
で main と tail を同時に変える方向に寄せる。

4x 目標への再設定:

- 4x は `8.0 bits/sample` が目標。
- crop512 の現行 total は約 `15.23 bits/sample`。
- tail以外が約 `4.07 bits/sample` なので、tail を `11.28 -> 3.9 bits/sample`
  付近まで落とせれば 4x が見える。
- raw/body low15 tail の条件付き entropy 下限は約 `2.856 bits/sample` なので、
  side-info と model cost を `~1 bps` 程度に抑えられる route が必要。

tail 狙い撃ち 初回追試:

- `probe_tail_image_adaptive_context.py` に `--tail-transforms` と `--no-save` を追加。
  raw / `xor_green` / `sub_green` / previous channel 系を試せるようにした。
- `probe_tail_static_prob_table.py` と `probe_tail_context_palette.py` に `--no-save` を追加。
- `probe_tail_context_palette.py` に `--scope image` と `--tail-transforms` を追加。
- `probe_tail_conditional_entropy.py` に `--tail-source body_payload` を追加し、GDX residual
  low tail そのものの下限を見られるようにした。

結果、`sample_hilberts`, crop128:

- image-global adaptive KT tail route:
  current `1.992x`, hybrid `1.992x`; adaptive は選ばれない。
- static bit probability table:
  current low `11.269 bps`, static low `12.361 bps`; current に負ける。
- image-global context palette:
  current low `11.269 bps`, palette low `11.379 bps`,
  adaptive dictionary low `12.069 bps`; current に僅差負け。
- RGB tail transform:
  `xor_green` / `sub_green` は raw とほぼ同等、previous-channel 系は悪化。
- GDX body payload low15 の条件付き entropy 下限も crop512 で `2.856 bps`。

判断更新: `2.856 bps` の下限は有望だが、単純な adaptive/table/palette では
side-info や初出値コストでほぼ全て失われる。4x の次候補は「context ごとの値表」
ではなく、値を送らずに tail を小さくする predictor / reversible transform:

- high/exponent から low tail を直接予測し、residual を GDX へ戻す。
- tail 専用 reversible wavelet/lifting を試す。
- PIZ-like な byte-plane / wavelet route を tail+main の一体表現として試す。
- 条件付きentropy probe は今後、in-sample ではなく held-out / causal lower bound も
  併記して、過学習した下限を採用しない。

tail predictor / transform 追試:

- `scripts/probe_tail_predictor_residual.py` を追加した。
  - low ordered tail を causal predictor の residual として符号化する。
  - predictor: zero / west / north / average / gradient / median / paeth /
    previous_channel / green
  - residual: modular sub / xor
- `scripts/probe_tail_wavelet_transform.py` を追加した。
  - low ordered tail または body payload low bits だけに可逆 2x2 pyramid をかける。
- `sample_hilberts`, crop128, low15:
  - predictor residual: current `1.992x`, hybrid `1.998x`, gain `1.0032`
  - best は `zero_sub`。つまり予測器ではなく、ordered low tail を直接別routeに
    するだけの微改善。
  - tail wavelet: current のまま、選択されない。
- `sample_hilberts`, crop512, low15:
  - predictor residual: current `2.060x`, hybrid `2.069x`, gain `1.0041`
  - `zero_sub` が全16 tileで選択。
- tail幅 `8/10/12/15` を `zero/green` 候補で見ても、crop512 はすべて
  current `2.060x`, hybrid `2.069x` で同程度。

判断更新: tail を current residual から ordered-tail direct split に変えるだけで
微小な exact gain はあるが、4x には全く足りない。単純な causal predictor や
tail-only 2x2 wavelet は、この sample では breakthrough ではない。

PIZ-like / byte-plane 追試:

- `scripts/probe_nonjpeg_routes.py` に `--no-save` を追加し、診断用の
  `typed_byte_entropy_lower` も標準出力へ出すようにした。
- `sample_hilberts`, crop128, body preset, zstd level3:
  current `1.992x`, hybrid `1.992x`, typed lower `1.427x`
- 同 crop128, `--search-candidates`:
  current `1.992x`, hybrid `1.992x`, typed lower `1.427x`
- 同 crop512, body preset, zstd level3:
  current `2.060x`, hybrid `2.060x`, typed lower `1.515x`
- 採択 route はすべて `gdx2`。typed byte-plane は entropy lower で見ても現行より悪い。

判断更新: 現行 GDX payload の後段を byte-plane/zstd 系へ差し替えるだけでは、この
hard float32 sample では改善しない。PIZ/ZIP 的な思想を続けるなら、後段 compressor
ではなく、payload 前の可逆表現そのものを変える必要がある。

hash / LSH protocol 追試:

- 神領域アイデアとして、暗号/LSH 的な hash で tail に構造を作れるかを検証する。
- 外部辞書や共有乱数プールを圧縮サイズに数えない route は除外する。
- exact-compatible な形だけを試す:
  - tail permutation route:
    - low-tail symbol に固定の可逆 permutation/hash をかける。
    - その後の causal residual が小さくなるかを見る。
  - decoder-visible hash/LSH context route:
    - exponent / high mantissa / payload high / channel / xy / neighbor high だけから
      bucket を作る。
    - tail symbol は sparse adaptive alphabet で逐次符号化する。
    - 値辞書を送らず、decoder も復号済み symbol で同じ model を更新する。
- `scripts/probe_tail_hash_lsh_protocol.py` を追加した。
- 既存 `probe_tail_conditional_entropy.py` は payload context 追加部が早期 `return` で
  到達していなかったため修正した。

`sample_hilberts`, crop128, body payload:

- low08: GDX `5.992 bps`, adaptive hash/LSH `6.013 bps`, best residual `6.287 bps`
- low10: GDX `7.500 bps`, adaptive hash/LSH `7.580 bps`, best residual `7.903 bps`
- low12: GDX `9.008 bps`, adaptive hash/LSH `9.149 bps`, best residual `9.526 bps`
- low15: GDX `11.269 bps`, adaptive hash/LSH `11.751 bps`, best residual `11.962 bps`
- raw mantissa low15 でも adaptive `11.751 bps`, best hash-permutation residual
  `12.087 bps` で GDX に届かない。
- 一方、条件付きエントロピー下限は payload context 修正後も low15
  `2.850 bps` (`hash12_exp_high_xy_channel`) が見える。

判断更新: hash/LSH は「下限の覗き窓」としては有用だが、実際の exact route にすると
未知 symbol / sparse alphabet のコストで GDX に負ける。神領域 sandbox として残すが、
C++ 実装候補ではなく、次の課題は `2.85 bps` の下限へ近づくための
model sharing / escape coding / side-info amortization。

top-K escape table 追試:

- `scripts/probe_tail_escape_table_route.py` を追加した。
- 目的は、神視点の conditional entropy と実装可能 route の中間を測ること。
- 各 decoder-visible context ごとに top-K tail 値だけ辞書として送り、外れ値は raw
  escape する。

`sample_hilberts`, crop128, body payload:

- low08: GDX `5.992 bps`, lower `2.706 bps`, top-K `6.001 bps`
- low10: GDX `7.500 bps`, lower `2.765 bps`, top-K `7.501 bps`
- low12: GDX `9.008 bps`, lower `2.808 bps`, top-K `9.001 bps`
- low15: GDX `11.269 bps`, lower `2.850 bps`, top-K `11.251 bps`
- crop128 low15 の best route は `hash4:channel` で、details は
  `table_contexts=1`, `raw_contexts=3`, `dictionary_values=1`。
  つまりほぼ定数の channel を拾っただけで、下限本体には届いていない。
- crop256 low15:
  - GDX `11.225 bps`, lower `4.654 bps`, top-K `11.250 bps`
  - cropを広げると top-K は GDX に負ける。

下限差分診断:

- crop128 `hash12_exp_high_xy_channel`:
  - samples `65536`, contexts `4084`
  - samples/context mean `16.0`, median `12`
  - unique tail/context mean `12.0`, median `12`
  - tail global unique `24936`
- crop256:
  - samples `262144`, contexts `4096`
  - samples/context mean `64.0`, median `47`
  - unique tail/context mean `48.0`, median `47`
  - tail global unique `32583`

判断更新: `2.85 bps` 付近の下限は、context を細かく割りすぎた in-sample な薄さに
かなり依存している。contextごとの辞書値を送ると tail をほぼ送るのと同じになる。
次に続けるなら top-K 辞書ではなく、複数 context で共有できる generative predictor
か、escape を raw tail ではなく low-rank/correction として送る route が必要。

prefix-cascade symbol coder 追試:

- `scripts/probe_tail_prefix_cascade.py` を追加した。
- top-K 辞書を送らずに whole-symbol 構造を使うため、tail symbol を MSB から LSB
  へ順に符号化する。
- 各bitの context は
  `decoder-visible high/exponent/payload_high context + 既に復号した同一symbol prefix`。
- context node は KT universal binary cost なので、decoder は side-info なしで同じ
  adaptive model を更新できる。

`sample_hilberts`, body payload, low15:

- crop128:
  - GDX `11.269 bps`
  - prefix-cascade `11.840 bps`
  - best `hash12_payload_exp_high_xy_channel:prefix15`
  - lower `2.850 bps`
- crop256, image scope:
  - GDX `11.225 bps`
  - prefix-cascade `11.891 bps`
  - best `hash12_payload_exp_high_xy_channel:prefix15`
  - lower `4.310 bps`

判断更新: prefix は効いているが、visited node 数が多く KT コストで GDX に負ける。
GDX の既存 bitplane family は tail prefix 情報をかなり上手く使っている。辞書なし
symbol coder でも hard 4x への突破口にはならない。

visible predictor residual 追試:

- `scripts/probe_tail_visible_predictor_residual.py` を追加した。
- 辞書値を送らず、decoder-visible な high/exponent/payload_high から固定関数で
  low-tail を直接予測し、residual だけを既存 bitplane context で符号化する。
- predictor:
  - high low bits
  - high hash
  - exponent/channel/xy mix hash
  - west/north/green/previous-channel の payload_high
- `sample_hilberts`, crop128, body payload, low `8/10/12/15`:
  - すべて selected `zero_sub`
  - gain は `1.0002-1.0003`
  - best visible candidates は `green_payload_high_low` 系だが GDX に届かない。
- raw mantissa low15:
  - current `1.992x`, hybrid `1.998x`, gain `1.0032`
  - selected は `zero_sub`
  - visible hash predictor は現行より悪い。

判断更新: high/exponent には conditional entropy 上の情報はあるが、単純な固定 hash /
prefix predictor では取り出せない。次に続けるなら learned 係数や木構造が必要だが、
係数・木を送る side-info が top-K と同じ問題を持つため、C++ 実装優先度は低い。

bounded / fixed-grid 入力メモ:

- `sample_hilberts` は RGB 最大値が約 `149` あり、0..4 前提の画像ではない。
- 本番入力で `0 <= value <= 4` が保証できるなら、sign bit と巨大 exponent 空間を
  codec assumption として削れる。
- ただし exact bit-for-bit では「数値の最小値が0」だけでは足りない。`-0.0` は
  数値比較では0でも sign bit が1なので、sign省略routeの条件は
  `all(sign_bit == 0)` とする。
- nonnegative tile/channel route:
  - image/tile/channelごとに sign bit が全0なら、1bit flagだけ送って sign payload を省略。
  - signが混ざる tile は従来routeへfallback。
  - 値域 `0..4` などが上流仕様で保証される場合は、range証明なしでこのrouteを
    優先候補にできる。
- ただし値域制限だけでは low mantissa tail は残る。4x以上へ効かせるには、
  非負だけでなく fixed-point / integer grid / lower mantissa zero など発生源側の
  制約が重要。
- 次の別系統候補として、bounded exact route:
  - nonnegative flagで sign省略
  - narrow exponent alphabet
  - fixed-grid detectorとscale signaling
  - range外 escape stream
  を検討する。

nonnegative / bounded route 追試:

- `scripts/probe_nonnegative_sign_route.py` を追加した。
- `scripts/probe_bounded_value_route.py` を追加した。
- exact 条件は数値 min ではなく `all(sign_bit == 0)`。
- `sample_hilberts`, crop128:
  - sign stream current `0.00013 bps`, tile signless `0.00002 bps`
  - signless route は成立するが、総圧縮率への寄与はほぼゼロ。
- `sample_*.exr`, crop128:
  - `sample_1920×1280`: sign current `0.00018 bps`, routed `0.00002 bps`
  - `sample_hilberts`: sign current `0.00013 bps`, routed `0.00002 bps`
- bounded exponent alphabet, range `0..4`, crop128:
  - `sample_1920×1280`: eligible, sign+exponent current `0.4007 bps`,
    routed `0.4007 bps`
  - `sample_hilberts`: crop128 は局所的に eligible, current `0.4921 bps`,
    routed `0.4921 bps`
- `sample_hilberts`, crop512:
  - 局所 range `[0.0036, 1.1617]`, current `0.3269 bps`,
    routed `0.3269 bps`

判断更新: nonnegative / bounded assumption は bitstream の安全な pruning 条件として
計画に残す。ただし sign/exponent は現行 GDX がすでにかなり圧縮しており、4x の主因
にはならない。実装するなら tail/main route を邪魔しない lightweight fallback。

shared-exponent integer route メモ:

- 「値が整数である」という仮定は破棄する。少数値のまま exact に扱う。
- 代わりに、float32 を tile/block ごとの共通 2冪スケールで整数へ写す route を
  計画に入れる。
- 正規化floatでは概念的に
  `value = significand_integer * 2^(exponent - 150)` なので、block内の最小exponentを
  共有スケールにすれば、各値は
  `significand_integer << (exponent - block_min_exponent)` の整数として exact に表せる。
- 必要bit幅はおおむね `24 + exponent_span`。`sample_hilberts` のような
  `0.002..149` 級でも block/tile 内の exponent span が狭ければ `uint64` に収まる
  可能性が高い。
- これは IEEE ordered bits をさらに煮詰めるのではなく、数値の連続性を保つ
  block fixed-point 表現へ変える route。tail 問題を mantissa low bits としてではなく、
  integer residual / bitplane / wavelet の問題へ移せるかを見る。
- 次に作る監査:
  - tile/block ごとの exponent span
  - `uint32` / `uint64` 収まり率
  - 必要bit幅分布
  - shared-scale integer plane の GDX cost
  - shared-scale integer residual / wavelet cost
- 採用条件:
  - `sample_hilberts` crop512 で current `2.06x` を明確に超える。
  - tailだけでなく main+tail を一体で削る兆候がある。
  - uint64 route の追加metadataが 4x 目標に対して許容範囲に収まる。

block fixed-point route 初回追試:

- `scripts/probe_block_fixed_point_route.py` を追加した。
- finite float32 を tile 内共有の 2冪スケールへ exact signed integer として写す。
  - normal: `significand * 2^(exponent - 150)`
  - subnormal: `mantissa * 2^-149`
  - tile 内の最小 scale field を共有し、各値を left-shift した整数へ変換。
- `--verify` では shared integer から float32 bit を復元し、bit完全一致を確認する。
- integer stream は `raw / west / north / average / gradient / paeth /
  previous_channel / reversible green/YCoCg` residual を作り、byte-plane zstd と
  byte entropy lower を測る。
- 初回の `green_delta` winner は invalidated:
  - R/B を G で予測していたが、G 自身を送っていなかった。
  - `current=2.06x -> fixed=2.25x` 級の数字は非可逆 route による幻なので採用しない。
  - 可逆形は `R-G, G, B-G, A` または `G, R-G, B-G, A` とする。

`sample_hilberts-mill-conference-room_2K.exr`:

- crop128, tile128:
  - current `1.992x`, fixed zstd `1.614x`, gain `0.8104`
  - exact復元 verify 通過
  - width `32`
- crop512, tile512:
  - current `2.073x`, fixed zstd `1.675x`, gain `0.8077`
  - best exact route は `paeth_delta`
  - width `33`

`sample_1920×1280.exr` half-derived:

- crop512, tile512:
  - current `6.463x`, fixed zstd `5.311x`, gain `0.8217`
  - best exact route は `green_reorder_lift`
  - width `29`

判断更新: shared-scale integer 表現は exact には作れるが、単純な byte-plane zstd /
spatial residual では GDX に負ける。非可逆 `green_delta` を除くと positive signal は
消える。Route としては C++ 化しない。ただし、重要な教訓は残る:

- `G` を decoder-visible にする decode schedule を組めば、R/B を G で予測する方向は
  まだ可能性がある。
- その場合は `G/A raw + R-G/B-G residual` のように、基準channelを明示的に送る必要がある。
- 次に試すなら shared fixed-point ではなく、現行 GDX payload の channel decode order を
  変えて、green-first channel route を exact に評価する方が筋がよい。

green-first mixed route 追試:

- `scripts/probe_block_fixed_point_route.py` に `green_ref_rb` を追加した。
- G/A は従来 GDX で先に送る。
- R/B は shared-scale integer に写し、`R-G` / `B-G` residual を byte-plane zstd する。
- この route は G を無料扱いしないので exact-compatible。

結果:

- `sample_hilberts`, crop128, tile128:
  - current `1.992x`, green-ref fixed `1.796x`, gain `0.9019`
- `sample_hilberts`, crop512, tile512:
  - current `2.073x`, green-ref fixed `1.927x`, gain `0.9294`
- `sample_1920×1280`, crop512, tile512:
  - current `6.463x`, green-ref fixed `8.495x`, gain `1.3144`

判断更新: hard float32 sample では負けるので 4x 本命ではない。一方、half-derived /
lower-precision high-res sample では `+31%` の exact signal がある。Route C の
source-precision / half-like fallback と統合して、green-first mixed channel route として
再評価する価値がある。

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

shared tree 追試:

- `probe_signaled_context_tree_mdl.py` に `--tree-scope image` を追加した。
- `tile` scope は tile/bitplane ごとに木を送る想定、`image` scope は同一画像内の
  同じ bitplane に木を共有する想定。
- `ph_abandoned_tiled_room_1k`, crop256, bits15-17, leaves8:
  - tile scope: existing `28.230x`, hybrid `29.027x`, gain `1.0282`
  - image scope: existing `28.230x`, hybrid `28.588x`, gain `1.0127`
- `oexr_ScanLines_CandleGlass`, crop256, bits15-17, image scope:
  existing `20.451x`, hybrid `20.525x`, gain `1.0036`

判断更新: tree は tile-local だと数%の信号があるが、実装しやすい image-shared tree
では弱くなる。C++ に大きな signaled tree を入れる前に、tree から頻出splitを抽出し、
固定 context family として追加する方が実装コストに対する期待値が高い。

fixed family 抽出 追試:

- `scripts/probe_fixed_context_family_candidates.py` を追加した。
- C++ 最新の固定 context family 相当を baseline にし、tree 由来の
  `p1/wp1/np1/channel/xy` 系候補を同じ selector cost で比較する。
- `ph_abandoned_tiled_room_1k`, crop64, bits15-20:
  current `9.928x`, extended `9.939x`, gain `1.0011`
- `oexr_ScanLines_CandleGlass`, crop64, bits15-20:
  current `9.161x`, extended `9.162x`, gain `1.0001`
- 前に tile tree が効いた `ph_abandoned`, crop256, bits15-17:
  current `28.553x`, extended `28.662x`, gain `1.0038`
- `oexr_ScanLines_CandleGlass`, crop256, bits15-17:
  current `20.508x`, extended `20.523x`, gain `1.0007`
- decoder schedule 変更を要する `ordered_high` 候補も crop128 では
  `1.001-1.002x` 程度に留まった。

判断更新: tree の数% signal は単純な固定 context family 追加ではほぼ回収できない。
固定 family 抽出は優先度を下げる。Route A を続けるなら、leaf/family を送る小さい
signaled structural choice か、bitplane 表現そのものを変える方向に絞る。

`sample_hilberts` hard-float 追試:

- crop128, tile scope, leaves16:
  - tail bits0-14: existing `2.839x`, hybrid `2.863x`, gain `1.0084`
  - mid bits15-20: existing `8.052x`, hybrid `8.070x`, gain `1.0023`
  - high bits21-30: existing `38.837x`, hybrid `38.837x`, gain `1.0000`
- tail bits0-14 に `--feature-set with_ordered_high` を足しても hybrid `2.862x`,
  gain `1.0079`。

判断更新: この sample でも MDL tree は tail に小さく効くが、4x に必要な
`~2x -> 4x` 級の改善とは桁が違う。Route A は C++ 化候補ではなく、次の
表現変更 route の scorer / diagnostic として残す。

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

初回の実装方針:

- 上位 ordered-body actual bits を可逆 2x2 pyramid で変換する。
- 下位 `split_bit` 未満は現行 GDX residual bit として保持する。
- これにより puresky の random tail を壊さず、half/bfloat-like な main payload の
  表現だけを試す。

軽量スモーク:

```text
pixi run python scripts/probe_ordered_body_block_transform.py --glob 'ph_abandoned_tiled_room_1k.exr' --crop-size 64 --split-bit 15 --max-levels 4 --no-save
```

初回スモーク結果:

- `ph_abandoned_tiled_room_1k`, crop64, split15: GDX `7.158x`, split `5.120x`, hybrid `7.158x`
- `ph_abandoned_tiled_room_1k`, crop64, split21: GDX `7.158x`, split `5.751x`, hybrid `7.158x`
- `oexr_ScanLines_CandleGlass`, crop64, split15 average: GDX `7.025x`, split `6.069x`, hybrid `7.025x`
- `oexr_ScanLines_CandleGlass`, crop64, split15 anchor: GDX `7.025x`, split `6.346x`, hybrid `7.025x`

解釈: whole-body Lorenzo より壊しにくい分割形にしても、現時点では GDX fallback
しか選ばれない。Route B は「単純 2x2 pyramid」では薄く、続けるなら predictor
aligned な block basis か、source precision route の後段に限定する。

`sample_hilberts` hard-float 追試:

- `scripts/probe_ordered_body_block_transform.py` を、同一画像準備を使い回して
  `--split-bits` / `--lowpasses` で sweep できるようにした。
- crop128, split `8,10,12,15,18,21,24`, lowpass `anchor,average`:
  - 全パターンで current GDX が選択。
  - route 単体は `1.885x-1.943x` 程度で、GDX `1.992x` に届かない。

判断更新: PIZ-like な単純 2x2 high-body pyramid は、この true float32 sample では
採択されない。Route B を続けるなら、2x2 wavelet ではなく Lorenzo/plane predictor
residual や channel decorrelation を含む、より payload-aligned な変換へ寄せる。

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

既存 `half16_route` full 結果:

- full 13 image: current geomean `7.348x`, half-routed geomean `7.458x`
- byte-weighted total gain `1.0073`
- selected half tiles `225/299`

`probe_source_precision_routes.py` では half16 に加えて bf16/coarse exact route も
候補に入れる。軽量スモーク:

```text
pixi run python scripts/probe_source_precision_routes.py --glob 'ph_abandoned_tiled_room_1k.exr' --crop-size 64 --no-save
```

初回スモーク結果:

- `ph_abandoned_tiled_room_1k`, crop64: current `7.142x`, routed `7.165x`,
  eligible half16/bf16, selected half16, gain `1.0033`
- `ph_spruit_sunrise_1k`, crop64: current `24.822x`, routed `26.435x`,
  eligible half16/bf16, selected bf16, gain `1.0650`
- `oexr_ScanLines_CandleGlass`, crop64: current `7.013x`, routed `7.026x`,
  selected half16, gain `1.0018`

解釈: crop64 では小さいが、bf16/coarse route は `ph_spruit` で half16 より良い
候補になった。Route C は Route B より実装価値が高く、次は full ではなく
crop128/realistic-no-puresky の軽量比較で tile 選択の安定性を見る。

crop128 代表スモーク、mode を
`raw,delta_west,delta_channel_green,delta_med,delta_channel_previous,delta_planar`
に絞った軽量比較:

- `ph_abandoned_tiled_room_1k`: current `7.294x`, routed `7.407x`,
  selected half16, gain `1.0155`
- `ph_spruit_sunrise_1k`: current `27.661x`, routed `30.067x`,
  selected half16, gain `1.0870`
- `ph_studio_small_03_1k`: current `6.106x`, routed `6.113x`,
  selected bf16, gain `1.0012`
- `oexr_ScanLines_CandleGlass`: current `6.685x`, routed `7.104x`,
  selected half16, gain `1.0627`
- `oexr_ScanLines_Cannon`: current `5.325x`, routed `5.354x`,
  selected half16, gain `1.0056`
- `oexr_ScanLines_Tree`: current `4.270x`, routed `4.371x`,
  selected half16, gain `1.0236`

判断: Route C は「12x突破の単独本命」ではないが、確実な exact gain として
GDXB に入れる価値がある。次は C++ 側で half16/bf16 route を tile/channel mode
selection に統合し、route metadata を含めた実バイトで再測定する。

C++ 実験メモ:

- half16 route は既に GDXB に実装済みだった。
- bf16/coarse route は新しい mode id を増やせないため、既存 `Half*` mode を
  source16 route として再利用し、body tile selector `2` で bf16 を信号する実験形にした。
- selector `1` は従来通り tail split、selector `2` は bf16 source route。
- 実バイトでは、微差の bf16 採択は負けやすかったため、bf16候補に `512 bits/tile`
  の保守ペナルティを入れた。
- 軽量確認:
  - `ph_studio_small_03_1k`, crop128, effort12: bf16微差採択を抑制し、half routeで `6.184x`
  - `ph_spruit_sunrise_1k`, crop128, effort12: 従来tail split routeが勝ち、`31.257x`
  - `ph_spruit_sunrise_1k`, crop64, effort12: half routeが勝ち、`27.292x`

判断更新: bf16/coarse route は Python 推定では兆候があったが、現行GDXBの実コストでは
half16またはtail splitに負けることが多い。採択ペナルティ付きで安全なfallback候補に
留める。Route C の主価値は、現時点では既存half16 routeとsource-class pruning。

`sample_hilberts` hard-float 追試:

- crop128: eligible `{}`、selected `{'gdx': 1}`、gain `1.0000`

判断更新: 予想通り、この sample は half/bf16 完全一致 tile がなく、Route C は
4x 達成の主因にならない。

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

既存の近い probe:

```text
scripts/probe_tail_conditional_entropy.py
```

`probe_tail_conditional_entropy.py` に `--no-save` を追加し、保存なしで軽量再確認できる
ようにした。

軽量スモーク:

```text
pixi run python scripts/probe_tail_conditional_entropy.py --glob 'ph_*puresky*.exr' --limit 2 --crop-size 128 --tail-widths 15 --hash-bits 12 --no-save
```

結果:

- `ph_belfast_sunset_puresky_1k`, crop128, low15:
  GDX low `11.296 bits/sample`, best conditional `2.933 bits/sample`
  (`hash12_exp_high_xy_channel`)
- `ph_kloppenheim_06_puresky_1k`, crop128, low15:
  GDX low `11.294 bits/sample`, best conditional `2.887 bits/sample`
  (`hash12_exp_high_xy_channel`)

判断: high/exponent/xy/channel には低位 tail をかなり説明する情報がある。ただしこれは
条件付きエントロピー下限で、model/table/tree の送信費をまだ含まない。既存の
static/adaptive table probe ではこの差を十分に回収できていないため、次の Route D は
「もう一つの table」ではなく、side-info lower bound と support 分布を明示する
certificate として整理する。

### Route E: AI context mixer

優先度: 中

目的:

- 生成復元ではなく、exact entropy model の probability predictor としてだけ使う。

条件:

- Route A-D で hard bitplanes と feature set を確定してから。
- model weights の送信はしない。固定モデルまたは小さな signaled calibration のみ。
- decode speed と deterministic reproducibility を最優先。

現時点では、画像 autoencoder や lossy base + correction stream は exact 12x の本命ではない。
correction が大きくなり、結局 low tail を保存する問題に戻る。

tile-split MLP context mixer 追試:

- `scripts/probe_tail_mlp_tile_split.py` を追加した。
- 同一 hard image 内で train tile / held-out eval tile を分ける。
- 予測対象は exact payload bit の `P(bit=1)`。entropy coder は全bitを可逆に送る。
- 評価:
  - direct MLP: 固定モデル確率の cross entropy
  - MLP-bin KT: MLP 確率binを adaptive universal context として使う
  - augmented KT: 既存 GDX context family と MLP 確率binを掛け合わせる

`sample_hilberts`, crop256, bits0-14, checker split:

- strict features:
  - GDX `0.7512 bits/bit`
  - direct MLP `0.7495`, gain `1.0023`
  - MLP-bin KT `0.7484`, gain `1.0037`
  - augmented KT `0.7497`, hybrid gain `1.0020`
- oracle/high features:
  - GDX `0.7512`
  - direct MLP `0.7497`, gain `1.0019`
  - MLP-bin KT `0.7483`, gain `1.0038`
  - augmented KT `0.7490`, hybrid gain `1.0029`

`sample_hilberts`, crop256, bits0-14, bottom split, strict features:

- GDX `0.7511`
- direct MLP `0.7497`, gain `1.0018`
- MLP-bin KT `0.7483`, gain `1.0037`
- augmented KT `0.7494`, hybrid gain `1.0023`

判断更新: held-out tile で学習モデルは GDX をわずかに超える。固定 route が負け続けた中で
これは本物の signal だが、gain は `0.2-0.4%` 程度。4x の突破口というより、
GDX context family の最後の上乗せ候補。次に続けるなら:

- MLP を C++ 化するのではなく、MLP 出力binから頻出splitを抽出して小さな固定 family
  に落とす。
- 8K/super-tile で model calibration cost を償却できるか測る。
- train/test を画像間に分け、hard image 固有の過学習でないか確認する。

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
