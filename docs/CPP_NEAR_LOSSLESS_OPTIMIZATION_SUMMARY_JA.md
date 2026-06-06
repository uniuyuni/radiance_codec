# C++ Near-Lossless Router Optimization Summary

作成日: 2026-06-06
更新日: 2026-06-07

この資料は、near-lossless router を C++ codec stage に移した直後から、
現在の速度優先実装までの変更点と実測値を一本化するためのメモ。

主な根拠資料:

- `docs/NEAR_LOSSLESS_QUALITY_CRITERIA_JA.md`
- `results/near_lossless_router_v1_cpp_compact_benchmark.json`
- `results/near_lossless_router_v1_cpp_compact_visual_guard_benchmark.json`
- `results/near_lossless_router_v1_cpp_dscf_final_22m.json`
- `results/sample_near_lossless_router_v2_capacity_bench.json`
- `results/router_lowfreq_modes_dscf0009_metal_guided_r2_pool.json`
- `results/router_payload_positions_rans_dscf0009.json`
- `results/router_speed_guided_downsample_fused_dscf0009.json`
- `results/router_speed_guided_struct_r2_dscf0009.json`
- `results/router_speed_guided_struct_r2_light_snow.json`

## 現在の結論

現在の本命は `Stage.NEAR_LOSSLESS_ROUTER` / `encode_near_lossless_router_v1(...)`。
品質方針は、目視で許容された `Y8/CL8/H5 + signed-log anchor10 + dark refine G9 +
visual guard` 系を維持し、容量よりエンコード速度を優先している。

`sample_DSCF0009.EXR` の直近値:

| state | encoded | ratio | encode | decode | note |
|---|---:|---:|---:|---:|---|
| C++ compact 初期 | 39,621,623 B / 37.79 MiB | 12.06x | 19.07s | 2.65s | visual guardなし |
| internal visual guard | 31,039,702 B / 29.60 MiB | 15.39x | 36.46s | 2.63s | 品質本命化、ただし遅い |
| stream compaction | 23,204,463 B / 22.13 MiB | 20.59x | 125.02s | 5.57s | 最小サイズ寄り、探索/比較が重い |
| speed default bench | 24,038,281 B / 22.92 MiB | 19.88x | 3.13s | 1.04s | CPU fallback系の速度基準 |
| Metal all before payload opt | 23,731,469 B / 22.63 MiB | 20.13x | 3.31s | - | Metal guided/downsample/highpass/visual |
| current payload opt | 23,704,128 B / 22.61 MiB | 20.16x | 4.02s | - | 測定時に裏作業あり、容量は改善 |
| guided/downsample fused | 23,704,285 B / 22.61 MiB | 20.16x | 4.40s | - | guided と downsample を同一command化 |
| guided r2 structural | 23,704,278 B / 22.61 MiB | 20.16x | 2.79s | - | radius=2構造変更、dispatch削減 |

注意: encode/decode 時間は単発 wall time で、Metal 初期化、システム負荷、裏タスクの影響を受ける。
容量と stream method は安定値として扱える。

## 変更履歴

| commit | date | summary | 狙い |
|---|---|---|---|
| `8ccd870` | 2026-06-05 | Add near-lossless router stage | C++ compact bitstream stage 化 |
| `6f42549` | 2026-06-05 | Speed up near-lossless router | C++ encode の共通計算削減 |
| `618c6c9` | 2026-06-05 | Optimize near-lossless router decode path | decode 目標1秒級へ |
| `55c6795` | 2026-06-05 | Add experimental Metal router acceleration | guided/downsample/highpass/visual のGPU化 |
| `9849fed` | 2026-06-05 | Optimize experimental Metal router encode path | Metal経路の実用化 |
| `41fbf9a` | 2026-06-05 | Add dark refine stream | 暗部の黒浮き/階調救済 |
| `542bf18` | 2026-06-06 | Tune speed modes | 速度優先 default の整理 |
| `9c1f472` | 2026-06-06 | Make tiled router masks default | mask decode/encode を軽くする |
| `e13f374` | 2026-06-06 | Pack tiled router mask modes | tiled mask のmode stream縮小 |
| `e957e5c` | 2026-06-06 | Speed up encode | index/range/guard周辺のCPU高速化 |
| `46852b3` | 2026-06-06 | Reduce Metal transfer overhead | Metalバッファ転送を削減 |
| `512f4d4` | 2026-06-06 | Optimize Metal guided radius two path | guided r=2専用kernel |
| `244aba6` | 2026-06-06 | Speed up router payload encoding | sparse payload/list化、dark refine mask tiled化 |

## C++化直後の素の状態

最初の C++ stage は、Python研究で確立した VST/YCoCg router を codec pipeline に
載せることが主目的だった。

実装された payload:

- route mask
- high-pass mask
- VST/YCoCg の `Y` index
- guided chroma low `Co/Cg` index
- sparse chroma high `Co/Cg` index
- route領域の signed-log RGB index
- 追加チャンネルの constant/raw stream

初期の実測:

| image | raw | encoded | ratio | encode | decode |
|---|---:|---:|---:|---:|---:|
| `sample_1920×1280.exr` | 29.49 MB | 2.94 MB | 10.04x | 0.92s | 0.16s |
| `sample_DSCF0009.EXR` | 477.78 MB | 39.62 MB | 12.06x | 19.07s | 2.65s |
| `sample_bright_park.EXR` | 477.78 MB | 28.56 MB | 16.73x | 17.38s | 2.47s |
| `sample_middle_flower.EXR` | 477.78 MB | 20.21 MB | 23.64x | 16.68s | 2.20s |
| `sample_hilberts-mill-conference-room_2K.exr` | 33.55 MB | 1.70 MB | 19.69x | 0.85s | 0.14s |

この時点の問題:

- `sample_DSCF0009` の暗部と境界品質には visual guard が必要。
- encode はまだ Python研究経路より現実的だが、本番目標の2秒級には遠い。
- 汎用stream候補比較が多く、速度最適化の余地が大きい。

## Visual Guard と容量改善

品質確認で、DSCFの暗部階調、境界、ハイライトを守るには visual guard が必要になった。
内部 visual guard は、候補と安全版を表示空間 `white=4, gamma=2.2` で比較し、
表示輝度差 `L >= 0.010` の画素を route mask に追加する。

意外な結果として、DSCFでは visual guard によって容量も下がった。

| stream | guardなし | guard込み |
|---|---:|---:|
| route mask | 1.26 MB | 1.43 MB |
| high mask | 2.15 MB | 2.10 MB |
| Y | 8.10 MB | 8.07 MB |
| chroma low | 2.93 MB | 2.94 MB |
| chroma high | 2.35 MB | 2.14 MB |
| signed-log RGB | 22.83 MB | 14.35 MB |

解釈:

- guardで route 領域は増えるが、通常側のrange/高周波/escape分布が楽になった。
- signed-log RGB の量が大きく減り、DSCFでは `39.62 MB -> 31.04 MB` へ改善した。
- ただし basic候補生成とguard判定が二重気味になり、encodeは `19.07s -> 36.46s` と悪化した。

## Stream Compaction

品質を維持したまま、payload形式を詰めた段階。

採用:

- generic stream: raw / rANS order0 / rANS order1 / zstd の最小選択
- index stream: indexを直接 symbol rANS
- mask stream: west/north adaptive binary rANS
- chroma high: MED residualではなく selected raw index を symbol rANS

不採用:

- bitpack index + rANS: DSCFで改善なし、遅い
- fixed bit-position adaptive binary: DSCFで改善なし、遅い
- signed-log raw index fallback: DSCFで選ばれず、遅い

DSCFは `23,204,463 B / 22.13 MiB / 20.59x` まで到達した。
ただしこの時点の encode は候補比較が重く、`125s` の記録が残っている。

## Speed Priority 化

目標は「22MiB級を維持しつつ、エンコード最悪2秒級、少なくとも重い裏タスク込みで5秒級」。
ここからは容量最小より、品質固定・速度優先に寄せた。

主な変更:

- CPU guided estimation を標準で scale2 にする。
- 8x8 tiled route/high masks を default にする。
- tiled mask mode stream を2bit pack化する。
- display-space visual check に LUT を使う。
- Metal guided/downsample/high-pass/visual guard を追加。
- Metal側の入力/low bufferを再利用し、CPU/GPU転送を削減する。
- guided radius=2 専用kernelでbox filterをunrollする。

容量ベンチ代表値:

| image | encoded | ratio | encode | decode |
|---|---:|---:|---:|---:|
| `sample_1920×1280.exr` | 2.72 MiB | 10.34x | 0.443s | 0.102s |
| `sample_DSCF0009.EXR` | 22.92 MiB | 19.88x | 3.134s | 1.040s |
| `sample_bright_park.EXR` | 21.64 MiB | 21.06x | 2.947s | 1.153s |
| `sample_cat_noisy.EXR` | 54.09 MiB | 8.42x | 3.647s | 1.332s |
| `sample_hilberts-mill-conference-room_2K.exr` | 1.50 MiB | 21.35x | 0.407s | 0.084s |
| `sample_light_snow.EXR` | 10.79 MiB | 25.45x | 1.817s | 0.675s |
| `sample_middle_flower.EXR` | 17.65 MiB | 25.81x | 3.011s | 1.125s |
| `sample_night_city.EXR` | 29.55 MiB | 19.96x | 5.238s | 1.368s |

このベンチはCPU fallback系の注記があり、Metal実機値とは直接比較しない。

## 直近 Payload 最適化

`244aba6` で payload generation 側を改善した。

変更:

- `dark_refine_mask` を binary mask ではなく tiled mask へ固定。
- encode時の residual predictor は、ゼロ初期化した `decoded` 作業配列ではなく
  確定済みの `indices` から直接参照する。
- sparse stream (`route`, `high`, `dark_refine`) は全画素scanではなく positions list を使う。
- order1 rANS の中間 `payload` コピーを削減。
- payload stream別 trace を追加し、重いstreamを見える化。

DSCFの stream 内訳:

| stream | method | framed bytes |
|---|---|---:|
| route_mask | mask_tiled | 929,832 |
| high_mask | mask_tiled | 1,756,677 |
| y | rans1 | 7,649,281 |
| co_low | index_symbol_rans | 1,321,841 |
| cg_low | index_symbol_rans | 1,621,926 |
| co_high | index_symbol_rans | 563,672 |
| cg_high | index_symbol_rans | 667,432 |
| signed_r | index_symbol_rans | 3,252,032 |
| signed_g | index_symbol_rans | 2,611,256 |
| signed_b | index_symbol_rans | 2,924,140 |
| dark_refine_mask | mask_tiled | 166,565 |
| dark_refine_g | index_symbol_rans | 239,358 |

直近 trace の改善:

- `dark_refine_mask`: binary時は秒級に膨らむことがあったが、tiledで約 `98-128ms`。
- `dark_refine_g`: 全画素scanからpositions化し、約 `430ms -> 24ms`。
- `co_high/cg_high`: positions化で大きく短縮。
- payload全体はDSCF traceで約 `790ms -> 713ms`。

現在残る最大stream:

- `y`: order1 rANS、約7.65MB。速度と容量の綱引きが強い。
- `signed RGB`: 合計約8.79MB。容量の本丸だが品質影響も大きい。

## 2026-06-07 追加最適化

`sample_cat_noisy.exr` の確認で、元からノイズが全域に強い画像では、
通常の near-lossless router がノイズ除去風の模様を作りやすいことが分かった。
この画像は「きれいにする」対象ではなく、ソースのノイズも含めて守るべき対象なので、
品質救済と速度改善を分けて扱った。

### ノイズ画像の救済実験

実装に追加した実験用スイッチ:

- `RADIANCE_CODEC_ROUTER_FORCE_ROUTE_ALL=1`
- `RADIANCE_CODEC_ROUTER_ANCHOR_BITS=<1..16>`

`sample_cat_noisy.exr` では、全画素 route + anchor12 にすると見た目と数値は大きく改善した。
ただし容量は約 `119 MiB` 級まで増え、通常のnear-lossless routerとしては重すぎる。

採用したdefault挙動:

- route率が `95%` 以上になった場合だけ、anchor bits を最低 `12` へ上げる。
- 強制route allは環境変数のみで、defaultにはしない。

解釈:

- ノイズ全域画像は「圧縮しやすくする」より「壊さない」方が優先。
- ただし通常写真にまで anchor12 を広げると容量が厳しいので、route率で限定する。

### Dark Refine の早期skip

dark refine は暗部だけを見る処理だが、従来は明るい画素でも表示輝度計算まで進むことがあった。
そこで各RGBチャンネルから、明らかに `dark_refine_luma_max` を超える画素を早期にskipする。

効果:

- `sample_light_snow.exr` で dark refine が約 `230ms -> 60ms` 級まで短縮。
- encoded size は実質維持。
- 暗部候補だけを落とさない conservative なskipなので、品質リスクは低い。

### Visual Metal の source luma 化

`visual_guard_dilate_radius == 0` かつ `visual_guard_rgb_threshold <= 0` の場合、
Metal kernel は元RGB全体ではなく、CPU側で事前計算した source display luma だけを受け取る。

狙い:

- GPU側で元RGBから毎回 display luma を再計算しない。
- raw RGB転送を luma float buffer に縮める。

結果:

- DSCFで `encoded_bytes` は `+157 B` 程度、light_snowで `+5 B` 程度の差。
- 画質方針としては conservative。
- `visual-metal` はまだ1秒前後残ることがあり、次の大きな対象。

### Guided + Downsample の融合

最初の一段では、Metal guided low-pass と block mean downsample を同じ
`MTLCommandBuffer` / encoder sequence にまとめた。

結果:

| image | encoded | encode |
|---|---:|---:|
| `sample_DSCF0009.EXR` | 23,704,285 B / 22.606 MiB | 4.402s |
| `sample_light_snow.exr` | 11,484,076 B / 10.952 MiB | 1.642s |

これはCPU/GPU同期とbuffer往復を減らす変更で、数値構造はほぼ従来通り。

### Guided radius=2 の構造変更

さらに、defaultの `guide_radius=2` 専用で guided filter の構造を変えた。

従来の流れ:

- `prepare_guide`
- guide / guide^2 の horizontal + vertical box
- `prepare_planes_pair`
- plane / guide*plane の horizontal + vertical box
- `compute_ab_pair`
- a/b の horizontal + vertical box
- `reconstruct_low_pair`
- `block_mean_downsample_pair`

新しいradius=2専用の流れ:

- `guided_stats_h_pair_r2`
  - guide, guide^2, plane0, guide*plane0, plane1, guide*plane1 の横統計をまとめる。
- `guided_stats_v_ab_pair_r2`
  - 縦統計から a0/b0/a1/b1 を直接出す。
- `box_h_quad_r2`
  - a/b の横boxだけを行う。
- `box_v_reconstruct_low_pair_r2`
  - a/b の縦boxと low復元を同時に行う。
- `block_mean_downsample_pair`

dispatch数は概ね `11 -> 5`。
不要になった `guide_sq_buffer` 確保も radius=2 経路では省いた。
radiusが2以外の場合は従来経路にfallbackする。

最終計測:

| image | encoded | ratio | encode | note |
|---|---:|---:|---:|---|
| `sample_DSCF0009.EXR` | 23,704,278 B / 22.606 MiB | 20.16x | 2.785s | fused比で -7 B、かなり高速化 |
| `sample_light_snow.exr` | 11,484,085 B / 10.952 MiB | 25.08x | 1.57-1.64s | fused比で +9 B、速度は同等から微改善 |

DSCF trace の代表値:

- `guided-metal-down`: `601.493ms`
- encode total: `3507.491ms`

`sample_light_snow.exr` は最終版で encode/decode を確認済み。
decode結果は shape `(4000, 6000, 3)`、全値finite。

## 試したが戻したもの

NoCopy/Metal共有メモリ:

- CPU側でMetal shared bufferへ直接書く実験は、予想に反して遅くなった。
- GPUに都合の良いcopy経路の方が速いケースがあり、採用しなかった。

Metal guided の単純なkernel融合:

- `prepare_guide + prepare_planes_pair`、reflect最適化、horizontal box統合などを試した。
- 単純融合は速度悪化または改善が小さく、戻した。
- その後、radius=2専用の統計構造変更は採用した。

order0 byte stream:

- `RADIANCE_CODEC_ROUTER_RANS0_BYTE_STREAMS=1` は `y` stream をorder0へ変える。
- DSCFで約 `+0.40 MiB` 増え、速度改善も決定打ではないため default にはしない。

## Artifact Cleanup Plan

現状:

- `outputs`: 約15GB
- `data`: 約2.2GB
- `results`: 約13MB
- `docs`: 約452KB
- `outputs/previews`: 371 files
- `outputs/previews` の50MB超ファイル: 98 files
- `outputs/previews` の100MB超ファイル: 89 files

容量削減の主戦場は `outputs/previews`。
`results` は軽いので、削除よりも「重要結果だけ資料から参照」する運用でよい。

大きい preview directory:

| path | size | 判断 |
|---|---:|---|
| `outputs/previews/near_lossless_quality_audit` | 6.0GB | 監査用PNG。必要なら直近finalだけ残してarchive候補 |
| `outputs/previews/vst_chroma_dark_protect` | 4.4GB | 研究試行錯誤が多い。最終accepted以外はarchive候補 |
| `outputs/previews/near_lossless_router_gpu` | 1.2GB | 現行GPU確認用。残す価値あり |
| `outputs/previews/dither_breakthrough` | 776MB | bits8/9/dither研究。資料化後archive候補 |
| `outputs/previews/vst_chroma_nr` | 344MB | NR系研究。archive候補 |
| `outputs/previews/ycocg_*` | 約953MB | YCoCg比較。最終採用外はarchive候補 |

直近の未追跡一時結果:

- `results/router_payload_mask_default_light_snow.json`
- `results/router_payload_mask_packed_light_snow.json`
- `results/router_payload_mask_default_dscf0009.json`
- `results/router_payload_mask_packed_dscf0009.json`
- `results/router_payload_dark_refine_tiled_dscf0009.json`
- `results/router_payload_predictor_direct_light_snow.json`
- `results/router_payload_positions_light_snow.json`
- `results/router_payload_rans_copy_light_snow.json`
- `results/router_payload_positions_rans_dscf0009.json`

これらは今回の資料に要点を転記済みなので、コミット対象にはしない。
削除しても研究履歴の大筋は失われない。

推奨整理方針:

1. `outputs/previews/near_lossless_router_gpu` は残す。
2. `outputs/previews/near_lossless_quality_audit` は最新品質監査だけ残し、古いPNGはarchive。
3. `outputs/previews/vst_chroma_dark_protect` は `sample_DSCF0009_full_original` と
   現行candidate/guard比較だけ残し、古いthreshold sweepはarchive。
4. `outputs/previews/dither_breakthrough`, `vst_chroma_nr`, `ycocg_*` は
   この資料と既存docsから再現できるものとしてarchive候補。
5. root直下の古い `sample_DSCF0009_crop0_*` preview PNG は、現在のcodec評価では不要。

実削除はまだ行わない。削除するなら、先に `outputs/archive_20260606/` へ移動してから、
数日後に本削除するのが安全。

## 次の一段

ここからさらに進めるなら、優先順位は以下。

1. `y` stream の速度改善
   - order1 rANS の高速化、または `Y` 専用の2D予測 + symbol rANS化。
   - 容量増を避けるなら order0 fallback はまだ使わない。
2. `signed RGB` の容量/速度両面改善
   - route positions の局所性を使ったcontext分割。
   - ただし品質の要なので、容量だけを追いすぎない。
3. Visual guard のMetal/CPU境界をもう一段整理
   - `visual-metal` がまだ1秒前後出ることがある。
   - 既に大きく削った guided より、次は guard 判定/indices 周辺が効きやすい。
4. Cleanup
   - 古いpreviewをarchiveし、評価対象を `near_lossless_router_gpu` と
     最新auditに絞る。
