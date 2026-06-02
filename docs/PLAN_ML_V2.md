# Float32 HDR Lossless Codec V2: Learned Hierarchical Representation

## 1. Why the current ML prototype should stop here

The byte-LM prototype proved that a learned model can beat the classical
predictor on some HDR images, but it is not a viable product architecture:

- It receives a one-dimensional bitshuffled byte stream, so it cannot directly
  learn two-dimensional image structure.
- It uses autoregressive byte decoding, requiring one network inference step
  per decoded byte.
- The file-level ML-or-Blosc estimate is only `4.73x` geomean versus `4.63x`
  for Blosc alone. This does not justify the model cost.
- Lower mantissa planes in sky images remain high entropy after the existing
  MED-XOR transform. A stronger entropy coder alone cannot remove information
  that the representation fails to expose.

The current prototype remains useful as a correctness harness for fixed-point
probability quantization, ANS round trips, and float32 bit-exact verification.

## 2. New goal

The learned codec must provide a clear reason to exist:

| Metric | Go threshold | Stretch goal |
|---|---:|---:|
| Bit-exact lossless compression on real HDR hold-out | `>= 1.5x` smaller than Blosc | `>= 2.0x` smaller than Blosc |
| Decode speed on Apple Silicon | `>= 100 MB/s` | `>= 300 MB/s` |
| Network invocations per tile | `<= 8` | `<= 4` |
| Round trip | bit-exact float32 | bit-exact float32 |

The first question is feasibility, not implementation speed. If an oracle
hierarchical predictor cannot approach the compression target, the lossless
target must be revised before building a production codec.

## 3. Core design: coarse-to-fine HDR reconstruction

Do not model bytes in raster order. Encode a reversible hierarchy and let a
network predict missing samples from already decoded lower-resolution context.

```text
float32 RGB(A)
  -> reinterpret as structured fields: sign | exponent | mantissa
  -> reversible channel lifting transform
  -> spatial anchor pyramid
       level 0: sparse anchors
       level 1: predict horizontal/vertical midpoints
       level 2: predict centers
       level 3+: repeat until full resolution
  -> for each level:
       tiny CNN predicts conditional distributions in parallel
       rANS encodes exact correction symbols
  -> unpredictable tail planes:
       choose raw / zstd / Blosc per tile and plane group
```

This is lossless: the model predicts probabilities, not final pixel values.
Every correction symbol required to reconstruct the original float32 bits is
still encoded.

### 3.1 Structured float representation

Treat each float32 value as fields rather than four unrelated bytes:

```text
sign:      1 bit
exponent:  8 bits
mantissa: 23 bits
```

For each tile and channel:

1. Encode sign and exponent fields first.
2. Encode mantissa from most significant to least significant groups.
3. Condition later groups on reconstructed earlier groups.
4. Stop invoking ML when the measured gain becomes negligible; use the best
   classical fallback for the tail.

The lower mantissa tail is expected to be partly incompressible in noisy HDR
sources. Spending neural inference on it is wasteful.

### 3.2 Reversible channel lifting

Use an integer lifting transform over the uint32 field representation:

```text
G
R - predict_R(G)
B - predict_B(G, R)
A
```

The initial implementation should compare simple fixed lifting with a learned
small predictor. Each subtraction is modular integer arithmetic, preserving all
float32 bit patterns including NaN payloads, signed zero, and infinities.

### 3.3 Parallel spatial hierarchy

Use a checkerboard or interpolation pyramid:

```text
known anchors -> predict independent missing positions -> decode corrections
              -> newly decoded samples become context for the next pass
```

All symbols in one pass are independent given previous passes, so CNN inference
and ANS table generation are parallel across the tile. Avoid per-byte inference.

The first prototype should use `128x128` tiles and four to eight passes. The
tile size, number of levels, and field grouping are ablation parameters.

### 3.4 Learned side information

After the basic hierarchy works, test a compact hyperprior:

```text
tile -> small encoder -> quantized latent z
z -> entropy-coded side information
z + decoded coarse fields -> conditional distributions for fine fields
```

Keep `z` only when its byte cost is smaller than the corrections it saves.
This gives the model content awareness without sequential decoding.

### 3.5 Tile-level fallback

Each tile and field group records a small mode:

```text
ML hierarchy | Blosc bitshuffle+zstd | raw
```

The codec should never lose badly on noise, gradients, or unsupported content.
The mode map is also a diagnostic: it shows exactly where learned modeling adds
value.

## 4. Feasibility experiments before production work

### Experiment A: entropy ceiling audit

Train a deliberately oversized offline teacher. It may be slow, but its
conditioning must match information available to the real decoder.

Measure hold-out cross entropy by:

- field: sign, exponent, high/mid/low mantissa;
- hierarchy level;
- image class: indoor, landscape, sky, synthetic, noise;
- tile.

Go condition:

```text
oracle hybrid <= 0.67 * Blosc bytes
```

If the oversized teacher cannot save at least 33 percent, a production model is
unlikely to justify its cost. A `0.50 * Blosc` stretch result would validate the
two-times-smaller goal.

### Experiment B: hierarchical minimal model

Build a small PyTorch prototype:

- float field extraction;
- fixed reversible lifting;
- checkerboard pyramid;
- shallow CNN distribution heads;
- exact rANS round trip;
- per-tile mode selection.

Go condition:

```text
prototype hybrid <= 0.75 * Blosc bytes
network invocations <= 8 per tile
```

### Experiment C: deployment model

Only after Experiment B passes:

- quantize inference to integer arithmetic;
- cache ANS tables;
- convert the predictor to Core ML;
- measure CPU, GPU, and ANE where available;
- implement tile-parallel decoding.

## 5. Work that should not continue yet

Do not spend time optimizing the current autoregressive byte decoder. It is a
correctness prototype, not the production path.

Do not assume a larger byte-LM will reach the target. The representation and
decode schedule are the limiting factors.

Do not claim a fixed lossless ratio for arbitrary float32 inputs. Random lower
mantissa bits are real information and must be stored.

## 6. Immediate implementation sequence

1. Add `analyze_float_fields.py` to report entropy by float field, bit plane,
   tile, and hierarchy level.
2. Add reversible uint32 channel lifting experiments.
3. Add a checkerboard pyramid dataset generator.
4. Train the oversized teacher and produce the entropy ceiling report.
5. Decide Go/No-Go before implementing Core ML or C++ inference.

## 7. Current structural-codec findings

The `.pic`-like branch is now the more promising short-term direction than the
autoregressive byte model. It decomposes float32 values into fields, chooses a
local reversible predictor per tile/field, and codes the exact XOR residual.

The best pure entropy oracle is still optimistic because it assumes each tile's
bit probability is known for free. Explicit probability side information was too
expensive, and global probabilities lost the local structure. A decoder-
reproducible adaptive Bernoulli model is the best practical bridge so far:

```text
tile32 adaptive structural geomean: 4.10x
tile64 adaptive structural geomean: 4.20x
Blosc bitshuffle+zstd geomean:      4.63x
file-level hybrid geomean:          4.86x
```

This does not yet beat Blosc globally, but it beats Blosc on several real HDR
structure-heavy images and needs no probability side stream. The next useful
step is not a larger metadata MLP; it is a content-aware probability model that
uses decoder-available context, such as decoded exponent/high-mantissa fields,
coarse spatial anchors, and neighboring reconstructed samples.

A first decoder-context CNN probe now predicts structural residual probabilities
for later mantissa fields from already decoded float32 high bits plus the stored
predictor mode. With per-tile fallback to the adaptive model:

```text
tile64 adaptive structural geomean:       4.20x
mantissa_mid context hybrid geomean:      4.29x
mantissa_lo context hybrid geomean:       4.26x
mantissa_mid+lo context hybrid geomean:   4.35x
Blosc bitshuffle+zstd geomean:            4.63x
context-or-Blosc file hybrid geomean:     4.86x
```

This is a positive but not sufficient signal. The learned model helps puresky
and RGBA HDR cases, where lower mantissa fields retain structure, but it does
not yet move the file-level hybrid beyond the structural+Blosc baseline. The
next probe should condition on more decoder-available spatial context: decoded
neighbor residuals within a scan order or hierarchy pass, not only high float
field planes.

The stronger result came from that decoder-available context. A PIC-like
adaptive bit-plane model using west/north/north-west/north-east residual bits
plus three already decoded higher residual bits gives:

```text
tile128 spatial context, no previous residual bits: 5.10x
tile128 spatial context, previous_bits=1:          5.85x
tile128 spatial context, previous_bits=2:          5.97x
tile128 spatial context, previous_bits=3:          6.01x
tile128 spatial context, previous_bits=4:          6.01x
Blosc bitshuffle+zstd geomean:                     4.63x
```

At `tile128, previous_bits=3`, the structural model beats Blosc on almost every
test image and reaches a `6.02x` file-level hybrid geomean. This is the current
best lossless direction. The remaining gap to a 2x-smaller-than-Blosc goal is
mostly lower mantissa information in sky/noise-like content and high mantissa
residuals in detailed images. A practical codec should now prototype this
context model in an actual entropy coder before adding more neural machinery.

Decoder-safety matters. The `float_med` and `float_planar` predictor estimates
are attractive, but they need a carefully defined full-pixel decode schedule.
With only immediately decoder-safe bit predictors
(`zero`, `west`, `north`, `northwest`, `northeast`, `bit_xor_planar`,
`bit_majority`), the same context model gives:

```text
tile128 previous_bits=3 safe structural geomean:        4.97x
tile128 previous_bits=3 safe-or-Blosc hybrid geomean:   5.06x
```

A Python rANS roundtrip confirms that this safe subset is not just an entropy
estimate:

```text
ph_studio_small_03 128x128 crop:
estimate:      38,733 bytes (5.076x)
rANS payload:  38,683 bytes (5.083x)
roundtrip:     bit-exact

ph_belfast_sunset_puresky 128x128 crop:
estimate:      119,336 bytes (2.197x)
rANS payload:  119,403 bytes (2.195x)
roundtrip:     bit-exact
```

The C++ codec now has an internal adaptive binary rANS helper with tests. This
is the bridge needed for a production structural stage:

```text
test_rans_binary:
uniform     0.997x
sparse      4.998x
structured  1.986x
roundtrip   passed
```

The safe subset has now been integrated as a self-contained C++ pipeline stage
(`StageStructuralContext`). It excludes predictors that require unresolved
full-pixel scheduling, and it treats `northeast` as tile-local so decoding never
depends on a future tile. On the five real `ph_*.exr` hold-out files:

```text
ph_abandoned_tiled_room      4.82x
ph_belfast_sunset_puresky    2.21x
ph_kloppenheim_06_puresky    2.17x
ph_spruit_sunrise            6.17x
ph_studio_small_03           5.80x

C++ structural geomean:      3.83x
Blosc geomean on same files: 3.38x
```

This is slower than a production codec because mode search and context coding
are still scalar research code, but decode already works end-to-end through the
public Python/ctypes binding.

The structural stage now uses `PipelineConfig.effort` for mode search:

```text
effort < 7: zero / west / north
effort >= 7: zero / west / north / northwest / northeast / bit_xor_planar / bit_majority
```

On the five `ph_*.exr` hold-out files, the fast default preserved the measured
ratios while cutting encode time materially. For `ph_studio_small_03_1k`, the
full image roundtrip is now about:

```text
ratio: 5.80x vs Blosc 4.76x
encode: ~4.3s
decode: ~1.3s
```

The standard benchmark harness now includes `our_v5_structural_context`.

## 8. Toward 8x: ordered-float deltas

XOR residuals leave some smooth ULP-scale structure on the table. A new probe
maps float32 bit patterns into an order-preserving uint32 domain and tests
spatial deltas there. This is the first experiment that puts the 8x target in
range:

```text
ordered-delta field-wise upper probe, tile256 prev4:
  all 13 images geomean: 7.93x
  ph hold-out geomean:   4.64x

Blosc all 13 geomean:    4.63x
Blosc ph geomean:        3.38x
```

The float mapping is the standard sortable-float / FloatFlip transform:
negative float bit patterns are inverted, non-negative patterns have the sign
bit flipped. This makes nearby positive HDR float values nearby in unsigned
integer space while remaining reversible for every float32 bit pattern.

The initial catch was carry handling: a normal delta residual is a whole-word
value because addition carries across bit groups. Several variants were tested:

```text
whole-word delta, one delta mode per tile:       7.08x all-image geomean
field-local modular delta, per-field modes:      6.72x all-image geomean
signed zigzag whole-word delta:                  6.79x all-image geomean
carry-aware low-to-high fields:                  7.01x all-image geomean
```

The important correction is that field-wise whole residuals are reconstructable
without storing carry bits if the decoder reconstructs each pixel's fields from
low to high. For a field using predictor `p`, the incoming carry can be computed
from already reconstructed lower actual bits and `p`'s lower bits. A Python
roundtrip now verifies this representation:

```text
ph_studio_small_03_1k full image:
  field-wise ordered-delta representation roundtrip: bit-exact

ph_spruit_sunrise_1k 128x128 crop:
  field-wise ordered-delta representation roundtrip: bit-exact
```

A second correction: the old Python `shifted()` helper used in exploratory
predictors has the opposite sign convention from the mode names. The new
ordered-delta probes therefore have explicit causal predictors and ordered-XOR
fallbacks. With causal predictors and a single ordered representation:

```text
field-wise ordered-delta optimistic context, tile256 prev4:
  all 13 images geomean: 7.94x
  ph_studio_small_03:    7.44x
  ph_spruit_sunrise:     7.76x
```

This is still optimistic because it scores each field against the higher
residual bits of the same candidate predictor. The actual payload stream mixes
fields selected by different predictors, so those context bits change. When the
payload word is built first and then scored, the benefit drops:

```text
actual mixed payload context, tile256 prev4:
  all 13 images geomean: 7.08x
  ph_studio_small_03:    6.10x
  ph_spruit_sunrise:     6.45x

rANS actual payload, ph_studio 128x128 crop:
  4.98x, bit-exact
```

An alternate valid design stores explicit incoming-carry side bits so each field
can keep the candidate-specific higher residual context:

```text
side-carry ordered-delta, tile256 prev4:
  all 13 images geomean: 6.91x
  ph_studio_small_03:    6.83x
  ph_spruit_sunrise:     6.85x

side-carry ordered-delta, tile128 prev3:
  all 13 images geomean: 6.96x
  ph_studio_small_03:    7.19x
  ph_spruit_sunrise:     6.99x

side-carry ordered-delta, tile64 prev3:
  all 13 images geomean: 6.58x
  hybrid geomean:        6.62x
  ph_studio_small_03:    7.22x
  ph_spruit_sunrise:     7.00x

image-level payload/sidecarry/tile-family hybrid:
  all 13 images geomean: 7.27x
```

So the current map is now:

- `~8x` is visible as a representation upper bound.
- `~7.1x` is the current best all-image valid estimate for the simpler
  no-side-stream payload schedule, but it is only `~6.1-6.5x` on the two
  strongest real HDR examples.
- `~6.9x` is the best all-image side-carry estimate; smaller tiles lift the
  strongest real images above `7.1x`, but hurt gradients and simple synthetic
  cases through overhead.
- The missing piece is no longer basic carry correctness. It is context
  alignment: making the entropy context see candidate-specific higher residuals
  without paying too many carry side bits. The next best probes are stronger
  carry-bit coding and a production-shaped side-carry rANS roundtrip.

An exponent-aligned float-field residual was also tested: encode exponent
separately, then predict mantissa after scaling the predictor's significand into
the actual exponent's units. It is clean and reversible, but weaker than the
ordered-float path on the strongest real images:

```text
float exponent-aligned mantissa residual, tile256 prev4:
  ph_studio_small_03:    5.86x
  ph_spruit_sunrise:     6.39x
```

So the current best non-AI direction remains ordered-float deltas, not a return
to independent exponent/mantissa coding.

### 8.1 Context alignment breakthrough: grouped body delta

The strongest non-AI improvement so far is to stop choosing predictor modes
independently for every float field. Instead, bind the numerical body

```text
body = exponent | mantissa_hi | mantissa_mid | mantissa_lo
sign = separate
```

to one ordered-delta predictor per tile, while coding sign with XOR-like modes.
This keeps candidate-specific higher residual bits available as context across
the full exponent+mantissa body and avoids explicit carry side bits because the
body starts at bit 0.

Measured with causal predictors and ordered-XOR modes:

```text
grouped body ordered-delta, tile128 prev3:
  all 13 images geomean: 7.51x
  ph_abandoned_tiled_room: 6.18x
  ph_studio_small_03:      7.74x
  ph_spruit_sunrise:       7.84x

grouped exp_hi/tail split, tile128 prev3:
  all 13 images geomean: 7.20x
  ph_studio_small_03:      7.74x
  ph_spruit_sunrise:       7.84x

image-level grouped-body + payload + sidecarry hybrid:
  all 13 images geomean: 7.72x
```

This is a major improvement over the previous `7.27x` family hybrid. It also
explains the context-alignment failure more cleanly: too much field freedom
breaks useful residual context; too little freedom loses local adaptation. The
right constraint is to keep the numerical exponent+mantissa body together and
let only sign split away.

Representation roundtrips now pass:

```text
ph_studio_small_03_1k full image:
  grouped body representation roundtrip: bit-exact

ph_spruit_sunrise_1k full image:
  grouped body representation roundtrip: bit-exact
```

A first grouped-body rANS stream also roundtrips on 128x128 crops:

```text
ph_studio_small_03 128x128:
  rANS payload: 5.30x, bit-exact

ph_spruit_sunrise 128x128:
  rANS payload: 22.93x, bit-exact
```

The crop ratios are not directly comparable to full-image estimates, but they
validate the grouped payload decode schedule. The next production-shaped step is
to integrate the grouped body transform into the C++ structural stage and then
measure full-image rANS payloads with the same mode map and context schedule.

### 8.2 Corrected grouped-body result and C++ GDX2/GDX3 stage

The first `7.5x-7.8x` grouped-body estimate was too optimistic. The estimator
masked the stored body bits only after mode selection, so the cost model for
`body = bits 0..30` could accidentally use bit 31 from the full 32-bit residual
as a previous-bit context. That bit is not present in the actual body payload.
After masking before scoring, the Python estimate matches the real C++ rANS
payload almost exactly.

The codec now has a second self-contained C++ research stage:

```text
StageGroupedDelta / GDX3
  raw float32 bits
  -> FloatFlip ordered uint32
  -> body bits 0..30: grouped ordered-delta payload
  -> sign bit 31: XOR-like payload
  -> adaptive binary rANS with decoder-safe spatial + higher-bit contexts
```

`GDX3` uses `128x128` tiles and `previous_bits=4`. It also includes safe
body-only channel and edge predictors:

- `delta_channel_green`: decodes body channels in green-first order for tiles
  that choose that mode.
- `delta_select`: JPEG XL / WebP-style select predictor, evaluated on body
  bits only so the encoder and decoder agree before the sign bit is decoded.
- `delta_paeth`: PNG/JPEG XL-style Paeth predictor, also evaluated on body
  bits only.
- bitplane-level context family selection: each payload bitplane chooses one
  decoder-reproducible adaptive context family, packed at four bits per
  bitplane. This is the first MANIAC-like step: not a full learned/tree model,
  but a signaled choice among context shapes.

Full-image C++ roundtrip measurements after the packed-family and encoder-speed
updates:

```text
all 13 images, GDX3/GDX4 normal grouped_delta geomean: 7.317x
all 13 images, pre-GDX4 top-K mode refinement:         7.346x
Blosc bitshuffle+zstd geomean:                  4.633x

oexr_ScanLines_CandleGlass:        5.77x vs Blosc 4.37x
oexr_ScanLines_Cannon:             5.19x vs Blosc 3.85x
oexr_ScanLines_Tree:               4.04x vs Blosc 3.48x
oexr_TestImages_GrayRampsHorizontal: 487.99x vs Blosc 262.16x
ph_abandoned_tiled_room_1k:        5.97x vs Blosc 4.12x
ph_belfast_sunset_puresky_1k:      2.42x vs Blosc 2.12x
ph_kloppenheim_06_puresky_1k:      2.38x vs Blosc 2.10x
ph_spruit_sunrise_1k:              6.99x vs Blosc 5.05x
ph_studio_small_03_1k:             6.93x vs Blosc 4.76x
synth_rgba_hdr_1k:                 2.07x vs Blosc 1.99x
```

The Python estimator's selected-mode histogram over the 13 images includes
`body:delta_channel_green` on 48 tiles, `body:delta_select` on 19 tiles, and
`body:delta_paeth` on 3 tiles. So the new JPEG XL-style predictors are not
dominant, but they are genuinely selected in the production-shaped schedule.

Context-family probing before C++ integration showed:

```text
fixed context baseline estimate:       6.928x
family-only estimate:                  7.330x
tree/family hybrid on ph_studio only:  6.949x vs 6.409x baseline

C++ GDX2 packed-family actual:         7.302x
C++ GDX2 fast mode-search actual:      7.253x
C++ GDX2 fast + added hi-channel families: 7.293x
```

The first byte-per-family implementation worked but hurt very small payloads
(`GrayRamps` dropped to `422.79x`). Packing each family id recovered the loss;
the original eight-family format used three bits, and the current GDX3 format
uses four bits to make room for one additional parity/source-class-like family.

The original 3-bit family namespace was fully used. Moving the research
bitstream to GDX3 with four-bit family ids keeps the earlier useful high-context
families and adds one more tree-derived shape:

- `SpatialHiChannel`: current W/N plus channel id plus high-neighbor residual
  bits.
- `SpatialHiPc`: current spatial context plus previous-channel current bit plus
  high-neighbor residual bits.
- `PrevXYChannel`: previous two higher payload bits plus x/y parity and channel
  id. This came from the MANIAC/context-tree probe and specifically helps
  puresky/RGBA hard cases without requiring a full signaled tree.

The `PrevXYChannel` C++ integration is the first fixed-family extraction from
the MANIAC-style probe. Representative effort-9 changes:

```text
ph_belfast_sunset_puresky_1k:  2.37x -> 2.42x
ph_kloppenheim_06_puresky_1k:  2.33x -> 2.38x
synth_rgba_hdr_1k:             2.03x -> 2.07x
all 13 images geomean:         7.293x -> 7.317x
```

The first C++ integration used family-aware mode search for every candidate
predictor. That was too slow: representative `ph_*` images took about
`30-35s` to encode, and large OpenEXR images took up to about `56s`. The current
fast version chooses predictor modes using only the base context, then chooses
bitplane context families once for the final payload. This keeps most of the
compression gain. The encoder was then sped up without changing the bitstream
semantics:

- base-context mode scoring now counts all bitplanes in one tile pass instead
  of rescanning once per bit;
- context-family selection now counts all candidate families in one bitplane
  pass instead of rescanning once per family.

On an otherwise idle machine this moved representative 1K `ph_*` encode times
from about `5-7s` to roughly `2.5-3s`, while preserving the ratios.

A top-K refinement was tested in C++: choose modes with the fast base context,
then re-score only the top candidate modes with family-aware contexts. Before
the encoder-speed work it was too slow for the tiny gain:

```text
ph_studio fast:       6.88x, encode ~6-8s
ph_studio top-K=3:    6.92x, encode ~28s
ph_spruit fast:       6.95x, encode ~6-8s
ph_spruit top-K=3:    6.95x, encode ~28s
ph_abandoned fast:    5.88x, encode ~6-8s
ph_abandoned top-K=3: 5.89x, encode ~28s
```

After the speed work and the added families, top-K is more plausible as a
high-compression preset:

```text
effort 9, all 13 images:  7.317x
pre-GDX4 effort 11, all 13 images: 7.346x
```

It is still not the default because the gain is only about `0.4%` for several
times the encode work.

The MANIAC-style tree then pointed at a more implementation-shaped improvement:
split context-family selection by channel for selected body bitplanes. A Python
probe showed that this recovers most of the 8-leaf tree gain without storing a
full tree:

```text
ph full, current shared family estimate:   4.409x
ph full, channel-split hybrid estimate:    4.481x
all 13 estimate, shared family:            7.348x
all 13 estimate, channel-split hybrid:     7.430x
```

This has now been integrated as `GDX4`. Each bitplane stores either one shared
family id or a channel-split marker followed by one family id per channel. It is
enabled only for `effort >= 10` so the normal effort-9 preset keeps the previous
speed/ratio behavior.

```text
GDX4 effort 9, all 13 images:   7.317x  (unchanged normal preset)
GDX4 effort 10, all 13 images:  7.413x
GDX4 effort 11, all 13 images:  7.424x
GDX4 effort 10, ph geomean:     4.487x

representative effort-10 changes vs effort-9:
  CandleGlass:     5.77x -> 5.91x
  Cannon:          5.19x -> 5.29x
  Tree:            4.04x -> 4.11x
  ph_abandoned:    5.97x -> 6.20x
  ph_belfast:      2.42x -> 2.45x
  ph_kloppenheim:  2.38x -> 2.40x
  ph_spruit:       6.99x -> 7.12x
  ph_studio:       6.93x -> 7.01x
  synth_noise:     1.29x -> 1.29x
```

This is a small but real exact-lossless gain. It also validates the tree probe's
diagnosis: the useful next context axis is not simply "more spatial features",
but allowing different channels to use different context families on the same
bitplane.

The first C++ implementation was correct but slow because it rescanned the tile
for every `family x channel` pair during channel-split selection. The selector
now gathers shared-family and per-channel-family counts in one pass over the
tile/bitplane. This preserves the `7.413x` effort-10 ratio while moving the ph
full-image effort-10 encode times from roughly `20-33s` per image down to about
`7-9s` per image on the same machine. Full-corpus effort-10 encode times are now
roughly `5-24s` per image in this benchmark run, with the largest OpenEXR files
at the high end.

`GDX5` adds the next exact-lossless extraction: body predictor mode can now be
chosen per channel for high effort. Same-pixel channel dependencies are kept
decoder-safe by using a green-first order (`G, R, B, A`) and allowing only
dependency modes whose source channel is already reconstructed.

Probe result before integration:

```text
all 13 shared-mode estimate:        7.430x
all 13 channel-mode estimate:       7.516x
all 13 hybrid estimate:             7.524x
ph crop128 C++ effort 10:           5.946x
ph crop128 C++ effort 11 / GDX5:    6.006x
synth crop128 C++ effort 10/11:     4.315x  (no regression)
```

Measured full-corpus C++ result:

```text
GDX5 effort 11, all 13 images:      7.493x
Blosc all 13 images:                4.633x
```

The gain is modest (`GDX4 effort 11: 7.424x -> GDX5 effort 11: 7.493x`) but it
is bit-exact and confirms a useful pattern: the winning exact-lossless moves
are signaled structural choices with small side information, not generic bigger
contexts.

`GDX6` adds an exact half16 tile route for `effort >= 12`. If every float32
value in a tile is exactly representable as binary16, the encoder can code a
16-bit ordered-half residual instead of the normal 32-bit body/sign payload.
The decoder expands the reconstructed half bits back to the original float32
bits, so this remains exact lossless. This route is intentionally high-effort:
it is useful for half-like EXR sources, but it does not attack true/dithered
float tails.

Probe and C++ checks:

```text
half16 full-corpus probe:          7.348x -> 7.458x
oexr crop128 C++ effort 12:        15.983x geomean
oexr full C++ effort 12:           16.340x geomean
approx all-13 effort-12 geomean:   7.54x
approx target corpus without pure noise: 8.73x
ph crop128 effort 12:              6.049x
synth crop128 effort 12:           4.317x
```

The exact-lossless lesson is now sharper: half/bfloat-like sources keep giving
clean structural wins, while `synth_noise`, `synth_mixed`, `synth_rgba`, and the
puresky HDRs are dominated by real low-mantissa entropy.

Decision: keep `synth_noise` as a torture/stress test, not as part of the main
realistic target corpus. It is generated as pure Gaussian float32 noise and is
not representative of production HDR imagery. Do not use one blended geomean as
the headline number; the correct evaluation split is:

```text
realistic-core:  ph_* + oexr_ScanLines_*; no synth, no GrayRamps (~4.78x full)
realistic-no-puresky:
                 realistic-core without ph_*puresky*
puresky-hard:    ph_belfast + ph_kloppenheim puresky files
target-no-noise: all files except synth_noise (~8.73x full, includes easy cases)
easy:            synth_gradient + GrayRamps
synthetic:       synth_* files
noise-stress:    synth_noise only
```

The repeatable target benchmark is now the realistic core:

```text
pixi run bench-grouped-delta-target-cpp
pixi run bench-grouped-delta-target-no-puresky-cpp
pixi run bench-grouped-delta-puresky-cpp
pixi run bench-grouped-delta-no-noise-cpp
pixi run bench-grouped-delta-easy-cpp
pixi run bench-grouped-delta-synthetic-cpp
pixi run bench-grouped-delta-noise-stress-cpp
```

Puresky-specific follow-up after this split:

```text
scripts/probe_puresky_predictors.py
pixi run ml-probe-puresky-predictors
```

The first puresky-focused probes were negative but useful:

```text
puresky 256, body grouped estimate:          2.188x
puresky 256, exp_hi/tail split:              2.181x
puresky 256, hi_mid_lo_exp split:            2.181x
puresky 128, second-order smooth predictors: no tile selected them
puresky 256, tile64 channel-mode probe:      2.290x vs tile128 2.313x
feature-hash probe, puresky 256:             ~0.4-0.5% estimated gain
GDX7 HashAllXY C++ effort13, ph crop128:     no measured gain vs effort12
zero-mask raw tail route, puresky 256:       ~0.7% estimated gain
tile affine/quadratic surfaces:              no tile selected them
conditional tail entropy lower bound:
  puresky low15 GDX:                         ~11.29 bits/sample
  puresky low15 high-feature hash lower:     ~2.87-2.89 bits/sample
context-palette/adaptive dictionary route:   no gain once side cost is included
adaptive high-feature bit model:
  tile-reset and image-carried variants:     no gain; current low route selected
low-tail high-body surface residual:
  puresky crop256 low15:                     2.282x -> 2.290x
  puresky full low15:                        2.401x -> 2.407x
  ph crop128 low15:                          5.794x -> 5.803x
  progressive high-to-low surface:           about tied with one-shot surface
combined low-tail exact routes:
  puresky crop256 low15:                     2.282x -> 2.299x
  puresky full low15:                        2.401x -> 2.414x
  ph crop128 low15:                          5.794x -> 5.816x
decoder-shaped split roundtrip:
  puresky crop128 low15:                     2.299x -> 2.316x
  puresky full low15:                        2.401x -> 2.414x
  ph crop128 low15:                          5.794x -> 5.816x
```

Interpretation: for these two puresky files, ordinary smooth-gradient predictors
are not the missing route. The gradient is already mostly captured; the remaining
cost is dominated by low/mid mantissa tail bits that stay expensive even after
tile and body-field splits. A bounded hash context had a tiny estimated gain but
did not survive the C++ codec-shaped test. The exact-lossless path should next
test routes that use reconstructed high-value features for lower planes. The
first lower-bound result is tantalizing, but naive palette/dictionary side
information erases it, and straightforward adaptive hash models do not learn
fast enough. The positive exact signals are still small: zero-mask raw-tail
coding is the stronger one, while low-tail residual coding against a surface
derived from the already decoded high body is a weaker backup and mostly
overlaps the same structure. The surface route is not yet worth a C++ high/low
body split by itself, but it points to the right family: use high-body structure
to predict low-tail phase instead of treating low mantissa bits as independent
bitplanes. Otherwise the big puresky escape hatch is the separate near-lossless
tail option.

This exact split has now been implemented in C++ as `GDX8` for `effort >= 12`.
The body record can signal a low-tail split: high body bits stay in the normal
grouped-delta rANS payload, while the low `15` ordered-body bits are coded in a
separate zero-mask tail stream. Decode remains bit-exact because delta-mode high
bits can be reconstructed from the predictor high bits, encoded high residual,
and the borrow implied by the decoded low tail.

Initial C++ results:

```text
puresky-hard full, effort12:
  ph_belfast_puresky:      2.48x
  ph_kloppenheim_puresky:  2.42x
  geomean:                 2.450x

realistic-core crop128, effort12:
  geomean:                 5.893x

realistic-no-puresky crop128, effort12:
  geomean:                 8.010x

realistic-no-puresky full, effort12:
  geomean:                 5.984x
```

An encoded-stream budget audit was added:

```text
scripts/audit_gdx8_stream_budget.py
pixi run ml-audit-gdx8-stream-budget
```

It parses the self-contained `HDR0`/`GDX8` byte stream and reports top header,
GDX header, mode selectors, tail selectors, family selectors, main payload, and
tail payload bytes. The first full/crop audits changed the diagnosis:

```text
puresky-hard crop128, effort12:
  geomean:                 2.347x
  aggregate main payload:  17.9%
  aggregate tail payload:  82.1%

puresky-hard full, effort12:
  geomean:                 2.450x
  aggregate main payload:  34.4%
  aggregate tail payload:  65.6%

realistic-no-puresky crop128, effort12:
  geomean:                 8.009x
  aggregate main payload:  99.8%
  aggregate tail payload:  ~0.0%

realistic-no-puresky full, effort12:
  geomean:                 5.984x
  aggregate main payload:  99.8%
  aggregate tail payload:   0.1%
```

So there are now two distinct exact-lossless problems:

- `puresky-hard`: the low ordered-body tail dominates the actual compressed
  stream.
- `realistic-no-puresky`: the main grouped-delta payload dominates; low-tail
  work is not the next lever.

The puresky lower-tail audit is severe. For the two full puresky images, the
ordered low15 tail by channel is essentially full entropy in RGB and constant in
alpha:

```text
ph_belfast_sunset_puresky_1k:
  R/G/B low15 entropy:     ~14.87 bits/value
  A low15 entropy:          0.00 bits/value

ph_kloppenheim_06_puresky_1k:
  R/G/B low15 entropy:     ~14.87 bits/value
  A low15 entropy:          0.00 bits/value
```

That is about `11.15 bits/float sample` before any other image information is
stored. A `12x` exact-lossless target has only `2.67 bits/float sample` total
budget, so puresky cannot reach it unless the low tail is predicted from other
decoder-visible information. Follow-up attempts did not yet find that route:

```text
static transmitted tail probability table, crop128:
  no gain; best table route stayed slightly above current low coding

context palette/dictionary route, full puresky:
  no gain; all tiles kept `current_low`

raw ordered-tail bytes through zlib/bz2/lzma, full:
  lzma best at ~11.67 bits/value, still near the entropy floor

top16 Rice/Golomb residual route on bfloat-like crop128:
  worse than current GDX8 bitplane/context coding
```

`GDX9` adds four fixed context families inspired by the successful parts of the
MANIAC-like tree probe:

- `Prev4Channel`: previous four higher payload bits plus channel id;
- `WNPrev4Channel`: west/north current-bit context plus previous four higher
  bits plus channel id;
- `WNPrev4Pc`: west/north current-bit context plus previous-channel bit plus
  previous four higher bits;
- `SpatialXY`: spatial current-bit context plus x/y parity and two higher bits.

The Python fixed-family probe predicted modest but real gains, especially for
OpenEXR scanline samples:

```text
oexr crop128:
  fixed families:          5.339x -> 5.398x

ph crop128:
  fixed families:          5.794x -> 5.805x
```

Initial C++ `GDX9` crop128 result:

```text
realistic-no-puresky crop128, effort12:
  GDX8 geomean:            8.009x
  GDX9 geomean:            8.028x

realistic-no-puresky full, effort12:
  GDX8 geomean:            5.984x
  GDX9 geomean:            6.001x

puresky-hard crop128, effort12:
  unchanged geomean:       2.347x
```

`GDXA` keeps the `GDX9` context families and adds one exact half-source cleanup:
channel-mode split may now choose half-float predictor modes when every channel
in the tile is exactly representable as binary16. Mixed half/non-half splits are
rejected so the body remains a clean 16-bit half route and the sign record can
stay `XorZero`.

This is not a large ratio jump, but it fixes a real route-selection blind spot:

```text
realistic-no-puresky crop128, effort12:
  GDX9 geomean:            8.028x
  GDXA geomean:            8.030x

realistic-no-puresky full, effort12:
  GDX9 geomean:            6.001x
  GDXA geomean:            6.001x

GDXA crop128 stream audit:
  Cannon:                  half split selected, 1/1 tile
  ph_abandoned:            half split selected, 1/1 tile
  ph_studio:               half split selected, 1/1 tile
  puresky-hard geomean:    unchanged at 2.347x
```

`GDXB` adds a `ConstantZero` context family using the last spare 4-bit family
id (`14`; `15` remains channel split). If a record bitplane, or a split channel
bitplane, is entirely zero, the payload coder emits no rANS symbols for it and
the decoder leaves that plane at its initialized zero value. This is exact
lossless and mostly helps half-like/quantized sources where some high or masked
planes vanish.

```text
realistic-no-puresky crop128, effort12:
  GDXA geomean:            8.030x
  GDXB geomean:            8.043x
  aggregate ratio:         6.780x

puresky-hard crop128, effort12:
  unchanged geomean:       2.347x

realistic-no-puresky full, effort12:
  first GDXB full run:      6.004x
  after mode-search pruning:
    ph_spruit spot:        7.377x
```

An attempted variable tile split probe was added:

```text
scripts/probe_variable_tile_split.py
```

It estimates a per-128-tile choice between one `128x128` tile and four `64x64`
subtiles. On crop128 realistic inputs it was too weak to justify a C++ bitstream
change:

```text
oexr crop128:
  tile128 geomean:         5.605x
  dynamic geomean:         5.611x

ph crop128:
  tile128 geomean:         6.064x
  dynamic geomean:         6.064x
```

Additional probes after the correction:

```text
word/full-32-bit grouped delta:       about equal to body+sign, not a win
tile256:                              worse; loses local adaptation
tile64:                               helps some photos, hurts gradients badly
previous_bits=5/6:                    flat versus previous_bits=4
bit-index family pruning:             faster but lost compression, rejected
constant-bitplane family:             worse than `SpatialHiPc`, rejected
green color-difference predictor:      not selected well enough, rejected
extra W2/N2 bitplane contexts:        worse due context fragmentation
order-0 residual value entropy:       worse than bitplane context coding
simple second-order spatial modes:    not selected
```

The corrected conclusion is more conservative but stronger technically:
ordered-float grouped deltas are a real production-shaped improvement over
Blosc. The current non-AI path is now around `7.32x` geomean at normal effort
and roughly `7.5x` at very high effort, not yet `8x`.

The remaining cost is partly a real lossless floor. In the difficult puresky
RGBA images, many low mantissa residual bitplanes sit near `0.75` bits/sample:
the RGB channels are essentially random in those planes while alpha is constant.
In `synth_noise` and parts of `synth_mixed`, many low bitplanes are at
`~1.0` bits/sample. A predictor/context model cannot compress those bits much
without either a better numeric transform that makes the low mantissa less
random, or an explicitly near-lossless mode that discards/quantizes them.
Further lossless progress should therefore focus on changing the payload
representation for the hard low/mid mantissa planes, not merely adding ordinary
spatial contexts.

### 8.3 Literature pickup and next probes

The useful paper/codec ideas map cleanly onto the current grouped-delta design:

- JPEG XL modular / WebP / PNG predictors: Select and Paeth are simple,
  decoder-safe local predictors. These are integrated in `GDX2`; before context
  family selection they gave a small but real C++ improvement
  (`6.883x -> 6.910x` geomean).
- JPEG XL self-correcting weighted predictor: promising, but it needs
  per-pixel error feedback and tie-breaking that remains identical at encode
  and decode. Treat it as a separate predictor probe, not a quick patch.
- FLIF / MANIAC: the strongest lesson is adaptive context selection from
  decoder-visible properties. A signaled context-family version is now in C++
  and moves the actual fast geomean to `7.317x`; the first tree-derived fixed
  family is `PrevXYChannel`. The next version should test a compact tree or a
  representation change for the hard low/mid mantissa planes, not just more
  fixed context families.
- TDT typed data transformation: the codec is already doing a stronger
  type-aware split than byte shuffle by separating ordered-float body and sign.
  The next useful TDT-like probe is byte/bit clustering inside the residual
  body, especially around expensive bitplanes 16-20.
- Falcon-style adaptive sparse bit-plane coding: a simple transposed bitplane
  sparse-byte estimate was added to the grouped-delta estimator. It does not
  beat the current adaptive rANS context on `ph_studio`, `ph_spruit`,
  `GrayRamps`, or `synth_noise`; even all-zero-ish planes are already handled
  very cheaply by the KT/rANS path. Do not move this simple sparse-byte mode to
  C++ unless a stronger run/enumerative variant shows a clear win.
- FPC/Gorilla/fpzip family: value-delta, XOR, and leading-zero ideas are
  already represented by FloatFlip ordered deltas, XOR fallbacks, and bitplane
  coding. They are useful background, but less likely to give the next jump by
  themselves.

The next high-value non-AI experiment is therefore not another ordinary spatial
mode. It is either:

1. a compact MANIAC-like tree over decoder-visible residual properties for the
   expensive bitplanes, if it beats the fully-used 3-bit family set; or
2. a different reversible representation for low/mid mantissa residuals, since
   the current bitplanes show a real entropy floor on sky/noise-like images.

If either path gets close to `8x`, a learned model can then target only the
remaining hard tiles instead of carrying the whole codec.

### 8.4 Learned tail-probability probe

A first low/mid mantissa learned-probability probe now exists:

```text
scripts/probe_grouped_tail_mlp.py
pixi run ml-probe-grouped-tail
```

It trains a small MLP to predict exact grouped-delta payload bits. This remains
lossless in principle: the model only supplies probabilities for an entropy
coder; the actual bits are still coded exactly.

The probe compares two feature sets:

- `strict`: payload higher bits, same-bit spatial payload context, channel,
  mode, and local coordinates. These are compatible with the current GDX2
  payload decode schedule.
- `oracle`: `strict` plus actual ordered-float high body bits / exponent-like
  information. This is not available in the current payload-only schedule, but
  tests whether a progressive exponent-aware decoder would be worth designing.

Early smoke-test results with `128x128` crops and held-out eval images:

```text
synth_noise, bits 0..14:
  strict MLP: worse than tabular context
  oracle MLP: still worse than tabular context

ph_belfast_puresky, bits 0..14:
  strict/oracle MLP: worse than tabular context

ph_belfast_puresky, bits 15..20:
  tabular:    0.4549 bits/sample
  strict MLP: 0.4281 bits/sample  (~6.3% better)
  oracle MLP: 0.4280 bits/sample

ph_studio, bits 15..20:
  tabular:    1.0000 bits/sample
  strict MLP: 0.9660 bits/sample  (~3.5% better, early-stopped)
```

That first sample-level comparison was too optimistic because its tabular
baseline was weaker than the production GDX2 family model. A direct full-crop
comparison now checks:

- direct MLP probabilities;
- MLP probability bins used as adaptive KT contexts;
- existing GDX2 context families;
- a hybrid that chooses between existing family and MLP context per bitplane.

Results on `128x128` crops:

```text
ph_belfast_puresky, bits 15..20:
  existing family:       0.2635 bits/sample
  direct MLP:            0.4315 bits/sample
  MLP-bin KT context:    0.2960 bits/sample
  family/MLP hybrid:     0.2635 bits/sample

ph_studio, bits 15..20, held out:
  existing family:       0.7263 bits/sample
  direct MLP:            0.9648 bits/sample
  MLP-bin KT context:    0.8243 bits/sample
  family/MLP hybrid:     0.7261 bits/sample

ph_studio, bits 15..20, self-fit upper bound:
  existing family:       0.7263 bits/sample
  MLP-bin KT context:    0.7832 bits/sample
  family/MLP hybrid:     0.7209 bits/sample  (~0.8% local gain)
```

Interpretation: the useful learned signal exists, but the current small MLP is
not yet strong enough to replace GDX2's hand-built adaptive families. The best
codec-shaped use is not direct probabilities; it is an MLP-derived context bin
with adaptive counts. Even then, the current gain is tiny and mostly visible in
self-fit mode.

An additional float-tail structure audit was added:

```text
scripts/audit_float_tail_structure.py
pixi run ml-audit-float-tail
```

This gave a sharper explanation of the remaining floor:

```text
ph_studio full image:
  raw mantissa low15 entropy:      0.000 bits/bit
  grouped payload low15 entropy:   0.000 bits/bit
  raw low8 zero rate:              1.000

ph_belfast_puresky full image:
  raw mantissa low15 entropy:      0.850 bits/bit
  grouped payload low15 entropy:   0.853 bits/bit
  raw low8 zero rate:              0.254
  exponent 125/126 tail15 entropy: ~0.995 bits/bit

synth_noise crop:
  grouped payload low15 entropy:   0.995 bits/bit
  grouped payload bits15..20:      1.000 bits/bit
```

Many real HDR files are float32 containers holding half-like values; their low
15 mantissa bits are exactly zero and GDX2 already compresses them almost for
free. The hard puresky/noise cases are different: after splitting by exponent,
the RGB low tail remains nearly uniform. That strongly suggests real dither,
noise, or generated float precision, not merely a missing exponent law.

So the short-term AI conclusion is conservative: learned probabilities may help
around bits `15..20`, but they are unlikely to unlock `8x` on their own unless
the model becomes much stronger or the codec allows near-lossless treatment of
the truly random low tail.

MPS status: `torch.backends.mps.is_built()` is true, but
`torch.backends.mps.is_available()` is false in this Codex process. Metal is not
visible from the current sandbox, so serious training should run outside this
restricted process or in an environment where MPS devices are visible.

### 8.5 Tail route classification probe

A reversible route-classification probe now exists:

```text
scripts/probe_tail_class_routes.py
pixi run ml-probe-tail-routes
```

It classifies each tile's low mantissa tail into rough codec routes:

- `zero_tail`: raw low tail is all zero; candidate route is a fixed low-bit grid.
- `fixed_grid`: raw low tail has a shared number of trailing zero bits.
- `palette_tail`: low-tail values are few enough for a small dictionary.
- `plane_sparse`: individual payload bitplanes are mostly constant/sparse.
- `delta_structured`: grouped-delta payload tail is materially simpler than raw.
- `random_tail`: low and mid payload planes are both near entropy limit.
- `mixed_tail`: none of the above dominates.

The route costs are deliberately conservative enough to avoid mistaking a pure
entropy lower bound for an implementable route. `raw_tail_entropy` and
`payload_tail_entropy` are diagnostics only; the route chooser compares
`fixed_grid`, `plane_sparse`, `palette`, `payload_bitplane_kt`, and the current
`gdx2_tail`.

Full `1K` tile128 results:

```text
ph_abandoned:  zero_tail x32, fixed_grid route, gain 1.0008x on tail+mid
ph_spruit:     zero_tail x32, fixed_grid route, gain 1.0011x
ph_studio:     zero_tail x32, fixed_grid route, gain 1.0010x
ph_belfast:    mixed_tail x32, gdx2_tail route, gain ~1.0000x
ph_kloppenheim:mixed_tail x32, gdx2_tail route, gain ~1.0000x

synth_gradient: plane_sparse x21 + palette_tail x11, but gdx2_tail still wins
synth_mixed:    random_tail x12 + mixed_tail x20, plane_sparse route, gain 1.0013x
synth_noise:    random_tail x32, plane_sparse route, gain 1.0008x
synth_rgba:     mixed_tail x32, gdx2_tail route, gain ~1.0000x
```

Interpretation: block/tile classification is real, but low-tail route switching
alone is not an `8x` path. It explains source regimes and can eventually become
an encoder speed optimization or a tiny ratio cleanup. The high-value conclusion
is negative but useful: on puresky/noise-like data, the current GDX2 context
families are already close to the best reversible route found for the low tail.
For example, `ph_belfast` spends about `24.2M` bits on just the low 15 payload
planes (`~11.5` bits/sample), while an `8x` total target for the whole image is
only about `8.4M` bits. `synth_noise` is even harsher at `~15.0` low-tail
bits/sample. That makes bit-exact `8x` impossible on those images unless a new
representation finds conditional structure that these probes have not exposed.
For exact lossless `8x`, the next plausible target must either attack higher
structure outside the low tail, or explicitly accept a near-lossless/random-tail
discard mode as a separate product mode.

### 8.6 16x target audit

The aspirational target is now `16x`, which means only `2 bits` per float32
sample on average. A concrete budget audit was added:

```text
scripts/audit_16x_budget.py
pixi run ml-audit-16x-budget
```

Effort-9 full-corpus result with current GDX3:

```text
geomean: 7.317x
total bits above 16x target over 13 images: ~22.7 MiB

ph_belfast:      2.42x, low body 11.25 bits/sample
ph_kloppenheim:  2.38x, low body 11.27 bits/sample
synth_mixed:     1.61x, low body 15.03 bits/sample
synth_noise:     1.29x, low body 14.95 bits/sample
synth_rgba:      2.07x, low body 11.30 bits/sample
```

This reframes the problem: exact-lossless `16x` cannot be reached by ordinary
header cleanup or backend compression. The hard files spend many times the whole
`16x` budget in low body/mantissa bits alone. Continuing exact lossless means
searching for a new conditional structure in those bits, or building
impossibility certificates that justify a later near-lossless mode.

A bounded feature-hash context probe was also added:

```text
scripts/probe_feature_hash_context.py
pixi run ml-probe-feature-hash-context
```

Initial result:

```text
ph_belfast crop128: 2.353x -> 2.362x (~0.38% local gain)
synth_noise crop128: no gain
```

So a side-info-free fixed feature hash is weaker than the signaled tree result.
The next exact-lossless probe should either make the tree side information cheap
enough to use, or change the low-body representation itself.

### 8.7 12x recalibration: controlled low-tail quantization

The `16x` push was useful because it exposed the real bottleneck: not headers,
not generic backend compression, but low-body entropy in true/dithered float
tails plus upper-body cost in low-precision natural images. With the practical
stretch target relaxed to about `12x`, the correct split is:

- keep exact GDX3 as the main lossless track;
- measure a separate near-lossless mode that deliberately zeros low raw
  mantissa bits before the same GDX3 backend.

Two tools now cover this:

```text
scripts/estimate_mantissa_quantization.py
pixi run ml-estimate-mantissa-quantization

scripts/summarize_mantissa_target.py
pixi run ml-summarize-mantissa-target --target-ratio 12
```

Full 13-image effort-9 result:

```text
low00 exact:  7.317x geomean
low08:        8.650x geomean, max relative error about 3.04e-05
low12:       11.085x geomean, max relative error about 4.88e-04
low15:       12.540x geomean, max relative error about 3.89e-03
```

Target-summary result with the `reach-target` policy:

```text
exact geomean:          7.317x
selected geomean:      13.129x
exact total ratio:      3.317x
selected total ratio:   7.059x
images >= 12x:          4/13
remaining byte gap:     6480.3 KiB
```

The key split:

```text
ph_belfast_puresky: 2.42x -> 15.41x at low15, PSNR 84.4 dB
ph_kloppenheim:     2.38x -> 13.92x at low15, PSNR 88.4 dB

ph_abandoned:       unchanged at 5.97x, low bits already zero
ph_spruit:          unchanged at 6.99x, low bits already zero
ph_studio:          unchanged at 6.93x, low bits already zero

synth_noise:        1.29x -> 3.19x at low15, still not a 12x case
synth_mixed:        1.61x -> 6.62x at low15
synth_rgba:         2.07x -> 7.71x at low15
```

Interpretation: low-tail quantization is an excellent targeted product escape
hatch for puresky-like true-float tails, but it is not a universal 12x switch.
It does not help files whose low bits are already zero, and it cannot rescue
synthetic random data at this error level. The exact-lossless next step should
therefore attack upper-body representation, not just low-tail contexts:
reversible ordered-body block transforms, cheap signaled context trees, and
source-precision-aware mode pruning. If AI returns, it should be a tiny context
mixer for the residual hard classes after this source classifier, not a
full-image byte model.

### 8.8 Puresky exact-tail follow-up: fixed priors and adaptive dictionaries

After `GDXB`, puresky remains the clearest exact-lossless wall:

```text
GDXB puresky-hard crop128:
  geomean ratio: 2.347x
  aggregate sections: main=17.9%, tail=82.1%
```

The payload budget says the problem is not the visible smooth gradient. It is
the exact low ordered/mantissa tail. Several targeted probes now pin this down:

```text
puresky conditional entropy, crop128:
  low15 current GDX low tail: ~11.30 bits/sample
  best high-feature conditional entropy: ~2.9 bits/sample

tail adaptive high-context:
  no gain; sparse contexts reset/learn too slowly

tail image-adaptive high-context:
  no gain; bitwise adaptive counts still lose

low-tail surface residual:
  geomean 2.299x -> 2.316x at low15 estimate only

fixed decoder prior, leave-one-out:
  low15 geomean 2.299x -> 2.307x

fixed decoder prior, self-oracle:
  low15 geomean 2.299x -> 2.415x
```

A new reusable probe was added:

```text
scripts/probe_tail_fixed_prior.py
pixi run ml-probe-tail-fixed-prior
```

Interpretation: the low-tail information is highly image-local. If the decoder
is allowed an oracle table for the same image, whole-symbol contexts approach
the low conditional entropy; but cross-image fixed priors and causal adaptive
dictionaries do not get close enough. That makes an exact-lossless `12x` claim
for these puresky files unlikely without either a very cheap signaled model, a
new reversible representation of the tail values, or a separate near-lossless
tail option.

### 8.9 Fast exact tier: move half/tail split down to effort 11

The first effort-12 speed attempt, prefiltering channel modes before family
scoring, was rejected: it slightly hurt ratio and did not speed up the hard
cases. A better fast-tier change is to let effort 11 use the exact
source-precision routes that were previously effort-12-only:

- half-convertible body routes;
- low-tail split routes.

This keeps effort 12 as the maximum ratio path, but makes effort 11 much closer
without enabling the heaviest effort-12 channel-family refinement.

```text
realistic-no-puresky crop128:
  old effort10: 7.865x, ~0.6-0.8s/image
  old effort11: 7.955x, ~1.2-2.0s/image
  effort12:     8.043x, ~4.5-5.6s/image

new effort11:
  geomean:      8.015x
  enc times:    ~1.5-3.5s/image

puresky-hard crop128:
  new effort11: 2.345x, ~1.6s/image
  effort12:     2.347x
```

This is not a compression breakthrough, but it is a useful operational point:
effort 11 now captures most of effort 12's exact-lossless gain at much lower
encode time. The new convenience task is:

```text
pixi run bench-grouped-delta-fast-cpp
```

### 8.10 Dream-path audit: why the low-tail entropy gap is hard

The research path deliberately tested several exact-tail ideas that would have
been breakthrough-shaped if they worked. They did not:

```text
high-feature sorted tail sequence:
  low15 current ~11.30 bps
  sorted delta best ~13.33 bps

tail-only spatial/channel prediction:
  raw tail remains best

two-part context distribution table:
  low15 best ~12.43-12.47 bps including model cost

fixed-pattern two-stage training:
  low15 best ~12.15-12.21 bps

anchor/interpolation tail model:
  low15 best ~14.26 bps
```

The failure mode is now clear: conditional entropy is low only when the model
already knows a large image-local support. For low15 crop128, the useful
high-feature table needs about `49k` dictionary values for `65k` samples, and
the dictionary alone costs roughly `9.5` bits/sample. That erases the
`~2.9` bits/sample oracle entropy.

The remaining positive exact-lossless signal is context trees:

```text
puresky crop128:
  family geomean:      2.298x
  tree leaves16:       2.311x
  hybrid leaves16:     2.321x
  hybrid leaves32:     2.322x

ph_* crop128:
  family geomean:      5.792x
  hybrid leaves16:     5.880x
```

This is not enough for `12x`, but it is the most alive exact research thread.
The next dream experiment should focus on making signaled tree/model side
information cheaper, not on larger raw tail dictionaries.

### 8.11 Near-lossless implementation: low mantissa quantizer

Two final exact-lossless checks were run before implementing near-lossless:

```text
FPC/Gorilla-style XOR zero-region coding:
  weaker than current bitplane contexts on body/tail streams,
  except for zero-tail cases already handled by ConstantZero/tail split.

near-lossless base + exact correction stream:
  quantized base compresses well, but storing the discarded low bits exactly
  brings puresky back to roughly the original exact ratio.
```

So the useful branch is not "lossy base plus exact low-bit correction"; it is a
separate product mode that deliberately accepts bounded low-mantissa error.

The C++ codec now has `StageMantissaQuantize`:

```text
raw float32
  -> zero N low mantissa bits for finite values only
  -> grouped-delta backend
  -> decode returns the quantized pixels
```

The outer frame header is now version `2` and records `near_lossless_bits`.
Version `1` frames still decode with the default value. Python exposes:

```text
radiance_codec.encode_near_lossless(pixels, low_bits, effort=11)
radiance_codec.quantize_mantissa(pixels, low_bits)
pixi run bench-near-lossless-cpp
```

Crop128, effort11 results on the five real `ph_*_1k.exr` files:

```text
low00 exact:  6.076x
low08:        7.634x
low12:        9.314x
low15:       12.071x
```

On the two puresky-hard files specifically:

```text
low00 exact:  2.345x
low08:        4.151x
low12:        6.825x
low15:       13.051x
```

This is the clearest practical route to the requested `12x` class on puresky:
`low15` crosses it, while `low12` is a conservative quality point. It also
confirms the exact-lossless wall: once those low bits are stored as a correction
stream, the gain disappears. Exact research should therefore keep chasing cheap
context trees / upper-body representation changes, while near-lossless becomes a
separate selectable mode rather than a replacement for the lossless codec.
