# Float32 HDR Lossless Research Reboot

## Position

Near-lossless can be a later product mode, but exact lossless should remain the
main research target for now. The important change is methodological: stop
asking "which single trick gets us to 8x?" and rebuild the map from evidence.

The current GDX3 direction is not a dead end. It already found several pieces
that agree with floating-point compression literature:

- floats must be transformed as typed numerical data, not as raw bytes;
- prediction residuals are more compressible after an order-preserving or XOR
  transform;
- bit/byte transposition and field grouping matter;
- lower mantissa tails in real HDR can be either trivial, structured, or close
  to random depending on source.

The unsolved part is the exact-lossless floor on puresky/noise-like images.

## Current Local Evidence

### What Works

GDX3 grouped ordered-delta is currently the strongest exact-lossless path:

```text
Current accepted C++ stage, effort 9:
  13-image geomean: 7.317x
  Blosc geomean:    4.63x

Representative hard cases:
  ph_belfast_puresky:   2.42x
  ph_kloppenheim:       2.38x
  synth_noise:          ~1.29x
  synth_mixed:          ~1.61x
```

The useful ingredients are:

- order-preserving float mapping;
- grouped exponent+mantissa body rather than independent fields;
- tile-local predictor choice;
- bitplane rANS with adaptive decoder-reproducible contexts;
- previous higher payload bits plus spatial/channel context;
- packed per-bitplane context-family selection, including the tree-derived
  `PrevXYChannel` family.

### What Did Not Move The Needle Enough

Small learned probability models did find some signal around bits `15..20`, but
they did not beat the existing GDX2 context families in a codec-shaped
comparison. Direct MLP probabilities were worse; MLP probability bins plus
adaptive counts were closer but still not a breakthrough.

Low-tail route classification also works diagnostically, but not as a ratio
path:

```text
zero-tail real images: fixed-grid route helps only about 0.08-0.11%
random/noise images:  plane-sparse route helps only about 0.08-0.13%
puresky images:       current gdx2_tail route still wins
```

### The Hard Floor

For exact lossless, every random-looking low mantissa bit must be stored. The
current audit found:

```text
ph_belfast full image:
  low 15 payload planes: ~24.2M bits
  low-tail cost:         ~11.5 bits/sample
  8x whole-image budget: ~8.4M bits

synth_noise full image:
  low 15 payload planes: ~23.5M bits
  low-tail cost:         ~15.0 bits/sample
```

This does not prove 8x is impossible, but it does prove that an 8x exact codec
must expose conditional structure not found by the current predictors, contexts,
MLP probe, exponent split, or low-tail route probe.

## Literature Signals To Incorporate

### OpenEXR

OpenEXR's technical documentation says most built-in compression methods are
lossless, but photographic images with significant grain typically shrink only
to 35-55% of uncompressed size. It also explicitly explains why PXR24 helps:
for 32-bit float data it rounds to 24 bits and removes the 8 least significant
bits, which are often noisy and difficult to compress.

Useful takeaway: the low float tail being painful is not surprising. Industry
practice often solves it by making that part lossy.

Source: https://openexr.com/en/latest/TechnicalIntroduction.html

### JPEG AI and High-Fidelity JPEG Activities

JPEG AI should be tracked, but it is not a direct solution for exact lossless
float32 HDR compression today. The published version 1 standard is explicitly
targeted at human visual consumption and machine consumption from a compact
learned compressed-domain representation. The normative core specifies the
codestream syntax and reconstruction process for decoded images, not bit-exact
preservation of an arbitrary input float32 buffer.

Key JPEG AI facts:

- ISO/IEC 6048-1:2025 / ITU-T T.840.1 is published.
- JPEG describes it as the first international image coding standard based on an
  end-to-end learning-based approach.
- The stated target is better subjective quality at the same rate, plus
  compressed-domain processing for machine vision.
- The first version has practical engineering features worth watching:
  tile-based random access, progressive decoding, conditional colour separation,
  multiple decoder complexity branches, HDR/wide-gamut metadata support, and a
  reference software project.
- The performance claim is about roughly 30% rate reduction at equivalent
  subjective quality versus strong conventional image coding standards, not
  bit-exact lossless ratio.

Useful takeaway for this project: JPEG AI validates learned latent image coding
as a deployable technology, but its v1 design is lossy/visual. It is more
relevant to a future near-lossless/random-tail-discard mode than to current
exact lossless. The exact-lossless lesson is architectural rather than directly
algorithmic: use tile independence, progressive latent ordering, low decoder
memory, and explicit profiles/complexity levels.

JPEG AIC is also relevant, but as an evaluation framework. It covers the range
from high quality up to mathematically lossless and includes nearly-lossless and
high-fidelity subjective/objective assessment work. This is useful if/when we
define a near-lossless float-tail mode, because it gives a disciplined way to
measure "visually indistinguishable" rather than relying only on PSNR or max
absolute error.

Sources:

- https://jpeg.org/jpegai/index.html
- https://jpeg.org/jpegai/workplan.html
- https://www.iso.org/standard/88911.html
- https://jpeg.org/jpegai/software.html
- https://jpeg.org/aic/index.html
- https://jpeg.org/aic/aic3.html

### Adjacent JPEG Standards

JPEG XL is the JPEG-family baseline most worth benchmarking for our exact
lossless matrix. It supports lossy, lossless, progressive coding, HDR/high bit
depth, alpha/extra channels, and lossless recompression of existing JPEG files.
It is not a float32 scientific-data codec, but its Modular lossless tools are
high quality and should be included as a competitor.

JPEG XR is more conceptually relevant than JPEG AI for HDR float containers: it
supports fixed-point and floating-point decoded numerical representations,
lossless/lossy coding, bit-exact decoder results, reversible colour transforms,
and a reversible hierarchical lifting-based lapped transform. However, its
official overview notes that 32-bit formats are supported using lossy
compression and that up to 24 bits are retained through transforms, so it should
not be assumed to solve exact float32 preservation.

JPEG 2000 and HTJ2K remain useful baselines. JPEG 2000 supports lossless coding,
up to 38 bits/sample, many components, HDR sample representations, ROI,
progressive decoding, and high-throughput variants. The wavelet/lifting family
is still relevant to our possible reversible ordered-float S-transform/Haar lane.

JPEG-LS is relevant as a predictor/context baseline for low-complexity lossless
and near-lossless continuous-tone images. Its Part 2 includes lossless
multi-component transforms and near-lossless tuning.

JPEG XS is mainly visually lossless/low latency. It supports up to 16-bit
components and lossless up to 12 bits/component, so it is not directly aligned
with exact float32 HDR, but its low-latency transform design is useful context
for a future fast near-lossless mode.

Sources:

- https://jpeg.org/jpegxl/index.html
- https://jpeg.org/jpegxr/index.html
- https://jpeg.org/jpeg2000/index.html
- https://jpeg.org/jpegls/index.html
- https://jpeg.org/jpegxs/index.html

### Learned Lossless / Near-Lossless Image Coding

Most learned image compression is lossy, but there is a separate learned
lossless thread:

- "Learning Better Lossless Compression Using Lossy Compression" stores a lossy
  reconstruction plus a learned exact residual. This is strongly relevant to a
  future exact codec if the "lossy base" is replaced by our reversible
  high-structure GDX2/base predictor and only the residual tail is modeled.
- PILC focuses on practical learned lossless compression and explicitly calls
  out the speed problem: many generative lossless codecs are far below practical
  throughput, while their target is 100+ MB/s. It uses a lightweight model plus
  an efficient entropy coder to reach practical GPU speed on 8-bit image data.
- Recent fitted neural lossless codecs show neural models can beat classical
  image codecs on some 8-bit datasets, but even current papers benchmark
  against JPEG XL and report significant CPU/GPU inference cost. These methods
  are not immediately applicable to bit-exact float32 tails, but they are useful
  as context-mixer designs.

Useful takeaway: for exact lossless HDR, learned coding should probably not be a
full-image autoencoder. The promising shape remains "strong deterministic
float-aware representation first, learned probability/context model only for the
residual classes where it beats GDX2."

Sources:

- https://arxiv.org/abs/2003.10184
- https://openaccess.thecvf.com/content/CVPR2022/papers/Kang_PILC_Practical_Image_Lossless_Compression_With_an_End-to-End_GPU_Oriented_CVPR_2022_paper.pdf
- https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_Fitted_Neural_Lossless_Image_Compression_CVPR_2025_paper.pdf

### FPZIP

FPZIP is a lossless/lossy multidimensional floating-point array compressor based
on predictive coding. It predicts floating-point values, converts residuals to
integers, and entropy-codes them.

Useful takeaway: our predictor/residual framing is aligned with established
float compressors, but we should compare against fpzip directly and borrow its
multidimensional prediction ideas where they fit images.

Source: https://computing.llnl.gov/projects/fpzip

### Scientific Float Compressors: The Main Exact-Lossless Mine

The closest non-JPEG family is scientific floating-point array compression, not
photographic image compression. These systems assume regular grids of float32 or
float64 values, exploit smoothness in one or more dimensions, and stay exact by
working on bit/integer representations rather than doing floating-point
subtraction.

Important mechanisms to port or test:

- `Lorenzo` / multidimensional predictors: predict a sample from corners of the
  local grid. In 2D, this is the familiar `west + north - northwest` structure,
  but scientific compressors generalize it cleanly and evaluate it as a primary
  transform.
- Integer-domain residuals: lossless compressors avoid FP subtraction because it
  is not bijective. They use XOR, integer subtraction, sign-aware mappings, or
  sortable/rotated float bit representations.
- Residual bit-width coding: good predictions create leading redundant bits.
  Many compressors encode the number of leading zeros/sign bits and then copy the
  remaining tail.
- Bit-matrix transposition + zero-word elimination: instead of coding each value
  independently, transpose residual bits over a block and remove all-zero words
  with masks.
- Block independence: fixed-size blocks expose parallelism and bound working
  memory, at the cost of some context loss.

The `ndzip` paper is especially relevant because its pipeline is almost a
checklist for the next GDX2 probe:

```text
block subdivision
-> reversible integer mapping of float bits
-> integer Lorenzo transform
-> sign-bit/sign-magnitude handling
-> bit matrix transposition
-> zero-word elimination
-> packed nonzero words
```

Two details are worth testing immediately:

1. ndzip reports that a simple left rotation of the float representation, moving
   the sign bit to the least-significant position, worked best for their data.
   Our ordered-float mapping helped images, but sign-rotation should be included
   in the representation search.
2. Their residual coder is not a per-bit adaptive context model. It is a block
   bit-matrix compactor. This could beat GDX2 on easy/medium blocks and, more
   importantly, can fail cheaply on random tails.

Sources:

- https://dps.uibk.ac.at/~fabian/publications/2021-ndzip-a-high-throughput-parallel-lossless-compressor-for-scientific-data.pdf
- https://sc21.supercomputing.org/proceedings/tech_paper/tech_paper_pages/pap595.html

### FPC

FPC predicts a value, XORs the prediction with the true IEEE float, and encodes
the result using leading-zero structure. The paper emphasizes recurring
difference patterns rather than only small consecutive differences.

Useful takeaway: test context-history predictors, not only local spatial
predictors. For image tiles, this may mean repeated residual-pattern dictionaries
conditioned on exponent/high mantissa/channel, not LZ over bytes.

Source: https://userweb.cs.txstate.edu/~mb92/papers/dcc06.pdf

### Time-Series Float Compression

Time-series float compressors are 1D, but they have useful bit-level ideas for
image scanlines, channel streams, and residual-history modeling.

Relevant families:

- `Gorilla`: XOR current value with previous value, then encode leading/trailing
  zero regions and the meaningful bits.
- `Chimp` / `Chimp128`: improves the Gorilla trade-off for floating-point time
  series using better reuse of prior XOR structure.
- `Elf` / `Elf+`: "erasing" transforms set some low bits to zero in an
  analytically reversible way, making XOR values expose trailing zeros.
- `ALP` in DuckDB: adaptively detects decimal-origin floats and losslessly
  encodes them as integers; otherwise it compresses front bits vectorized.

Why this matters for HDR float32:

- Some EXR values may be decimal/export-origin or half-like, not genuine
  measured float32. ALP-style source classification could route those tiles to a
  decimal/integer representation.
- Gorilla/Chimp-style XOR metadata may be useful for ordered scanlines where
  GDX2's per-bit context overhead is too high.
- Elf-style erasing is conceptually close to detecting reversible "already
  implied" trailing bits. It is not the same as near-lossless bit dropping; the
  published idea is exact recovery under conditions derived from float format
  properties.

Sources:

- https://www.vldb.org/pvldb/vol8/p1816-teller.pdf
- https://www.vldb.org/pvldb/vol15/p3058-liakos.pdf
- https://arxiv.org/abs/2306.16053
- https://duckdb.org/library/alp/

### ZFP Reversible Mode

ZFP's reversible mode guarantees bit-for-bit reconstruction for integer and
floating-point data, including IEEE special values, using block-independent
compression.

Useful takeaway: block transforms are worth revisiting, but only if they are
fully reversible in integer/bit space. A reversible S-transform/Haar-like pass
over ordered-float integers may expose structure that scalar deltas miss.

Source: https://zfp.readthedocs.io/en/release1.0.0/modes.html

### NDZip / Recent CPU-GPU Float Compressors

Recent lossless scientific-data compressors use high-throughput transformations
such as Lorenzo coding, difference coding, byte shuffling, and bit
transposition. A 2025 ASPLOS paper summarizes related systems: FPZIP, ZFP, SPDP,
GFC, MPC, nvCOMP Bitcomp/ANS, and NDZip.

Useful takeaway: add a "scientific compressor replication" lane. The current
codec already uses some of these ideas, but not all combinations:

- Lorenzo predictors in 2D image order;
- bit transposition after residual coding;
- byte-position grouping / typed byte streams;
- zero maps plus packed nonzero residual tails;
- CPU/GPU-friendly block independence.

Sources:

- https://sigport.org/documents/ndzip-high-throughput-parallel-lossless-compressor-scientific-data
- https://userweb.cs.txstate.edu/~burtscher/papers/asplos25.pdf

### HDF5 / Blosc / Bitshuffle Ecosystem

The HDF5 compression ecosystem is useful because it treats compression as a
filter pipeline: chunking, shuffling/bitshuffling, optional n-bit/scale-offset
filters, then a backend compressor such as LZ4, Zstd, Blosc, or ZFP.

Useful takeaway: our codec should keep representation transforms separable from
the entropy backend. A route like:

```text
GDX2 residuals -> bit transpose -> zero map -> zstd/lz4/rANS
```

should be benchmarked as a first-class competitor to the current bitplane rANS
model. Bitshuffle is not magic; it simply exposes common bit positions across
typed values. But that is exactly the kind of redundancy float32 fields often
have.

Sources:

- https://support.hdfgroup.org/documentation/hdf5-docs/hdf5_topics/UsingCompressionInHDF5.html
- https://docs.hdfgroup.org/archive/support/services/filters.html
- https://blosc.org/c-blosc2/reference/blosc1.html
- https://arxiv.org/abs/1503.00638

### FLIF / MANIAC and Non-JPEG Lossless Image Codecs

FLIF is discontinued as a deployed format, but its MANIAC entropy model is still
one of the best conceptual matches for our current context problem. It uses
adaptive decision trees over local context properties, not just adaptive
probabilities inside fixed contexts.

Useful takeaway: GDX2 has exhausted its tiny fixed context-family namespace.
Instead of adding one more hand-coded family, the next exact-lossless context
experiment should look like a bounded MANIAC/PAQ-style tree over
decoder-visible features:

- bit index;
- previous higher payload bits;
- west/north/northwest/northeast bits;
- channel;
- tile source class;
- predictor mode;
- local coordinates;
- exponent/high-body buckets when decoder-visible.

WebP lossless, PNG, and FFV1 are also worth mining for simple robust ideas:
block predictors, color transforms, small color caches, median/gradient
predictors, and range coding. Their sample depth limits are not ideal for
float32 HDR, but the predictor/context patterns are battle-tested.

Sources:

- https://flif.info/spec.html
- https://developers.google.com/speed/webp/docs/webp_lossless_bitstream_specification
- https://www.libpng.org/pub/png/spec/1.2/PNG-Filters.html
- https://www.ffmpeg.org/~michael/ffv1.html

### RAW / CFA / Hyperspectral Predictive Compression

Another non-JPEG vein is sensor and hyperspectral compression. These codecs care
about high bit depth, spatial/spectral correlation, hardware simplicity, and
lossless or near-lossless guarantees.

Important signals:

- CCSDS-123 is a low-complexity lossless/near-lossless standard for multispectral
  and hyperspectral imagery. It uses a configurable predictive preprocessor for
  spatial/spectral decorrelation followed by entropy coding.
- Recent deep hyperspectral work avoids full autoencoders and instead uses a
  predictive neural network plus entropy-coded residuals, even beating
  CCSDS-123 in reported lossless/near-lossless experiments.
- RAWIC, a 2026 learned raw-image compressor, explicitly conditions on patch bit
  depth and reports an average `7.7%` bitrate reduction over JPEG XL for Bayer
  raw images.

Useful takeaway: source-bit-depth classification is not just a heuristic; it is
showing up in recent learned raw compression. For EXR float32, a parallel idea
is per-tile "effective precision" or "source precision" conditioning:

```text
half-like / PXR24-like / decimal-origin / true-float / dithered-tail
```

This should feed both route selection and learned context models.

Sources:

- https://www.mdpi.com/2072-4292/11/11/1390
- https://ntrs.nasa.gov/citations/20240012764
- https://arxiv.org/abs/2403.17677
- https://arxiv.org/abs/2603.28105

### Content-Adaptive Learned Lossless

Beyond JPEG AI, learned lossless image work is moving toward content-adaptive
models and test-time specialization. CALLIC, for example, fine-tunes low-rank
incremental weights during encoding using an MDL/rate-guided objective.

Useful takeaway: if AI returns to this project, the best fit may not be a global
model trained once. It may be a tiny per-image/per-tile adaptation whose update
cost is explicitly charged as side information. That is expensive, but it gives
a clean test: if adaptation cannot beat GDX2 after paying its model delta, it is
not worth deploying.

Source: https://arxiv.org/abs/2412.17464

### Lossless Float Preprocessing

A 2023 paper studies invertible floating-point preprocessing and reports
compression improvements up to 40% on some real datasets. Its operations are
aimed at increasing shared bits while preserving exact recoverability.

Useful takeaway: although the paper is 1D and mostly double/time-series
oriented, the idea of storing side metadata to move values into more
compressible regions is worth testing per tile/exponent bucket. Do it on integer
representations first to avoid relying on FP arithmetic.

Source: https://arxiv.org/abs/2308.03623

### Typed Data Transformation

A 2025 arXiv paper proposes grouping related bytes of floating-point values
before generic compression, reporting a 1.16x geometric mean improvement over
state-of-the-art generic tools such as zstd and throughput gains.

Useful takeaway: this is close to "bitshuffle, but typed/contextual." We should
test byte/bit stream layouts after GDX2 residuals, not only before prediction.

Source: https://arxiv.org/abs/2506.18062

## New Information-Gathering Plan

### Lane A: External Baseline Matrix

Run the same 13-image corpus through float-aware and EXR-aware baselines:

- OpenEXR ZIP, PIZ, HTJ2K if available;
- fpzip lossless;
- zfp reversible;
- zstd/lz4/xz on byte shuffle and bit shuffle variants;
- current GDX2 effort 9/11.

Record ratio, encode time, decode time, and whether the codec is exact for every
float32 bit pattern.

### Lane B: Source-Precision Forensics

Classify each tile/channel by likely source:

- true float32;
- half-like low 13/15 zero;
- PXR24-like / 24-bit float;
- decimal/text-export quantized;
- dithered/noise tail;
- generated smooth math tail;
- alpha/constant/sparse channels.

This should produce a source map before choosing compression modes. The aim is
not just better ratio; it prevents wasting search on impossible tails.

### Lane C: Representation Search

Try reversible transforms not yet fully explored:

- 2D Lorenzo residuals in ordered-float body space;
- reversible integer S-transform / Haar over ordered body values;
- Hilbert/Z-order/tile traversal changes before residual coding;
- predictor-history dictionary similar to FPC;
- residual byte/bit typed stream layouts after GDX2;
- zero/nonzero maps plus packed nonzero residual bytes.

Use oracle-ish scoring first, then only port winners to C++.

### Lane D: Stronger Adaptive Contexts

The fixed 3-bit family namespace is exhausted. Next context work should be a
proper adaptive mixer/tree, not another hand-added family:

- MANIAC/PAQ-style context tree over decoder-visible features;
- online feature hashing with bounded tables;
- per-bitplane context pruning using held-out tile cost;
- separate models for source classes from Lane B.

The go condition is strict: it must beat current GDX2 on hard puresky/noise
tiles, not only on easy zero-tail images.

### Lane E: Exact-Lossless Impossibility Certificates

For each hard tile, estimate lower bounds under progressively richer
conditioning:

1. raw tail entropy;
2. exponent/channel/tile class;
3. spatial neighbors;
4. GDX2 higher payload bits;
5. predictor-history contexts;
6. learned oracle features.

If the bound remains above the target budget, mark the tile as exact-lossless
hard. This gives a principled bridge to a later near-lossless mode without
mixing product goals.

## Immediate Next Steps

1. Build the external baseline matrix.
2. Add source-precision forensics for each tile/channel.
3. Add one or two representation probes from Lane C, starting with 2D Lorenzo
   and residual typed-byte layouts because they are closest to published
   successful float compressors.
4. Only after those, revisit AI as a context mixer for the remaining hard tiles.

## First Non-JPEG Integration Probe

Two probes now exist:

```text
scripts/probe_nonjpeg_routes.py
pixi run ml-probe-nonjpeg-routes

scripts/audit_source_precision.py
pixi run ml-audit-source-precision
```

`probe_nonjpeg_routes.py` keeps the current GDX2 grouped ordered-delta payload
and tests non-JPEG/scientific-data entropy routes on top:

- raw fallback;
- ndzip-like bit-matrix zero-word elimination;
- bit-matrix constant-word elimination;
- centered zigzag residuals before bit-matrix coding;
- typed byte-plane entropy lower bounds;
- optional actual byte-plane `zstd` route.

It also has `--search-candidates` for crop-sized checks where predictor/mode
selection is redone with the new routes in the loop.

Initial result: on the current GDX2 payload, these routes do not improve exact
lossless ratio. Representative full `1K`, tile128, prev4, word8 results:

```text
ph_belfast_puresky:
  GDX2:              28.28M bits  (2.373x)
  byte-plane zstd:   35.05M bits
  zigzag zstd:       32.49M bits
  zigzag bitmatrix:  37.09M bits
  selected routes:   GDX2 for all 32 body tiles

ph_studio:
  GDX2:              7.26M bits   (6.930x)
  byte-plane zstd:   10.26M bits
  zigzag zstd:       11.79M bits
  selected routes:   GDX2 for all 32 body tiles

synth_noise:
  GDX2:              39.04M bits  (1.289x)
  byte-plane zstd:   41.28M bits
  zigzag zstd:       41.66M bits
  zigzag entropy LB: 38.91M bits
  selected routes:   GDX2 for all 32 body tiles
```

The only tempting number is `synth_noise`'s zigzag byte entropy lower bound,
which is about `0.3%` below GDX2, but actual zstd does not realize it and the
gain is too small to justify a production route. This strongly suggests that a
simple ndzip-style post-entropy route is not the missing `8x` mechanism once
GDX2 contexts are already active.

`--search-candidates` was also run on `128x128` crops for ph and synthetic
images. Even when predictor/mode selection is redone with bit-matrix routes
available, GDX2 remains selected for all tested crops.

The source-precision audit gave a more useful signal:

```text
ph_abandoned:    all tiles bfloat_or_coarser, low16 zero, half-exact rate 1.000
ph_spruit:       all tiles bfloat_or_coarser, low16 zero, half-exact rate 1.000
ph_studio:       all tiles bfloat_or_coarser, low16 zero, half-exact rate 1.000
ph_belfast:      all tiles true_or_dithered_float, half-exact rate 0.250
ph_kloppenheim:  all tiles true_or_dithered_float, half-exact rate 0.250
synth_noise:     all tiles true_or_dithered_float, best decimal rate 0.761
```

Interpretation:

- Some real HDR files are effectively low-precision float containers. GDX2 is
  already exploiting this well; a source-precision route may help speed and mode
  pruning more than ratio.
- The hard puresky images are not just missing a fixed-grid/decimal route. RGB
  behaves like true or dithered float; the `0.250` half-exact rate is likely the
  alpha/constant channel.
- The next non-JPEG candidate should not be a post-GDX2 byte/bit compactor. It
  should be a different predictor/context family that exposes new conditional
  structure before the entropy stage, especially FPC/Gorilla-style residual
  history or a bounded MANIAC-style adaptive context tree.

### MANIAC-style context tree check

The existing MANIAC-like context tree probe was rerun with `max_leaves=8`:

```text
scripts/evaluate_grouped_context_tree.py
pixi run ml-probe-context-tree -- --max-leaves 8
```

This is still a slow Python oracle-ish probe, but unlike the post-GDX2
bit-matrix routes it improves the hard puresky images:

```text
ph_abandoned:   family 5.88x -> hybrid 6.12x
ph_belfast:     family 2.37x -> hybrid 2.44x
ph_kloppenheim: family 2.32x -> hybrid 2.39x
ph_spruit:      family 6.95x -> hybrid 7.11x
ph_studio:      family 6.88x -> hybrid 6.95x
ph geomean:     family 4.344x -> hybrid 4.461x

synth_gradient: family 422.33x -> hybrid 426.65x
synth_mixed:    family 1.61x   -> hybrid 1.61x
synth_noise:    family 1.29x   -> hybrid 1.29x
synth_rgba:     family 2.02x   -> hybrid 2.08x
synth geomean:  family 6.489x -> hybrid 6.558x
```

Interpretation: the tree is the first non-JPEG idea in this pass that actually
moves puresky in the right direction. It is not enough for `8x`, and the current
Python search is far too slow, but it identifies a real direction: bounded
adaptive context construction over decoder-visible features. The next useful
implementation work is to make a production-shaped, cheap version of this tree
or feature-hash mixer and test it inside the GDX2 C++ stage.

### First C++ extraction from the tree

The cheapest tree-derived fixed family was extracted and integrated into the C++
stage:

```text
PrevXYChannel =
  previous payload bit + next previous payload bit
  + x parity + y parity
  + channel id flags
```

This required bumping the internal grouped-delta stream from `GDX2` to `GDX3`
because the eight 3-bit family ids were exhausted. Family ids are now packed in
four bits.

Before C++ integration, `probe_context_family_variants.py` predicted stable
full-image gains:

```text
ph_belfast:      2.373x -> 2.424x
ph_kloppenheim:  2.327x -> 2.377x
synth_rgba:      2.028x -> 2.073x
```

The C++ effort-9 benchmark confirmed the direction:

```text
ph_belfast:      2.42x
ph_kloppenheim:  2.38x
synth_rgba:      2.07x
all 13 images:   7.317x geomean

pre-GDX4 effort 11 all 13 images: 7.346x geomean
Blosc all 13 images:     4.633x geomean
```

Interpretation: this is a small global gain, but an important qualitative win:
a MANIAC-style feature did move the hard puresky cases after being reduced to a
production-shaped fixed context. The next step should continue in this direction
with either a tiny signaled tree or a bounded feature-hash mixer, while charging
its side information explicitly.

### GDX4: Channel-Split Context Families

The next tree inspection showed that many useful splits were channel-oriented.
A direct one-split tree was too weak, and a reversible ordered-body Lorenzo /
S-transform pyramid lost to GDX3 on every tested crop. The useful extraction was
instead simpler and more production-shaped: keep the same grouped-delta payload,
but allow selected body bitplanes to choose one context family per channel.

New probe:

```text
scripts/probe_channel_split_families.py
pixi run ml-probe-channel-split-families
```

Probe result:

```text
ph full shared-family estimate:      4.409x
ph full channel-split hybrid:        4.481x
all 13 shared-family estimate:       7.348x
all 13 channel-split hybrid:         7.430x
```

This has been integrated into the C++ grouped-delta stage as `GDX4`. The stream
stores a normal family id for shared bitplanes, or a split marker plus one family
id per channel. It is enabled only for `effort >= 10`; effort 9 remains the fast
normal preset.

Measured full-corpus C++ results:

```text
GDX4 effort 9, all 13 images:   7.317x
GDX4 effort 10, all 13 images:  7.413x
GDX4 effort 11, all 13 images:  7.424x
Blosc all 13 images:            4.633x

ph effort 9 geomean:            4.408x
ph effort 10 geomean:           4.487x
```

Representative effort-10 improvements over effort 9:

```text
CandleGlass:     5.77x -> 5.91x
Cannon:          5.19x -> 5.29x
Tree:            4.04x -> 4.11x
ph_abandoned:    5.97x -> 6.20x
ph_belfast:      2.42x -> 2.45x
ph_kloppenheim:  2.38x -> 2.40x
ph_spruit:       6.99x -> 7.12x
ph_studio:       6.93x -> 7.01x
synth_rgba:      2.07x -> 2.08x
synth_noise:     unchanged at 1.29x
```

Interpretation: this does not change the low-tail impossibility story, but it
is a clean exact-lossless win on upper/body context coding. The next exact track
should optimize the high-effort encoder cost and then look for another
tree-derived extraction with a similar implementation/side-info shape.

Encoder-cost follow-up: the initial channel-split selector rescanned each
tile/bitplane for every `family x channel` candidate. It now gathers the shared
and per-channel counts in one tile pass. The effort-10 ratio stays at `7.413x`,
while representative ph full-image encode times dropped from about `20-33s` to
about `7-9s` per image.

### GDX5: Channel-Split Body Predictor Modes

The next exact-lossless probe tested whether the body predictor mode itself
should split by channel. The unsafe oracle was large, but it allowed cyclic
same-pixel dependencies. The integrated version keeps only decoder-safe
green-first dependencies: RGB(A) body channels are reconstructed in `G, R, B, A`
order, and channel modes are allowed only when their source channel is already
available.

Full-corpus probe:

```text
all 13 shared-mode estimate:    7.430x
all 13 channel-mode estimate:   7.516x
all 13 hybrid estimate:         7.524x
```

C++ result after integration:

```text
GDX5 effort 11, all 13 images:  7.493x
GDX4 effort 11, all 13 images:  7.424x
ph crop128 effort 10 -> 11:     5.946x -> 6.006x
synth crop128 effort 10 -> 11:  4.315x -> 4.315x
```

Interpretation: this is another small but clean exact-lossless win. The shape is
good: a compact signaled structural choice beats a larger unsignaled context.
It still does not solve the hard low-mantissa planes, so the next large move
probably needs a representation change rather than another ordinary predictor.

### GDX6: Exact Half16 Tile Route

The next representation route targets float32 files whose values are actually
binary16 values widened to float32. A tile is eligible only when every sample can
be converted to binary16 and back bit-exactly. Eligible body records can then
store a 16-bit ordered-half residual; the decoder reconstructs the half bits and
expands them back to the exact float32 bit pattern.

Probe and C++ results:

```text
half16 full-corpus probe:          7.348x -> 7.458x
oexr crop128 C++ effort 12:        15.983x geomean
oexr full C++ effort 12:           16.340x geomean
approx all-13 effort-12 geomean:   7.54x
approx target corpus without pure noise: 8.73x
ph crop128 effort 12:              6.049x
synth crop128 effort 12:           4.317x
```

This is useful and exact, but not the missing 8x breakthrough. It improves
half-like sources and leaves the true/dithered float stress cases essentially
unchanged. The synthetic data generator confirms why: `synth_noise` is pure
Gaussian noise, while `synth_mixed` and `synth_rgba` explicitly add random
float32 noise. Without near-lossless tail handling or generator-specific tricks,
those low mantissa planes have to be stored.

Target-corpus decision: `synth_noise` remains useful as a pure incompressibility
torture test, but it is excluded from the main realistic target metric. The
numbers are now reported as a split, not as a single blended headline:

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

The repeatable target benchmark task is the realistic core:

```text
pixi run bench-grouped-delta-target-cpp
pixi run bench-grouped-delta-target-no-puresky-cpp
pixi run bench-grouped-delta-puresky-cpp
pixi run bench-grouped-delta-no-noise-cpp
pixi run bench-grouped-delta-easy-cpp
pixi run bench-grouped-delta-synthetic-cpp
pixi run bench-grouped-delta-noise-stress-cpp
```

Puresky-specific tile-split probes:

```text
scripts/probe_puresky_predictors.py
pixi run ml-probe-puresky-predictors

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

Takeaway: the intuitive "smooth sky gradient" is not enough for exact lossless
Float32 compression here. Current predictors already capture most of that
smoothness; the remaining puresky cost behaves like real low/mid mantissa
payload entropy. A bounded hash context produced a tiny probe gain, but the
C++ codec-shaped version did not move the measured ratio. The next exact route
should try lower-plane prediction conditioned on reconstructed high-value
features. The lower-bound signal is strong, but straightforward
palette/dictionary side information destroys it, and adaptive hash models do
not learn fast enough. The small positive exact result is still zero-mask
raw-tail coding; low-tail residual coding against a surface derived from the
already decoded high body is a weaker backup and mostly overlaps the same
structure. That is not a breakthrough yet, but it is the right shape: predict
low-tail phase from high float structure without transmitting a large
dictionary. Near-lossless tail control remains the practical large-ratio escape
hatch.

This exact split is now integrated in C++ as `GDX8` for `effort >= 12`. The body
record can signal a low-tail split: high body bits remain in the grouped-delta
rANS payload, while low `15` ordered-body bits are coded in a separate
zero-mask tail stream. Decode is still bit-exact because delta-mode high bits
are reconstructed using the predictor high bits, encoded high residual, and the
borrow implied by the decoded low tail.

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

The `GDX8` byte stream can now be audited directly:

```text
scripts/audit_gdx8_stream_budget.py
pixi run ml-audit-gdx8-stream-budget
```

The first budget split is the key result:

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

So the exact-lossless roadmap must branch. Puresky is tail dominated; ordinary
real HDR images are main-payload dominated.

The puresky low-tail entropy audit is nearly a hard certificate. In both full
puresky files, ordered low15 entropy is about `14.87 bits/value` in each RGB
channel and `0` in alpha. Averaged over four channels, the low tail alone costs
about `11.15 bits/float sample` before any other image data is stored. A `12x`
exact-lossless target has only `2.67 bits/float sample` total budget. Therefore
the exact puresky route must either predict that low tail from other
decoder-visible data or accept that the puresky files cap the ratio near the
current range.

Follow-up exact probes did not yet find a usable predictor:

```text
static transmitted tail probability table, crop128:
  no gain; table route stayed slightly worse than current low coding

context palette/dictionary route, full puresky:
  no gain; all tiles kept `current_low`

raw ordered-tail bytes through zlib/bz2/lzma, full:
  lzma best at ~11.67 bits/value, still near the entropy floor

top16 Rice/Golomb residual route on bfloat-like crop128:
  worse than current GDX8 bitplane/context coding
```

`GDX9` adds four cheap fixed context families derived from the useful parts of
the MANIAC-like tree probe:

- `Prev4Channel`;
- `WNPrev4Channel`;
- `WNPrev4Pc`;
- `SpatialXY`.

These are not a breakthrough, but they are production-shaped: no dynamic tree
metadata, just extra 4-bit family selectors. The first C++ crop result is small
and positive:

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

`GDXA` is a small format bump for a source-precision cleanup. It keeps `GDX9`
but allows channel-mode split to choose half-float predictor modes when all
channels in the tile are exactly binary16-representable. Mixed half/non-half
channel splits remain invalid.

```text
realistic-no-puresky crop128, effort12:
  GDX9 geomean:            8.028x
  GDXA geomean:            8.030x

realistic-no-puresky full, effort12:
  GDX9 geomean:            6.001x
  GDXA geomean:            6.001x

GDXA crop128 audit:
  Cannon / ph_abandoned / ph_studio select all-half channel splits
  puresky-hard remains unchanged at 2.347x
```

`GDXB` uses the last spare normal family id (`14`) as `ConstantZero`; selector
`15` remains the channel-split marker. Entirely zero bitplanes are skipped
instead of being adaptively coded. Decoder correctness is simple because payload
buffers already start at zero.

```text
realistic-no-puresky crop128, effort12:
  GDXA geomean:            8.030x
  GDXB geomean:            8.043x

puresky-hard crop128, effort12:
  unchanged geomean:       2.347x

realistic-no-puresky full, effort12:
  first GDXB full run:      6.004x
  ph_spruit spot after pruning: 7.377x
```

The variable-tile-split probe (`scripts/probe_variable_tile_split.py`) was
negative: dynamic `128` versus four `64` subtiles moved oexr crop128 only
`5.605x -> 5.611x` and did not move the ph crop128 geomean. That is currently
not enough ratio for the required format complexity.

## 16x Target Audit

The aspirational target was raised to `16x`. For exact lossless, that means an
average of only `2 bits/float32 sample`. A new budget audit exists:

```text
scripts/audit_16x_budget.py
pixi run ml-audit-16x-budget
```

It runs the current C++ grouped-delta stage, computes the exact `16x` bit
budget, and estimates where the grouped-delta model spends bits.

Effort-9 full-corpus result:

```text
current geomean: 7.317x
total bits still above 16x target over 13 files: ~22.7 MiB

hard blockers:
  ph_belfast:     2.42x, needs -2.80 MiB, low body 11.25 bits/sample
  ph_kloppenheim: 2.38x, needs -2.87 MiB, low body 11.27 bits/sample
  synth_mixed:    1.61x, needs -3.34 MiB, low body 15.03 bits/sample
  synth_noise:    1.29x, needs -4.28 MiB, low body 14.95 bits/sample
  synth_rgba:     2.07x, needs -3.36 MiB, low body 11.30 bits/sample
```

Interpretation: `16x` exact lossless on every image is not a matter of shaving
headers or choosing a better backend compressor. The hard images spend far more
than the entire `16x` budget in low mantissa/body bits alone. To keep the exact
lossless target alive, the next research must either:

- find a new conditional law inside those low body bits, or
- prove that the low body bits are incompressible under a chosen conditioning
  set, turning those tiles into exact-lossless hard certificates.

A bounded feature-hash context probe was added:

```text
scripts/probe_feature_hash_context.py
pixi run ml-probe-feature-hash-context
```

Initial crop results were weak:

```text
ph_belfast crop128:
  current family baseline: 2.353x
  best feature-hash:       2.362x  (~0.38% local gain)

synth_noise crop128:
  current family baseline: 1.290x
  best feature-hash:       1.290x
```

So a fixed side-info-free hash is not yet the `16x` path. The stronger direction
remains an explicitly signaled tiny tree or a different payload representation
for the low body bits.

## 12x Recalibration and Near-Lossless Mantissa Audit

After the intentionally aggressive `16x` push, the practical stretch target was
relaxed to around `12x`. That does not erase what the `16x` audit found: exact
lossless is blocked by real low-body entropy on some files. The useful move is
to separate two questions:

1. How far can exact GDX3 keep moving?
2. If a near-lossless product mode is allowed, how much does controlled
   mantissa-tail quantization buy?

Two scripts now support that split:

```text
scripts/estimate_mantissa_quantization.py
pixi run ml-estimate-mantissa-quantization

scripts/summarize_mantissa_target.py
pixi run ml-summarize-mantissa-target --target-ratio 12
```

Full 13-image effort-9 result from zeroing raw mantissa low bits and then using
the same GDX3 backend:

```text
geomean by low bits:
  low00:  7.317x
  low08:  8.650x   max relative error about 3.04e-05 where changed
  low12: 11.085x   max relative error about 4.88e-04 where changed
  low15: 12.540x   max relative error about 3.89e-03 where changed
```

The `12x` target summary with a reach-target policy gives:

```text
exact geomean:         7.317x
selected geomean:     13.129x
exact total ratio:     3.317x
selected total ratio:  7.059x
images >= 12x:         4/13  (exact 2/13)
remaining gap to 12x:  6480.3 KiB
```

The important per-image behavior:

```text
ph_belfast_puresky:    2.42x -> 15.41x at low15, PSNR 84.4 dB
ph_kloppenheim:        2.38x -> 13.92x at low15, PSNR 88.4 dB

ph_abandoned:          unchanged at 5.97x; low bits already zero
ph_spruit:             unchanged at 6.99x; low bits already zero
ph_studio:             unchanged at 6.93x; low bits already zero

oexr_CandleGlass:      5.77x -> 6.40x at low15
oexr_Cannon:           5.19x -> 5.46x at low15
oexr_Tree:             4.04x -> 4.93x at low15

synth_mixed:           1.61x -> 6.62x at low15
synth_noise:           1.29x -> 3.19x at low15
synth_rgba:            2.07x -> 7.71x at low15
```

Interpretation:

- Low-tail quantization is a real escape hatch for true/dithered puresky float
  tails. It can push those hard images past `12x` with bounded relative error.
- It does almost nothing for bfloat/half-like HDR files because their low
  mantissa bits are already zero; their remaining cost lives in upper body
  structure and headers/modes, not random tail.
- It is not enough for synthetic random/noise stress cases. Those need either
  much more aggressive loss, exclusion from a photographic-HDR target, or an
  exact-lossless hard certificate.
- The geomean can cross `12x`, but byte-weighted total ratio does not. A product
  claim must state which metric is being targeted.

The next exact-lossless work should therefore pivot from ordinary low-tail
contexts to upper-body representation changes: reversible block transforms over
ordered-float bodies, cheaper signaled trees for expensive bitplanes, and
source-precision-aware mode pruning. The next near-lossless product probe should
be a source-class selector:

```text
exact GDX3 for half/bfloat-like tiles
low-tail quantization for true/dithered puresky-like tiles
raw/noise fallback or stronger loss setting for pathological random tiles
```

## Puresky Exact-Tail Follow-Up

`GDXB` makes the no-puresky crop set respectable, but puresky remains hard:

```text
GDXB puresky-hard crop128:
  geomean ratio: 2.347x
  aggregate sections: main=17.9%, tail=82.1%
```

So the bottleneck is the exact low-tail payload, not the visible gradient.
Follow-up probes tested the obvious escape routes:

```text
puresky conditional entropy, crop128:
  low15 current GDX low tail: ~11.30 bits/sample
  best high-feature conditional entropy: ~2.9 bits/sample

tail adaptive high-context:
  no gain

tail image-adaptive high-context:
  no gain

low-tail surface residual:
  geomean 2.299x -> 2.316x at low15 estimate only

fixed decoder prior, leave-one-out:
  low15 geomean 2.299x -> 2.307x

fixed decoder prior, self-oracle:
  low15 geomean 2.299x -> 2.415x
```

The new fixed-prior probe is:

```text
scripts/probe_tail_fixed_prior.py
pixi run ml-probe-tail-fixed-prior
```

Conclusion: the promising low conditional entropy is mostly image-local. A
same-image/oracle whole-symbol table can expose it, but cross-image fixed priors
and causal adaptive dictionaries do not recover enough gain. For exact lossless,
the remaining path is a much cheaper signaled image-local model or a different
reversible representation of the tail values. For product compression targets,
this strengthens the case for keeping puresky tail quantization as a separate
near-lossless option.

## Fast Exact Tier

The first effort-12 speed experiment, prefiltering channel modes before family
scoring, was rejected because it did not speed up the hard cases. The useful
change was smaller: enable the exact half-convertible body route and low-tail
split route at effort 11 instead of effort 12.

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

So effort 11 is now the practical exact-lossless fast tier. It captures most of
the effort-12 compression gain while avoiding the heaviest channel-family
refinement work.

```text
pixi run bench-grouped-delta-fast-cpp
```

## Dream-Path Negative/Positive Audit

After the fixed-prior work, several more exact-tail ideas were tested against
puresky crop128. These are important because they explain why the apparent
conditional-entropy gap is hard to turn into a codec.

Rejected routes:

```text
high-feature sorted tail sequence:
  low15 current ~11.30 bps
  sorted delta best ~13.33 bps

tail-only spatial/channel prediction:
  raw tail remains best for low08/10/12/15

two-part context distribution table:
  low15 best ~12.43-12.47 bps including model cost

fixed-pattern two-stage training:
  low15 best ~12.15-12.21 bps

anchor/interpolation tail model:
  low15 best ~14.26 bps
```

The reason is consistent: the low-tail distribution is biased if the decoder is
given the whole image's future histogram, but the support is too large. For
low15 crop128, a high-feature context table needs roughly `49k` dictionary
values for `65k` samples; compressing those dictionary values costs about
`9.5` bits/sample before the actual entropy-coded indices. Fixed training
samples also fail because most later symbols are unseen.

The only exact-lossless research route that still shows positive signal is a
small signaled context tree:

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

This is far from a `12x` puresky breakthrough, but it is the least-dead exact
research path. The next dream experiment should not be another raw low-tail
dictionary. It should try to make context-tree side information cheap enough
for the main payload, or discover a new source-derived latent variable whose
support is much smaller than the current high-feature tail table.

## Near-Lossless Stage Implementation

Before opening the near-lossless branch, two exact-lossless variants were checked
as a guardrail:

```text
FPC/Gorilla-style XOR leading/trailing-zero route:
  body31 and tail15 estimates stayed worse than current GDX bitplane contexts,
  except for zero-tail cases already covered by existing modes.

quantized base + exact low-bit correction:
  puresky quantized-only ratios rose sharply,
  but adding exact correction bits returned the total ratio to ~2.36x.
```

That says the discarded low mantissa bits are the value of the near-lossless
mode. They are not a cheap side stream waiting to be coded exactly.

The product-shaped near-lossless path is now implemented in C++ and Python:

```text
StageMantissaQuantize:
  finite float32 values only
  clear configurable low mantissa bits, 0..23
  preserve NaN/Inf bit patterns
  decode is a passthrough because the payload already stores quantized pixels

Outer frame:
  header version 2
  stores near_lossless_bits
  keeps version 1 decode compatibility

Python:
  radiance_codec.encode_near_lossless(...)
  radiance_codec.quantize_mantissa(...)
```

Effort11 crop128 results:

```text
five real ph_* files:
  low00 exact:  6.076x
  low08:        7.634x
  low12:        9.314x
  low15:       12.071x

puresky-hard only:
  low00 exact:  2.345x
  low08:        4.151x
  low12:        6.825x
  low15:       13.051x
```

The strategic split is now clear:

- Exact lossless remains the research dream path: cheaper signaled context trees,
  upper-body representation changes, and source-class routing.
- Near-lossless is a separate option for true/dithered float tails, especially
  puresky. It can reach the `12x` class on those files because it removes the
  exact information that dominates their payload.
- The next near-lossless step should be a selector/policy layer, not just a fixed
  global bit count: exact for half-like tiles, `low12` or `low15` for puresky-like
  true-float tails, and explicit fallback handling for pathological random data.
