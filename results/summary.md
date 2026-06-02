# Float32 HDR EXR lossless compression

- Test set: 13 EXR images (Poly Haven + OpenEXR samples)
- Median of 3 runs (1 warm-up) per measurement
- Apple Silicon, single thread per tool (default settings)

## Aggregate results

| method                          | ratio (geomean)   |   bpp mean | bpp range   |   enc MB/s |   dec MB/s | lossless   |
|---------------------------------|-------------------|------------|-------------|------------|------------|------------|
| blosc_zstd9_bitshuf             | 4.63x             |      37.4  | 0.1-77.2    |       25.5 |      859.6 | 13/13      |
| xz_9                            | 4.31x             |      39.72 | 0.1-82.3    |        1.7 |       32.8 | 13/13      |
| planes_zstd19                   | 3.90x             |      43.26 | 0.1-76.3    |        5.2 |       49.7 | 13/13      |
| blosc_zstd9_shuf                | 3.80x             |      43.57 | 0.1-76.4    |       18.7 |     1071.4 | 13/13      |
| zstd_22                         | 3.43x             |      47.52 | 0.2-85.0    |        2   |      150.6 | 13/13      |
| zstd_19                         | 3.43x             |      47.52 | 0.2-85.0    |        2.2 |      156.1 | 13/13      |
| blosc_zstd9_noshuf              | 3.37x             |      47.42 | 0.2-85.0    |       14.8 |      862.8 | 13/13      |
| exr_zip                         | 3.30x             |      50.16 | 0.4-95.4    |      235   |      827.5 | 13/13      |
| bzip2_9                         | 3.17x             |      47.8  | 0.5-87.3    |        8.6 |       18.8 | 13/13      |
| our_v4a_pred_order0             | 3.16x             |      48.11 | 0.3-85.7    |       54.4 |       56.6 | 13/13      |
| our_v4d_pred_bitshuf_order1     | 3.15x             |      42.21 | 1.7-87.2    |       34.7 |       30.7 | 13/13      |
| our_v4c_pred_bitshuf_order0     | 3.10x             |      48.59 | 0.2-89.8    |       37.3 |       40.7 | 13/13      |
| blosc_lz4hc9_shuf               | 3.05x             |      50.56 | 0.2-86.0    |      111.9 |     1342.7 | 13/13      |
| exr_piz                         | 2.93x             |      47.56 | 1.6-96.0    |      226.4 |      528.5 | 13/13      |
| our_v4b_pred_order1             | 2.86x             |      46.26 | 1.8-86.3    |       50.6 |       32.3 | 13/13      |
| zstd_3                          | 2.84x             |      53.1  | 0.2-85.0    |       95.2 |      156.5 | 13/13      |
| our_v3c_rct_pred_bitshuf_order1 | 2.76x             |      46.22 | 1.7-90.3    |       34.7 |       28.1 | 13/13      |
| our_v2b_bitshuf_order1          | 2.74x             |      43.31 | 6.1-83.9    |       40   |       35.7 | 13/13      |
| blosc_lz4_shuf                  | 2.74x             |      54.9  | 0.3-93.5    |     1002.7 |     1233.7 | 13/13      |
| our_v3b_rct_bitshuf_order1      | 2.72x             |      43.58 | 6.1-85.0    |       39.6 |       34.2 | 13/13      |
| exr_zips                        | 2.34x             |      56.36 | 2.3-96.1    |      159.1 |      727.6 | 13/13      |
| lz4_9                           | 2.30x             |      63.12 | 0.5-100.9   |       28.8 |      204.7 | 13/13      |
| our_v1c_rans_order1             | 1.87x             |      58.95 | 14.8-90.4   |       67.2 |       32   | 13/13      |
| zfp                             | 1.86x             |      65.21 | 5.1-114.2   |      179.1 |      160   | 13/13      |
| our_v2a_bitshuf_order0          | 1.81x             |      61.4  | 11.3-89.6   |       41.7 |       51.6 | 13/13      |
| our_v3a_rct_order1              | 1.78x             |      61.82 | 14.8-91.8   |       65.7 |       32.8 | 13/13      |
| our_v1b_rans_order0             | 1.54x             |      69.56 | 19.8-101.2  |       68   |       75.5 | 13/13      |
| exr_rle                         | 1.43x             |      81.57 | 7.0-113.3   |      259.5 |     1038.2 | 13/13      |
| exr_none                        | 1.23x             |      88.75 | 16.2-128.1  |      305.6 |     1275.6 | 13/13      |
| our_v0_passthrough              | 1.00x             |     103.38 | 32.0-128.0  |      467.9 |      747.3 | 13/13      |
| our_v1a_rans_static             | 1.00x             |     103.39 | 32.0-128.0  |       91.6 |       79.6 | 13/13      |
| jxl_e7_nl                       | 7.87x             |      18.41 | 0.5-37.8    |        3.8 |       46.1 | 2/12       |
| jxl_e3_nl                       | 6.23x             |      19.68 | 4.1-38.4    |       26.5 |       53.6 | 2/12       |
| jxl_e7_ll                       | 2.12x             |      54.58 | 13.3-75.2   |        8.7 |       50.1 | 4/12       |
| jxl_e3_ll                       | 1.77x             |      62.15 | 34.9-78.8   |       25.8 |       55.5 | 4/12       |

Columns:
- **ratio (geomean)**: geometric mean of compression ratio across all test images (higher = better)
- **bpp**: bits per pixel of the compressed output
- **enc/dec MB/s**: throughput vs raw float32 data (higher = faster)
- **lossless**: number of images verified byte-exact

## Per-image results

See `results.csv` / `results.json` for per-image breakdown.
