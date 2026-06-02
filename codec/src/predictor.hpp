// Stage: spatial predictor (MED / LOCO-I rule, adapted for float32).
//
// For each pixel x at (row, col, channel), uses three neighbors:
//   W  = x[row,   col-1]  (West)
//   N  = x[row-1, col]    (North)
//   NW = x[row-1, col-1]  (Northwest)
//
// Predicts via the JPEG-LS MED rule (Median Edge Detection) computed
// in IEEE 754 float arithmetic:
//   if NW >= max(N, W):  P = min(N, W)   (edge sloping up-right)
//   elif NW <= min(N, W): P = max(N, W)  (edge sloping down-right)
//   else:                P = N + W - NW  (planar interpolation)
//
// Residual is encoded as a BITWISE XOR between the actual float bits
// (uint32 view) and the predicted float bits. XOR is exactly invertible:
//   actual_bits = predicted_bits XOR residual_bits
//
// For HDR images where neighbors are similar, predicted_bits is close
// to actual_bits, so residual_bits has many high zero bits. After
// downstream bitshuffle these zero bits cluster into long zero runs
// that the entropy coder collapses.
//
// Determinism: float arithmetic on the same hardware and compiler
// produces identical bit patterns, so encoder and decoder agree exactly.
//
// Layout: input is interleaved float32 (channels packed per pixel).
// We process each channel independently in raster order, emitting
// residuals in the SAME interleaved layout. Boundary pixels use the
// available neighbors (e.g., first column uses only N).

#pragma once

#include "pipeline.hpp"

namespace radiance_codec {

class PredictStage final : public IStage {
public:
    Status encode(std::span<const std::uint8_t> in,
                  const ImageMeta& meta,
                  std::vector<std::uint8_t>& out) noexcept override;
    Status decode(std::span<const std::uint8_t> in,
                  const ImageMeta& meta,
                  std::vector<std::uint8_t>& out) noexcept override;
    const char* name() const noexcept override { return "predict_med"; }
};

} // namespace radiance_codec
