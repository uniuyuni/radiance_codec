#include "predictor.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>

namespace radiance_codec {

namespace {

// IEEE 754 float bit-pattern XOR via memcpy (avoids strict aliasing UB).
inline uint32_t f_to_u32(float v) noexcept {
    uint32_t u;
    std::memcpy(&u, &v, 4);
    return u;
}
inline float u32_to_f(uint32_t u) noexcept {
    float v;
    std::memcpy(&v, &u, 4);
    return v;
}

// JPEG-LS MED predictor in float space.
//
// Notes:
//   - std::min/std::max on float: for NaN inputs the behaviour is
//     implementation defined but is consistent within a single binary,
//     so encoder and decoder still agree. HDR data in our domain has
//     no NaNs in practice.
//   - The planar branch (N + W - NW) may produce a value that is not
//     bit-exactly recoverable in the *value* domain, but we don't care
//     because we only use it as a bit-string to XOR against.
inline float med_predict(float W, float N, float NW) noexcept {
    const float mx = std::max(N, W);
    const float mn = std::min(N, W);
    if (NW >= mx) return mn;
    if (NW <= mn) return mx;
    return N + W - NW;
}

// For boundary pixels use simpler predictors:
//   (0,0)           : P = 0
//   (0, col>0)      : P = W
//   (row>0, 0)      : P = N
inline float predict_at(const float* plane, uint32_t W_pix, uint32_t H_pix,
                        uint32_t row, uint32_t col) noexcept {
    if (row == 0 && col == 0) return 0.0f;
    if (row == 0) return plane[col - 1];
    if (col == 0) return plane[(row - 1) * W_pix];
    const float w  = plane[ row      * W_pix + (col - 1)];
    const float n  = plane[(row - 1) * W_pix +  col     ];
    const float nw = plane[(row - 1) * W_pix + (col - 1)];
    return med_predict(w, n, nw);
}

} // namespace

Status PredictStage::encode(std::span<const std::uint8_t> in,
                             const ImageMeta& meta,
                             std::vector<std::uint8_t>& out) noexcept {
    const std::size_t expected = meta.raw_size();
    if (in.size() != expected) return Status::SizeMismatch;
    if (meta.format != PixelFormat::Float32) return Status::UnsupportedFormat;

    const uint32_t W = meta.width;
    const uint32_t H = meta.height;
    const uint8_t  C = meta.channels;
    const std::size_t pixels = std::size_t(W) * H;

    // De-interleave channels into separate planes for prediction.
    std::vector<float> planes(pixels * C);  // [c * pixels + idx] layout
    {
        const float* src = reinterpret_cast<const float*>(in.data());
        for (std::size_t i = 0; i < pixels; ++i) {
            for (uint8_t c = 0; c < C; ++c) {
                planes[c * pixels + i] = src[i * C + c];
            }
        }
    }

    // Predict & XOR per channel, then re-interleave residuals.
    out.assign(in.size(), 0);
    uint32_t* dst = reinterpret_cast<uint32_t*>(out.data());

    for (uint8_t c = 0; c < C; ++c) {
        const float* plane = planes.data() + c * pixels;
        for (uint32_t row = 0; row < H; ++row) {
            for (uint32_t col = 0; col < W; ++col) {
                const float actual = plane[row * W + col];
                const float pred   = predict_at(plane, W, H, row, col);
                const uint32_t r = f_to_u32(actual) ^ f_to_u32(pred);
                dst[(row * W + col) * C + c] = r;
            }
        }
    }
    return Status::Ok;
}

Status PredictStage::decode(std::span<const std::uint8_t> in,
                             const ImageMeta& meta,
                             std::vector<std::uint8_t>& out) noexcept {
    const std::size_t expected = meta.raw_size();
    if (in.size() != expected) return Status::SizeMismatch;
    if (meta.format != PixelFormat::Float32) return Status::UnsupportedFormat;

    const uint32_t W = meta.width;
    const uint32_t H = meta.height;
    const uint8_t  C = meta.channels;
    const std::size_t pixels = std::size_t(W) * H;

    // Residuals are also interleaved (matching the encode output).
    const uint32_t* res_inter =
        reinterpret_cast<const uint32_t*>(in.data());

    // We reconstruct per-channel planes incrementally so the predictor
    // has access to already-decoded neighbors.
    std::vector<float> planes(pixels * C);

    for (uint8_t c = 0; c < C; ++c) {
        float* plane = planes.data() + c * pixels;
        for (uint32_t row = 0; row < H; ++row) {
            for (uint32_t col = 0; col < W; ++col) {
                const float pred = predict_at(plane, W, H, row, col);
                const uint32_t r = res_inter[(row * W + col) * C + c];
                const uint32_t a = r ^ f_to_u32(pred);
                plane[row * W + col] = u32_to_f(a);
            }
        }
    }

    // Re-interleave to match the original byte order.
    out.assign(in.size(), 0);
    float* dst = reinterpret_cast<float*>(out.data());
    for (std::size_t i = 0; i < pixels; ++i) {
        for (uint8_t c = 0; c < C; ++c) {
            dst[i * C + c] = planes[c * pixels + i];
        }
    }
    return Status::Ok;
}

} // namespace radiance_codec
