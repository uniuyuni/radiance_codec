#include "color_transform.hpp"

#include <cstdint>
#include <cstring>

namespace radiance_codec {

namespace {

// Note: the XOR transform is its own inverse, so encode and decode share
// the same body. We keep two methods to satisfy the IStage interface and
// document the directionality.
void apply_xor_rct(const uint8_t* in_bytes, uint8_t* out_bytes,
                    std::size_t pixels, uint8_t channels) {
    const uint32_t* src = reinterpret_cast<const uint32_t*>(in_bytes);
    uint32_t*       dst = reinterpret_cast<uint32_t*>(out_bytes);
    for (std::size_t p = 0; p < pixels; ++p) {
        const std::size_t i = p * channels;
        const uint32_t r = src[i + 0];
        const uint32_t g = src[i + 1];
        const uint32_t b = src[i + 2];
        dst[i + 0] = r ^ g;
        dst[i + 1] = g;
        dst[i + 2] = b ^ g;
        if (channels >= 4) {
            dst[i + 3] = src[i + 3];   // alpha untouched
        }
    }
}

} // namespace

Status ColorTransformStage::encode(std::span<const std::uint8_t> in,
                                    const ImageMeta& meta,
                                    std::vector<std::uint8_t>& out) noexcept {
    if (in.size() != meta.raw_size()) return Status::SizeMismatch;
    if (meta.format != PixelFormat::Float32) return Status::UnsupportedFormat;

    if (meta.channels < 3) {
        // Nothing to decorrelate; passthrough so the pipeline framing
        // still works without us touching the bytes.
        out.assign(in.begin(), in.end());
        return Status::Ok;
    }
    out.assign(in.size(), 0);
    const std::size_t pixels = std::size_t(meta.width) * meta.height;
    apply_xor_rct(in.data(), out.data(), pixels, meta.channels);
    return Status::Ok;
}

Status ColorTransformStage::decode(std::span<const std::uint8_t> in,
                                    const ImageMeta& meta,
                                    std::vector<std::uint8_t>& out) noexcept {
    // XOR is involutive: same body.
    return encode(in, meta, out);
}

} // namespace radiance_codec
