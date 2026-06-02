#include "mantissa_quantize.hpp"

#include <cstdint>

namespace radiance_codec {
namespace {

std::uint32_t read_le32(const std::uint8_t* p) noexcept {
    return static_cast<std::uint32_t>(p[0])
        | (static_cast<std::uint32_t>(p[1]) << 8)
        | (static_cast<std::uint32_t>(p[2]) << 16)
        | (static_cast<std::uint32_t>(p[3]) << 24);
}

void write_le32(std::uint8_t* p, std::uint32_t value) noexcept {
    p[0] = static_cast<std::uint8_t>(value & 0xffu);
    p[1] = static_cast<std::uint8_t>((value >> 8) & 0xffu);
    p[2] = static_cast<std::uint8_t>((value >> 16) & 0xffu);
    p[3] = static_cast<std::uint8_t>((value >> 24) & 0xffu);
}

} // namespace

Status MantissaQuantizeStage::encode(
    std::span<const std::uint8_t> in,
    const ImageMeta& meta,
    std::vector<std::uint8_t>& out) noexcept {
    if (meta.format != PixelFormat::Float32) return Status::UnsupportedFormat;
    if (in.size() != meta.raw_size()) return Status::SizeMismatch;

    out.assign(in.begin(), in.end());
    if (low_bits_ == 0) return Status::Ok;

    const std::uint32_t clear_mask =
        low_bits_ >= 23
            ? 0xff800000u
            : (0xffffffffu << low_bits_);
    const std::size_t count = out.size() / 4;
    for (std::size_t i = 0; i < count; ++i) {
        auto* p = out.data() + i * 4;
        auto bits = read_le32(p);
        const auto exponent = (bits >> 23) & 0xffu;
        if (exponent != 0xffu) {
            bits &= clear_mask;
            write_le32(p, bits);
        }
    }
    return Status::Ok;
}

Status MantissaQuantizeStage::decode(
    std::span<const std::uint8_t> in,
    const ImageMeta& meta,
    std::vector<std::uint8_t>& out) noexcept {
    if (meta.format != PixelFormat::Float32) return Status::UnsupportedFormat;
    out.assign(in.begin(), in.end());
    return Status::Ok;
}

} // namespace radiance_codec
