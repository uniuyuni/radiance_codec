#include "bitshuffle.hpp"

#include <cstdint>
#include <cstring>
#include <vector>

namespace radiance_codec {

namespace {

template <typename T>
void put_le(std::vector<uint8_t>& v, T x) {
    for (std::size_t i = 0; i < sizeof(T); ++i) {
        v.push_back(static_cast<uint8_t>((x >> (8 * i)) & 0xFF));
    }
}

template <typename T>
T get_le(const uint8_t* p) {
    T x = 0;
    for (std::size_t i = 0; i < sizeof(T); ++i) {
        x |= static_cast<T>(p[i]) << (8 * i);
    }
    return x;
}

// Transpose 8 items (each `typesize` bytes) into `typesize` bytes ×
// 8 bit-planes. Output is written to `dst[plane * stride + block_idx]`
// where stride is the per-plane stride (n_items / 8 bytes).
//
// Specifically, after this call:
//   dst[bit_pos * stride + block_idx] gets a byte whose bit i is
//   (src[i * typesize + (bit_pos / 8)] >> (bit_pos % 8)) & 1.
void shuffle_block_8(const uint8_t* src, uint8_t* dst,
                      std::size_t typesize, std::size_t stride,
                      std::size_t block_idx) {
    for (std::size_t bit_pos = 0; bit_pos < typesize * 8; ++bit_pos) {
        const std::size_t byte_in = bit_pos / 8;
        const std::size_t bit_in  = bit_pos % 8;
        uint8_t out = 0;
        for (std::size_t i = 0; i < 8; ++i) {
            uint8_t b = (src[i * typesize + byte_in] >> bit_in) & 1u;
            out |= static_cast<uint8_t>(b << i);
        }
        dst[bit_pos * stride + block_idx] = out;
    }
}

// Inverse of shuffle_block_8.
void unshuffle_block_8(const uint8_t* src, uint8_t* dst,
                        std::size_t typesize, std::size_t stride,
                        std::size_t block_idx) {
    // Zero destination items first
    std::memset(dst, 0, 8 * typesize);
    for (std::size_t bit_pos = 0; bit_pos < typesize * 8; ++bit_pos) {
        const std::size_t byte_in = bit_pos / 8;
        const std::size_t bit_in  = bit_pos % 8;
        uint8_t plane_byte = src[bit_pos * stride + block_idx];
        for (std::size_t i = 0; i < 8; ++i) {
            uint8_t b = (plane_byte >> i) & 1u;
            dst[i * typesize + byte_in] |= static_cast<uint8_t>(b << bit_in);
        }
    }
}

} // namespace

Status BitshuffleStage::encode(std::span<const std::uint8_t> in,
                                const ImageMeta& /*meta*/,
                                std::vector<std::uint8_t>& out) noexcept {
    if (typesize_ == 0) return Status::InvalidArg;
    if (in.size() % typesize_ != 0) return Status::SizeMismatch;
    const std::size_t n_items = in.size() / typesize_;
    const std::size_t n_blocks = n_items / 8;
    const std::size_t tail_items = n_items % 8;
    const std::size_t body_size = n_blocks * 8 * typesize_;
    const std::size_t tail_size = tail_items * typesize_;
    const std::size_t header_size = 4 + 1 + 1;

    out.clear();
    out.resize(header_size + body_size + tail_size);

    // Header
    {
        std::vector<uint8_t> hdr;
        put_le<uint32_t>(hdr, static_cast<uint32_t>(n_items));
        hdr.push_back(typesize_);
        hdr.push_back(static_cast<uint8_t>(tail_items));
        std::memcpy(out.data(), hdr.data(), header_size);
    }

    // Body: bit-transposed planes
    if (n_blocks > 0) {
        uint8_t* body = out.data() + header_size;
        // stride per bit-plane = n_blocks bytes (one byte per block of 8 items)
        const std::size_t stride = n_blocks;
        for (std::size_t b = 0; b < n_blocks; ++b) {
            const uint8_t* src = in.data() + b * 8 * typesize_;
            shuffle_block_8(src, body, typesize_, stride, b);
        }
    }

    // Tail: copy verbatim
    if (tail_size > 0) {
        const uint8_t* tail_src = in.data() + n_blocks * 8 * typesize_;
        std::memcpy(out.data() + header_size + body_size,
                    tail_src, tail_size);
    }

    return Status::Ok;
}

Status BitshuffleStage::decode(std::span<const std::uint8_t> in,
                                const ImageMeta& /*meta*/,
                                std::vector<std::uint8_t>& out) noexcept {
    constexpr std::size_t header_size = 6;
    if (in.size() < header_size) return Status::DecompressFailed;

    uint32_t n_items   = get_le<uint32_t>(in.data());
    uint8_t  ts        = in[4];
    uint8_t  tail_items = in[5];
    if (ts == 0 || ts != typesize_) return Status::DecompressFailed;
    if (tail_items >= 8) return Status::DecompressFailed;

    const std::size_t n_blocks  = (n_items - tail_items) / 8;
    if (n_blocks * 8 + tail_items != n_items) {
        return Status::DecompressFailed;
    }
    const std::size_t body_size = n_blocks * 8 * typesize_;
    const std::size_t tail_size = tail_items * typesize_;
    if (in.size() != header_size + body_size + tail_size) {
        return Status::DecompressFailed;
    }

    out.assign(static_cast<std::size_t>(n_items) * typesize_, 0);

    if (n_blocks > 0) {
        const uint8_t* body = in.data() + header_size;
        const std::size_t stride = n_blocks;
        for (std::size_t b = 0; b < n_blocks; ++b) {
            uint8_t* dst = out.data() + b * 8 * typesize_;
            unshuffle_block_8(body, dst, typesize_, stride, b);
        }
    }

    if (tail_size > 0) {
        std::memcpy(out.data() + n_blocks * 8 * typesize_,
                    in.data() + header_size + body_size,
                    tail_size);
    }

    return Status::Ok;
}

} // namespace radiance_codec
