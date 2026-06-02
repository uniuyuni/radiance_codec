// C ABI thin wrapper around the C++ API.
// All ownership crosses through radiance_codec_buffer_t / radiance_codec_buffer_free.

#include "radiance_codec/c_api.h"
#include "radiance_codec/codec.hpp"

#include <cstdlib>
#include <cstring>
#include <new>
#include <vector>

// Helpers live in an anonymous namespace with C++ linkage so that
// overloading works. The C ABI is the extern "C" block below.
namespace {

radiance_codec::ImageMeta meta_to_cpp(const radiance_codec_meta_t& m) {
    radiance_codec::ImageMeta out;
    out.width    = m.width;
    out.height   = m.height;
    out.channels = m.channels;
    out.format   = static_cast<radiance_codec::PixelFormat>(m.format);
    return out;
}

radiance_codec::PipelineConfig config_to_cpp(const radiance_codec_config_t& c) {
    radiance_codec::PipelineConfig out;
    out.stages    = c.stages;
    out.effort    = c.effort;
    out.rans_mode = c.rans_mode;
    out.near_lossless_bits = c.near_lossless_bits;
    return out;
}

int populate_buffer(const std::vector<std::uint8_t>& src,
                    radiance_codec_buffer_t* out) {
    out->data = static_cast<std::uint8_t*>(std::malloc(src.size()));
    if (!out->data) return RADIANCE_CODEC_DECOMPRESS_FAILED;
    std::memcpy(out->data, src.data(), src.size());
    out->size = src.size();
    return RADIANCE_CODEC_OK;
}

} // namespace

extern "C" {

const char* radiance_codec_version(void) {
    return radiance_codec::version();
}

void radiance_codec_buffer_free(radiance_codec_buffer_t* buf) {
    if (buf && buf->data) {
        std::free(buf->data);
        buf->data = nullptr;
        buf->size = 0;
    }
}

int radiance_codec_encode(
    const uint8_t* raw, size_t raw_size,
    const radiance_codec_meta_t* meta,
    const radiance_codec_config_t* config,
    radiance_codec_buffer_t* out) {

    if (!raw || !meta || !config || !out) return RADIANCE_CODEC_INVALID_ARG;
    out->data = nullptr;
    out->size = 0;

    std::vector<std::uint8_t> compressed;
    auto status = radiance_codec::encode(
        std::span<const std::uint8_t>(raw, raw_size),
        meta_to_cpp(*meta), config_to_cpp(*config), compressed);

    if (status != radiance_codec::Status::Ok) {
        return static_cast<int>(status);
    }
    return populate_buffer(compressed, out);
}

int radiance_codec_decode(
    const uint8_t* compressed, size_t compressed_size,
    const radiance_codec_meta_t* meta,
    const radiance_codec_config_t* config,
    radiance_codec_buffer_t* out) {

    if (!compressed || !meta || !config || !out) return RADIANCE_CODEC_INVALID_ARG;
    out->data = nullptr;
    out->size = 0;

    std::vector<std::uint8_t> raw;
    auto status = radiance_codec::decode(
        std::span<const std::uint8_t>(compressed, compressed_size),
        meta_to_cpp(*meta), config_to_cpp(*config), raw);

    if (status != radiance_codec::Status::Ok) {
        return static_cast<int>(status);
    }
    return populate_buffer(raw, out);
}

} // extern "C"
