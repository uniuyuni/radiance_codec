// Adaptive binary rANS helpers for bit-plane context coding.
//
// This is an internal building block for structural HDR coding. It encodes
// binary symbols with a decoder-reproducible adaptive probability per context.

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace radiance_codec::rans {

// Encode `bits[i]` using context `contexts[i]`. Contexts must be reproducible
// by the decoder before decoding symbol i. The returned vector is the raw rANS
// payload, without any container header.
std::vector<std::uint8_t> encode_adaptive_binary(
    std::span<const std::uint8_t> bits,
    std::span<const std::uint16_t> contexts,
    std::uint32_t context_count);

// Decode a payload produced by encode_adaptive_binary.
bool decode_adaptive_binary(
    std::span<const std::uint8_t> payload,
    std::span<const std::uint16_t> contexts,
    std::uint32_t context_count,
    std::vector<std::uint8_t>& bits);

} // namespace radiance_codec::rans
