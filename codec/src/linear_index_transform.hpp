#pragma once

#include "radiance_codec/codec.hpp"

#include <cstdint>

namespace radiance_codec {

enum class LinearIndexTransformMode : std::uint8_t {
    Linear = 0,
    SignedLog = 1,
    Sqrt = 2,
    Gamma075 = 3,
    Gamma025 = 4,
    Asinh = 5,
};

LinearIndexTransformMode linear_index_transform_from_policy(
    std::uint8_t policy) noexcept;
bool linear_index_transform_mode_is_valid(std::uint8_t mode) noexcept;
double linear_index_transform_value(
    double value,
    LinearIndexTransformMode mode) noexcept;
double linear_index_inverse_transform_value(
    double value,
    LinearIndexTransformMode mode) noexcept;

} // namespace radiance_codec
