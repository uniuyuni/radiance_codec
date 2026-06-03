#include "linear_index_transform.hpp"

#include <cmath>

namespace radiance_codec {

LinearIndexTransformMode linear_index_transform_from_policy(
    std::uint8_t policy) noexcept {
    switch (static_cast<NearLosslessPolicy>(policy)) {
        case NearLosslessPolicy::LogRange:
            return LinearIndexTransformMode::SignedLog;
        case NearLosslessPolicy::SqrtRange:
            return LinearIndexTransformMode::Sqrt;
        case NearLosslessPolicy::Gamma075Range:
            return LinearIndexTransformMode::Gamma075;
        case NearLosslessPolicy::Gamma025Range:
            return LinearIndexTransformMode::Gamma025;
        case NearLosslessPolicy::AsinhRange:
            return LinearIndexTransformMode::Asinh;
        default:
            return LinearIndexTransformMode::Linear;
    }
}

bool linear_index_transform_mode_is_valid(std::uint8_t mode) noexcept {
    return mode <= static_cast<std::uint8_t>(LinearIndexTransformMode::Asinh);
}

double linear_index_transform_value(
    double value,
    LinearIndexTransformMode mode) noexcept {
    switch (mode) {
        case LinearIndexTransformMode::SignedLog:
            return std::copysign(std::log2(1.0 + std::abs(value)), value);
        case LinearIndexTransformMode::Sqrt:
            return std::copysign(std::sqrt(std::abs(value)), value);
        case LinearIndexTransformMode::Gamma075:
            return std::copysign(std::pow(std::abs(value), 0.75), value);
        case LinearIndexTransformMode::Gamma025:
            return std::copysign(std::pow(std::abs(value), 0.25), value);
        case LinearIndexTransformMode::Asinh:
            return std::asinh(value);
        case LinearIndexTransformMode::Linear:
            return value;
    }
    return value;
}

double linear_index_inverse_transform_value(
    double value,
    LinearIndexTransformMode mode) noexcept {
    switch (mode) {
        case LinearIndexTransformMode::SignedLog:
            return std::copysign(std::exp2(std::abs(value)) - 1.0, value);
        case LinearIndexTransformMode::Sqrt:
            return std::copysign(value * value, value);
        case LinearIndexTransformMode::Gamma075:
            return std::copysign(std::pow(std::abs(value), 1.0 / 0.75), value);
        case LinearIndexTransformMode::Gamma025:
            return std::copysign(std::pow(std::abs(value), 4.0), value);
        case LinearIndexTransformMode::Asinh:
            return std::sinh(value);
        case LinearIndexTransformMode::Linear:
            return value;
    }
    return value;
}

} // namespace radiance_codec
