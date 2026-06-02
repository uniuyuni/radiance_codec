#include "passthrough.hpp"

namespace radiance_codec {

Status PassthroughStage::encode(std::span<const std::uint8_t> in,
                                 const ImageMeta& /*meta*/,
                                 std::vector<std::uint8_t>& out) noexcept {
    out.assign(in.begin(), in.end());
    return Status::Ok;
}

Status PassthroughStage::decode(std::span<const std::uint8_t> in,
                                 const ImageMeta& /*meta*/,
                                 std::vector<std::uint8_t>& out) noexcept {
    out.assign(in.begin(), in.end());
    return Status::Ok;
}

} // namespace radiance_codec
