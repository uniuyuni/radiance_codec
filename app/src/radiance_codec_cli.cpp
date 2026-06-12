#include <radiance_codec/codec.hpp>

#include <OpenEXR/ImfChannelList.h>
#include <OpenEXR/ImfFrameBuffer.h>
#include <OpenEXR/ImfHeader.h>
#include <OpenEXR/ImfInputFile.h>
#include <OpenEXR/ImfOutputFile.h>
#include <OpenEXR/ImfRgbaFile.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace fs = std::filesystem;
namespace exr = OPENEXR_IMF_NAMESPACE;

namespace {

struct Image {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint8_t channels = 0;
    std::vector<float> pixels;
};

[[noreturn]] void fail(std::string_view message) {
    throw std::runtime_error(std::string(message));
}

bool has_channel(const exr::ChannelList& channels, const char* name) {
    return channels.findChannel(name) != nullptr;
}

std::vector<std::uint8_t> read_file(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) fail("can't open input file: " + path.string());
    in.seekg(0, std::ios::end);
    const auto size = in.tellg();
    if (size < 0) fail("can't stat input file: " + path.string());
    in.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
    if (!bytes.empty()) {
        in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }
    if (!in && !bytes.empty()) fail("can't read input file: " + path.string());
    return bytes;
}

void write_file(const fs::path& path, std::span<const std::uint8_t> bytes) {
    if (path.has_parent_path()) fs::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary);
    if (!out) fail("can't open output file: " + path.string());
    if (!bytes.empty()) {
        out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }
    if (!out) fail("can't write output file: " + path.string());
}

Image read_exr(const fs::path& path) {
    exr::InputFile file(path.string().c_str());
    const exr::Header& header = file.header();
    const auto dw = header.dataWindow();
    const int width = dw.max.x - dw.min.x + 1;
    const int height = dw.max.y - dw.min.y + 1;
    if (width <= 0 || height <= 0) fail("invalid EXR data window");

    const exr::ChannelList& channel_list = header.channels();
    std::vector<std::string> names;
    if (has_channel(channel_list, "Y")) {
        names = {"Y"};
    } else if (has_channel(channel_list, "R")
               && has_channel(channel_list, "G")
               && has_channel(channel_list, "B")) {
        names = {"R", "G", "B"};
        if (has_channel(channel_list, "A")) names.push_back("A");
    } else {
        fail("EXR must contain Y or RGB channels");
    }

    Image image;
    image.width = static_cast<std::uint32_t>(width);
    image.height = static_cast<std::uint32_t>(height);
    image.channels = static_cast<std::uint8_t>(names.size());
    image.pixels.assign(
        static_cast<std::size_t>(width) * height * image.channels,
        0.0f);

    exr::FrameBuffer fb;
    char* base = reinterpret_cast<char*>(image.pixels.data());
    const std::size_t x_stride = sizeof(float) * image.channels;
    const std::size_t y_stride = x_stride * image.width;
    for (std::size_t c = 0; c < names.size(); ++c) {
        fb.insert(
            names[c].c_str(),
            exr::Slice(
                exr::FLOAT,
                base + sizeof(float) * c
                    - dw.min.x * static_cast<std::ptrdiff_t>(x_stride)
                    - dw.min.y * static_cast<std::ptrdiff_t>(y_stride),
                x_stride,
                y_stride));
    }
    file.setFrameBuffer(fb);
    file.readPixels(dw.min.y, dw.max.y);
    return image;
}

void write_exr(const fs::path& path, const Image& image) {
    if (image.width == 0 || image.height == 0 || image.channels < 1 || image.channels > 4) {
        fail("invalid image shape for EXR output");
    }
    if (path.has_parent_path()) fs::create_directories(path.parent_path());

    exr::Header header(static_cast<int>(image.width), static_cast<int>(image.height));
    std::vector<std::string> names;
    if (image.channels == 1) {
        names = {"Y"};
    } else {
        names = {"R", "G", "B"};
        if (image.channels == 4) names.push_back("A");
    }
    for (const auto& name : names) {
        header.channels().insert(name.c_str(), exr::Channel(exr::FLOAT));
    }

    exr::OutputFile file(path.string().c_str(), header);
    exr::FrameBuffer fb;
    char* base = const_cast<char*>(reinterpret_cast<const char*>(image.pixels.data()));
    const std::size_t x_stride = sizeof(float) * image.channels;
    const std::size_t y_stride = x_stride * image.width;
    for (std::size_t c = 0; c < names.size(); ++c) {
        fb.insert(
            names[c].c_str(),
            exr::Slice(exr::FLOAT, base + sizeof(float) * c, x_stride, y_stride));
    }
    file.setFrameBuffer(fb);
    file.writePixels(static_cast<int>(image.height));
}

radiance_codec::PipelineConfig config_for(
    const std::string& mode,
    const std::string& preset,
    int effort,
    int low_bits) {
    radiance_codec::PipelineConfig cfg;
    cfg.rans_mode = 1;
    if (mode == "lossless") {
        if (preset == "fast") {
            cfg.stages = radiance_codec::StageByteplaneRans;
            cfg.effort = effort >= 0 ? effort : 5;
        } else if (preset == "compact") {
            cfg.stages = radiance_codec::StageByteplaneRans;
            cfg.effort = effort >= 0 ? effort : 6;
        } else {
            cfg.stages = radiance_codec::StageGroupedDelta;
            if (preset == "balanced") cfg.effort = effort >= 0 ? effort : 10;
            else if (preset == "quality") cfg.effort = effort >= 0 ? effort : 11;
            else if (preset == "max") cfg.effort = effort >= 0 ? effort : 12;
            else fail("unknown lossless preset: " + preset);
        }
    } else if (mode == "near") {
        if (low_bits < 0 || low_bits > 23) fail("--low-bits must be in 0..23");
        cfg.stages = radiance_codec::StageMantissaQuantize | radiance_codec::StageGroupedDelta;
        cfg.effort = effort >= 0 ? effort : 11;
        cfg.near_lossless_bits = static_cast<std::uint8_t>(low_bits);
        cfg.near_lossless_policy =
            static_cast<std::uint8_t>(radiance_codec::NearLosslessPolicy::Fixed);
    } else if (mode == "router") {
        cfg.stages = radiance_codec::StageNearLosslessRouter;
        cfg.effort = effort >= 0 ? effort : 11;
    } else {
        fail("unknown mode: " + mode);
    }
    return cfg;
}

std::vector<std::uint8_t> image_bytes(const Image& image) {
    std::vector<std::uint8_t> raw(image.pixels.size() * sizeof(float));
    std::memcpy(raw.data(), image.pixels.data(), raw.size());
    return raw;
}

Image image_from_bytes(std::span<const std::uint8_t> raw, const radiance_codec::ImageMeta& meta) {
    if (meta.format != radiance_codec::PixelFormat::Float32) fail("unsupported pixel format");
    if (raw.size() != meta.raw_size()) fail("decoded raw size mismatch");
    Image image;
    image.width = meta.width;
    image.height = meta.height;
    image.channels = meta.channels;
    image.pixels.resize(raw.size() / sizeof(float));
    std::memcpy(image.pixels.data(), raw.data(), raw.size());
    return image;
}

void usage() {
    std::cerr
        << "radiance-codec encode input.exr output.rcodec [--mode lossless|near|router]\n"
        << "                                             [--preset fast|compact|balanced|quality|max]\n"
        << "                                             [--effort N] [--low-bits N]\n"
        << "radiance-codec decode input.rcodec output.exr\n"
        << "radiance-codec info input.rcodec\n";
}

int command_encode(int argc, char** argv) {
    if (argc < 4) {
        usage();
        return 2;
    }
    const fs::path input = argv[2];
    const fs::path output = argv[3];
    std::string mode = "lossless";
    std::string preset = "quality";
    int effort = -1;
    int low_bits = 12;
    for (int i = 4; i < argc; ++i) {
        const std::string arg = argv[i];
        auto take_value = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) fail(std::string("missing value for ") + flag);
            return argv[++i];
        };
        if (arg == "--mode") mode = take_value("--mode");
        else if (arg == "--preset") preset = take_value("--preset");
        else if (arg == "--effort") effort = std::stoi(take_value("--effort"));
        else if (arg == "--low-bits") low_bits = std::stoi(take_value("--low-bits"));
        else fail("unknown argument: " + arg);
    }

    const Image image = read_exr(input);
    const auto raw = image_bytes(image);
    const radiance_codec::ImageMeta meta{
        .width = image.width,
        .height = image.height,
        .channels = image.channels,
        .format = radiance_codec::PixelFormat::Float32,
    };
    const auto cfg = config_for(mode, preset, effort, low_bits);
    std::vector<std::uint8_t> encoded;
    const auto status = radiance_codec::encode(raw, meta, cfg, encoded);
    if (status != radiance_codec::Status::Ok) {
        fail("encode failed with status " + std::to_string(static_cast<int>(status)));
    }
    write_file(output, encoded);
    std::cout << "encoded " << input << " -> " << output
              << " raw=" << raw.size()
              << " encoded=" << encoded.size()
              << " ratio=" << (static_cast<double>(raw.size()) / encoded.size())
              << "x\n";
    return 0;
}

int command_decode(int argc, char** argv) {
    if (argc != 4) {
        usage();
        return 2;
    }
    const auto encoded = read_file(argv[2]);
    std::vector<std::uint8_t> raw;
    radiance_codec::ImageMeta meta;
    const auto status = radiance_codec::decode(encoded, raw, &meta);
    if (status != radiance_codec::Status::Ok) {
        fail("decode failed with status " + std::to_string(static_cast<int>(status)));
    }
    const Image image = image_from_bytes(raw, meta);
    write_exr(argv[3], image);
    std::cout << "decoded " << argv[2] << " -> " << argv[3]
              << " shape=(" << image.height << "," << image.width << ","
              << unsigned(image.channels) << ")\n";
    return 0;
}

int command_info(int argc, char** argv) {
    if (argc != 3) {
        usage();
        return 2;
    }
    const auto encoded = read_file(argv[2]);
    std::vector<std::uint8_t> raw;
    radiance_codec::ImageMeta meta;
    const auto status = radiance_codec::decode(encoded, raw, &meta);
    if (status != radiance_codec::Status::Ok) {
        fail("decode failed with status " + std::to_string(static_cast<int>(status)));
    }
    std::cout << "width=" << meta.width << "\n"
              << "height=" << meta.height << "\n"
              << "channels=" << unsigned(meta.channels) << "\n"
              << "raw_bytes=" << raw.size() << "\n"
              << "encoded_bytes=" << encoded.size() << "\n";
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            usage();
            return 2;
        }
        const std::string command = argv[1];
        if (command == "encode") return command_encode(argc, argv);
        if (command == "decode") return command_decode(argc, argv);
        if (command == "info") return command_info(argc, argv);
        usage();
        return 2;
    } catch (const std::exception& exc) {
        std::cerr << "error: " << exc.what() << "\n";
        return 1;
    }
}
