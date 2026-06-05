cmake_minimum_required(VERSION 3.20)
project(radiance_codec
    VERSION 0.0.1
    DESCRIPTION "Lossless float32 HDR image compression — research codec"
    LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Build configuration default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ─── core library (header-only consumer; shared lib for FFI) ───
add_library(radiance_codec SHARED
    src/codec.cpp
    src/pipeline.cpp
    src/passthrough.cpp
    src/rans.cpp
    src/rans_binary.cpp
    src/rans_models.cpp
    src/bitshuffle.cpp
    src/predictor.cpp
    src/color_transform.cpp
    src/structural_context.cpp
    src/grouped_delta.cpp
    src/mantissa_quantize.cpp
    src/linear_index.cpp
    src/linear_index_transform.cpp
    src/near_lossless_router.cpp
    src/c_api.cpp
)
find_path(ZSTD_INCLUDE_DIR zstd.h)
find_library(ZSTD_LIBRARY zstd)
target_include_directories(radiance_codec
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)
if(ZSTD_INCLUDE_DIR AND ZSTD_LIBRARY)
    target_include_directories(radiance_codec PRIVATE ${ZSTD_INCLUDE_DIR})
    target_link_libraries(radiance_codec PRIVATE ${ZSTD_LIBRARY})
    target_compile_definitions(radiance_codec PRIVATE RADIANCE_CODEC_HAS_ZSTD=1)
endif()
set_target_properties(radiance_codec PROPERTIES
    OUTPUT_NAME "radiance_codec"
    VERSION ${PROJECT_VERSION}
)

# ─── round-trip test executable ───
add_executable(test_codec
    tests/test_passthrough.cpp
)
target_link_libraries(test_codec PRIVATE radiance_codec)

add_executable(test_rans
    tests/test_rans.cpp
)
target_link_libraries(test_rans PRIVATE radiance_codec)

add_executable(test_rans_binary
    tests/test_rans_binary.cpp
)
target_link_libraries(test_rans_binary PRIVATE radiance_codec)
target_include_directories(test_rans_binary PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

add_executable(test_structural_context
    tests/test_structural_context.cpp
)
target_link_libraries(test_structural_context PRIVATE radiance_codec)

add_executable(test_grouped_delta
    tests/test_grouped_delta.cpp
)
target_link_libraries(test_grouped_delta PRIVATE radiance_codec)

add_executable(test_bitshuffle
    tests/test_bitshuffle.cpp
)
target_link_libraries(test_bitshuffle PRIVATE radiance_codec)

add_executable(test_predictor
    tests/test_predictor.cpp
)
target_link_libraries(test_predictor PRIVATE radiance_codec)

add_executable(test_near_lossless_router
    tests/test_near_lossless_router.cpp
)
target_link_libraries(test_near_lossless_router PRIVATE radiance_codec)
target_include_directories(test_near_lossless_router PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# ─── install ───
install(TARGETS radiance_codec
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(DIRECTORY include/ DESTINATION include)
