cmake_minimum_required(VERSION 3.20)
project(radiance_codec_app LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(radiance_codec CONFIG REQUIRED)
find_package(OpenEXR CONFIG REQUIRED)

add_executable(radiance-codec
    src/radiance_codec_cli.cpp
)

target_link_libraries(radiance-codec PRIVATE
    radiance_codec::radiance_codec
    OpenEXR::OpenEXR
)

if(APPLE)
    set_target_properties(radiance-codec PROPERTIES
        INSTALL_RPATH "@executable_path/../lib"
        BUILD_RPATH "@executable_path/../lib"
    )
endif()
