cmake_minimum_required(VERSION 3.17)
project(triton-preprocess-cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Generates compile_commands.json for clangd/IDE support.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(TRITON_INCLUDE_DIR "/opt/tritonserver/include"
    CACHE PATH "Path to Triton server include directory")

find_library(JPEG_LIB jpeg REQUIRED)

add_library(triton_preprocess_cpp SHARED preprocess.cpp)

target_include_directories(triton_preprocess_cpp PRIVATE
    ${TRITON_INCLUDE_DIR}
)

target_link_libraries(triton_preprocess_cpp PRIVATE
    ${JPEG_LIB}
)

set_target_properties(triton_preprocess_cpp PROPERTIES
    OUTPUT_NAME "triton_preprocess_cpp"
    PREFIX "lib"
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
)
