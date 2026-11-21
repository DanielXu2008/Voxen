cmake_minimum_required(VERSION 3.24)
project(Voxen LANGUAGES CXX CUDA)

# Language standards
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build GPU code for the P620 (compute capability 6.1)
set(CMAKE_CUDA_ARCHITECTURES 61)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")

# Smoke tests
add_executable(hello smoke/hello.cpp)
add_executable(hello_cuda smoke/hello.cu)
add_executable(gen_tone tools/gen_tone.cpp)
