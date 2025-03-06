cmake_minimum_required(VERSION 3.20)
project(cudaSamples LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -I${CMAKE_SOURCE_DIR}/Common --expt-relaxed-constexpr --expt-extended-lambda")
set(CMAKE_CUDA_ARCHITECTURES 89)

add_subdirectory(Samples)
