file(GLOB_RECURSE srcs CONFIGURE_DEPENDS *.cu *.cpp *.cuh)
add_executable(GEMM ${srcs})
target_include_directories(GEMM PRIVATE ${CMAKE_SOURCE_DIR}/Common)
target_compile_options(GEMM PRIVATE -lineinfo)