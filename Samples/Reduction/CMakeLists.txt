file(GLOB_RECURSE srcs CONFIGURE_DEPENDS *.cu *.cpp *.cuh)
add_executable(Reduction ${srcs})
target_include_directories(Reduction PRIVATE ${CMAKE_SOURCE_DIR}/Common)
target_compile_options(Reduction PRIVATE -lineinfo)