file(GLOB_RECURSE srcs CONFIGURE_DEPENDS *.cu *.cpp)
add_executable(Convolution ${srcs})
target_include_directories(Convolution PRIVATE ${CMAKE_SOURCE_DIR}/Common)
target_compile_options(Convolution PRIVATE -lineinfo)