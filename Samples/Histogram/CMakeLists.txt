file(GLOB_RECURSE srcs CONFIGURE_DEPENDS *.cu *.cpp *.cuh)
add_executable(Histogram ${srcs})
target_include_directories(Histogram PRIVATE ${CMAKE_SOURCE_DIR}/Common)
target_compile_options(Histogram PRIVATE -lineinfo)