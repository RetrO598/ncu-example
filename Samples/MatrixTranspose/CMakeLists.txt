file(GLOB_RECURSE srcs CONFIGURE_DEPENDS *.cu *.cpp *.cuh)
add_executable(MatrixTranspose ${srcs})
target_include_directories(MatrixTranspose PRIVATE ${CMAKE_SOURCE_DIR}/Common)
target_compile_options(MatrixTranspose PRIVATE -lineinfo)