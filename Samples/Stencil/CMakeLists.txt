file(GLOB_RECURSE srcs CONFIGURE_DEPENDS *.cu *.cpp *.cuh)
add_executable(Stencil ${srcs})
target_include_directories(Stencil PRIVATE ${CMAKE_SOURCE_DIR}/Common)
target_compile_options(Stencil PRIVATE -lineinfo)