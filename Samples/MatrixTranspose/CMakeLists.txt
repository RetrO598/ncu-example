file(GLOB_RECURSE srcs CONFIGURE_DEPENDS *.cu *.cpp)
add_executable(MatrixTranspose ${srcs})
target_include_directories(MatrixTranspose PRIVATE ${CMAKE_SOURCE_DIR}/Common)