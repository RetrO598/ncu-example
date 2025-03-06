file(GLOB_RECURSE srcs CONFIGURE_DEPENDS *.cu *.cpp)
add_executable(VectorAdd ${srcs})
target_include_directories(VectorAdd PRIVATE ${CMAKE_SOURCE_DIR}/Common)