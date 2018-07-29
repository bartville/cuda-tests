# CMakeLists.txt to build hellocuda.cu
cmake_minimum_required(VERSION 2.8)
find_package(CUDA QUIET REQUIRED)

include_directories(src)

add_subdirectory(src)
add_subdirectory(test)
