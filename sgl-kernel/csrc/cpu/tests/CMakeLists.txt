# CMakeLists.txt for SVE functional tests
# Cross-compile: cmake -DCMAKE_TOOLCHAIN_FILE=aarch64-toolchain.cmake ..
# Or directly:   cmake -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc \
#                      -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ ..
cmake_minimum_required(VERSION 3.16)
project(sglang_sve_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# aarch64 SVE+BF16 flags
set(SVE_FLAGS "-march=armv8.6-a+sve+bf16")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SVE_FLAGS} -O2")

add_executable(test_sve_kernels test_sve_kernels.cpp)
