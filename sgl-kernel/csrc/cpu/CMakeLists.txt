cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(sgl_kernel)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python COMPONENTS Interpreter Development.Module ${SKBUILD_SABI_COMPONENT} REQUIRED)

execute_process(
    COMMAND ${Python_EXECUTABLE}
            -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_PY_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS ${TORCH_PY_PREFIX})
list(APPEND CMAKE_PREFIX_PATH ${TORCH_PY_PREFIX}/Torch)
find_package(Torch REQUIRED)

include_directories(
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_INSTALL_PREFIX}/include
    ${Python_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../csrc
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Platform-specific library directory
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(PLAT_LIB_DIR "/usr/lib/x86_64-linux-gnu")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(PLAT_LIB_DIR "/usr/lib/aarch64-linux-gnu")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "ppc64le|ppc64")
    set(PLAT_LIB_DIR "/usr/lib/powerpc64le-linux-gnu")
else()
    set(PLAT_LIB_DIR "/usr/lib/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu")
endif()
link_directories(${PLAT_LIB_DIR})

# Conda library path support
if(DEFINED ENV{CONDA_PREFIX})
    set(CONDA_LIB_DIR "$ENV{CONDA_PREFIX}/lib")
    message(STATUS "Using Conda lib dir: ${CONDA_LIB_DIR}")
    link_directories(${CONDA_LIB_DIR})
    set(CONDA_INCLUDE_DIR "$ENV{CONDA_PREFIX}/include")
    include_directories(${CONDA_INCLUDE_DIR})

    # Look for libnuma in Conda's lib directory
    find_library(NUMA_LIB numa HINTS "${CONDA_LIB_DIR}")
    if(NUMA_LIB)
        message(STATUS "Found libnuma: ${NUMA_LIB}")
    else()
        message(FATAL_ERROR "libnuma not found in Conda environment at ${CONDA_LIB_DIR}\n"
                            "Please install it using: conda install libnuma numactl\n")
    endif()
else()
    if(DEFINED ENV{VIRTUAL_ENV})
        set(VENV_LIB_DIR "$ENV{VIRTUAL_ENV}/lib")
        message(STATUS "Using venv lib dir: ${VENV_LIB_DIR}")
        link_directories(${VENV_LIB_DIR})
        set(VENV_INCLUDE_DIR "$ENV{VIRTUAL_ENV}/include")
        include_directories(${VENV_INCLUDE_DIR})
    endif()
    # Look for libnuma in system env paths
    find_library(NUMA_LIB numa)
    if(NUMA_LIB)
        message(STATUS "Found libnuma: ${NUMA_LIB}")
    else()
        message(FATAL_ERROR "libnuma not found in system environment\n"
                            "Please install it using: apt-get install libnuma numactl\n")
    endif()
endif()

file(GLOB_RECURSE SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
list(FILTER SOURCES EXCLUDE REGEX "tests/.*")

if(NOT DEFINED ENV{SGLANG_CPU_FP8_CVT_FTZ})
    set(ENV{SGLANG_CPU_FP8_CVT_FTZ} "1")
endif()

if("$ENV{SGLANG_CPU_FP8_CVT_FTZ}" STREQUAL "1")
    message(STATUS "Enabling macro: SGLANG_CPU_FP8_CVT_FTZ")
    add_compile_definitions(SGLANG_CPU_FP8_CVT_FTZ)
endif()

# Platform-specific compile options
add_compile_options(
    -O3
    -Wno-unknown-pragmas
    -fopenmp
)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    # Detect SVE + BF16 support on aarch64
    include(CheckCXXSourceCompiles)
    set(CMAKE_REQUIRED_FLAGS "-march=armv8.6-a+sve+bf16")
    check_cxx_source_compiles("
        #include <arm_sve.h>
        int main() {
            svfloat32_t a = svdup_f32(1.0f);
            svbfloat16_t b = svreinterpret_bf16_u16(svdup_u16(0));
            (void)a; (void)b;
            return 0;
        }
    " HAS_SVE_BF16)
    unset(CMAKE_REQUIRED_FLAGS)

    if(HAS_SVE_BF16)
        message(STATUS "Enabling SVE + BF16 support for aarch64")
        add_compile_definitions(SGLANG_WITH_SVE)
        if("$ENV{SGLANG_SVE512_VEC}" STREQUAL "1")
            message(STATUS "Enabling fixed-length SVE-512 Vectorized fallback")
            add_compile_definitions(SGLANG_SVE512_VEC)
            add_compile_options(-msve-vector-bits=512)
        endif()
        # Use explicit features to avoid missing fp16 on some native detections
        add_compile_options(-march=armv8.6-a+sve2+bf16+fp16)
    else()
        message(STATUS "SVE + BF16 not available, using NEON fallback")
        add_compile_options(-march=native)
    endif()
else()
    # x86_64: use -march=native to auto-enable AVX-512, etc.
    add_compile_options(-march=native)
endif()

Python_add_library(common_ops MODULE USE_SABI ${SKBUILD_SABI_VERSION} WITH_SOABI ${SOURCES})
target_link_libraries(common_ops PRIVATE ${TORCH_LIBRARIES} ${NUMA_LIB})
target_include_directories(common_ops PRIVATE ${TORCH_INCLUDE_DIRS})

install(TARGETS common_ops
    LIBRARY DESTINATION sgl_kernel/sm100
)
