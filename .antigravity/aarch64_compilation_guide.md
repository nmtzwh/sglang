# Compiling SGLang for AArch64 (SVE) 

This guide outlines the steps required to compile the `sgl-kernel` CPU extension natively on `aarch64` hardware, ensuring that Scalable Vector Extension (SVE) optimizations are fully enabled. This is crucial for achieving high performance when running MoE models like Qwen3.5 on ARM architectures.

## Prerequisites

1.  **Hardware**: An `aarch64` machine with support for SVE and BF16 instructions (e.g., AWS Graviton 3, Graviton 4, or similar ARMv8.6+ CPUs).
2.  **OS**: A compatible Linux distribution (e.g., Ubuntu 22.04/24.04).
3.  **Python & PyTorch**: A native `linux/arm64` Python environment with PyTorch installed.

```bash
# Recommended: Create a new Conda/mamba environment or venv
conda create -n sglang python=3.10
conda activate sglang

# Install PyTorch for AArch64 CPU
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Compilation Steps

The `sgl-kernel` uses CMake to detect `aarch64` features. If SVE and BF16 are present on the hardware, `-march=native` will automatically enable the `CPU_CAPABILITY_SVE` flags during compilation.

However, to ensure Qwen3.5 MoE models run optimally, we need to explicitly enable the **Fixed-Length SVE-512 Vectorized Fallback**. While PyTorch's default `aten::vec` API supports SVE-256 and NEON, it lacks comprehensive SVE-512 support. SGLang provides a customized, high-performance fallback for 512-bit fixed-length vectorization to bridge this gap.

### 1. Set Environment Variables

Before installing the kernel, set the following environment variables. The most critical flag is `SGLANG_SVE512_VEC`, which enables the specialized high-level operations used by Qwen3.5 MoE blocks.

```bash
# [CRITICAL] Enable fixed-length SVE-512 vectorization for PyTorch aten::vec fallback
export SGLANG_SVE512_VEC=1

# [OPTIONAL] Flush-to-zero for FP8 conversions (if supported/required by your model layer)
export SGLANG_CPU_FP8_CVT_FTZ=1
```

> [!NOTE]  
> Setting `SGLANG_SVE512_VEC=1` forces the compiler to rely on strict `512-bit` SVE vector lengths. Make sure your target hardware genuinely supports or effectively emulates 512-bit registers, or you are comfortable restricting vector lengths via hardware flags.

### 2. Compile `sgl-kernel`

Navigate to the `sgl-kernel` directory and build the python package from source:

```bash
cd sgl-kernel

# Build and install the kernel (this will trigger CMake)
python setup_cpu.py install
# Alternatively: pip install -e .
```

### 3. Verification

During the build process in your terminal output, look for the following CMake status messages to confirm the SVE paths are active:

```text
-- Enabling SVE + BF16 support for aarch64
-- Enabling fixed-length SVE-512 Vectorized fallback
```

Once installed, you can verify the kernels compiled correctly by running the SGLang CPU regression tests natively:

```bash
python -m pytest test/srt/cpu/
```

## Running Qwen3.5 Models

Once the kernel is compiled with `SGLANG_SVE512_VEC=1`, standard SGLang engine invocations will automatically dispatch to your SVE-optimized routines (such as GEMM, RMSNorm, RoPE, and MoE routing).

```bash
python -m sglang.launch_server \
   --model-path Qwen/Qwen3.5-35B-A3B \
   --device cpu \
   --port 30000
```

*(Note: Ensure you allocate sufficient RAM and NUMA node configurations to handle the 32B+ MoE model sizes appropriately on your CPU server).*
