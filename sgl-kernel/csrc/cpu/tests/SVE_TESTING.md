# SVE Functional Testing Guide

Instructions for cross-compiling and running the aarch64 SVE functional tests on an x86 host using QEMU user-mode emulation.

## Prerequisites

### Ubuntu/Debian (WSL or native Linux)

```bash
# 1. Install aarch64 cross-compiler
sudo apt-get update
sudo apt-get install -y gcc-aarch64-linux-gnu g++-aarch64-linux-gnu

# 2. Install QEMU user-mode emulation
sudo apt-get install -y qemu-user qemu-user-binfmt

# 3. Install aarch64 runtime libraries (for dynamic linking)
sudo apt-get install -y libc6-arm64-cross libstdc++-12-dev-arm64-cross
```

> [!TIP]
> If `libstdc++` version differs, adjust the package name (e.g., `libstdc++-13-dev-arm64-cross`).

## Option A: Build with CMake

```bash
cd sgl-kernel/csrc/cpu/tests

# Configure (cross-compile for aarch64)
cmake -B build \
  -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc \
  -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ \
  .

# Build
cmake --build build
```

## Option B: Build directly (no CMake)

```bash
cd sgl-kernel/csrc/cpu/tests

# Compile (static linking for easiest QEMU use)
aarch64-linux-gnu-g++ -O2 -march=armv8.6-a+sve+bf16 -static \
  -o test_sve_kernels test_sve_kernels.cpp
```

> [!IMPORTANT]
> Use `-static` to avoid needing aarch64 shared libraries at runtime. If the static build fails (e.g., glibc static not installed), use dynamic linking and set `QEMU_LD_PREFIX`.

## Running with QEMU

### Static binary (recommended)

```bash
# Run with SVE-512 (16 x fp32 lanes, matching AVX-512 width)
qemu-aarch64 -cpu max,sve512=on ./test_sve_kernels

# Run with SVE-256 (8 x fp32 lanes, verifies VL-agnostic correctness)
qemu-aarch64 -cpu max,sve256=on ./test_sve_kernels

# Run with SVE-128 (4 x fp32 lanes, minimum SVE width)
qemu-aarch64 -cpu max,sve128=on ./test_sve_kernels
```

### Dynamic binary

```bash
# Set library search path for aarch64 libs
export QEMU_LD_PREFIX=/usr/aarch64-linux-gnu

qemu-aarch64 -cpu max,sve512=on ./build/test_sve_kernels
```

## Expected Output

```
=== SVE Functional Tests ===
SVE vector length: 512 bits (16 x fp32)

[TEST] bf16 <-> fp32 conversion roundtrip
...
[TEST] GEMM with accumulate (add_C mode)

=== Results: 11 passed, 0 failed ===
```

```
=== High-Level SVE Algorithm Functional Tests ===
SVE vector length: 512 bits (16 x fp32)

[TEST] RMSNorm and GemmaRMSNorm
[TEST] SiLUAndMul and GeLUAndMul
[TEST] Rotary Embedding (RoPE)
[TEST] TopK Softmax (kernel logic)

=== Results: XX passed, 0 failed ===
```

## Test Coverage

### Low-Level Primitives (`test_sve_kernels`)

| Test | What it validates |
|---|---|
| `test_bf16_conversion` | `svcvt_bf16_f32_x` / `svcvt_f32_bf16_x` roundtrip |
| `test_reduce` | `svaddv_f32` / `svmaxv_f32` horizontal reductions |
| `test_bfdot` | `svbfdot_f32` bf16 dot product accumulation |
| `test_gemm_portable` | GEMM with VNNI-packed B (identity matrix) |
| `test_vnni_pack` | VNNI [K/2, N, 2] pack/unpack correctness |
| `test_sve_fexp` | Fast exponential polynomial (Cephes-style) |
| `test_predicated_tail` | `svwhilelt_b32` tail masking |
| `test_gemm_general` | General GEMM (non-trivial A and B) |
| `test_silu` | SiLU activation: `x / (1 + exp(-x))` |
| `test_vl_agnostic_loop` | VL-agnostic iteration pattern used in all kernels |

## Testing at Different Vector Lengths

Run all three VLs to verify VL-agnostic correctness:

```bash
for vl in 128 256 512; do
  echo "--- SVE-${vl} ---"
  qemu-aarch64 -cpu max,sve${vl}=on ./test_sve_kernels
done
```

All tests should pass at every vector length.
