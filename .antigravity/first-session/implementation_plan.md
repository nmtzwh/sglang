# SVE Support for `extend_attention` and `flash_linear_attention`

## Problem

Both `extend.cpp` and `fla.cpp` call `at::native::cpublas::brgemm()` **unconditionally** (not guarded by `can_use_brgemm()`). Since `brgemm` is an x86 AMX instruction, these kernels will **fail at runtime** on aarch64.

> [!CAUTION]
> Unlike `gemm.cpp`/`moe.cpp` where `tinygemm_kernel` checks `can_use_brgemm()` and falls back to SVE micro-kernels, `extend.cpp` and `fla.cpp` call `brgemm` directly. This is the highest-priority fix.

## Analysis Summary

### `extend.cpp` — Extend Attention
- **4 `brgemm` calls** — Q@K^T and S@V for both prefix and extend stages
- Uses `pack_vnni`/`pack_vnni2` from `vec_pack.h` — scalar `#else` fallback exists ✅
- Uses `flash_attn_softmax` — SVE already ported ✅
- Uses `fill_stub`/`copy_stub` via PyTorch `Vectorized<>` — works on aarch64 ✅

### `fla.cpp` — Flash Linear Attention (Gated Delta Rule)

#### `chunk_gated_delta_rule_kernel_impl` (lines 30-795)
- **8 `brgemm` calls** — Multiple GEMM operations in the chunked delta rule algorithm
- Uses `pack_vnni`/`pack_vnni2` — scalar fallback exists ✅
- Uses `at::native::utils::transpose` — architecture-independent ✅
- Uses PyTorch `Vectorized<>` for elementwise ops — works on aarch64 ✅
- Uses `vec_reduce_sum` — SVE already ported ✅

#### `fused_sigmoid_gating_delta_rule_update_kernel_impl` (lines 817-974)
- **No `brgemm` calls** — purely `Vectorized<>` based ✅
- Uses `vec_reduce_sum` — SVE already ported ✅
- **Already works on aarch64** ✅

#### `fused_gdn_gating_kernel_impl` (lines 976-1022)
- **No `brgemm` calls** — purely `Vectorized<>` + scalar ✅
- **Already works on aarch64** ✅

## Proposed Changes

### 1. Replace `brgemm` with architecture-aware GEMM wrapper

> [!IMPORTANT]
> We need a wrapper that dispatches to `brgemm` on x86 and to an SVE-based GEMM on aarch64. The wrapper handles both bf16 input→fp32 accumulation and bf16 input→bf16 output cases.

#### [MODIFY] [gemm.h](file:///wsl.localhost/Ubuntu/home/tom/workspace/sglang/sgl-kernel/csrc/cpu/gemm.h)

Add a portable `gemm_kernel()` wrapper function that:
- On x86: calls `at::native::cpublas::brgemm()` directly (existing fast path)
- On aarch64/SVE: calls our SVE `tinygemm_kernel_nn` micro-kernel with tile-blocking

```cpp
// Portable GEMM wrapper — dispatches to brgemm (x86) or SVE tinygemm (aarch64)
template <typename scalar_t>
void gemm_kernel(
    int M, int N, int K,
    int lda, int ldb, int ldc,
    bool add_C,
    const scalar_t* A,    // [M, K] in VNNI format for brgemm, raw for SVE
    const scalar_t* B,    // [K/2, N, 2] VNNI packed
    float* C);            // [M, N] fp32 output
```

---

#### [MODIFY] [extend.cpp](file:///wsl.localhost/Ubuntu/home/tom/workspace/sglang/sgl-kernel/csrc/cpu/extend.cpp)

Replace 4 `at::native::cpublas::brgemm()` calls with `gemm_kernel()` wrapper. Also replace the `brgemm_release()` call with a conditional guard.

---

#### [MODIFY] [fla.cpp](file:///wsl.localhost/Ubuntu/home/tom/workspace/sglang/sgl-kernel/csrc/cpu/mamba/fla.cpp)

Replace 8 `at::native::cpublas::brgemm()` calls with `gemm_kernel()` wrapper.

---

### 2. SVE fast path for `vec_pack.h`

#### [MODIFY] [vec_pack.h](file:///wsl.localhost/Ubuntu/home/tom/workspace/sglang/sgl-kernel/csrc/cpu/vec_pack.h)

Add `#elif defined(CPU_CAPABILITY_SVE)` path for `pack_vnni` and `pack_vnni2`. The scalar fallback works but is slow; an SVE path using `svld1`/`svst1` with stride-based gather/scatter will improve packing throughput.

## PyTorch ATen SVE-512 Vector Fallback

The user requested a compiler-enabled fallback path for PyTorch's `at::vec` API to fixed-length SVE-512 vectors, as SVE-512 is poorly supported natively.

### Architecture
1. **Toggle Switch**: We will add a CMake option `SGLANG_SVE512_VEC` to enable/disable the SVE-512 `Vectorized` fallback. When enabled, it will set `-DSGLANG_SVE512_VEC` and `-msve-vector-bits=512` during compilation.
2. **Namespace Alias**: We will replace all hardcoded occurrences of `at::vec` with `sgl_vec` across the codebase.
3. **Dispatch Header**: In `vec.h`, we will add logic:
   ```cpp
   #if defined(CPU_CAPABILITY_SVE) && defined(SGLANG_SVE512_VEC)
   #include "vec_sve512.h"
   namespace sgl_vec = sgl::vec;
   #else
   namespace sgl_vec = at::vec;
   #endif
   ```
4. **SVE-512 Implementation (`vec_sve512.h`)**: We will implement `sgl::vec::Vectorized<T>` for `float`, `at::BFloat16`, and `at::Half`. These will use ACLE fixed-length `__attribute__((arm_sve_vector_bits(512)))` types implicitly or raw `float arr[16]` loaded via `svld1_f32(svptrue_b32(), arr)`.

### Verification Plan
- Compile the SGLang CPU core with SGLANG_SVE512_VEC enabled.
- Verify our test suite works seamlessly using the fallback vector types.

### Build Test
```bash
# Cross-compile for aarch64
cmake -DCMAKE_TOOLCHAIN_FILE=... -DCMAKE_SYSTEM_PROCESSOR=aarch64 ..
make -j
```

### Functional Test  
```bash
# QEMU user-mode emulation
qemu-aarch64 -cpu max,sve512=on ./test_extend_attention
qemu-aarch64 -cpu max,sve512=on ./test_fla
```
