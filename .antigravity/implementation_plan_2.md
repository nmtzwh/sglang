# SVE Support for [extend_attention](file://wsl.localhost/Ubuntu/home/tom/workspace/sglang/sgl-kernel/csrc/cpu/extend.cpp#310-433) and `flash_linear_attention`

## Problem

Both [extend.cpp](file://wsl.localhost/Ubuntu/home/tom/workspace/sglang/sgl-kernel/csrc/cpu/extend.cpp) and [fla.cpp](file://wsl.localhost/Ubuntu/home/tom/workspace/sglang/sgl-kernel/csrc/cpu/mamba/fla.cpp) call `at::native::cpublas::brgemm()` **unconditionally** (not guarded by [can_use_brgemm()](file://wsl.localhost/Ubuntu/home/tom/workspace/sglang/sgl-kernel/csrc/cpu/gemm.h#19-24)). Since [brgemm](file://wsl.localhost/Ubuntu/home/tom/workspace/sglang/sgl-kernel/csrc/cpu/gemm_fp8.cpp#239-259) is an x86 AMX instruction, these kernels will **fail at runtime** on aarch64.

> [!CAUTION]
> Unlike [gemm.cpp](file://wsl.localhost/Ubuntu/home/tom/workspace/sglang/sgl-kernel/csrc/cpu/gemm.cpp)/[moe.cpp](file://wsl.localhost/Ubuntu/home/tom/workspace/sglang/sgl-kernel/csrc/cpu/moe.cpp) where [tinygemm_kernel](file://wsl.localhost/Ubuntu/home/tom/workspace/sglang/sgl-kernel/csrc/cpu/gemm_int8.cpp#344-362) checks [can_use_brgemm()](file://wsl.localhost/Ubuntu/home/tom/workspace/sglang/sgl-kernel/csrc/cpu/gemm.h#19-24) and falls back to SVE micro-kernels, [extend.cpp](file://wsl.localhost/Ubuntu/home/tom/workspace/sglang/sgl-kernel/csrc/cpu/extend.cpp) and [fla.cpp](file://wsl.localhost/Ubuntu/home/tom/workspace/sglang/sgl-kernel/csrc/cpu/mamba/fla.cpp) call [brgemm](file://wsl.localhost/Ubuntu/home/tom/workspace/sglang/sgl-kernel/csrc/cpu/gemm_fp8.cpp#239-259) directly. This is the highest-priority fix.

## Analysis Summary

### [extend.cpp](file://wsl.localhost/Ubuntu/home/tom/workspace/sglang/sgl-kernel/csrc/cpu/extend.cpp) — Extend Attention
- **4 [brgemm](file://wsl.localhost/Ubuntu/home/tom/workspace/sglang/sgl-kernel/csrc/cpu/gemm_fp8.cpp#239-259) calls** — Q@K^T and S@V for both prefix and extend stages
- Uses [pack_vnni](file://wsl.localhost/Ubuntu/home/tom/workspace/sglang/sgl-kernel/csrc/cpu/vec_pack.h#198-202)/[pack_vnni2](file://wsl.localhost/Ubuntu/home/tom/workspace/sglang/sgl-kernel/csrc/cpu/vec_pack.h#216-275) from [vec_pack.h](file://wsl.localhost/Ubuntu/home/tom/workspace/sglang/sgl-kernel/csrc/cpu/vec_pack.h) — scalar `#else` fallback exists ✅
- Uses [flash_attn_softmax](file://wsl.localhost/Ubuntu/home/tom/workspace/sglang/sgl-kernel/csrc/cpu/flash_attn.h#91-143) — SVE already ported ✅
- Uses [fill_stub](file://wsl.localhost/Ubuntu/home/tom/workspace/sglang/sgl-kernel/csrc/cpu/mamba/fla.cpp#15-29)/[copy_stub](file://wsl.localhost/Ubuntu/home/tom/workspace/sglang/sgl-kernel/csrc/cpu/gemm_int4.cpp#740-753) via PyTorch `Vectorized<>` — works on aarch64 ✅

### [fla.cpp](file://wsl.localhost/Ubuntu/home/tom/workspace/sglang/sgl-kernel/csrc/cpu/mamba/fla.cpp) — Flash Linear Attention (Gated Delta Rule)

#### [chunk_gated_delta_rule_kernel_impl](file://wsl.localhost/Ubuntu/home/tom/workspace/sglang/sgl-kernel/csrc/cpu/mamba/fla.cpp#30-796) (lines 30-795)
- **8 [brgemm](file://wsl.localhost/Ubuntu/home/tom/workspace/sglang/sgl-kernel/csrc/cpu/gemm_fp8.cpp#239-259) calls** — Multiple GEMM operations in the chunked delta rule algorithm
- Uses [pack_vnni](file://wsl.localhost/Ubuntu/home/tom/workspace/sglang/sgl-kernel/csrc/cpu/vec_pack.h#198-202)/[pack_vnni2](file://wsl.localhost/Ubuntu/home/tom/workspace/sglang/sgl-kernel/csrc/cpu/vec_pack.h#216-275) — scalar fallback exists ✅
- Uses `at::native::utils::transpose` — architecture-independent ✅
- Uses PyTorch `Vectorized<>` for elementwise ops — works on aarch64 ✅
- Uses [vec_reduce_sum](file://wsl.localhost/Ubuntu/home/tom/workspace/sglang/sgl-kernel/csrc/cpu/vec.h#319-322) — SVE already ported ✅

#### [fused_sigmoid_gating_delta_rule_update_kernel_impl](file://wsl.localhost/Ubuntu/home/tom/workspace/sglang/sgl-kernel/csrc/cpu/mamba/fla.cpp#817-975) (lines 817-974)
- **No [brgemm](file://wsl.localhost/Ubuntu/home/tom/workspace/sglang/sgl-kernel/csrc/cpu/gemm_fp8.cpp#239-259) calls** — purely `Vectorized<>` based ✅
- Uses [vec_reduce_sum](file://wsl.localhost/Ubuntu/home/tom/workspace/sglang/sgl-kernel/csrc/cpu/vec.h#319-322) — SVE already ported ✅
- **Already works on aarch64** ✅

#### [fused_gdn_gating_kernel_impl](file://wsl.localhost/Ubuntu/home/tom/workspace/sglang/sgl-kernel/csrc/cpu/mamba/fla.cpp#976-1023) (lines 976-1022)
- **No [brgemm](file://wsl.localhost/Ubuntu/home/tom/workspace/sglang/sgl-kernel/csrc/cpu/gemm_fp8.cpp#239-259) calls** — purely `Vectorized<>` + scalar ✅
- **Already works on aarch64** ✅

## Proposed Changes

### 1. Replace [brgemm](file://wsl.localhost/Ubuntu/home/tom/workspace/sglang/sgl-kernel/csrc/cpu/gemm_fp8.cpp#239-259) with architecture-aware GEMM wrapper

> [!IMPORTANT]
> We need a wrapper that dispatches to [brgemm](file://wsl.localhost/Ubuntu/home/tom/workspace/sglang/sgl-kernel/csrc/cpu/gemm_fp8.cpp#239-259) on x86 and to an SVE-based GEMM on aarch64. The wrapper handles both bf16 input→fp32 accumulation and bf16 input→bf16 output cases.

#### [MODIFY] [gemm.h](file:///wsl.localhost/Ubuntu/home/tom/workspace/sglang/sgl-kernel/csrc/cpu/gemm.h)

Add a portable [gemm_kernel()](file://wsl.localhost/Ubuntu/home/tom/workspace/sglang/sgl-kernel/csrc/cpu/gemm_int8.cpp#344-362) wrapper function that:
- On x86: calls `at::native::cpublas::brgemm()` directly (existing fast path)
- On aarch64/SVE: calls our SVE [tinygemm_kernel_nn](file://wsl.localhost/Ubuntu/home/tom/workspace/sglang/sgl-kernel/csrc/cpu/decode.cpp#478-494) micro-kernel with tile-blocking

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

Replace 4 `at::native::cpublas::brgemm()` calls with [gemm_kernel()](file://wsl.localhost/Ubuntu/home/tom/workspace/sglang/sgl-kernel/csrc/cpu/gemm_int8.cpp#344-362) wrapper. Also replace the `brgemm_release()` call with a conditional guard.

---

#### [MODIFY] [fla.cpp](file:///wsl.localhost/Ubuntu/home/tom/workspace/sglang/sgl-kernel/csrc/cpu/mamba/fla.cpp)

Replace 8 `at::native::cpublas::brgemm()` calls with [gemm_kernel()](file://wsl.localhost/Ubuntu/home/tom/workspace/sglang/sgl-kernel/csrc/cpu/gemm_int8.cpp#344-362) wrapper.

---

### 2. SVE fast path for [vec_pack.h](file://wsl.localhost/Ubuntu/home/tom/workspace/sglang/sgl-kernel/csrc/cpu/vec_pack.h)

#### [MODIFY] [vec_pack.h](file:///wsl.localhost/Ubuntu/home/tom/workspace/sglang/sgl-kernel/csrc/cpu/vec_pack.h)

Add `#elif defined(CPU_CAPABILITY_SVE)` path for [pack_vnni](file://wsl.localhost/Ubuntu/home/tom/workspace/sglang/sgl-kernel/csrc/cpu/vec_pack.h#198-202) and [pack_vnni2](file://wsl.localhost/Ubuntu/home/tom/workspace/sglang/sgl-kernel/csrc/cpu/vec_pack.h#216-275). The scalar fallback works but is slow; an SVE path using `svld1`/`svst1` with stride-based gather/scatter will improve packing throughput.

## Verification Plan

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
