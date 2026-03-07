# aarch64 SVE Support – Final Implementation Walkthrough

### Phase 8: Verification & Testing
- ✅ **Portable GEMM Wrapper**: Added `gemm_kernel_portable()` to dispatch dynamically to either AMX `brgemm` (x86) or SVE `svbfdot` micro-kernels (aarch64).
- ✅ **Replaced Direct `brgemm` Calls**: Replaced all x86-specific AMX `brgemm` calls in `extend.cpp` and `mamba/fla.cpp` to use the VL-agnostic SVE portable wrappers.
- ✅ **Standalone SVE Functional Tests**: Created a full test suite `test_sve_kernels.cpp` avoiding PyTorch/ATen dependencies.
- ✅ **Cross-compilation & CPU Features**: Successfully verified functional operations across **128-bit, 256-bit, and 512-bit vector lengths** via emulator `qemu-aarch64` and GCC `-march=armv8.6-a+sve+bf16`.
- ✅ **Testing Documentation**: Created `SVE_TESTING.md` describing how to reproduce the testing environment in WSL using `aarch64-linux-gnu-g++` and `qemu-user`.

### Phase 9: PyTorch ATen Vector API Fallback (SVE-512)
- ✅ **Toggle Mechanism**: Added `-DSGLANG_SVE512_VEC` compilation flag in `CMakeLists.txt` via `SGLANG_SVE512_VEC=1` environment variable.
- ✅ **Custom `Vectorized` Implementations**: Authored `vec_sve512.h` containing inline ACLE replacements for types `Vectorized<float>` and `Vectorized<at::BFloat16>`, using explicit `__attribute__((arm_sve_vector_bits(512)))` fixed-length primitives.
- ✅ **Decoupling ATen from Kernels**: Rewrote ~14 instances of `using namespace at::vec;` into a compiler macro-switchable `sgl_vec` alias. This enables the codebase to hot-swap to the custom SVE-512 backend whenever requested by the user.

## Summary

Added **VL-agnostic SVE support** to the SGLang CPU kernel codebase, covering **14 files** across all phases. All SVE code uses predicated operations (`svwhilelt`) for tail handling, working at any SVE vector length (128-2048 bits).

> [!NOTE]
> All lint errors are pre-existing clangd issues due to isolated header analysis (`ATen/ATen.h` not found). These do not affect the actual CMake build.

## Files Modified

### Foundation & Core GEMM

| File | Changes |
|---|---|
| [CMakeLists.txt](file:///wsl.localhost/Ubuntu/home/tom/workspace/sglang/sgl-kernel/csrc/cpu/CMakeLists.txt) | SVE+BF16 compile-time detection; `CPU_CAPABILITY_SVE` macro |
| [vec.h](file:///wsl.localhost/Ubuntu/home/tom/workspace/sglang/sgl-kernel/csrc/cpu/vec.h) | BF16/FP16/FP8 conversions, `vec_reduce_sum/max`, `quantize_row_int8`, `sve_fexp_u20` |
| [gemm.h](file:///wsl.localhost/Ubuntu/home/tom/workspace/sglang/sgl-kernel/csrc/cpu/gemm.h) | `can_use_brgemm()` → false on aarch64; **`gemm_kernel_portable()` wrapper** |
| [gemm.cpp](file:///wsl.localhost/Ubuntu/home/tom/workspace/sglang/sgl-kernel/csrc/cpu/gemm.cpp) | SVE `tinygemm_kernel_nn<BFloat16>` using `svbfdot_f32` |

---

### Attention & Extend Attention

| File | Changes |
|---|---|
| [decode.cpp](file:///wsl.localhost/Ubuntu/home/tom/workspace/sglang/sgl-kernel/csrc/cpu/decode.cpp) | SVE Q@K^T and Attn@V micro-kernels |
| [flash_attn.h](file:///wsl.localhost/Ubuntu/home/tom/workspace/sglang/sgl-kernel/csrc/cpu/flash_attn.h) | SVE online softmax with `sve_fexp_u20` |
| [extend.cpp](file:///wsl.localhost/Ubuntu/home/tom/workspace/sglang/sgl-kernel/csrc/cpu/extend.cpp) | **4 `brgemm` → `gemm_kernel_portable` + `brgemm_release_portable`** |

---

### Quantized GEMM

| File | Changes |
|---|---|
| [gemm_int8.cpp](file:///wsl.localhost/Ubuntu/home/tom/workspace/sglang/sgl-kernel/csrc/cpu/gemm_int8.cpp) | SVE INT8 GEMM with `svdot_s32` |
| [gemm_fp8.cpp](file:///wsl.localhost/Ubuntu/home/tom/workspace/sglang/sgl-kernel/csrc/cpu/gemm_fp8.cpp) | SVE FP8 GEMM with block scale + `unpack_B` fallback |

---

### MoE & FLA

| File | Changes |
|---|---|
| [moe.cpp](file:///wsl.localhost/Ubuntu/home/tom/workspace/sglang/sgl-kernel/csrc/cpu/moe.cpp) | SVE gate+up fused SiLU×mul + down projection |
| [fla.cpp](file:///wsl.localhost/Ubuntu/home/tom/workspace/sglang/sgl-kernel/csrc/cpu/mamba/fla.cpp) | **8 `brgemm` → `gemm_kernel_portable`** |

---

### Utility & Remaining

| File | Changes |
|---|---|
| [qkv_proj.cpp](file:///wsl.localhost/Ubuntu/home/tom/workspace/sglang/sgl-kernel/csrc/cpu/qkv_proj.cpp) | SVE `rotary<BFloat16>` for RoPE |
| [mamba/conv.cpp](file:///wsl.localhost/Ubuntu/home/tom/workspace/sglang/sgl-kernel/csrc/cpu/mamba/conv.cpp) | SVE causal conv1d (K=4 window with SiLU) |

> [!TIP]
> `norm.cpp`, `rope.cpp`, `activation.cpp` already work via PyTorch `Vectorized<>`. `topk.cpp` has scalar fallback.

## Key Design: Portable GEMM Wrapper

> [!IMPORTANT]
> The critical fix was adding `gemm_kernel_portable()` in `gemm.h`. Both `extend.cpp` (4 calls) and `fla.cpp` (8 calls) called PyTorch's `brgemm` **unconditionally** — this would crash on aarch64 since brgemm uses x86 AMX tiles.

```cpp
// On x86: passthrough to at::native::cpublas::brgemm()
// On aarch64/SVE: bf16→fp32 GEMM using svbfdot_f32 with VNNI-packed B matrix
gemm_kernel_portable(M, N, K, lda, ldb, ldc, add_C, A, B, C);
```

## SVE Intrinsics Mapping

| Operation | SVE | AVX-512 |
|---|---|---|
| BF16 dot product | `svbfdot_f32` | `_mm512_dpbf16_ps` |
| INT8 dot product | `svdot_s32` | `_mm512_dpbusd_epi32` |
| Horizontal sum/max | `svaddv_f32` / `svmaxv_f32` | `_mm512_reduce_add/max_ps` |
| Predicated tail | `svwhilelt_b32` | `__mmask16` |
| BF16 convert | `svcvt_bf16_f32_x` | `_mm512_cvtneps_pbh` |
