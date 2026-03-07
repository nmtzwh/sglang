# aarch64 SVE-512 Support for SGLang CPU Kernels

## Research
- [x] Explore `sgl-kernel/csrc/cpu/` directory structure
- [x] Analyze all AVX-512 intrinsic usage patterns across files
- [x] Identify existing aarch64 support (only `aarch64/shm.h` with NEON)
- [x] Catalog all intrinsic categories: VNNI dot-product, FP8 conv, transpose, reduce, exp
- [x] Check build system (`CMakeLists.txt`) for aarch64 handling
- [x] Review brgemm/AMX usage and PyTorch integration

## Planning
- [x] Write comprehensive implementation plan for SVE-512 support
- [x] Get user review on the plan (approved: VL-agnostic SVE, QEMU testing, tinygemm fallback)

## Implementation (pending plan approval)
- [x] Phase 1: Foundation – `vec.h` SVE primitives + `CMakeLists.txt` build infra
- [x] Phase 2: Core GEMM – `gemm.cpp` SVE bf16 dot-product micro-kernels
- [x] Phase 3: Attention – `decode.cpp`, `flash_attn.h`/`.cpp`, `extend.cpp`
- [x] Phase 4: Quantized GEMM – `gemm_fp8.cpp`, `gemm_int8.cpp`, `gemm_int4.cpp`
- [x] Phase 5: MoE – `moe.cpp`, `moe_int8.cpp`, `moe_fp8.cpp`, `moe_int4.cpp`
- [x] Phase 6: Utility kernels – `norm.cpp`, `rope.cpp`, `topk.cpp`, `activation.cpp`
- [x] Phase 7: Remaining – `qkv_proj.cpp`, `mamba/conv.cpp`, `vec_pack.h`, `shm`
- [x] Phase 8: Verification
- [x] Portable GEMM wrapper – `gemm.h` `gemm_kernel_portable()` for brgemm replacement
- [x] Extend Attention SVE – `extend.cpp` brgemm → gemm_kernel_portable
- [x] FLA SVE – `fla.cpp` brgemm → gemm_kernel_portable
- [x] Functional tests – Standalone test suite via QEMU cross-compilation (`test_sve_kernels.cpp` & `SVE_TESTING.md`)
- [x] SVE-512 ATen Vector API Fallback – Implement fixed-length SVE-512 `Vectorized<T>` overrides.s
