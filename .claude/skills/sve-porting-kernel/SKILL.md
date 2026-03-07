---
name: sve-porting-kernel
description: Best practices and technical patterns for extending heavily optimized AVX-512 C++ PyTorch custom kernels to ARM SVE using ACLE intrinsics.
---

# ARM SVE Porting Guide for PyTorch Kernels

This skill encapsulates the technical experience and patterns discovered while porting an optimized AVX-512 PyTorch/ATen C++ kernel application (like MoE, Attention, and GEMMs) to the ARM Scalable Vector Extension (SVE) architecture.

## 1. Core Philosophy: VL-Agnostic vs Fixed-Length

When porting to SVE, prioritize **Vector-Length (VL) Agnostic** implementations. Unlike AVX-512 where registers are strictly 512 bits, SVE registers can range from 128 to 2048 bits depending on the hardware.

### Predicated Tail Handling (The VL-Agnostic Way)
Instead of hardcoding loop unrolls based on a fixed vector size (e.g., `constexpr int VEC_ELEM = 16;`), use SVE predicates (`svbool_t`) to handle arbitrary array sizes seamlessly without separate scalar tail loops:

```cpp
int i = 0;
while (i < N) {
    // Generate a predicate for the remaining elements
    svbool_t pg = svwhilelt_b32(i, N);
    
    // Load safely relying on the predicate
    svfloat32_t va = svld1_f32(pg, a + i);
    svfloat32_t vb = svld1_f32(pg, b + i);
    
    // Compute & store
    svfloat32_t vc = svadd_f32_x(pg, va, vb);
    svst1_f32(pg, c + i, vc);
    
    // Advance by the active vector length
    i += svcntw(); 
}
```

## 2. ATen Vector API Fallbacks (SVE-512)

PyTorch's `at::vec::Vectorized<T>` API natively supports AVX-512 but lacks deep SVE-512 native optimization. When exact parity with a fixed 512-bit width is necessary (e.g., rigid mathematical structs), you can mimic the ATen API.

**To implement a compiler-switched SVE-512 fallback:**
1. Isolate `at::vec` usages using a namespace alias (e.g., `namespace sgl_vec = at::vec;`).
2. Add a fallback header mapping `Vectorized<T>` to fixed SVE types compiled only when an environment flag specifies it.
3. Use ACLE compiler directives to fix sizes: `__attribute__((arm_sve_vector_bits(512)))`.

```cpp
#if defined(__ARM_FEATURE_SVE_BITS) && __ARM_FEATURE_SVE_BITS == 512
typedef svfloat32_t fixed_svfloat32_t __attribute__((arm_sve_vector_bits(512)));

template <> struct Vectorized<float> {
  fixed_svfloat32_t vec_;
  Vectorized(fixed_svfloat32_t v) : vec_(v) {}
  
  static Vectorized<float> loadu(const void* ptr) {
    return svld1_f32(svptrue_b32(), (const float*)ptr);
  }
  // Implement operators overloading...
};
#endif
```

## 3. High-Performance GEMM Translations

Kernels optimized for Intel often rely on AMX's `brgemm`. This is platform-specific. Implement a **Portable GEMM Wrapper** to conditionally dispatch:
* **x86**: Dispatch to `at::native::cpublas::brgemm`
* **aarch64 SVE**: Dispatch to a custom inner-kernel utilizing `svbfdot_f32` (BF16 dot product matrix multiplication instruction). SVE requires VNNI packing for optimal performance (e.g., interleaving blocks of 2/4 for BF16/INT8).

## 4. Key SVE Intrinsic Mappings from AVX-512

| Feature | AVX-512 / ATen | SVE Equivalent |
|---------|---------------|----------------|
| BF16 Dot Product | `_mm512_dpbf16_ps` | `svbfdot_f32` (Use `svptrue_b32()` predicate) |
| INT8 Dot Product | `_mm512_dpbusd_epi32` | `svdot_s32` |
| BFloat16 Convert | `_mm512_cvtneps_pbh` | `svcvt_bf16_f32_x` |
| Masked Load | `_mm512_maskz_loadu_ps` | `svld1_f32(pg, ptr)` |
| Fused Add-Mul | `_mm512_fmadd_ps` | `svmla_f32_m` / `svmla_f32_x` |
| Horizontal Max | `_mm512_reduce_max_ps` | `svmaxv_f32` |
| Gather with Indices | `_mm512_i32gather_ps` | `svld1_gather_u32index_f32` |

## 5. Quick Testing Environment with QEMU & WSL

Do not rely on full PyTorch compilation cycles to test SVE logic syntax. Set up a quick mock environment:

1. **Standalone C++ Tests**: Write small functional C++ tests bypassing ATen headers, mimicking `bfloat16` and matrix loads.
2. **Mocking Headers**: For deep integrations, write dummy ATen headers `namespace at { struct BFloat16 { uint16_t x; }; }` to check syntax.
3. **Cross Compilation**: Compile with `aarch64-linux-gnu-g++ -msve-vector-bits=512 -march=armv8.6-a+sve+bf16`.
4. **QEMU Emulation**: Execute tests via `qemu-aarch64 -L /usr/aarch64-linux-gnu -cpu max,sve512=on ./test_binary`.

## 6. SVE Type Nuances
- **No Implicit Casts**: SVE strongly types its intrinsics (`_x` vs `_m` vs `_z` for unpredicated, merging, and zeroing respectively).
- **Fast Math**: Complex ops like `exp` or `tanh` require custom polynomial approximation kernels (e.g., using Cephes polynomials) because standard math libraries don't natively map these to deep SIMD intrinsics cross-platform.
- **Predicates Everywhere**: Almost every instruction takes a predicate `svbool_t`. Use `svptrue_b32()` when no masking is needed.
