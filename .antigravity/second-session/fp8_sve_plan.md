# FP8 SVE Migration Plan

## Objective
SVE lacks a native FP8 dot-product/MMA instruction. The approach will remain converting FP8 to BF16 on-the-fly and using `svbfdot_f32`, but we need to resolve existing bugs and introduce SVE-specific vectorization for the conversion.

## 1. Bug Fix in `tinygemm_kernel_nn`
The current SVE fallback in `gemm_fp8.cpp` incorrectly casts `uint8_t* B` to a `float*` and attempts to load it directly via `svld1_f32`. This reads 4 bytes per 1-byte FP8 element (out of bounds memory read) and treats the raw integer bytes as floats.
We need to change the load to `svld1_u8` and correctly convert it.

## 2. SVE On-the-fly Unpacking (`unpack_B`)
Replace the scalar FP8 $\rightarrow$ BF16 loop with SVE intrinsics. We will widen `svuint8_t` to `svuint16_t`, and use bitwise shifts (`svlsl_n_u16`, `svlsr_n_u16`) combined with `svadd_n_u16` to inject the FP8 E4M3 exponent bias into a BF16 format.

```cpp
// Example logical flow for SVE FP8 -> BF16
svuint8_t v_fp8 = svld1_u8(pg, b_ptr);
// Widen to 16-bit
svuint16_t v_u16 = svunpklo_u16(v_fp8); // (and unpkhi)
// Shift left to align mantissa/exponent to BF16 position
svuint16_t v_bf16_bits = svlsl_n_u16(v_u16, 8);
// Apply FP8 to BF16 exponent bias adjustment (add or subtract)
// svadd_n_u16...
svbfloat16_t v_bf16 = svreinterpret_bf16_u16(v_bf16_bits);
```

## 3. Micro-kernel (`svbfdot_f32`)
Once `B` is expanded to BF16 (either in registers for small `M` or in `Btmp` via large M path), use `svbfdot_f32` which computes a 2-element dot product of BF16 into an FP32 accumulator.
We will ensure the VNNI packing of B (`[K/2, N, 2]`) perfectly aligns with the 2-element structural requirement of `svbfdot`.

## Next Steps
1. Fix the bug in `gemm_fp8.cpp` SVE micro-kernel.
2. Implement SVE vectorized `unpack_B`.
3. Verify the layout alignment for `svbfdot_f32`.