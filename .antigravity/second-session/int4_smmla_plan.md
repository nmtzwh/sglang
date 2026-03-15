# INT4 SVE SMMLA Migration Plan

## Objective
SVE does not have a native INT4 matrix multiply. However, we can achieve high performance by unpacking INT4 to INT8 on the fly (or in blocks) and leveraging the newly developed INT8 SMMLA logic (`svusmmla_s32`).

## 1. SVE Dequantization Primitives (`_dequant_weight_zp_only`)
We need to implement an SVE version of the `_dequant_weight_zp_only` function which unpacks INT4 weights to INT8 and subtracts the zero points.

```cpp
template <int64_t N, int64_t ldb>
void _dequant_weight_zp_only_sve(const uint8_t* __restrict__ B, int8_t* dqB, const int8_t* __restrict__ qzeros, int64_t K) {
  // SVE implementation of INT4 -> INT8 dequantization
  // Load INT4 packed bytes
  // Expand to 8-bit using `svlsr_n_u8` (logical shift right)
  // Bitwise AND with `0x0F` using `svand_x`
  // Subtract zero points using `svsub_x`
  // Store to dqB
}
```

## 2. Small M Path (`_dequant_gemm_accum_small_M`)
For memory-bound small $M$ shapes (like decode $M=1$), we can perform on-the-fly decompression.
We will load INT4 from `B`, unpack to `svint8_t` vectors in registers, and immediately feed them into the `svusmmla_s32` instruction alongside the broadcasted `svuint8_t` `A` vectors.

*Important:* The INT4 packing format must map smoothly into the `[K/8, N/2, 16]` SMMLA structure *after* unpacking in registers.

## 3. Large M Path (BRGEMM fallback)
For compute-bound large $M$ shapes (prefill), we will follow a similar approach to the AVX512 path:
1. Decompress an entire block of INT4 weight to an INT8 buffer (`dqB`) using the SVE `_dequant_weight_zp_only_sve`.
2. Call the highly optimized INT8 SVE SMMLA kernel on the unpacked buffer.
3. Apply standard FP32/BF16 scales and activations.

## Next Steps
1. Implement the SVE INT4 to INT8 unpacking loop (`_dequant_weight_zp_only`).
2. Adapt the INT8 SMMLA micro-kernel to accept the on-the-fly unpacked INT4 vectors for small M.