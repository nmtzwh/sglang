# INT8 SVE SMMLA Migration Plan

## Objective
Migrate the `gemm_int8.cpp` micro-kernel from its current `svdot_s32` implementation (which computes a 1D dot product 4 elements at a time) to the high-throughput `svusmmla_s32` instruction.

`svusmmla_s32` (Unsigned by Signed Matrix Multiply-Accumulate) computes a $2 \times 2$ block of 32-bit integers from a $2 \times 8$ block of unsigned 8-bit integers (Matrix A) and an $8 \times 2$ block of signed 8-bit integers (Matrix B).

## 1. Weight Packing Format Update
The SMMLA instruction expects Matrix B to be an $8 \times 2$ matrix. Currently, `pack_vnni<int8_t>` in `gemm.cpp` packs `int8_t` weights for AVX512. We need to define an SVE SMMLA specific packing format: `[K/8, N/2, 16]`.

```cpp
template <>
inline void pack_vnni<int8_t>(int8_t* __restrict__ packed, const int8_t* __restrict__ weight, int N, int K) {
#if defined(CPU_CAPABILITY_SVE)
  // SVE SMMLA expects 8x2 blocks (16 bytes)
  const int VNNI_ROW = 8;
  const int VNNI_COL = 2;
  for (int n = 0; n < N / VNNI_COL; ++n) {
    for (int k = 0; k < K / VNNI_ROW; ++k) {
      for (int r = 0; r < VNNI_ROW; ++r) {
        for (int c = 0; c < VNNI_COL; ++c) {
          // Dest: [K/8, N/2, 8, 2] -> linear 16 byte blocks
          packed[k * (N/2) * 16 + n * 16 + r * 2 + c] = weight[(n * 2 + c) * K + (k * 8 + r)];
        }
      }
    }
  }
  // (Don't forget to handle compensation similarly, or retain existing s8s8 compensation logic)
#else
  // ... existing AVX512 VNNI packing (4x4 or 4x2 depending on instruction) ...
#endif
}
```

## 2. The Compute Loop ($M > 1$)
For standard prefill operations where we have multiple rows of `A`:
We load a $2 \times 8$ tile from `A` and broadcast it to all 128-bit segments of an SVE vector using `svld1rq_u8`.

```cpp
// Example: M=2, loading 8 elements for row 0 and 8 elements for row 1
// We need A to look like: [a00, a01.. a07, a10, a11.. a17] (16 bytes)
svuint8_t va = svld1rq_u8(svptrue_b8(), a_ptr + k); // Assumes A is interleaved for SMMLA, or we load and interleave on the fly.
```
*Note: Since A is dynamically quantized per token (row), it's typically stored row-major `[M, K]`. Loading a $2 \times 8$ block requires reading 8 bytes from `row0` and 8 bytes from `row1` and combining them into a 128-bit vector. SVE `svtrn1`/`svtrn2` or `svtbl` might be needed if A is not pre-interleaved.*

## 3. The Decode Fast-Path ($M = 1$)
For autoregressive decode, `M=1`. We only have 8 bytes of `A` per $K$-step. We can broadcast this $1 \times 8$ chunk to both the top and bottom 64-bits of the 128-bit segment.

```cpp
// 1. Broadcast A (8 bytes) to every 64-bit lane
uint64_t a_val = *reinterpret_cast<const uint64_t*>(a_ptr + k); // 8 bytes of A
svuint8_t va = svreinterpret_u8_u64(svdup_n_u64(a_val));

// 2. Load B (8x2 block per 128-bit segment)
svint8_t vb = svld1_s8(pg, b_ptr + n); // loads [K/8, N/2, 16] packed layout

// 3. Compute 2x2 matrix
acc0 = svusmmla_s32(acc0, va, vb);

// ... after K loop ...
// 4. Extract valid results (since Row0 and Row1 of A were identical, we computed everything twice)
// Accumulator holds [c0, c1, c0, c1, c2, c3, c2, c3] per 256 bits.
// Unzip even 64-bit elements:
svint32_t c_valid = svreinterpret_s32_s64(
    svuzp1_s64(svreinterpret_s64_s32(acc0), svreinterpret_s64_s32(acc0))
);
// c_valid is now [c0, c1, c2, c3]
```

## Next Steps
1. Implement SMMLA specific `pack_vnni` for SVE.
2. Draft the $M=1$ micro-kernel in `gemm_int8.cpp`.
3. Draft the $M>1$ (e.g. $M=2, M=4$) micro-kernels interleaving `A` rows dynamically.