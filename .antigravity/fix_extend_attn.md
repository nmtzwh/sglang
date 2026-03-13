# CPU Attention Kernel Fixes for `test_extend.py`

This document summarizes the root causes and fixes implemented for the failing `test_extend.py` in the `sgl-kernel` CPU backend.

## 1. Incorrect `m_prime` Initialization
In `extend.cpp` and `flash_attn.cpp`, `m_prime` (a `float` array storing row-wise maximums) was improperly initialized:
```cpp
// Bug: 
fill_stub(m_prime, -std::numeric_limits<scalar_t>::infinity(), m_size);

// Fix:
fill_stub(m_prime, -std::numeric_limits<float>::infinity(), m_size);
```
Since `scalar_t` was instantiated as `c10::BFloat16` (which does not have a well-defined `infinity()` in `std::numeric_limits` on some platforms/versions), it resolved to `0.0`. Consequently, `m_prime` was initialized to `0.0` instead of `-inf`. This artificially clipped all intermediate logit maximums (`m_i`) to a minimum of `0.0`, vastly skewing the `softmax` probabilities in cases where the true query-key dot products were entirely negative.

## 2. Incorrect Causal Mask Condition
The condition to decide when to apply the causal mask was functionally flawed and tied to block sizes:
```cpp
// Bug:
if (num_keys - n <= BLOCK_N) { ... }

// Fix:
if (n + n_size > m) { ... }
```
The buggy code assumed that the causal masking would only ever happen on the very last `BLOCK_N` iteration. This is only true when `BLOCK_M` perfectly divides `num_keys` or is strictly correlated with `BLOCK_N`. For block configurations like `BLOCK_M=512, BLOCK_N=768` (which happens for seq len > 4096), this condition failed to cover all overlap blocks or tried to access memory out-of-bounds. The updated condition precisely checks if the current `n`-block intersects with the active query `m`-block.

Additionally, bounds checking was added inside the causal mask logic to strictly restrict writes to `n_size`:
```cpp
int start_col = std::max(last_col + 1, 0);
if (start_col < n_size) {
    float* row_ptr = s_i + row * BLOCK_N;
    fill_stub(row_ptr + start_col, -std::numeric_limits<float>::infinity(), n_size - start_col);
}
```

## 3. Broken VNNI Read Strides on ARM (SVE)
The portable fallback `brgemm` implementation in `sgl-kernel/csrc/cpu/gemm.h` correctly expected the key matrix to be packed in a VNNI layout. However, when decoding the values, it used the query loop bound `N` instead of the physical layout stride `ldb`:

```cpp
// Bug:
svbfloat16_t vb = svreinterpret_bf16(svld1_f32(pg, reinterpret_cast<const float*>(B) + k * N + n));

// Fix:
svbfloat16_t vb = svreinterpret_bf16(svld1_f32(pg, reinterpret_cast<const float*>(B) + k * ldb + n));
```
And similarly in the scalar loop fallback:
```cpp
// Bug:
float b_val = static_cast<float>(B[(k >> 1) * N * 2 + n * 2 + (k & 1)]);

// Fix:
float b_val = static_cast<float>(B[(k >> 1) * ldb * 2 + n * 2 + (k & 1)]);
```
Because `N` is frequently smaller than `ldb` (due to leftover block boundaries like `n_size`), using `N` as a stride read garbage memory from the VNNI array instead of the corresponding key vectors, completely corrupting the attention scores for edge blocks.

## Result
Applying these three fixes allowed `test_extend_attention` within `test/srt/cpu/test_extend.py` to pass perfectly across various sequence lengths and multi-head attention configurations.