# PyTorch `at::vec` API Usage in SGLang CPU Kernels

Based on the analysis of the `sgl-kernel/csrc/cpu/` codebase, the implementation heavily relies on PyTorch's ATen vectorization (`at::vec`) primitives for SIMD abstractions. Since PyTorch currently has poor support for SVE-512, any attempt to bypass or natively implement these fallbacks must account for the following APIs used throughout the kernels.

## Core Types
- `at::vec::Vectorized<T>`: The fundamental ATen vector struct.
  - SGLang uses this templated on `float`, `scalar_t` (`at::BFloat16`, `at::Half`), and `int`.

## Memory & Initialization
- `Vectorized<T>::loadu(const void* ptr)`: Unaligned loads of vectors from memory.
- `Vectorized<T>::store(void* ptr) const`: Unaligned stores of vectors back to memory.
- `Vectorized<T>(val)`: Broadcasting constructors.
- `Vectorized<T>::size()`: `constexpr` method returning the number of elements in the vector.

## Conversion & Casting
- `at::vec::convert_to_float(...)`: Used extensively (e.g. in MoE and TopK) to cast `Vectorized<scalar_t>` (like bf16 or fp16) into high/low components of `Vectorized<float>`.
- `at::vec::convert<InType, OutType>(...)`: Array-to-array explicit conversions used before reductions.

## Mathematical Operations
These are used primarily as overloaded operators and standard math functions across `Vectorized<float>` combinations:
- **Operators**: `+`, `-`, `*`, `/`, `>`, `<`, `==`
- **Math Intents**:
  - `at::vec::maximum(a, b)`: Element-wise max.
  - `at::vec::minimum(a, b)`: Element-wise min.
  - `at::vec::clamp_max(a, max_val)`: Element-wise upper clamp.
  - `at::vec::clamp_min(a, min_val)`: Element-wise lower clamp.
  - `at::vec::exp(a)`: Fast exponential computation.
  - `at::vec::log2(a)`: Base-2 logarithm.
  - `at::vec::fmadd(a, b, c)`: Fused multiply-add (`a * b + c`).
  
## Logical & Blend Operations
- `at::vec::blendv(a, b, mask)`: Selects elements from `a` or `b` depending on a vectorized logical mask operation.

## Reductions
- `at::vec::reduce_all<float>(lambda, a, size)`: Horizontally reduces elements via a lambda function (often wrapping `at::vec::maximum` or basic sum `x + y`). SGLang creates custom wrappers (like `vec_reduce_sum` and `vec_reduce_max`) in `vec.h` that dispatch to either AVX-512, SVE-agnostic loops, or this ATen fallback.
