# High-Level SVE Op Testing Plan

## Goal

Add standalone SVE functional tests for the core algorithm patterns used by the sgl-kernel high-level CPU ops, following the same cross-compile + QEMU strategy as the existing [test_sve_kernels.cpp](file:///home/tom/workspace/sglang/sgl-kernel/csrc/cpu/tests/test_sve_kernels.cpp).

> [!IMPORTANT]
> The actual kernel functions (e.g., `rmsnorm_cpu`, `silu_and_mul_cpu`) depend on PyTorch ATen and cannot be compiled standalone.
> Instead, we reimplement the **core vectorized algorithm** from each kernel using raw SVE intrinsics, and compare against a scalar reference.

## Proposed Changes

### Testing Component

#### [NEW] [test_sve_highlevel_ops.cpp](file:///home/tom/workspace/sglang/sgl-kernel/csrc/cpu/tests/test_sve_highlevel_ops.cpp)

Standalone C++ tests covering the **algorithm patterns** from these kernels:

| Test | Kernel Source | Algorithm Tested |
|------|-------------|-----------------|
| `test_rmsnorm` | `norm.cpp` | VL-agnostic variance+normalize loop |
| `test_fused_add_rmsnorm` | `norm.cpp` | Residual add + RMSNorm (fused) |
| `test_gemma_rmsnorm` | `norm.cpp` | RMSNorm with `weight + 1` (Gemma variant) |
| `test_silu_and_mul` | `activation.cpp` | Split input, SiLU(x) * y |
| `test_gelu_tanh_and_mul` | `activation.cpp` | Split input, GELU_tanh(x) * y |
| `test_gelu_and_mul` | `activation.cpp` | Split input, GELU(x) * y |
| `test_rotary_embedding` | `rope.cpp` | cos/sin rotation of query/key heads |
| `test_topk_softmax` | `topk.cpp` | Softmax + top-K selection |
| `test_fused_rmsnorm_gated` | `norm.cpp` | RMSNorm(x) * SiLU(gate) |

Each test: (1) generates deterministic input data, (2) runs the SVE-vectorized kernel implementation, (3) compares against a scalar reference, (4) reports pass/fail.

#### [MODIFY] [CMakeLists.txt](file:///home/tom/workspace/sglang/sgl-kernel/csrc/cpu/tests/CMakeLists.txt)

Add `test_sve_highlevel_ops` as a second test executable.

#### [NEW] [SVE_TESTING.md](file:///home/tom/workspace/sglang/sgl-kernel/csrc/cpu/tests/SVE_TESTING.md) (modify)

Add section documenting the high-level op tests.

## Verification Plan

### Automated Tests

Build and run with QEMU at SVE-512, SVE-256, and SVE-128:

```bash
cd sgl-kernel/csrc/cpu/tests

# Build
aarch64-linux-gnu-g++ -O2 -march=armv8.6-a+sve+bf16 -static \
  -o test_sve_highlevel_ops test_sve_highlevel_ops.cpp

# Run at all VLs
for vl in 128 256 512; do
  echo "--- SVE-${vl} ---"
  qemu-aarch64 -cpu max,sve${vl}=on ./test_sve_highlevel_ops
done
```

All tests should pass at every vector length.
