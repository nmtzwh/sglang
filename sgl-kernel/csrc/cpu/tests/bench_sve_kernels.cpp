#include <arm_sve.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>

// ============================================================================
// Lightweight bfloat16 type
// ============================================================================
struct bf16_t {
  uint16_t val;
  bf16_t() : val(0) {}
  explicit bf16_t(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, 4);
    val = static_cast<uint16_t>(bits >> 16);
  }
  explicit operator float() const {
    uint32_t bits = static_cast<uint32_t>(val) << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
  }
};

// ============================================================================
// GCC-compatible SVE bf16 <-> fp32 helpers
// ============================================================================
static inline svfloat32_t sve_bf16_to_f32(svbool_t pg, svbfloat16_t vb) {
  svuint16_t vu16 = svreinterpret_u16(vb);
  svuint32_t vu32 = svunpklo_u32(vu16);
  vu32 = svlsl_n_u32_x(pg, vu32, 16);
  return svreinterpret_f32(vu32);
}

static inline svbfloat16_t sve_f32_to_bf16(svbool_t pg, svfloat32_t vf) {
  svuint32_t vu32 = svreinterpret_u32(vf);
  svuint16_t vu16_wide = svreinterpret_u16(svlsr_n_u32_x(pg, vu32, 16));
  svuint16_t vu16 = svuzp1_u16(vu16_wide, svdup_u16(0));
  return svreinterpret_bf16(vu16);
}

// ============================================================================
// Benchmarking Utility
// ============================================================================
struct Timer {
  std::chrono::high_resolution_clock::time_point start_time;
  Timer() { start_time = std::chrono::high_resolution_clock::now(); }
  double elapsed_sec() {
    auto end_time = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end_time - start_time).count();
  }
};

// ============================================================================
// BF16 GEMM Implementation (Register Accumulator)
// ============================================================================
template <int BLOCK_M, int BLOCK_N>
void bf16_gemm_new(
    const bf16_t* A, const bf16_t* B, bf16_t* C, const float* bias,
    int64_t K, int64_t lda, int64_t ldb, int64_t ldc) {
  const uint64_t vl_f32 = svcntw();
  const int64_t K2 = K >> 1;
  const float* a_ptr = reinterpret_cast<const float*>(A);
  const int64_t lda2 = lda >> 1;
  const int64_t ldb2 = ldb;

  for (int64_t n = 0; n < BLOCK_N; n += vl_f32) {
    svbool_t pg = svwhilelt_b32((uint32_t)n, (uint32_t)BLOCK_N);
    
    // Avoid array of SVE types for compatibility
    svfloat32_t acc0 = svdup_n_f32(0.f);
    svfloat32_t acc1 = svdup_n_f32(0.f);
    svfloat32_t acc2 = svdup_n_f32(0.f);
    svfloat32_t acc3 = svdup_n_f32(0.f);

    for (int64_t k = 0; k < K2; ++k) {
      const float* b_row = reinterpret_cast<const float*>(B) + k * ldb2;
      svbfloat16_t vb = svreinterpret_bf16(svld1_f32(pg, b_row + n));
      
      if (BLOCK_M >= 1) acc0 = svbfdot_f32(acc0, svreinterpret_bf16(svdup_f32(a_ptr[0 * lda2 + k])), vb);
      if (BLOCK_M >= 2) acc1 = svbfdot_f32(acc1, svreinterpret_bf16(svdup_f32(a_ptr[1 * lda2 + k])), vb);
      if (BLOCK_M >= 3) acc2 = svbfdot_f32(acc2, svreinterpret_bf16(svdup_f32(a_ptr[2 * lda2 + k])), vb);
      if (BLOCK_M >= 4) acc3 = svbfdot_f32(acc3, svreinterpret_bf16(svdup_f32(a_ptr[3 * lda2 + k])), vb);
    }

    if (BLOCK_M >= 1) svst1_bf16(svwhilelt_b16((uint32_t)n, (uint32_t)BLOCK_N), reinterpret_cast<bfloat16_t*>(C + 0 * ldc + n), sve_f32_to_bf16(pg, acc0));
    if (BLOCK_M >= 2) svst1_bf16(svwhilelt_b16((uint32_t)n, (uint32_t)BLOCK_N), reinterpret_cast<bfloat16_t*>(C + 1 * ldc + n), sve_f32_to_bf16(pg, acc1));
    if (BLOCK_M >= 3) svst1_bf16(svwhilelt_b16((uint32_t)n, (uint32_t)BLOCK_N), reinterpret_cast<bfloat16_t*>(C + 2 * ldc + n), sve_f32_to_bf16(pg, acc2));
    if (BLOCK_M >= 4) svst1_bf16(svwhilelt_b16((uint32_t)n, (uint32_t)BLOCK_N), reinterpret_cast<bfloat16_t*>(C + 3 * ldc + n), sve_f32_to_bf16(pg, acc3));
  }
}

// ============================================================================
// BF16 GEMM Implementation (Memory Accumulator - Old Baseline)
// ============================================================================
template <int BLOCK_M, int BLOCK_N>
void bf16_gemm_old(
    const bf16_t* A, const bf16_t* B, bf16_t* C, const float* bias,
    int64_t K, int64_t lda, int64_t ldb, int64_t ldc) {
  const uint64_t vl_f32 = svcntw();
  float acc[BLOCK_M][BLOCK_N];
  for (int m = 0; m < BLOCK_M; ++m)
    for (int n = 0; n < BLOCK_N; ++n)
      acc[m][n] = 0.f;

  const int64_t K2 = K >> 1;
  const float* a_ptr = reinterpret_cast<const float*>(A);
  const int64_t lda2 = lda >> 1;
  const int64_t ldb2 = ldb;

  for (int64_t k = 0; k < K2; ++k) {
    for (int m = 0; m < BLOCK_M; ++m) {
      float a_val = a_ptr[m * lda2 + k];
      svbfloat16_t va = svreinterpret_bf16(svdup_f32(a_val));
      const float* b_row = reinterpret_cast<const float*>(B) + k * ldb2;
      for (int64_t n = 0; n < BLOCK_N; n += vl_f32) {
        svbool_t pg = svwhilelt_b32((uint32_t)n, (uint32_t)BLOCK_N);
        svfloat32_t vc = svld1_f32(pg, acc[m] + n);
        svbfloat16_t vb = svreinterpret_bf16(svld1_f32(pg, b_row + n));
        vc = svbfdot_f32(vc, va, vb);
        svst1_f32(pg, acc[m] + n, vc);
      }
    }
  }

  for (int m = 0; m < BLOCK_M; ++m) {
    for (int64_t n = 0; n < BLOCK_N; n += vl_f32) {
      svbool_t pg = svwhilelt_b32((uint32_t)n, (uint32_t)BLOCK_N);
      svfloat32_t vf = svld1_f32(pg, acc[m] + n);
      svbfloat16_t vbf = sve_f32_to_bf16(pg, vf);
      svst1_bf16(svwhilelt_b16((uint32_t)n, (uint32_t)BLOCK_N), reinterpret_cast<bfloat16_t*>(C + m * ldc + n), vbf);
    }
  }
}

// ============================================================================
// Extended Attention Attn@V (Register Accumulator)
// ============================================================================
void attn_v_new(
    const float* A, const bf16_t* B, float* C, const int64_t* indices,
    const float* scale, int64_t lda, int64_t ldb, int64_t ldc,
    int64_t K, int64_t BLOCK_M, int64_t BLOCK_N) {
  const uint64_t vl_f32 = svcntw();
  for (int64_t n = 0; n < BLOCK_N; n += vl_f32) {
    svbool_t pg = svwhilelt_b32((uint32_t)n, (uint32_t)BLOCK_N);
    
    svfloat32_t vc0 = svld1_f32(pg, C + 0 * ldc + n);
    if (BLOCK_M >= 1) vc0 = svmul_f32_x(pg, vc0, svdup_f32(scale[0]));
    svfloat32_t vc1 = (BLOCK_M >= 2) ? svld1_f32(pg, C + 1 * ldc + n) : svdup_n_f32(0.f);
    if (BLOCK_M >= 2) vc1 = svmul_f32_x(pg, vc1, svdup_f32(scale[1]));
    svfloat32_t vc2 = (BLOCK_M >= 3) ? svld1_f32(pg, C + 2 * ldc + n) : svdup_n_f32(0.f);
    if (BLOCK_M >= 3) vc2 = svmul_f32_x(pg, vc2, svdup_f32(scale[2]));
    svfloat32_t vc3 = (BLOCK_M >= 4) ? svld1_f32(pg, C + 3 * ldc + n) : svdup_n_f32(0.f);
    if (BLOCK_M >= 4) vc3 = svmul_f32_x(pg, vc3, svdup_f32(scale[3]));

    for (int64_t k = 0; k < K; ++k) {
      int64_t b_idx = indices[k];
      svbfloat16_t vb_bf16 = svld1_bf16(
          svwhilelt_b16((uint32_t)n, (uint32_t)BLOCK_N), reinterpret_cast<const bfloat16_t*>(B + b_idx * ldb + n));
      svfloat32_t vb = sve_bf16_to_f32(pg, vb_bf16);
      if (BLOCK_M >= 1) vc0 = svmla_f32_x(pg, vc0, svdup_f32(A[0 * lda + k]), vb);
      if (BLOCK_M >= 2) vc1 = svmla_f32_x(pg, vc1, svdup_f32(A[1 * lda + k]), vb);
      if (BLOCK_M >= 3) vc2 = svmla_f32_x(pg, vc2, svdup_f32(A[2 * lda + k]), vb);
      if (BLOCK_M >= 4) vc3 = svmla_f32_x(pg, vc3, svdup_f32(A[3 * lda + k]), vb);
    }

    if (BLOCK_M >= 1) svst1_f32(pg, C + 0 * ldc + n, vc0);
    if (BLOCK_M >= 2) svst1_f32(pg, C + 1 * ldc + n, vc1);
    if (BLOCK_M >= 3) svst1_f32(pg, C + 2 * ldc + n, vc2);
    if (BLOCK_M >= 4) svst1_f32(pg, C + 3 * ldc + n, vc3);
  }
}

void attn_v_old(
    const float* A, const bf16_t* B, float* C, const int64_t* indices,
    const float* scale, int64_t lda, int64_t ldb, int64_t ldc,
    int64_t K, int64_t BLOCK_M, int64_t BLOCK_N) {
  const uint64_t vl_f32 = svcntw();
  for (int m = 0; m < BLOCK_M; ++m) {
    float s = scale[m];
    for (int64_t n = 0; n < BLOCK_N; n += vl_f32) {
      svbool_t pg = svwhilelt_b32((uint32_t)n, (uint32_t)BLOCK_N);
      svfloat32_t vc = svld1_f32(pg, C + m * ldc + n);
      vc = svmul_f32_x(pg, vc, svdup_f32(s));
      for (int64_t k = 0; k < K; ++k) {
        int64_t b_idx = indices[k];
        svbfloat16_t vb_bf16 = svld1_bf16(
            svwhilelt_b16((uint32_t)n, (uint32_t)BLOCK_N), reinterpret_cast<const bfloat16_t*>(B + b_idx * ldb + n));
        svfloat32_t vb = sve_bf16_to_f32(pg, vb_bf16);
        vc = svmla_f32_x(pg, vc, svdup_f32(A[m * lda + k]), vb);
      }
      svst1_f32(pg, C + m * ldc + n, vc);
    }
  }
}

// ============================================================================
// Benchmarking
// ============================================================================
void bench_gemm() {
  printf("--- BF16 GEMM Performance ---\n");
  const int M = 4, N = 64, K = 4096;
  const int ITERS = 100;

  std::vector<bf16_t> A(M * K, bf16_t(1.0f));
  std::vector<bf16_t> B(K * N, bf16_t(1.0f));
  std::vector<bf16_t> C(M * N);
  std::vector<float> bias(N, 0.0f);

  {
    Timer t;
    for (int i = 0; i < ITERS; i++) {
      bf16_gemm_old<M, N>(A.data(), B.data(), C.data(), bias.data(), K, K, N, N);
    }
    double sec = t.elapsed_sec();
    printf("  Old (Memory Acc): %.4f s (%.2f GFLOPS)\n", sec, (2.0 * M * N * K * ITERS) / sec / 1e9);
  }

  {
    Timer t;
    for (int i = 0; i < ITERS; i++) {
      bf16_gemm_new<M, N>(A.data(), B.data(), C.data(), bias.data(), K, K, N, N);
    }
    double sec = t.elapsed_sec();
    printf("  New (Register Acc): %.4f s (%.2f GFLOPS)\n", sec, (2.0 * M * N * K * ITERS) / sec / 1e9);
  }
}

void bench_attn_v() {
  printf("--- Extended Attention (Attn@V) Performance ---\n");
  const int M = 4, N = 128, K = 512;
  const int ITERS = 1000;

  std::vector<float> A(M * K, 1.0f);
  std::vector<bf16_t> B(K * N, bf16_t(1.0f));
  std::vector<float> C(M * N, 0.0f);
  std::vector<int64_t> indices(K);
  for (int i = 0; i < K; i++) indices[i] = i;
  std::vector<float> scale(M, 1.0f);

  {
    Timer t;
    for (int i = 0; i < ITERS; i++) {
      attn_v_old(A.data(), B.data(), C.data(), indices.data(), scale.data(), K, N, N, K, M, N);
    }
    double sec = t.elapsed_sec();
    printf("  Old (Memory Acc): %.4f s\n", sec);
  }

  {
    Timer t;
    for (int i = 0; i < ITERS; i++) {
      attn_v_new(A.data(), B.data(), C.data(), indices.data(), scale.data(), K, N, N, K, M, N);
    }
    double sec = t.elapsed_sec();
    printf("  New (Register Acc): %.4f s\n", sec);
  }
}

int main() {
  printf("SVE vector length: %lu bits\n", svcntw() * 32);
  bench_gemm();
  bench_attn_v();
  return 0;
}
