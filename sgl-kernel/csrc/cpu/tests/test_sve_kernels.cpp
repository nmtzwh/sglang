// Standalone functional tests for SVE aarch64 kernels
// Build: aarch64-linux-gnu-g++ -O2 -march=armv8.6-a+sve+bf16 -static \
//        -o test_sve_kernels test_sve_kernels.cpp
// Run:   qemu-aarch64 -cpu max,sve512=on ./test_sve_kernels
//
// No PyTorch dependency — uses lightweight bf16 helpers.
// Compatible with GCC 13+ (avoids svcvt_f32_bf16_x which is not in GCC).

#include <arm_sve.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

// ============================================================================
// Lightweight bfloat16 type (no ATen dependency)
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
// GCC 13 does not have svcvt_f32_bf16_x / svcvt_bf16_f32_x.
// We do the conversion manually via uint16/uint32 bit manipulation.
// ============================================================================

// bf16 vector -> fp32 vector (widen lower half)
// bf16 is just fp32 with the bottom 16 bits zeroed.
static inline svfloat32_t sve_bf16_to_f32(svbool_t pg, svbfloat16_t vb) {
  svuint16_t vu16 = svreinterpret_u16(vb);
  // Unpack lower half of u16 to u32 (zero-extends)
  svuint32_t vu32 = svunpklo_u32(vu16);
  // Shift left 16 to place bf16 bits in the upper 16 bits of fp32
  vu32 = svlsl_n_u32_x(pg, vu32, 16);
  return svreinterpret_f32(vu32);
}

// fp32 vector -> bf16 vector (truncate, store into lower half)
// This is a simple truncation (no rounding).
static inline svbfloat16_t sve_f32_to_bf16(svbool_t pg, svfloat32_t vf) {
  svuint32_t vu32 = svreinterpret_u32(vf);
  // Shift right 16 to get upper 16 bits (bf16 bits)
  svuint16_t vu16_wide = svreinterpret_u16(svlsr_n_u32_x(pg, vu32, 16));
  // Pack even-indexed u16 elements (interleave with zeros)
  svuint16_t vu16 = svuzp1_u16(vu16_wide, svdup_u16(0));
  return svreinterpret_bf16(vu16);
}

// Store bf16 elements using a scalar loop (avoids predicated bf16 store issues)
static inline void store_bf16_from_f32(float* src, bfloat16_t* dst, int n) {
  for (int i = 0; i < n; i++) {
    bf16_t v(src[i]);
    std::memcpy(&dst[i], &v.val, 2);
  }
}

// Load bf16 elements to fp32 using a scalar loop
static inline void load_bf16_to_f32(const bfloat16_t* src, float* dst, int n) {
  for (int i = 0; i < n; i++) {
    bf16_t v;
    std::memcpy(&v.val, &src[i], 2);
    dst[i] = (float)v;
  }
}

// ============================================================================
// Test infrastructure
// ============================================================================
static int g_pass = 0, g_fail = 0;

#define TEST_ASSERT(cond, msg)                         \
  do {                                                 \
    if (!(cond)) {                                     \
      printf("  FAIL: %s (line %d)\n", msg, __LINE__); \
      g_fail++;                                        \
    } else {                                           \
      g_pass++;                                        \
    }                                                  \
  } while (0)

#define TEST_ASSERT_NEAR(a, b, tol, msg)                                                         \
  do {                                                                                           \
    float _a = (a), _b = (b), _t = (tol);                                                        \
    if (std::fabs(_a - _b) > _t) {                                                               \
      printf("  FAIL: %s: got %f, expected %f (tol %f) (line %d)\n", msg, _a, _b, _t, __LINE__); \
      g_fail++;                                                                                  \
    } else {                                                                                     \
      g_pass++;                                                                                  \
    }                                                                                            \
  } while (0)

// ============================================================================
// Test 1: bf16 <-> fp32 conversion roundtrip (scalar)
// ============================================================================
void test_bf16_conversion() {
  printf("[TEST] bf16 <-> fp32 conversion roundtrip\n");
  const int N = 32;
  float src[N], dst[N];
  alignas(64) bfloat16_t bf_buf[N];

  for (int i = 0; i < N; i++)
    src[i] = 1.0f + 0.125f * i;

  // fp32 -> bf16
  store_bf16_from_f32(src, bf_buf, N);

  // bf16 -> fp32
  load_bf16_to_f32(bf_buf, dst, N);

  for (int i = 0; i < N; i++) {
    // bf16 truncates mantissa; allow ~1% tolerance
    TEST_ASSERT_NEAR(dst[i], src[i], std::fabs(src[i]) * 0.01f + 0.01f, "bf16 roundtrip");
  }
}

// ============================================================================
// Test 2: SVE horizontal reduce (svaddv, svmaxv)
// ============================================================================
void test_reduce() {
  printf("[TEST] SVE horizontal reduce (svaddv, svmaxv)\n");
  const int N = 16;
  alignas(64) float data[N];
  for (int i = 0; i < N; i++)
    data[i] = (float)(i + 1);

  svbool_t pg = svwhilelt_b32((uint32_t)0, (uint32_t)N);
  svfloat32_t v = svld1_f32(pg, data);
  float sum = svaddv_f32(pg, v);
  float mx = svmaxv_f32(pg, v);

  int actual_elements = std::min((uint64_t)N, svcntw());
  float expected_sum = (float)(actual_elements * (actual_elements + 1) / 2);
  TEST_ASSERT_NEAR(sum, expected_sum, 0.01f, "svaddv_f32");
  TEST_ASSERT_NEAR(mx, (float)actual_elements, 0.01f, "svmaxv_f32");
}

// ============================================================================
// Test 3: SVE bfdot (bf16 dot product -> fp32)
// ============================================================================
void test_bfdot() {
  printf("[TEST] SVE bfdot (bf16 dot product)\n");
  // bfdot processes pairs: acc[i] += a[2i]*b[2i] + a[2i+1]*b[2i+1]
  const int N = 8;  // fp32 output elements
  float a_f32[N * 2], b_f32[N * 2];
  alignas(64) bfloat16_t a_bf16[N * 2], b_bf16[N * 2];
  alignas(64) float result[N];

  for (int i = 0; i < N * 2; i++) {
    a_f32[i] = (float)(i + 1);
    b_f32[i] = 1.0f;
  }

  store_bf16_from_f32(a_f32, a_bf16, N * 2);
  store_bf16_from_f32(b_f32, b_bf16, N * 2);

  svbool_t pg = svwhilelt_b32((uint32_t)0, (uint32_t)N);
  svfloat32_t acc = svdup_f32(0.f);

  // Load bf16 pairs as fp32 words (2 bf16 = 1 fp32 slot)
  svbfloat16_t va = svreinterpret_bf16(svld1_f32(pg, reinterpret_cast<const float*>(a_bf16)));
  svbfloat16_t vb = svreinterpret_bf16(svld1_f32(pg, reinterpret_cast<const float*>(b_bf16)));
  acc = svbfdot_f32(acc, va, vb);

  svst1_f32(pg, result, acc);

  const uint64_t vl = svcntw();
  for (uint64_t i = 0; i < std::min((uint64_t)N, vl); i++) {
    float expected = a_f32[2 * i] + a_f32[2 * i + 1];
    TEST_ASSERT_NEAR(result[i], expected, 0.5f, "bfdot element");
  }
}

// ============================================================================
// Test 4: Portable GEMM (bf16 A @ VNNI-packed B -> fp32 C)
// ============================================================================
void test_gemm_portable() {
  printf("[TEST] Portable GEMM kernel (bf16 matmul, identity B)\n");
  const int M = 2, N = 4, K = 4;

  float A_f32[M * K] = {1, 2, 3, 4, 5, 6, 7, 8};
  float B_f32[K * N];
  for (int k = 0; k < K; k++)
    for (int n = 0; n < N; n++)
      B_f32[k * N + n] = (k == n) ? 1.0f : 0.0f;

  // Convert A to bf16
  alignas(64) bfloat16_t A_bf16[M * K];
  store_bf16_from_f32(A_f32, A_bf16, M * K);

  // Pack B into VNNI [K/2, N, 2]
  alignas(64) bfloat16_t B_vnni[K / 2 * N * 2];
  for (int k2 = 0; k2 < K / 2; k2++) {
    for (int n = 0; n < N; n++) {
      bf16_t v0(B_f32[(k2 * 2 + 0) * N + n]);
      bf16_t v1(B_f32[(k2 * 2 + 1) * N + n]);
      std::memcpy(&B_vnni[k2 * N * 2 + n * 2 + 0], &v0.val, 2);
      std::memcpy(&B_vnni[k2 * N * 2 + n * 2 + 1], &v1.val, 2);
    }
  }

  alignas(64) float C[M * N] = {};

  // SVE bfdot-based GEMM
  const uint64_t vl_f32 = svcntw();
  for (int m = 0; m < M; ++m) {
    for (int64_t n = 0; n < N; n += vl_f32) {
      svbool_t pg = svwhilelt_b32((uint32_t)n, (uint32_t)N);
      svfloat32_t vc = svdup_f32(0.f);
      for (int k = 0; k < K / 2; ++k) {
        float a_pair;
        std::memcpy(&a_pair, reinterpret_cast<const char*>(A_bf16 + m * K + k * 2), sizeof(float));
        svbfloat16_t va = svreinterpret_bf16(svdup_f32(a_pair));
        svbfloat16_t vb = svreinterpret_bf16(svld1_f32(pg, reinterpret_cast<const float*>(B_vnni) + k * N + n));
        vc = svbfdot_f32(vc, va, vb);
      }
      svst1_f32(pg, C + m * N + n, vc);
    }
  }

  // With B = identity, C should ≈ A
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      TEST_ASSERT_NEAR(C[m * N + n], A_f32[m * K + n], 0.5f, "gemm identity");
    }
  }
}

// ============================================================================
// Test 5: VNNI packing correctness
// ============================================================================
void test_vnni_pack() {
  printf("[TEST] VNNI pack format correctness\n");
  const int K = 4, N = 4;
  float src[K * N];
  for (int k = 0; k < K; k++)
    for (int n = 0; n < N; n++)
      src[k * N + n] = (float)(k * 10 + n);

  bfloat16_t packed[K / 2 * N * 2];
  for (int k2 = 0; k2 < K / 2; k2++) {
    for (int n = 0; n < N; n++) {
      bf16_t v0(src[(k2 * 2 + 0) * N + n]);
      bf16_t v1(src[(k2 * 2 + 1) * N + n]);
      std::memcpy(&packed[k2 * N * 2 + n * 2 + 0], &v0.val, 2);
      std::memcpy(&packed[k2 * N * 2 + n * 2 + 1], &v1.val, 2);
    }
  }

  for (int k = 0; k < K; k++) {
    for (int n = 0; n < N; n++) {
      bf16_t v;
      std::memcpy(&v.val, &packed[(k >> 1) * N * 2 + n * 2 + (k & 1)], 2);
      TEST_ASSERT_NEAR((float)v, src[k * N + n], 0.5f, "vnni unpack");
    }
  }
}

// ============================================================================
// SVE VL-agnostic fast exp logic isolated for tests
// ============================================================================
inline svfloat32_t sve_fexp_u20_test(svbool_t pg, svfloat32_t values) {
  const svfloat32_t vc0 = svdup_f32(0.00010703434948458272f);
  const svfloat32_t vc1 = svdup_f32(0.30354260500649682f);
  const svfloat32_t vc2 = svdup_f32(-0.22433836478672356f);
  const svfloat32_t vc3 = svdup_f32(-0.079204240219773236f);

  union {
    uint32_t u;
    float f;
  } log2ef_u = {0x3fb8aa3b};
  const svfloat32_t vec_exp_log2ef = svdup_f32(log2ef_u.f);

  const svfloat32_t vec_a = svdup_f32(std::pow(2.f, 23.f) / std::log2(2.f));
  const svfloat32_t vec_b = svdup_f32(std::pow(2.f, 23.f) * 127.f);

  union {
    uint32_t u;
    float f;
  } ln_min_u = {0xc2aeac50};
  union {
    uint32_t u;
    float f;
  } ln_max_u = {0x42b17218};
  const svfloat32_t vec_ln_flt_min = svdup_f32(ln_min_u.f);
  const svfloat32_t vec_ln_flt_max = svdup_f32(ln_max_u.f);
  const svuint32_t vec_infinity = svdup_u32(0x7F800000);
  const svuint32_t vec_zero = svdup_u32(0);

  svbool_t min_mask = svcmplt_f32(pg, values, vec_ln_flt_min);
  svbool_t max_mask = svcmpgt_f32(pg, values, vec_ln_flt_max);

  svfloat32_t vec_src = svmul_f32_x(pg, values, vec_exp_log2ef);
  svfloat32_t vec_floor = svrintm_f32_x(pg, vec_src);  // floor
  svfloat32_t vec_fractional = svsub_f32_x(pg, vec_src, vec_floor);

  svfloat32_t vec_res = svmla_f32_x(pg, vc2, vec_fractional, vc3);
  vec_res = svmla_f32_x(pg, vc1, vec_fractional, vec_res);
  vec_res = svmla_f32_x(pg, vc0, vec_fractional, vec_res);

  vec_src = svsub_f32_x(pg, vec_src, vec_res);
  svfloat32_t tmp = svmla_f32_x(pg, vec_b, vec_a, vec_src);
  svuint32_t casted = svreinterpret_u32(svcvt_s32_f32_z(pg, tmp));

  casted = svsel_u32(min_mask, vec_zero, casted);
  casted = svsel_u32(max_mask, vec_infinity, casted);
  return svreinterpret_f32(casted);
}

// ============================================================================
// Test 6: SVE fast exponential approximation
// ============================================================================
void test_sve_fexp() {
  printf("[TEST] SVE fast exponential approximation\n");
  const int N = 8;
  alignas(64) float input[N], result[N];

  for (int i = 0; i < N; i++)
    input[i] = -3.0f + 0.75f * i;

  svbool_t pg = svwhilelt_b32((uint32_t)0, (uint32_t)N);
  svfloat32_t vx = svld1_f32(pg, input);
  svfloat32_t vy = sve_fexp_u20_test(pg, vx);

  svst1_f32(pg, result, vy);

  const uint64_t vl = svcntw();
  for (uint64_t i = 0; i < std::min((uint64_t)N, vl); i++) {
    float expected = std::exp(input[i]);
    float tol = std::fabs(expected) * 0.05f + 0.005f;  // fast approx: ~5% relative
    TEST_ASSERT_NEAR(result[i], expected, tol, "fast exp");
  }
}

// ============================================================================
// Test 7: SVE predicated tail handling
// ============================================================================
void test_predicated_tail() {
  printf("[TEST] SVE predicated tail handling\n");
  const int N = 7;
  alignas(64) float src[32] = {}, dst[32];

  for (int i = 0; i < N; i++)
    src[i] = (float)(i + 1);
  for (int i = 0; i < 32; i++)
    dst[i] = -1.0f;

  const uint64_t vl = svcntw();
  for (uint64_t i = 0; i < (uint64_t)N; i += vl) {
    svbool_t pg = svwhilelt_b32((uint32_t)i, (uint32_t)N);
    svfloat32_t v = svld1_f32(pg, src + i);
    svfloat32_t v2 = svmul_f32_x(pg, v, svdup_f32(2.0f));
    svst1_f32(pg, dst + i, v2);
  }

  for (int i = 0; i < N; i++) {
    TEST_ASSERT_NEAR(dst[i], src[i] * 2.0f, 0.01f, "tail element");
  }
  for (int i = N; i < 32; i++) {
    TEST_ASSERT_NEAR(dst[i], -1.0f, 0.01f, "beyond-tail untouched");
  }
}

// ============================================================================
// Test 8: General bf16 GEMM (non-identity B)
// ============================================================================
void test_gemm_general() {
  printf("[TEST] General bf16 GEMM (non-identity)\n");
  const int M = 3, N = 4, K = 4;

  float A_f32[M * K] = {1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0};
  float B_f32[K * N] = {2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 0, 0, 0, 0, 0};
  float expected[M * N] = {2, 3, 4, 5, 6, 7, 8, 9, 8, 10, 12, 14};

  alignas(64) bfloat16_t A_bf16[M * K];
  store_bf16_from_f32(A_f32, A_bf16, M * K);

  alignas(64) bfloat16_t B_vnni[K / 2 * N * 2];
  for (int k2 = 0; k2 < K / 2; k2++) {
    for (int n = 0; n < N; n++) {
      bf16_t v0(B_f32[(k2 * 2 + 0) * N + n]);
      bf16_t v1(B_f32[(k2 * 2 + 1) * N + n]);
      std::memcpy(&B_vnni[k2 * N * 2 + n * 2 + 0], &v0.val, 2);
      std::memcpy(&B_vnni[k2 * N * 2 + n * 2 + 1], &v1.val, 2);
    }
  }

  alignas(64) float C[M * N] = {};

  const uint64_t vl_f32 = svcntw();
  for (int m = 0; m < M; ++m) {
    for (int64_t n = 0; n < N; n += vl_f32) {
      svbool_t pg = svwhilelt_b32((uint32_t)n, (uint32_t)N);
      svfloat32_t vc = svdup_f32(0.f);
      for (int k = 0; k < K / 2; ++k) {
        float a_pair;
        std::memcpy(&a_pair, reinterpret_cast<const char*>(A_bf16 + m * K + k * 2), sizeof(float));
        svbfloat16_t va = svreinterpret_bf16(svdup_f32(a_pair));
        svbfloat16_t vb = svreinterpret_bf16(svld1_f32(pg, reinterpret_cast<const float*>(B_vnni) + k * N + n));
        vc = svbfdot_f32(vc, va, vb);
      }
      svst1_f32(pg, C + m * N + n, vc);
    }
  }

  for (int i = 0; i < M * N; i++) {
    TEST_ASSERT_NEAR(C[i], expected[i], 1.0f, "general gemm");
  }
}

// ============================================================================
// Test 9: SVE SiLU activation
// ============================================================================
void test_silu() {
  printf("[TEST] SVE SiLU activation\n");
  const int N = 8;
  alignas(64) float input[N], result[N];
  for (int i = 0; i < N; i++)
    input[i] = -2.0f + 0.5f * i;

  svbool_t pg = svwhilelt_b32((uint32_t)0, (uint32_t)N);
  svfloat32_t vx = svld1_f32(pg, input);

  svfloat32_t vneg = svneg_f32_x(pg, vx);
  svfloat32_t vy = sve_fexp_u20_test(pg, vneg);
  svfloat32_t vdenom = svadd_f32_x(pg, svdup_f32(1.0f), vy);
  svfloat32_t vsilu = svdiv_f32_x(pg, vx, vdenom);
  svst1_f32(pg, result, vsilu);

  const uint64_t vl = svcntw();
  for (uint64_t i = 0; i < std::min((uint64_t)N, vl); i++) {
    float x = input[i];
    float expected = x / (1.0f + std::exp(-x));
    TEST_ASSERT_NEAR(result[i], expected, std::fabs(expected) * 0.05f + 0.005f, "silu");
  }
}

// ============================================================================
// Test 10: VL-agnostic iteration pattern
// ============================================================================
void test_vl_agnostic_loop() {
  printf("[TEST] VL-agnostic loop pattern\n");
  const int N = 37;  // intentionally not a power of 2
  alignas(64) float a[64], b[64], c[64] = {};
  for (int i = 0; i < N; i++) {
    a[i] = (float)(i + 1);
    b[i] = 2.0f;
  }

  const uint64_t vl = svcntw();
  for (uint64_t i = 0; i < (uint64_t)N; i += vl) {
    svbool_t pg = svwhilelt_b32((uint32_t)i, (uint32_t)N);
    svfloat32_t va = svld1_f32(pg, a + i);
    svfloat32_t vb = svld1_f32(pg, b + i);
    svfloat32_t vc = svmul_f32_x(pg, va, vb);
    svst1_f32(pg, c + i, vc);
  }

  for (int i = 0; i < N; i++) {
    TEST_ASSERT_NEAR(c[i], a[i] * 2.0f, 0.01f, "vl loop element");
  }
  for (int i = N; i < 64; i++) {
    TEST_ASSERT_NEAR(c[i], 0.0f, 0.01f, "vl loop guard");
  }
}

// ============================================================================
// Test 11: GEMM with accumulate (add_C = true)
// ============================================================================
void test_gemm_accumulate() {
  printf("[TEST] GEMM with accumulate (add_C mode)\n");
  const int M = 2, N = 4, K = 4;

  float A_f32[M * K] = {1, 0, 0, 0, 0, 1, 0, 0};
  float B_f32[K * N] = {2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 0, 0, 0, 0, 0};

  alignas(64) bfloat16_t A_bf16[M * K];
  store_bf16_from_f32(A_f32, A_bf16, M * K);

  alignas(64) bfloat16_t B_vnni[K / 2 * N * 2];
  for (int k2 = 0; k2 < K / 2; k2++)
    for (int n = 0; n < N; n++) {
      bf16_t v0(B_f32[(k2 * 2 + 0) * N + n]);
      bf16_t v1(B_f32[(k2 * 2 + 1) * N + n]);
      std::memcpy(&B_vnni[k2 * N * 2 + n * 2 + 0], &v0.val, 2);
      std::memcpy(&B_vnni[k2 * N * 2 + n * 2 + 1], &v1.val, 2);
    }

  // Pre-fill C with 100.0
  alignas(64) float C[M * N];
  for (int i = 0; i < M * N; i++)
    C[i] = 100.0f;

  // Run GEMM with add_C = true
  const uint64_t vl_f32 = svcntw();
  for (int m = 0; m < M; ++m) {
    for (int64_t n = 0; n < N; n += vl_f32) {
      svbool_t pg = svwhilelt_b32((uint32_t)n, (uint32_t)N);
      svfloat32_t vc = svld1_f32(pg, C + m * N + n);  // add_C: load existing
      for (int k = 0; k < K / 2; ++k) {
        float a_pair;
        std::memcpy(&a_pair, reinterpret_cast<const char*>(A_bf16 + m * K + k * 2), sizeof(float));
        svbfloat16_t va = svreinterpret_bf16(svdup_f32(a_pair));
        svbfloat16_t vb = svreinterpret_bf16(svld1_f32(pg, reinterpret_cast<const float*>(B_vnni) + k * N + n));
        vc = svbfdot_f32(vc, va, vb);
      }
      svst1_f32(pg, C + m * N + n, vc);
    }
  }

  // Expected: C = 100 + A@B
  float expected[M * N] = {102, 103, 104, 105, 106, 107, 108, 109};
  for (int i = 0; i < M * N; i++) {
    TEST_ASSERT_NEAR(C[i], expected[i], 1.0f, "gemm accumulate");
  }
}

// ============================================================================
int main() {
  printf("=== SVE Functional Tests ===\n");
  printf("SVE vector length: %lu bits (%lu x fp32)\n", svcntw() * 32, svcntw());
  printf("\n");

  test_bf16_conversion();
  test_reduce();
  test_bfdot();
  test_gemm_portable();
  test_vnni_pack();
  test_sve_fexp();
  test_predicated_tail();
  test_gemm_general();
  test_silu();
  test_vl_agnostic_loop();
  test_gemm_accumulate();

  printf("\n=== Results: %d passed, %d failed ===\n", g_pass, g_fail);
  return g_fail > 0 ? 1 : 0;
}
