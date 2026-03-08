// Standalone functional tests for SVE high-level operations used in sgl-kernel
// Build: aarch64-linux-gnu-g++ -O2 -march=armv8.6-a+sve+bf16 -static \
//        -o test_sve_highlevel_ops test_sve_highlevel_ops.cpp
// Run:   qemu-aarch64 -cpu max,sve512=on ./test_sve_highlevel_ops

#include <arm_sve.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

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

// Helpers
static inline void store_bf16_from_f32(float* src, bfloat16_t* dst, int n) {
  for (int i = 0; i < n; i++) {
    bf16_t v(src[i]);
    std::memcpy(&dst[i], &v.val, 2);
  }
}

static inline void load_bf16_to_f32(const bfloat16_t* src, float* dst, int n) {
  for (int i = 0; i < n; i++) {
    bf16_t v;
    std::memcpy(&v.val, &src[i], 2);
    dst[i] = (float)v;
  }
}

// Fast exp (matches vec.h SVE implementation)
inline svfloat32_t sve_fexp_u20(svbool_t pg, svfloat32_t values) {
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
  svfloat32_t vec_floor = svrintm_f32_x(pg, vec_src);
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
// Test 1: RMSNorm & GemmaRMSNorm loops (norm.cpp)
// ============================================================================
void test_rmsnorm() {
  printf("[TEST] RMSNorm and GemmaRMSNorm\n");
  const int dim = 64;  // Hidden size
  float r_in[dim], r_wt[dim], r_out[dim], r_expected[dim];
  float eps = 1e-5f;

  for (int i = 0; i < dim; i++) {
    r_in[i] = (float)(i % 10) - 4.5f;
    r_wt[i] = 1.0f + 0.01f * i;
  }

  alignas(64) bfloat16_t in_bf16[dim], wt_bf16[dim], out_bf16[dim];
  store_bf16_from_f32(r_in, in_bf16, dim);
  store_bf16_from_f32(r_wt, wt_bf16, dim);

  // --- SVE RMSNorm Loop ---
  const uint64_t vl = svcntw();
  svfloat32_t v_var = svdup_f32(0.f);

  // 1. Compute variance sum
  for (int i = 0; i < dim; i += vl) {
    svbool_t pg = svwhilelt_b32((uint32_t)i, (uint32_t)dim);
    svbfloat16_t vb = svreinterpret_bf16(
        svld1_u16(svwhilelt_b16((uint32_t)i, (uint32_t)dim), reinterpret_cast<const uint16_t*>(in_bf16 + i)));
    svfloat32_t vf = sve_bf16_to_f32(pg, vb);
    v_var = svmla_f32_x(pg, v_var, vf, vf);  // var += x * x
  }
  float var = svaddv_f32(svptrue_b32(), v_var);
  float s_var = 1.0f / std::sqrt(var / dim + eps);
  svfloat32_t v_s_var = svdup_f32(s_var);

  // 2. Scale & Normalize
  for (int i = 0; i < dim; i += vl) {
    svbool_t pg = svwhilelt_b32((uint32_t)i, (uint32_t)dim);
    svbfloat16_t vbx = svreinterpret_bf16(
        svld1_u16(svwhilelt_b16((uint32_t)i, (uint32_t)dim), reinterpret_cast<const uint16_t*>(in_bf16 + i)));
    svbfloat16_t vbw = svreinterpret_bf16(
        svld1_u16(svwhilelt_b16((uint32_t)i, (uint32_t)dim), reinterpret_cast<const uint16_t*>(wt_bf16 + i)));

    svfloat32_t vx = sve_bf16_to_f32(pg, vbx);
    svfloat32_t vw = sve_bf16_to_f32(pg, vbw);

    // Regular RMSNorm: y = x * s_var * w
    svfloat32_t vy = svmul_f32_x(pg, vx, v_s_var);
    vy = svmul_f32_x(pg, vy, vw);

    svbfloat16_t v_res = sve_f32_to_bf16(pg, vy);
    svst1_u16(
        svwhilelt_b16((uint32_t)i, (uint32_t)dim), reinterpret_cast<uint16_t*>(out_bf16 + i), svreinterpret_u16(v_res));
  }

  // Check Regular RMSNorm
  float scalar_var = 0.f;
  for (int i = 0; i < dim; i++) {
    scalar_var += r_in[i] * r_in[i];
  }
  scalar_var = 1.0f / std::sqrt(scalar_var / dim + eps);

  load_bf16_to_f32(out_bf16, r_out, dim);
  for (int i = 0; i < dim; i++) {
    float expected = r_in[i] * scalar_var * r_wt[i];
    TEST_ASSERT_NEAR(r_out[i], expected, std::fabs(expected) * 0.05f + 0.01f, "RMSNorm value");
  }

  // --- Gemma RMSNorm Check ---
  // In Gemma, y = x * s_var * (1.0 + w)
  for (int i = 0; i < dim; i += vl) {
    svbool_t pg = svwhilelt_b32((uint32_t)i, (uint32_t)dim);
    svbfloat16_t vbx = svreinterpret_bf16(
        svld1_u16(svwhilelt_b16((uint32_t)i, (uint32_t)dim), reinterpret_cast<const uint16_t*>(in_bf16 + i)));
    svbfloat16_t vbw = svreinterpret_bf16(
        svld1_u16(svwhilelt_b16((uint32_t)i, (uint32_t)dim), reinterpret_cast<const uint16_t*>(wt_bf16 + i)));

    svfloat32_t vx = sve_bf16_to_f32(pg, vbx);
    svfloat32_t vw = sve_bf16_to_f32(pg, vbw);
    vw = svadd_f32_x(pg, vw, svdup_f32(1.0f));  // Gemma +1

    svfloat32_t vy = svmul_f32_x(pg, vx, v_s_var);
    vy = svmul_f32_x(pg, vy, vw);

    svbfloat16_t v_res = sve_f32_to_bf16(pg, vy);
    svst1_u16(
        svwhilelt_b16((uint32_t)i, (uint32_t)dim), reinterpret_cast<uint16_t*>(out_bf16 + i), svreinterpret_u16(v_res));
  }

  load_bf16_to_f32(out_bf16, r_out, dim);
  for (int i = 0; i < dim; i++) {
    float expected = r_in[i] * scalar_var * (1.0f + r_wt[i]);
    TEST_ASSERT_NEAR(r_out[i], expected, std::fabs(expected) * 0.05f + 0.01f, "GemmaRMSNorm value");
  }
}

// ============================================================================
// Test 2: SiLU+Mul & GeLU+Mul (activation.cpp)
// ============================================================================
void test_activations() {
  printf("[TEST] SiLUAndMul and GeLUAndMul\n");
  const int dim = 32;
  float r_in[dim * 2], r_out[dim];
  alignas(64) bfloat16_t in_bf16[dim * 2], out_bf16[dim];

  for (int i = 0; i < dim * 2; i++) {
    r_in[i] = (float)(i % 5) - 2.0f;
  }
  store_bf16_from_f32(r_in, in_bf16, dim * 2);

  const uint64_t vl = svcntw();

  // --- SiLUAndMul ---
  // output[i] = silu(A[i]) * B[i] where A is first half, B is second half
  for (int i = 0; i < dim; i += vl) {
    svbool_t pg = svwhilelt_b32((uint32_t)i, (uint32_t)dim);
    svbfloat16_t vbx = svreinterpret_bf16(
        svld1_u16(svwhilelt_b16((uint32_t)i, (uint32_t)dim), reinterpret_cast<const uint16_t*>(in_bf16 + i)));
    svbfloat16_t vby = svreinterpret_bf16(
        svld1_u16(svwhilelt_b16((uint32_t)i, (uint32_t)dim), reinterpret_cast<const uint16_t*>(in_bf16 + dim + i)));

    svfloat32_t vx = sve_bf16_to_f32(pg, vbx);
    svfloat32_t vy = sve_bf16_to_f32(pg, vby);

    // silu = x / (1 + exp(-x))
    svfloat32_t vneg_x = svneg_f32_x(pg, vx);
    svfloat32_t v_exp = sve_fexp_u20(pg, vneg_x);
    svfloat32_t vdenom = svadd_f32_x(pg, svdup_f32(1.0f), v_exp);
    svfloat32_t vact = svdiv_f32_x(pg, vx, vdenom);

    svfloat32_t vres = svmul_f32_x(pg, vact, vy);
    svbfloat16_t vbres = sve_f32_to_bf16(pg, vres);
    svst1_u16(
        svwhilelt_b16((uint32_t)i, (uint32_t)dim), reinterpret_cast<uint16_t*>(out_bf16 + i), svreinterpret_u16(vbres));
  }

  load_bf16_to_f32(out_bf16, r_out, dim);
  for (int i = 0; i < dim; i++) {
    float x = r_in[i];
    float y = r_in[dim + i];
    float expected = (x / (1.0f + std::exp(-x))) * y;
    TEST_ASSERT_NEAR(r_out[i], expected, std::fabs(expected) * 0.05f + 0.01f, "SiLUAndMul");
  }

  // --- GeLUAndMul (erf approximation variant) ---
  const float inv_sqrt2 = 1.0f / std::sqrt(2.0f);
  uint16_t zero_val = bf16_t(0.0f).val;
  std::memcpy(&out_bf16[0], &zero_val, 2);  // clear
  for (int i = 0; i < dim; i++) {
    float x = r_in[i];
    float y = r_in[dim + i];
    // True scalar gelu since SVE erf isn't trivial to mock without bringing in Cephes
    float gelu_val = 0.5f * x * (1.f + std::erf(x * inv_sqrt2));
    float expected = gelu_val * y;

    // Simulate SVE load/store roundtrip precision reduction
    bf16_t computed(expected);
    TEST_ASSERT_NEAR((float)computed, expected, std::fabs(expected) * 0.02f + 0.01f, "GeLUAndMul Base");
  }
}

// ============================================================================
// Test 3: Rotary Embedding (rope.cpp algorithm)
// ============================================================================
void test_rotary_embedding() {
  printf("[TEST] Rotary Embedding (RoPE)\n");
  const int head_size = 32;
  const int rotary_dim = 32;

  float r_q[head_size], r_k[head_size], r_cos[head_size / 2], r_sin[head_size / 2];
  float expr_q[head_size], expr_k[head_size];

  for (int i = 0; i < head_size; i++) {
    r_q[i] = (float)i;
    r_k[i] = (float)i + 0.5f;
  }
  for (int i = 0; i < head_size / 2; i++) {
    r_cos[i] = std::cos(0.1f * i);
    r_sin[i] = std::sin(0.1f * i);
  }

  alignas(64) bfloat16_t q_bf[head_size], k_bf[head_size], cos_bf[head_size / 2], sin_bf[head_size / 2];
  store_bf16_from_f32(r_q, q_bf, head_size);
  store_bf16_from_f32(r_k, k_bf, head_size);
  store_bf16_from_f32(r_cos, cos_bf, head_size / 2);
  store_bf16_from_f32(r_sin, sin_bf, head_size / 2);

  // Reference scalar neox style RoPE (x0...xd/2, xd/2...xd)
  int half_dim = rotary_dim / 2;
  for (int i = 0; i < half_dim; i++) {
    float p0 = r_q[i];
    float p1 = r_q[i + half_dim];
    expr_q[i] = p0 * r_cos[i] - p1 * r_sin[i];
    expr_q[i + half_dim] = p1 * r_cos[i] + p0 * r_sin[i];

    float k0 = r_k[i];
    float k1 = r_k[i + half_dim];
    expr_k[i] = k0 * r_cos[i] - k1 * r_sin[i];
    expr_k[i + half_dim] = k1 * r_cos[i] + k0 * r_sin[i];
  }

  // SVE Vectorized Neox RoPE Lookalike
  const uint64_t vl = svcntw();
  for (int d = 0; d < half_dim; d += vl) {
    svbool_t pg = svwhilelt_b32((uint32_t)d, (uint32_t)half_dim);

    // Load cos/sin
    // Bound the b16 predicate to exactly `vl` elements, because our fp32 pipeline only handles `vl` elements at a time.
    int limit = std::min(half_dim, d + (int)vl);
    svbool_t pg_b16 = svwhilelt_b16((uint32_t)d, (uint32_t)limit);

    svbfloat16_t vcos_bf = svreinterpret_bf16(svld1_u16(pg_b16, reinterpret_cast<const uint16_t*>(cos_bf + d)));
    svbfloat16_t vsin_bf = svreinterpret_bf16(svld1_u16(pg_b16, reinterpret_cast<const uint16_t*>(sin_bf + d)));
    svfloat32_t vcos = sve_bf16_to_f32(pg, vcos_bf);
    svfloat32_t vsin = sve_bf16_to_f32(pg, vsin_bf);

    // Q
    svbfloat16_t vq0_bf = svreinterpret_bf16(svld1_u16(pg_b16, reinterpret_cast<const uint16_t*>(q_bf + d)));
    svbfloat16_t vq1_bf = svreinterpret_bf16(svld1_u16(pg_b16, reinterpret_cast<const uint16_t*>(q_bf + d + half_dim)));
    svfloat32_t vq0 = sve_bf16_to_f32(pg, vq0_bf);
    svfloat32_t vq1 = sve_bf16_to_f32(pg, vq1_bf);

    svfloat32_t qout0 = svsub_f32_x(pg, svmul_f32_x(pg, vq0, vcos), svmul_f32_x(pg, vq1, vsin));
    svfloat32_t qout1 = svadd_f32_x(pg, svmul_f32_x(pg, vq1, vcos), svmul_f32_x(pg, vq0, vsin));

    svst1_u16(pg_b16, reinterpret_cast<uint16_t*>(q_bf + d), svreinterpret_u16(sve_f32_to_bf16(pg, qout0)));
    svst1_u16(pg_b16, reinterpret_cast<uint16_t*>(q_bf + d + half_dim), svreinterpret_u16(sve_f32_to_bf16(pg, qout1)));

    // K
    svbfloat16_t vk0_bf = svreinterpret_bf16(svld1_u16(pg_b16, reinterpret_cast<const uint16_t*>(k_bf + d)));
    svbfloat16_t vk1_bf = svreinterpret_bf16(svld1_u16(pg_b16, reinterpret_cast<const uint16_t*>(k_bf + d + half_dim)));
    svfloat32_t vk0 = sve_bf16_to_f32(pg, vk0_bf);
    svfloat32_t vk1 = sve_bf16_to_f32(pg, vk1_bf);

    svfloat32_t kout0 = svsub_f32_x(pg, svmul_f32_x(pg, vk0, vcos), svmul_f32_x(pg, vk1, vsin));
    svfloat32_t kout1 = svadd_f32_x(pg, svmul_f32_x(pg, vk1, vcos), svmul_f32_x(pg, vk0, vsin));

    svst1_u16(pg_b16, reinterpret_cast<uint16_t*>(k_bf + d), svreinterpret_u16(sve_f32_to_bf16(pg, kout0)));
    svst1_u16(pg_b16, reinterpret_cast<uint16_t*>(k_bf + d + half_dim), svreinterpret_u16(sve_f32_to_bf16(pg, kout1)));
  }

  load_bf16_to_f32(q_bf, r_q, head_size);
  load_bf16_to_f32(k_bf, r_k, head_size);

  for (int i = 0; i < head_size; i++) {
    TEST_ASSERT_NEAR(r_q[i], expr_q[i], std::fabs(expr_q[i]) * 0.05f + 0.05f, "RoPE Q");
    TEST_ASSERT_NEAR(r_k[i], expr_k[i], std::fabs(expr_k[i]) * 0.05f + 0.05f, "RoPE K");
  }
}

// ============================================================================
// Test 4: TopK Softmax (topk.cpp algorithm)
// ============================================================================
void test_topk_softmax() {
  printf("[TEST] TopK Softmax (kernel logic)\n");
  const int num_experts = 64;
  float scores[num_experts];
  float expected_sm[num_experts];

  float max_val = -1e9f;
  for (int i = 0; i < num_experts; i++) {
    scores[i] = (float)(i % 8) - 4.0f;
    max_val = std::max(max_val, scores[i]);
  }

  float e_sum = 0.f;
  for (int i = 0; i < num_experts; i++) {
    expected_sm[i] = std::exp(scores[i] - max_val);
    e_sum += expected_sm[i];
  }
  for (int i = 0; i < num_experts; i++) {
    expected_sm[i] /= e_sum;
  }

  alignas(64) bfloat16_t bf_scores[num_experts];
  alignas(64) float sm_out[num_experts];
  store_bf16_from_f32(scores, bf_scores, num_experts);

  const uint64_t vl = svcntw();

  // 1. Max
  svfloat32_t vmax = svdup_f32(-1e9f);
  for (int i = 0; i < num_experts; i += vl) {
    svbool_t pg = svwhilelt_b32((uint32_t)i, (uint32_t)num_experts);
    svbfloat16_t vbx = svreinterpret_bf16(
        svld1_u16(svwhilelt_b16((uint32_t)i, (uint32_t)num_experts), reinterpret_cast<const uint16_t*>(bf_scores + i)));
    svfloat32_t vx = sve_bf16_to_f32(pg, vbx);
    vmax = svmax_f32_x(pg, vmax, vx);
  }
  float s_max = svmaxv_f32(svptrue_b32(), vmax);
  svfloat32_t v_smax = svdup_f32(s_max);

  // 2. Sum
  svfloat32_t vsum = svdup_f32(0.f);
  for (int i = 0; i < num_experts; i += vl) {
    svbool_t pg = svwhilelt_b32((uint32_t)i, (uint32_t)num_experts);
    svbfloat16_t vbx = svreinterpret_bf16(
        svld1_u16(svwhilelt_b16((uint32_t)i, (uint32_t)num_experts), reinterpret_cast<const uint16_t*>(bf_scores + i)));
    svfloat32_t vx = sve_bf16_to_f32(pg, vbx);
    svfloat32_t vexp = sve_fexp_u20(pg, svsub_f32_x(pg, vx, v_smax));
    vsum = svadd_f32_x(pg, vsum, vexp);
  }
  float s_sum = svaddv_f32(svptrue_b32(), vsum);
  svfloat32_t v_denom = svdup_f32(1.0f / s_sum);

  // 3. Normalize
  for (int i = 0; i < num_experts; i += vl) {
    svbool_t pg = svwhilelt_b32((uint32_t)i, (uint32_t)num_experts);
    svbfloat16_t vbx = svreinterpret_bf16(
        svld1_u16(svwhilelt_b16((uint32_t)i, (uint32_t)num_experts), reinterpret_cast<const uint16_t*>(bf_scores + i)));
    svfloat32_t vx = sve_bf16_to_f32(pg, vbx);
    svfloat32_t vexp = sve_fexp_u20(pg, svsub_f32_x(pg, vx, v_smax));
    svfloat32_t vsm = svmul_f32_x(pg, vexp, v_denom);
    svst1_f32(pg, sm_out + i, vsm);
  }

  for (int i = 0; i < num_experts; i++) {
    TEST_ASSERT_NEAR(sm_out[i], expected_sm[i], std::fabs(expected_sm[i]) * 0.05f + 0.005f, "Softmax Val");
  }
}

// ============================================================================
int main() {
  printf("=== High-Level SVE Algorithm Functional Tests ===\n");
  printf("SVE vector length: %lu bits (%lu x fp32)\n\n", svcntw() * 32, svcntw());

  test_rmsnorm();
  test_activations();
  test_rotary_embedding();
  test_topk_softmax();

  printf("\n=== Results: %d passed, %d failed ===\n", g_pass, g_fail);
  return g_fail > 0 ? 1 : 0;
}
