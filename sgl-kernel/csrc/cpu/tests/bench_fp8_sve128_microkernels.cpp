#include <arm_sve.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

constexpr int kN = 16;

inline uint16_t float_to_bf16_bits(float x) {
  uint32_t bits;
  std::memcpy(&bits, &x, sizeof(bits));
  return static_cast<uint16_t>(bits >> 16);
}

inline float bf16_bits_to_float(uint16_t x) {
  uint32_t bits = static_cast<uint32_t>(x) << 16;
  float out;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

inline float pack_bf16_pair(float a0, float a1) {
  uint32_t bits = static_cast<uint32_t>(float_to_bf16_bits(a1)) << 16 | float_to_bf16_bits(a0);
  float out;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

inline svbfloat16_t cvt_fp8_to_bf16_ext(svuint8_t a) {
  const svbool_t pg16 = svptrue_b16();
  svuint16_t x = svunpklo_u16(a);
  svuint16_t vsign = svand_u16_x(pg16, x, svdup_u16(0x80));
  vsign = svlsl_n_u16_x(pg16, vsign, 8);
  svuint16_t vexp_and_mant = svand_u16_x(pg16, x, svdup_u16(0x7F));
  vexp_and_mant = svlsl_n_u16_x(pg16, vexp_and_mant, 4);
  svuint16_t result = svorr_u16_x(pg16, vsign, svdup_u16(0x4000));
  result = svorr_u16_x(pg16, result, vexp_and_mant);
  return svreinterpret_bf16(result);
}

inline void cvt_fp8x16_to_2xbf16_ext(svuint8_t a, svbfloat16_t& lo, svbfloat16_t& hi) {
  const svbool_t pg16 = svptrue_b16();
  svuint16_t xlo = svunpklo_u16(a);
  svuint16_t xhi = svunpkhi_u16(a);

  auto cvt_half = [&](svuint16_t x) {
    svuint16_t vsign = svand_u16_x(pg16, x, svdup_u16(0x80));
    vsign = svlsl_n_u16_x(pg16, vsign, 8);
    svuint16_t vexp_and_mant = svand_u16_x(pg16, x, svdup_u16(0x7F));
    vexp_and_mant = svlsl_n_u16_x(pg16, vexp_and_mant, 4);
    svuint16_t result = svorr_u16_x(pg16, vsign, svdup_u16(0x4000));
    result = svorr_u16_x(pg16, result, vexp_and_mant);
    return svreinterpret_bf16(result);
  };

  lo = cvt_half(xlo);
  hi = cvt_half(xhi);
}

inline double now_sec() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

std::vector<uint8_t> pack_fp8_k2(const std::vector<uint8_t>& src, int K) {
  const int K2 = K / 2;
  std::vector<uint8_t> dst(K2 * kN * 2);
  for (int n = 0; n < kN; ++n) {
    for (int k2 = 0; k2 < K2; ++k2) {
      dst[k2 * kN * 2 + n * 2 + 0] = src[n * K + 2 * k2 + 0];
      dst[k2 * kN * 2 + n * 2 + 1] = src[n * K + 2 * k2 + 1];
    }
  }
  return dst;
}

std::vector<uint8_t> pack_fp8_k4_blocked(const std::vector<uint8_t>& src, int K) {
  const int K4 = K / 4;
  std::vector<uint8_t> dst(K4 * (kN / 4) * 16);
  for (int k4 = 0; k4 < K4; ++k4) {
    for (int g = 0; g < kN / 4; ++g) {
      uint8_t* block = dst.data() + k4 * (kN / 4) * 16 + g * 16;
      for (int c = 0; c < 4; ++c) {
        int n = g * 4 + c;
        block[c * 2 + 0] = src[n * K + 4 * k4 + 0];
        block[c * 2 + 1] = src[n * K + 4 * k4 + 1];
        block[8 + c * 2 + 0] = src[n * K + 4 * k4 + 2];
        block[8 + c * 2 + 1] = src[n * K + 4 * k4 + 3];
      }
    }
  }
  return dst;
}

std::vector<uint16_t> pack_bf16_decoded_k2(const std::vector<uint8_t>& src, int K) {
  const int K2 = K / 2;
  std::vector<uint16_t> dst(K2 * kN * 2);
  for (int k2 = 0; k2 < K2; ++k2) {
    for (int g = 0; g < kN / 4; ++g) {
      uint8_t fp8[8];
      for (int c = 0; c < 4; ++c) {
        int n = g * 4 + c;
        fp8[c * 2 + 0] = src[n * K + 2 * k2 + 0];
        fp8[c * 2 + 1] = src[n * K + 2 * k2 + 1];
      }
      svbool_t pg8 = svwhilelt_b8(0u, 8u);
      svbool_t pg16 = svptrue_b16();
      svbfloat16_t vbf = cvt_fp8_to_bf16_ext(svld1_u8(pg8, fp8));
      svst1_u16(pg16, dst.data() + k2 * kN * 2 + g * 8, svreinterpret_u16(vbf));
    }
  }
  return dst;
}

template <int ROWS>
void kernel_fp8_k2(const float* a_pairs, const uint8_t* b, int K, float* out) {
  const int K2 = K / 2;
  const svbool_t pg8 = svwhilelt_b8(0u, 8u);
  const svbool_t pgf = svptrue_b32();
  svfloat32_t acc00 = svdup_n_f32(0.f), acc01 = svdup_n_f32(0.f), acc02 = svdup_n_f32(0.f), acc03 = svdup_n_f32(0.f);
  svfloat32_t acc10 = svdup_n_f32(0.f), acc11 = svdup_n_f32(0.f), acc12 = svdup_n_f32(0.f), acc13 = svdup_n_f32(0.f);
  svfloat32_t acc20 = svdup_n_f32(0.f), acc21 = svdup_n_f32(0.f), acc22 = svdup_n_f32(0.f), acc23 = svdup_n_f32(0.f);
  svfloat32_t acc30 = svdup_n_f32(0.f), acc31 = svdup_n_f32(0.f), acc32 = svdup_n_f32(0.f), acc33 = svdup_n_f32(0.f);
  for (int k2 = 0; k2 < K2; ++k2) {
    svbfloat16_t vb0 = cvt_fp8_to_bf16_ext(svld1_u8(pg8, b + k2 * kN * 2 + 0 * 8));
    svbfloat16_t vb1 = cvt_fp8_to_bf16_ext(svld1_u8(pg8, b + k2 * kN * 2 + 1 * 8));
    svbfloat16_t vb2 = cvt_fp8_to_bf16_ext(svld1_u8(pg8, b + k2 * kN * 2 + 2 * 8));
    svbfloat16_t vb3 = cvt_fp8_to_bf16_ext(svld1_u8(pg8, b + k2 * kN * 2 + 3 * 8));
    if constexpr (ROWS >= 1) {
      svbfloat16_t va = svreinterpret_bf16(svdup_f32(a_pairs[0 * K2 + k2]));
      acc00 = svbfdot_f32(acc00, va, vb0);
      acc01 = svbfdot_f32(acc01, va, vb1);
      acc02 = svbfdot_f32(acc02, va, vb2);
      acc03 = svbfdot_f32(acc03, va, vb3);
    }
    if constexpr (ROWS >= 2) {
      svbfloat16_t va = svreinterpret_bf16(svdup_f32(a_pairs[1 * K2 + k2]));
      acc10 = svbfdot_f32(acc10, va, vb0);
      acc11 = svbfdot_f32(acc11, va, vb1);
      acc12 = svbfdot_f32(acc12, va, vb2);
      acc13 = svbfdot_f32(acc13, va, vb3);
    }
    if constexpr (ROWS >= 3) {
      svbfloat16_t va = svreinterpret_bf16(svdup_f32(a_pairs[2 * K2 + k2]));
      acc20 = svbfdot_f32(acc20, va, vb0);
      acc21 = svbfdot_f32(acc21, va, vb1);
      acc22 = svbfdot_f32(acc22, va, vb2);
      acc23 = svbfdot_f32(acc23, va, vb3);
    }
    if constexpr (ROWS >= 4) {
      svbfloat16_t va = svreinterpret_bf16(svdup_f32(a_pairs[3 * K2 + k2]));
      acc30 = svbfdot_f32(acc30, va, vb0);
      acc31 = svbfdot_f32(acc31, va, vb1);
      acc32 = svbfdot_f32(acc32, va, vb2);
      acc33 = svbfdot_f32(acc33, va, vb3);
    }
  }
  if constexpr (ROWS >= 1) {
    svst1_f32(pgf, out + 0 * kN + 0 * 4, acc00);
    svst1_f32(pgf, out + 0 * kN + 1 * 4, acc01);
    svst1_f32(pgf, out + 0 * kN + 2 * 4, acc02);
    svst1_f32(pgf, out + 0 * kN + 3 * 4, acc03);
  }
  if constexpr (ROWS >= 2) {
    svst1_f32(pgf, out + 1 * kN + 0 * 4, acc10);
    svst1_f32(pgf, out + 1 * kN + 1 * 4, acc11);
    svst1_f32(pgf, out + 1 * kN + 2 * 4, acc12);
    svst1_f32(pgf, out + 1 * kN + 3 * 4, acc13);
  }
  if constexpr (ROWS >= 3) {
    svst1_f32(pgf, out + 2 * kN + 0 * 4, acc20);
    svst1_f32(pgf, out + 2 * kN + 1 * 4, acc21);
    svst1_f32(pgf, out + 2 * kN + 2 * 4, acc22);
    svst1_f32(pgf, out + 2 * kN + 3 * 4, acc23);
  }
  if constexpr (ROWS >= 4) {
    svst1_f32(pgf, out + 3 * kN + 0 * 4, acc30);
    svst1_f32(pgf, out + 3 * kN + 1 * 4, acc31);
    svst1_f32(pgf, out + 3 * kN + 2 * 4, acc32);
    svst1_f32(pgf, out + 3 * kN + 3 * 4, acc33);
  }
}

template <int ROWS>
void kernel_fp8_k2_ld16(const float* a_pairs, const uint8_t* b, int K, float* out) {
  const int K2 = K / 2;
  const svbool_t pg16b = svptrue_b8();
  const svbool_t pgf = svptrue_b32();
  svfloat32_t acc00 = svdup_n_f32(0.f), acc01 = svdup_n_f32(0.f), acc02 = svdup_n_f32(0.f), acc03 = svdup_n_f32(0.f);
  svfloat32_t acc10 = svdup_n_f32(0.f), acc11 = svdup_n_f32(0.f), acc12 = svdup_n_f32(0.f), acc13 = svdup_n_f32(0.f);
  svfloat32_t acc20 = svdup_n_f32(0.f), acc21 = svdup_n_f32(0.f), acc22 = svdup_n_f32(0.f), acc23 = svdup_n_f32(0.f);
  svfloat32_t acc30 = svdup_n_f32(0.f), acc31 = svdup_n_f32(0.f), acc32 = svdup_n_f32(0.f), acc33 = svdup_n_f32(0.f);
  for (int k2 = 0; k2 < K2; ++k2) {
    svbfloat16_t vb0, vb1, vb2, vb3;
    cvt_fp8x16_to_2xbf16_ext(svld1_u8(pg16b, b + k2 * kN * 2 + 0), vb0, vb1);
    cvt_fp8x16_to_2xbf16_ext(svld1_u8(pg16b, b + k2 * kN * 2 + 16), vb2, vb3);
    if constexpr (ROWS >= 1) {
      svbfloat16_t va = svreinterpret_bf16(svdup_f32(a_pairs[0 * K2 + k2]));
      acc00 = svbfdot_f32(acc00, va, vb0);
      acc01 = svbfdot_f32(acc01, va, vb1);
      acc02 = svbfdot_f32(acc02, va, vb2);
      acc03 = svbfdot_f32(acc03, va, vb3);
    }
    if constexpr (ROWS >= 2) {
      svbfloat16_t va = svreinterpret_bf16(svdup_f32(a_pairs[1 * K2 + k2]));
      acc10 = svbfdot_f32(acc10, va, vb0);
      acc11 = svbfdot_f32(acc11, va, vb1);
      acc12 = svbfdot_f32(acc12, va, vb2);
      acc13 = svbfdot_f32(acc13, va, vb3);
    }
    if constexpr (ROWS >= 3) {
      svbfloat16_t va = svreinterpret_bf16(svdup_f32(a_pairs[2 * K2 + k2]));
      acc20 = svbfdot_f32(acc20, va, vb0);
      acc21 = svbfdot_f32(acc21, va, vb1);
      acc22 = svbfdot_f32(acc22, va, vb2);
      acc23 = svbfdot_f32(acc23, va, vb3);
    }
    if constexpr (ROWS >= 4) {
      svbfloat16_t va = svreinterpret_bf16(svdup_f32(a_pairs[3 * K2 + k2]));
      acc30 = svbfdot_f32(acc30, va, vb0);
      acc31 = svbfdot_f32(acc31, va, vb1);
      acc32 = svbfdot_f32(acc32, va, vb2);
      acc33 = svbfdot_f32(acc33, va, vb3);
    }
  }
  if constexpr (ROWS >= 1) {
    svst1_f32(pgf, out + 0 * kN + 0 * 4, acc00);
    svst1_f32(pgf, out + 0 * kN + 1 * 4, acc01);
    svst1_f32(pgf, out + 0 * kN + 2 * 4, acc02);
    svst1_f32(pgf, out + 0 * kN + 3 * 4, acc03);
  }
  if constexpr (ROWS >= 2) {
    svst1_f32(pgf, out + 1 * kN + 0 * 4, acc10);
    svst1_f32(pgf, out + 1 * kN + 1 * 4, acc11);
    svst1_f32(pgf, out + 1 * kN + 2 * 4, acc12);
    svst1_f32(pgf, out + 1 * kN + 3 * 4, acc13);
  }
  if constexpr (ROWS >= 3) {
    svst1_f32(pgf, out + 2 * kN + 0 * 4, acc20);
    svst1_f32(pgf, out + 2 * kN + 1 * 4, acc21);
    svst1_f32(pgf, out + 2 * kN + 2 * 4, acc22);
    svst1_f32(pgf, out + 2 * kN + 3 * 4, acc23);
  }
  if constexpr (ROWS >= 4) {
    svst1_f32(pgf, out + 3 * kN + 0 * 4, acc30);
    svst1_f32(pgf, out + 3 * kN + 1 * 4, acc31);
    svst1_f32(pgf, out + 3 * kN + 2 * 4, acc32);
    svst1_f32(pgf, out + 3 * kN + 3 * 4, acc33);
  }
}

template <int ROWS>
void kernel_fp8_k4(const float* a_pairs, const uint8_t* b, int K, float* out) {
  const int K4 = K / 4;
  const svbool_t pg8 = svwhilelt_b8(0u, 8u);
  const svbool_t pgf = svptrue_b32();
  svfloat32_t acc00 = svdup_n_f32(0.f), acc01 = svdup_n_f32(0.f), acc02 = svdup_n_f32(0.f), acc03 = svdup_n_f32(0.f);
  svfloat32_t acc10 = svdup_n_f32(0.f), acc11 = svdup_n_f32(0.f), acc12 = svdup_n_f32(0.f), acc13 = svdup_n_f32(0.f);
  svfloat32_t acc20 = svdup_n_f32(0.f), acc21 = svdup_n_f32(0.f), acc22 = svdup_n_f32(0.f), acc23 = svdup_n_f32(0.f);
  svfloat32_t acc30 = svdup_n_f32(0.f), acc31 = svdup_n_f32(0.f), acc32 = svdup_n_f32(0.f), acc33 = svdup_n_f32(0.f);
  for (int k4 = 0; k4 < K4; ++k4) {
    const uint8_t* bk = b + k4 * 4 * 16;
    svbfloat16_t vb0a = cvt_fp8_to_bf16_ext(svld1_u8(pg8, bk + 0 * 16 + 0));
    svbfloat16_t vb0b = cvt_fp8_to_bf16_ext(svld1_u8(pg8, bk + 0 * 16 + 8));
    svbfloat16_t vb1a = cvt_fp8_to_bf16_ext(svld1_u8(pg8, bk + 1 * 16 + 0));
    svbfloat16_t vb1b = cvt_fp8_to_bf16_ext(svld1_u8(pg8, bk + 1 * 16 + 8));
    svbfloat16_t vb2a = cvt_fp8_to_bf16_ext(svld1_u8(pg8, bk + 2 * 16 + 0));
    svbfloat16_t vb2b = cvt_fp8_to_bf16_ext(svld1_u8(pg8, bk + 2 * 16 + 8));
    svbfloat16_t vb3a = cvt_fp8_to_bf16_ext(svld1_u8(pg8, bk + 3 * 16 + 0));
    svbfloat16_t vb3b = cvt_fp8_to_bf16_ext(svld1_u8(pg8, bk + 3 * 16 + 8));
    if constexpr (ROWS >= 1) {
      svbfloat16_t va0 = svreinterpret_bf16(svdup_f32(a_pairs[0 * (K / 2) + 2 * k4 + 0]));
      svbfloat16_t va1 = svreinterpret_bf16(svdup_f32(a_pairs[0 * (K / 2) + 2 * k4 + 1]));
      acc00 = svbfdot_f32(acc00, va0, vb0a); acc00 = svbfdot_f32(acc00, va1, vb0b);
      acc01 = svbfdot_f32(acc01, va0, vb1a); acc01 = svbfdot_f32(acc01, va1, vb1b);
      acc02 = svbfdot_f32(acc02, va0, vb2a); acc02 = svbfdot_f32(acc02, va1, vb2b);
      acc03 = svbfdot_f32(acc03, va0, vb3a); acc03 = svbfdot_f32(acc03, va1, vb3b);
    }
    if constexpr (ROWS >= 2) {
      svbfloat16_t va0 = svreinterpret_bf16(svdup_f32(a_pairs[1 * (K / 2) + 2 * k4 + 0]));
      svbfloat16_t va1 = svreinterpret_bf16(svdup_f32(a_pairs[1 * (K / 2) + 2 * k4 + 1]));
      acc10 = svbfdot_f32(acc10, va0, vb0a); acc10 = svbfdot_f32(acc10, va1, vb0b);
      acc11 = svbfdot_f32(acc11, va0, vb1a); acc11 = svbfdot_f32(acc11, va1, vb1b);
      acc12 = svbfdot_f32(acc12, va0, vb2a); acc12 = svbfdot_f32(acc12, va1, vb2b);
      acc13 = svbfdot_f32(acc13, va0, vb3a); acc13 = svbfdot_f32(acc13, va1, vb3b);
    }
    if constexpr (ROWS >= 3) {
      svbfloat16_t va0 = svreinterpret_bf16(svdup_f32(a_pairs[2 * (K / 2) + 2 * k4 + 0]));
      svbfloat16_t va1 = svreinterpret_bf16(svdup_f32(a_pairs[2 * (K / 2) + 2 * k4 + 1]));
      acc20 = svbfdot_f32(acc20, va0, vb0a); acc20 = svbfdot_f32(acc20, va1, vb0b);
      acc21 = svbfdot_f32(acc21, va0, vb1a); acc21 = svbfdot_f32(acc21, va1, vb1b);
      acc22 = svbfdot_f32(acc22, va0, vb2a); acc22 = svbfdot_f32(acc22, va1, vb2b);
      acc23 = svbfdot_f32(acc23, va0, vb3a); acc23 = svbfdot_f32(acc23, va1, vb3b);
    }
    if constexpr (ROWS >= 4) {
      svbfloat16_t va0 = svreinterpret_bf16(svdup_f32(a_pairs[3 * (K / 2) + 2 * k4 + 0]));
      svbfloat16_t va1 = svreinterpret_bf16(svdup_f32(a_pairs[3 * (K / 2) + 2 * k4 + 1]));
      acc30 = svbfdot_f32(acc30, va0, vb0a); acc30 = svbfdot_f32(acc30, va1, vb0b);
      acc31 = svbfdot_f32(acc31, va0, vb1a); acc31 = svbfdot_f32(acc31, va1, vb1b);
      acc32 = svbfdot_f32(acc32, va0, vb2a); acc32 = svbfdot_f32(acc32, va1, vb2b);
      acc33 = svbfdot_f32(acc33, va0, vb3a); acc33 = svbfdot_f32(acc33, va1, vb3b);
    }
  }
  if constexpr (ROWS >= 1) {
    svst1_f32(pgf, out + 0 * kN + 0 * 4, acc00);
    svst1_f32(pgf, out + 0 * kN + 1 * 4, acc01);
    svst1_f32(pgf, out + 0 * kN + 2 * 4, acc02);
    svst1_f32(pgf, out + 0 * kN + 3 * 4, acc03);
  }
  if constexpr (ROWS >= 2) {
    svst1_f32(pgf, out + 1 * kN + 0 * 4, acc10);
    svst1_f32(pgf, out + 1 * kN + 1 * 4, acc11);
    svst1_f32(pgf, out + 1 * kN + 2 * 4, acc12);
    svst1_f32(pgf, out + 1 * kN + 3 * 4, acc13);
  }
  if constexpr (ROWS >= 3) {
    svst1_f32(pgf, out + 2 * kN + 0 * 4, acc20);
    svst1_f32(pgf, out + 2 * kN + 1 * 4, acc21);
    svst1_f32(pgf, out + 2 * kN + 2 * 4, acc22);
    svst1_f32(pgf, out + 2 * kN + 3 * 4, acc23);
  }
  if constexpr (ROWS >= 4) {
    svst1_f32(pgf, out + 3 * kN + 0 * 4, acc30);
    svst1_f32(pgf, out + 3 * kN + 1 * 4, acc31);
    svst1_f32(pgf, out + 3 * kN + 2 * 4, acc32);
    svst1_f32(pgf, out + 3 * kN + 3 * 4, acc33);
  }
}

template <int ROWS>
void kernel_bf16_upper(const float* a_pairs, const uint16_t* b, int K, float* out) {
  const int K2 = K / 2;
  const svbool_t pg16 = svptrue_b16();
  const svbool_t pgf = svptrue_b32();
  svfloat32_t acc00 = svdup_n_f32(0.f), acc01 = svdup_n_f32(0.f), acc02 = svdup_n_f32(0.f), acc03 = svdup_n_f32(0.f);
  svfloat32_t acc10 = svdup_n_f32(0.f), acc11 = svdup_n_f32(0.f), acc12 = svdup_n_f32(0.f), acc13 = svdup_n_f32(0.f);
  svfloat32_t acc20 = svdup_n_f32(0.f), acc21 = svdup_n_f32(0.f), acc22 = svdup_n_f32(0.f), acc23 = svdup_n_f32(0.f);
  svfloat32_t acc30 = svdup_n_f32(0.f), acc31 = svdup_n_f32(0.f), acc32 = svdup_n_f32(0.f), acc33 = svdup_n_f32(0.f);
  for (int k2 = 0; k2 < K2; ++k2) {
    svbfloat16_t vb0 = svreinterpret_bf16(svld1_u16(pg16, b + k2 * kN * 2 + 0 * 8));
    svbfloat16_t vb1 = svreinterpret_bf16(svld1_u16(pg16, b + k2 * kN * 2 + 1 * 8));
    svbfloat16_t vb2 = svreinterpret_bf16(svld1_u16(pg16, b + k2 * kN * 2 + 2 * 8));
    svbfloat16_t vb3 = svreinterpret_bf16(svld1_u16(pg16, b + k2 * kN * 2 + 3 * 8));
    if constexpr (ROWS >= 1) {
      svbfloat16_t va = svreinterpret_bf16(svdup_f32(a_pairs[0 * K2 + k2]));
      acc00 = svbfdot_f32(acc00, va, vb0);
      acc01 = svbfdot_f32(acc01, va, vb1);
      acc02 = svbfdot_f32(acc02, va, vb2);
      acc03 = svbfdot_f32(acc03, va, vb3);
    }
    if constexpr (ROWS >= 2) {
      svbfloat16_t va = svreinterpret_bf16(svdup_f32(a_pairs[1 * K2 + k2]));
      acc10 = svbfdot_f32(acc10, va, vb0);
      acc11 = svbfdot_f32(acc11, va, vb1);
      acc12 = svbfdot_f32(acc12, va, vb2);
      acc13 = svbfdot_f32(acc13, va, vb3);
    }
    if constexpr (ROWS >= 3) {
      svbfloat16_t va = svreinterpret_bf16(svdup_f32(a_pairs[2 * K2 + k2]));
      acc20 = svbfdot_f32(acc20, va, vb0);
      acc21 = svbfdot_f32(acc21, va, vb1);
      acc22 = svbfdot_f32(acc22, va, vb2);
      acc23 = svbfdot_f32(acc23, va, vb3);
    }
    if constexpr (ROWS >= 4) {
      svbfloat16_t va = svreinterpret_bf16(svdup_f32(a_pairs[3 * K2 + k2]));
      acc30 = svbfdot_f32(acc30, va, vb0);
      acc31 = svbfdot_f32(acc31, va, vb1);
      acc32 = svbfdot_f32(acc32, va, vb2);
      acc33 = svbfdot_f32(acc33, va, vb3);
    }
  }
  if constexpr (ROWS >= 1) {
    svst1_f32(pgf, out + 0 * kN + 0 * 4, acc00);
    svst1_f32(pgf, out + 0 * kN + 1 * 4, acc01);
    svst1_f32(pgf, out + 0 * kN + 2 * 4, acc02);
    svst1_f32(pgf, out + 0 * kN + 3 * 4, acc03);
  }
  if constexpr (ROWS >= 2) {
    svst1_f32(pgf, out + 1 * kN + 0 * 4, acc10);
    svst1_f32(pgf, out + 1 * kN + 1 * 4, acc11);
    svst1_f32(pgf, out + 1 * kN + 2 * 4, acc12);
    svst1_f32(pgf, out + 1 * kN + 3 * 4, acc13);
  }
  if constexpr (ROWS >= 3) {
    svst1_f32(pgf, out + 2 * kN + 0 * 4, acc20);
    svst1_f32(pgf, out + 2 * kN + 1 * 4, acc21);
    svst1_f32(pgf, out + 2 * kN + 2 * 4, acc22);
    svst1_f32(pgf, out + 2 * kN + 3 * 4, acc23);
  }
  if constexpr (ROWS >= 4) {
    svst1_f32(pgf, out + 3 * kN + 0 * 4, acc30);
    svst1_f32(pgf, out + 3 * kN + 1 * 4, acc31);
    svst1_f32(pgf, out + 3 * kN + 2 * 4, acc32);
    svst1_f32(pgf, out + 3 * kN + 3 * 4, acc33);
  }
}

template <typename Fn, typename PackT>
double run_bench(Fn&& fn, const float* a_pairs, const PackT* b, int K, int iters, int M, std::vector<float>& out) {
  const double t0 = now_sec();
  for (int it = 0; it < iters; ++it) {
    switch (M) {
      case 1: fn.template operator()<1>(a_pairs, b, K, out.data()); break;
      case 2: fn.template operator()<2>(a_pairs, b, K, out.data()); break;
      case 4: fn.template operator()<4>(a_pairs, b, K, out.data()); break;
      default: std::abort();
    }
  }
  return now_sec() - t0;
}

struct RunK2 {
  template <int M, typename PackT>
  void operator()(const float* a_pairs, const PackT* b, int K, float* out) const {
    kernel_fp8_k2<M>(a_pairs, reinterpret_cast<const uint8_t*>(b), K, out);
  }
};

struct RunK2Ld16 {
  template <int M, typename PackT>
  void operator()(const float* a_pairs, const PackT* b, int K, float* out) const {
    kernel_fp8_k2_ld16<M>(a_pairs, reinterpret_cast<const uint8_t*>(b), K, out);
  }
};

struct RunK4 {
  template <int M, typename PackT>
  void operator()(const float* a_pairs, const PackT* b, int K, float* out) const {
    kernel_fp8_k4<M>(a_pairs, reinterpret_cast<const uint8_t*>(b), K, out);
  }
};

struct RunBF16 {
  template <int M, typename PackT>
  void operator()(const float* a_pairs, const PackT* b, int K, float* out) const {
    kernel_bf16_upper<M>(a_pairs, reinterpret_cast<const uint16_t*>(b), K, out);
  }
};

void bench_case(int M, int K) {
  const int K2 = K / 2;
  std::mt19937 rng(1 + M + K);
  std::uniform_real_distribution<float> dist_a(-1.0f, 1.0f);
  std::uniform_int_distribution<int> dist_b(0, 254);

  std::vector<float> a_scalar(M * K);
  std::vector<float> a_pairs(M * K2);
  std::vector<uint8_t> b_fp8(kN * K);

  for (float& x : a_scalar) x = dist_a(rng);
  for (uint8_t& x : b_fp8) x = static_cast<uint8_t>(dist_b(rng));
  for (int m = 0; m < M; ++m) {
    for (int k2 = 0; k2 < K2; ++k2) {
      a_pairs[m * K2 + k2] = pack_bf16_pair(a_scalar[m * K + 2 * k2 + 0], a_scalar[m * K + 2 * k2 + 1]);
    }
  }

  auto packed_k2 = pack_fp8_k2(b_fp8, K);
  auto packed_k4 = pack_fp8_k4_blocked(b_fp8, K);
  auto packed_bf16 = pack_bf16_decoded_k2(b_fp8, K);

  std::vector<float> out0(M * kN), out1(M * kN), out2(M * kN), out3(M * kN);
  switch (M) {
    case 1:
      kernel_fp8_k2<1>(a_pairs.data(), packed_k2.data(), K, out0.data());
      kernel_fp8_k2_ld16<1>(a_pairs.data(), packed_k2.data(), K, out1.data());
      kernel_fp8_k4<1>(a_pairs.data(), packed_k4.data(), K, out2.data());
      kernel_bf16_upper<1>(a_pairs.data(), packed_bf16.data(), K, out3.data());
      break;
    case 2:
      kernel_fp8_k2<2>(a_pairs.data(), packed_k2.data(), K, out0.data());
      kernel_fp8_k2_ld16<2>(a_pairs.data(), packed_k2.data(), K, out1.data());
      kernel_fp8_k4<2>(a_pairs.data(), packed_k4.data(), K, out2.data());
      kernel_bf16_upper<2>(a_pairs.data(), packed_bf16.data(), K, out3.data());
      break;
    case 4:
      kernel_fp8_k2<4>(a_pairs.data(), packed_k2.data(), K, out0.data());
      kernel_fp8_k2_ld16<4>(a_pairs.data(), packed_k2.data(), K, out1.data());
      kernel_fp8_k4<4>(a_pairs.data(), packed_k4.data(), K, out2.data());
      kernel_bf16_upper<4>(a_pairs.data(), packed_bf16.data(), K, out3.data());
      break;
  }

  auto max_diff = [&](const std::vector<float>& lhs, const std::vector<float>& rhs) {
    float d = 0.f;
    for (size_t i = 0; i < lhs.size(); ++i) d = std::max(d, std::abs(lhs[i] - rhs[i]));
    return d;
  };

  const int iters = std::max(100, 32768 / K);
  double t_k2 = run_bench(RunK2{}, a_pairs.data(), packed_k2.data(), K, iters, M, out0);
  double t_k2_ld16 = run_bench(RunK2Ld16{}, a_pairs.data(), packed_k2.data(), K, iters, M, out1);
  double t_k4 = run_bench(RunK4{}, a_pairs.data(), packed_k4.data(), K, iters, M, out2);
  double t_bf = run_bench(RunBF16{}, a_pairs.data(), packed_bf16.data(), K, iters, M, out3);

  const double us_k2 = t_k2 * 1e6 / iters;
  const double us_k2_ld16 = t_k2_ld16 * 1e6 / iters;
  const double us_k4 = t_k4 * 1e6 / iters;
  const double us_bf = t_bf * 1e6 / iters;
  const double gflops_k2 = 2.0 * M * kN * K / (us_k2 * 1e3);
  const double gflops_k2_ld16 = 2.0 * M * kN * K / (us_k2_ld16 * 1e3);
  const double gflops_k4 = 2.0 * M * kN * K / (us_k4 * 1e3);
  const double gflops_bf = 2.0 * M * kN * K / (us_bf * 1e3);

  std::printf("\nM=%d N=%d K=%d iters=%d\n", M, kN, K, iters);
  std::printf("current k2 layout : %8.3f us  %8.2f GFLOP/s\n", us_k2, gflops_k2);
  std::printf("k2 full16-load    : %8.3f us  %8.2f GFLOP/s  diff_vs_k2=%.6f\n", us_k2_ld16, gflops_k2_ld16, max_diff(out1, out0));
  std::printf("blocked k4 layout : %8.3f us  %8.2f GFLOP/s  diff_vs_k2=%.6f\n", us_k4, gflops_k4, max_diff(out2, out0));
  std::printf("bf16 upper bound  : %8.3f us  %8.2f GFLOP/s  diff_vs_k2=%.6f\n", us_bf, gflops_bf, max_diff(out3, out0));
}

}  // namespace

int main() {
  const uint64_t vl = svcntw();
  std::printf("SVE vector length: %lu bits\n", vl * 32);
  if (vl != 4) {
    std::printf("This benchmark is intended for SVE-128 only.\n");
    return 0;
  }

  bench_case(1, 4096);
  bench_case(2, 4096);
  bench_case(4, 4096);
  bench_case(1, 8192);
  bench_case(2, 8192);
  bench_case(4, 8192);
  return 0;
}
