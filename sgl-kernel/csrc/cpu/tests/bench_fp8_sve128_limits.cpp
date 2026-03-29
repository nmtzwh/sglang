#include <arm_sve.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

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

inline double now_sec() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

template <typename T>
std::vector<T> make_buffer(size_t count, uint32_t seed) {
  std::vector<T> out(count);
  uint32_t x = seed;
  for (size_t i = 0; i < count; ++i) {
    x = x * 1664525u + 1013904223u;
    out[i] = static_cast<T>(x);
  }
  return out;
}

double bench_raw_fp8_load(const uint8_t* data, size_t bytes, int iters, uint64_t& sink) {
  const uint64_t vl_u8 = svcntb();
  const svbool_t pg_u8 = svptrue_b8();
  svuint8_t vacc = svdup_u8(0);
  double t0 = now_sec();
  for (int it = 0; it < iters; ++it) {
    for (size_t off = 0; off + vl_u8 <= bytes; off += vl_u8) {
      vacc = sveor_u8_x(pg_u8, vacc, svld1_u8(pg_u8, data + off));
    }
  }
  double t1 = now_sec();
  alignas(16) uint8_t tmp[32] = {};
  svst1_u8(pg_u8, tmp, vacc);
  for (uint64_t i = 0; i < vl_u8; ++i) sink += tmp[i];
  return t1 - t0;
}

double bench_fp8_decode(const uint8_t* data, size_t bytes, int iters, uint64_t& sink) {
  const uint64_t vl_u8 = svcntb();
  const svbool_t pg_u8 = svptrue_b8();
  const svbool_t pg_u16 = svptrue_b16();
  svuint16_t vacc = svdup_u16(0);
  double t0 = now_sec();
  for (int it = 0; it < iters; ++it) {
    for (size_t off = 0; off + vl_u8 <= bytes; off += vl_u8) {
      svbfloat16_t vbf = cvt_fp8_to_bf16_ext(svld1_u8(pg_u8, data + off));
      vacc = svadd_u16_x(pg_u16, vacc, svreinterpret_u16(vbf));
    }
  }
  double t1 = now_sec();
  alignas(16) uint16_t tmp[16] = {};
  svst1_u16(pg_u16, tmp, vacc);
  for (uint64_t i = 0; i < svcnth(); ++i) sink += tmp[i];
  return t1 - t0;
}

double bench_bfdot_only(const uint16_t* data, size_t elems, int iters, int rows, float& sink) {
  const uint64_t vl_bf16 = svcnth();
  const svbool_t pg_b16 = svptrue_b16();
  const svbool_t pg_f32 = svptrue_b32();
  svbfloat16_t va0 = svreinterpret_bf16(svdup_f32(0x1.0p0f));
  svbfloat16_t va1 = svreinterpret_bf16(svdup_f32(0x1.8p0f));
  svbfloat16_t va2 = svreinterpret_bf16(svdup_f32(0x1.0p1f));
  svbfloat16_t va3 = svreinterpret_bf16(svdup_f32(0x1.4p1f));
  svfloat32_t acc0 = svdup_n_f32(0.f);
  svfloat32_t acc1 = svdup_n_f32(0.f);
  svfloat32_t acc2 = svdup_n_f32(0.f);
  svfloat32_t acc3 = svdup_n_f32(0.f);
  double t0 = now_sec();
  for (int it = 0; it < iters; ++it) {
    for (size_t off = 0; off + vl_bf16 <= elems; off += vl_bf16) {
      svbfloat16_t vb = svreinterpret_bf16(svld1_u16(pg_b16, data + off));
      if (rows >= 1) acc0 = svbfdot_f32(acc0, va0, vb);
      if (rows >= 2) acc1 = svbfdot_f32(acc1, va1, vb);
      if (rows >= 3) acc2 = svbfdot_f32(acc2, va2, vb);
      if (rows >= 4) acc3 = svbfdot_f32(acc3, va3, vb);
    }
  }
  double t1 = now_sec();
  sink += svaddv_f32(pg_f32, acc0);
  sink += svaddv_f32(pg_f32, acc1);
  sink += svaddv_f32(pg_f32, acc2);
  sink += svaddv_f32(pg_f32, acc3);
  return t1 - t0;
}

double bench_fp8_decode_bfdot(const uint8_t* data, size_t bytes, int iters, int rows, float& sink) {
  const uint64_t vl_u8 = svcntb();
  const svbool_t pg_u8 = svptrue_b8();
  const svbool_t pg_f32 = svptrue_b32();
  svbfloat16_t va0 = svreinterpret_bf16(svdup_f32(0x1.0p0f));
  svbfloat16_t va1 = svreinterpret_bf16(svdup_f32(0x1.8p0f));
  svbfloat16_t va2 = svreinterpret_bf16(svdup_f32(0x1.0p1f));
  svbfloat16_t va3 = svreinterpret_bf16(svdup_f32(0x1.4p1f));
  svfloat32_t acc0 = svdup_n_f32(0.f);
  svfloat32_t acc1 = svdup_n_f32(0.f);
  svfloat32_t acc2 = svdup_n_f32(0.f);
  svfloat32_t acc3 = svdup_n_f32(0.f);
  double t0 = now_sec();
  for (int it = 0; it < iters; ++it) {
    for (size_t off = 0; off + vl_u8 <= bytes; off += vl_u8) {
      svbfloat16_t vb = cvt_fp8_to_bf16_ext(svld1_u8(pg_u8, data + off));
      if (rows >= 1) acc0 = svbfdot_f32(acc0, va0, vb);
      if (rows >= 2) acc1 = svbfdot_f32(acc1, va1, vb);
      if (rows >= 3) acc2 = svbfdot_f32(acc2, va2, vb);
      if (rows >= 4) acc3 = svbfdot_f32(acc3, va3, vb);
    }
  }
  double t1 = now_sec();
  sink += svaddv_f32(pg_f32, acc0);
  sink += svaddv_f32(pg_f32, acc1);
  sink += svaddv_f32(pg_f32, acc2);
  sink += svaddv_f32(pg_f32, acc3);
  return t1 - t0;
}

}  // namespace

int main() {
  const uint64_t vl_bits = svcntw() * 32;
  std::printf("SVE vector length: %lu bits\n", vl_bits);

  const size_t sizes[] = {
      32ull * 1024,
      512ull * 1024,
      4ull * 1024 * 1024,
      64ull * 1024 * 1024,
  };

  uint64_t sink_u64 = 0;
  float sink_f32 = 0.f;

  for (size_t fp8_bytes : sizes) {
    const size_t bf16_elems = fp8_bytes / sizeof(uint16_t);
    const int iters = std::max(20, static_cast<int>((256ull * 1024 * 1024) / fp8_bytes));
    auto fp8 = make_buffer<uint8_t>(fp8_bytes, 1u);
    auto bf16 = make_buffer<uint16_t>(bf16_elems, 2u);

    double t_load = bench_raw_fp8_load(fp8.data(), fp8_bytes, iters, sink_u64);
    double t_decode = bench_fp8_decode(fp8.data(), fp8_bytes, iters, sink_u64);
    double t_bfdot1 = bench_bfdot_only(bf16.data(), bf16_elems, iters, 1, sink_f32);
    double t_bfdot4 = bench_bfdot_only(bf16.data(), bf16_elems, iters, 4, sink_f32);
    double t_mix1 = bench_fp8_decode_bfdot(fp8.data(), fp8_bytes, iters, 1, sink_f32);
    double t_mix4 = bench_fp8_decode_bfdot(fp8.data(), fp8_bytes, iters, 4, sink_f32);

    const double total_fp8_gb = static_cast<double>(fp8_bytes) * iters / 1e9;
    const double total_bf16_gb = static_cast<double>(bf16_elems * sizeof(uint16_t)) * iters / 1e9;

    std::printf("\nWorking set: %.1f KiB, iters=%d\n", static_cast<double>(fp8_bytes) / 1024.0, iters);
    std::printf("Raw FP8 load       : %.2f GB/s\n", total_fp8_gb / t_load);
    std::printf("FP8 decode         : %.2f GB/s\n", total_fp8_gb / t_decode);
    std::printf("BF16 bfdot x1      : %.2f GB/s input\n", total_bf16_gb / t_bfdot1);
    std::printf("BF16 bfdot x4      : %.2f GB/s input\n", total_bf16_gb / t_bfdot4);
    std::printf("FP8 decode+bfdot x1: %.2f GB/s input\n", total_fp8_gb / t_mix1);
    std::printf("FP8 decode+bfdot x4: %.2f GB/s input\n", total_fp8_gb / t_mix4);
  }

  std::printf("\nSinks: %llu %.3f\n", static_cast<unsigned long long>(sink_u64), static_cast<double>(sink_f32));
  return 0;
}
