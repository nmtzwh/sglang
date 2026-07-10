#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/record_function.h>
#include <torch/all.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include "common.h"
#include "vec.h"

namespace {

template <typename scalar_t>
inline float load_logit(const scalar_t* ptr, int64_t idx) {
  return static_cast<float>(ptr[idx]);
}

template <typename scalar_t>
void sample_logits_kernel(
    const scalar_t* __restrict__ logits,
    const float* __restrict__ temperatures,
    const float* __restrict__ uniforms,
    int32_t* __restrict__ output,
    int64_t batch,
    int64_t vocab) {
  at::parallel_for(0, batch, 0, [&](int64_t begin, int64_t end) {
    for (int64_t b = begin; b < end; ++b) {
      const scalar_t* row = logits + b * vocab;
      const float inv_temp = 1.0f / temperatures[b];

      float row_max = -std::numeric_limits<float>::infinity();
      for (int64_t v = 0; v < vocab; ++v) {
        const float val = load_logit(row, v) * inv_temp;
        if (val > row_max) {
          row_max = val;
        }
      }

      float sum = 0.0f;
      for (int64_t v = 0; v < vocab; ++v) {
        sum += std::exp(load_logit(row, v) * inv_temp - row_max);
      }

      const float target = uniforms[b] * sum;
      float cumsum = 0.0f;
      int64_t selected = vocab - 1;
      for (int64_t v = 0; v < vocab; ++v) {
        cumsum += std::exp(load_logit(row, v) * inv_temp - row_max);
        if (cumsum >= target) {
          selected = v;
          break;
        }
      }
      output[b] = static_cast<int32_t>(selected);
    }
  });
}

#if defined(CPU_CAPABILITY_SVE)
inline svfloat32_t load_logits_sve(const float* row, int64_t offset, int64_t remaining, svbool_t pg) {
  return svld1_f32(pg, row + offset);
}

inline svfloat32_t load_logits_sve(const at::BFloat16* row, int64_t offset, int64_t remaining, svbool_t pg) {
  const uint64_t lanes = std::min<int64_t>(svcntw(), remaining);
  const svbool_t pg16 = svwhilelt_b16(0u, static_cast<uint32_t>(lanes));
  const svbfloat16_t values = svreinterpret_bf16(svld1_u16(pg16, reinterpret_cast<const uint16_t*>(row + offset)));
  return sve_cvt_bf16_to_fp32_low(values);
}

template <typename scalar_t>
void sample_logits_kernel_sve(
    const scalar_t* __restrict__ logits,
    const float* __restrict__ temperatures,
    const float* __restrict__ uniforms,
    int32_t* __restrict__ output,
    int64_t batch,
    int64_t vocab) {
  at::parallel_for(0, batch, 0, [&](int64_t begin, int64_t end) {
    const int64_t lanes = svcntw();
    std::vector<float> scratch(lanes);
    for (int64_t b = begin; b < end; ++b) {
      const scalar_t* row = logits + b * vocab;
      const float inv_temp = 1.0f / temperatures[b];
      const svfloat32_t vinv_temp = svdup_f32(inv_temp);

      svfloat32_t vmax = svdup_f32(-std::numeric_limits<float>::infinity());
      for (int64_t v = 0; v < vocab; v += lanes) {
        const int64_t remaining = vocab - v;
        const svbool_t pg = svwhilelt_b32(0u, static_cast<uint32_t>(remaining));
        svfloat32_t values = load_logits_sve(row, v, remaining, pg);
        values = svmul_f32_x(pg, values, vinv_temp);
        vmax = svmax_f32_m(pg, vmax, values);
      }
      const float row_max = svmaxv_f32(svptrue_b32(), vmax);
      const svfloat32_t vrow_max = svdup_f32(row_max);

      svfloat32_t vsum = svdup_f32(0.0f);
      for (int64_t v = 0; v < vocab; v += lanes) {
        const int64_t remaining = vocab - v;
        const svbool_t pg = svwhilelt_b32(0u, static_cast<uint32_t>(remaining));
        svfloat32_t values = load_logits_sve(row, v, remaining, pg);
        values = svmul_f32_x(pg, values, vinv_temp);
        const svfloat32_t weights = sve_fexp_u20(pg, svsub_f32_x(pg, values, vrow_max));
        vsum = svadd_f32_m(pg, vsum, weights);
      }
      const float target = uniforms[b] * svaddv_f32(svptrue_b32(), vsum);

      float cumsum = 0.0f;
      int32_t selected = static_cast<int32_t>(vocab - 1);
      bool found = false;
      for (int64_t v = 0; v < vocab && !found; v += lanes) {
        const int64_t remaining = vocab - v;
        const int64_t valid = std::min(lanes, remaining);
        const svbool_t pg = svwhilelt_b32(0u, static_cast<uint32_t>(remaining));
        svfloat32_t values = load_logits_sve(row, v, remaining, pg);
        values = svmul_f32_x(pg, values, vinv_temp);
        const svfloat32_t weights = sve_fexp_u20(pg, svsub_f32_x(pg, values, vrow_max));
        svst1_f32(pg, scratch.data(), weights);
        for (int64_t i = 0; i < valid; ++i) {
          cumsum += scratch[i];
          if (cumsum >= target) {
            selected = static_cast<int32_t>(v + i);
            found = true;
            break;
          }
        }
      }
      output[b] = selected;
    }
  });
}
#endif

struct Candidate {
  float logit;
  float weight;
  int32_t index;
};

template <typename scalar_t>
float scaled_row_max(const scalar_t* row, int64_t vocab, float inv_temp) {
#if defined(CPU_CAPABILITY_SVE)
  if constexpr (std::is_same_v<scalar_t, float> || std::is_same_v<scalar_t, at::BFloat16>) {
    const int64_t lanes = svcntw();
    const svfloat32_t vinv_temp = svdup_f32(inv_temp);
    svfloat32_t vmax = svdup_f32(-std::numeric_limits<float>::infinity());
    for (int64_t v = 0; v < vocab; v += lanes) {
      const int64_t remaining = vocab - v;
      const svbool_t pg = svwhilelt_b32(0u, static_cast<uint32_t>(remaining));
      svfloat32_t values = load_logits_sve(row, v, remaining, pg);
      values = svmul_f32_x(pg, values, vinv_temp);
      vmax = svmax_f32_m(pg, vmax, values);
    }
    return svmaxv_f32(svptrue_b32(), vmax);
  }
#endif
  float row_max = -std::numeric_limits<float>::infinity();
  for (int64_t v = 0; v < vocab; ++v) {
    row_max = std::max(row_max, load_logit(row, v) * inv_temp);
  }
  return row_max;
}

template <typename scalar_t>
float fill_candidate_weights(
    const scalar_t* row, std::vector<Candidate>& candidates, int64_t vocab, float inv_temp, float row_max) {
#if defined(CPU_CAPABILITY_SVE)
  if constexpr (std::is_same_v<scalar_t, float> || std::is_same_v<scalar_t, at::BFloat16>) {
    const int64_t lanes = svcntw();
    const svfloat32_t vinv_temp = svdup_f32(inv_temp);
    const svfloat32_t vrow_max = svdup_f32(row_max);
    svfloat32_t vsum = svdup_f32(0.0f);
    std::vector<float> scratch(lanes);
    for (int64_t v = 0; v < vocab; v += lanes) {
      const int64_t remaining = vocab - v;
      const int64_t valid = std::min(lanes, remaining);
      const svbool_t pg = svwhilelt_b32(0u, static_cast<uint32_t>(remaining));
      svfloat32_t values = load_logits_sve(row, v, remaining, pg);
      values = svmul_f32_x(pg, values, vinv_temp);
      const svfloat32_t weights = sve_fexp_u20(pg, svsub_f32_x(pg, values, vrow_max));
      vsum = svadd_f32_m(pg, vsum, weights);
      svst1_f32(pg, scratch.data(), weights);
      for (int64_t i = 0; i < valid; ++i) {
        candidates[v + i].weight = scratch[i];
      }
    }
    return svaddv_f32(svptrue_b32(), vsum);
  }
#endif
  float sum = 0.0f;
  for (int64_t v = 0; v < vocab; ++v) {
    candidates[v].weight = std::exp(load_logit(row, v) * inv_temp - row_max);
    sum += candidates[v].weight;
  }
  return sum;
}

template <typename scalar_t>
float sum_exp_weights(const scalar_t* row, int64_t vocab, float inv_temp, float row_max) {
#if defined(CPU_CAPABILITY_SVE)
  if constexpr (std::is_same_v<scalar_t, float> || std::is_same_v<scalar_t, at::BFloat16>) {
    const int64_t lanes = svcntw();
    const svfloat32_t vinv_temp = svdup_f32(inv_temp);
    const svfloat32_t vrow_max = svdup_f32(row_max);
    svfloat32_t vsum = svdup_f32(0.0f);
    for (int64_t v = 0; v < vocab; v += lanes) {
      const int64_t remaining = vocab - v;
      const svbool_t pg = svwhilelt_b32(0u, static_cast<uint32_t>(remaining));
      svfloat32_t values = load_logits_sve(row, v, remaining, pg);
      values = svmul_f32_x(pg, values, vinv_temp);
      const svfloat32_t weights = sve_fexp_u20(pg, svsub_f32_x(pg, values, vrow_max));
      vsum = svadd_f32_m(pg, vsum, weights);
    }
    return svaddv_f32(svptrue_b32(), vsum);
  }
#endif
  float sum = 0.0f;
  for (int64_t v = 0; v < vocab; ++v) {
    sum += std::exp(load_logit(row, v) * inv_temp - row_max);
  }
  return sum;
}

inline bool candidate_greater(const Candidate& lhs, const Candidate& rhs) {
  return lhs.logit > rhs.logit || (lhs.logit == rhs.logit && lhs.index < rhs.index);
}

Candidate find_top_p_boundary(std::vector<Candidate>& candidates, float limit) {
  auto first = candidates.begin();
  auto last = candidates.end();
  while (last - first > 1) {
    auto pivot = first + (last - first) / 2;
    std::nth_element(first, pivot, last, candidate_greater);

    float upper_sum = 0.0f;
    for (auto it = first; it != pivot; ++it) {
      upper_sum += it->weight;
    }

    if (upper_sum > limit) {
      last = pivot;
    } else if (upper_sum + pivot->weight > limit) {
      return *pivot;
    } else {
      limit -= upper_sum + pivot->weight;
      first = pivot + 1;
    }
  }
  return *first;
}

template <typename scalar_t>
void sample_top_k_top_p_logits_kernel(
    const scalar_t* __restrict__ logits,
    const float* __restrict__ temperatures,
    const int32_t* __restrict__ top_ks,
    const float* __restrict__ top_ps,
    const float* __restrict__ uniforms,
    int32_t* __restrict__ output,
    int64_t batch,
    int64_t vocab) {
  at::parallel_for(0, batch, 0, [&](int64_t begin, int64_t end) {
    std::vector<Candidate> candidates;
    candidates.reserve(vocab);
    for (int64_t b = begin; b < end; ++b) {
      const scalar_t* row = logits + b * vocab;
      const float inv_temp = 1.0f / temperatures[b];
      const int64_t top_k = std::max<int64_t>(1, std::min<int64_t>(static_cast<int64_t>(top_ks[b]), vocab));
      const float top_p = top_ps[b];

      if (top_k == 1) {
        int32_t selected = 0;
        float selected_logit = load_logit(row, 0);
        for (int64_t v = 1; v < vocab; ++v) {
          const float logit = load_logit(row, v);
          if (logit > selected_logit) {
            selected_logit = logit;
            selected = static_cast<int32_t>(v);
          }
        }
        output[b] = selected;
        continue;
      }

      candidates.clear();
      for (int64_t v = 0; v < vocab; ++v) {
        candidates.push_back({load_logit(row, v), 0.0f, static_cast<int32_t>(v)});
      }

      // Temperature scaling does not change rank, so select candidates using
      // the original logits. A finite top-k only needs O(vocab) selection and
      // an O(top-k log(top-k)) ordering step.
      if (top_k == vocab && top_p < 1.0f) {
        const float row_max = scaled_row_max(row, vocab, inv_temp);
        const float full_sum = fill_candidate_weights(row, candidates, vocab, inv_temp, row_max);

        // Find the first ranked token that crosses the top-p mass without a
        // full-vocabulary sort. nth_element recursively discards one side of
        // the rank space, giving expected O(vocab) selection.
        const Candidate boundary = find_top_p_boundary(candidates, top_p * full_sum);
        float retained_sum = 0.0f;
        for (const auto& candidate : candidates) {
          if (candidate_greater(candidate, boundary) || candidate.index == boundary.index) {
            retained_sum += candidate.weight;
          }
        }

        const float target = uniforms[b] * retained_sum;
        float cumsum = 0.0f;
        int32_t selected = boundary.index;
        for (const auto& candidate : candidates) {
          if (candidate_greater(candidate, boundary) || candidate.index == boundary.index) {
            cumsum += candidate.weight;
            if (cumsum >= target) {
              selected = candidate.index;
              break;
            }
          }
        }
        output[b] = selected;
        continue;
      }

      if (top_k < vocab) {
        std::nth_element(candidates.begin(), candidates.begin() + top_k, candidates.end(), candidate_greater);
        candidates.resize(top_k);
      }
      std::sort(candidates.begin(), candidates.end(), candidate_greater);

      const float row_max = candidates.front().logit * inv_temp;
      float full_sum = 0.0f;
      if (top_p < 1.0f) {
        // top-p is defined using probabilities from the full softmax, even
        // when top-k is also enabled.
        full_sum = sum_exp_weights(row, vocab, inv_temp, row_max);
      }

      const float nucleus_limit = top_p * full_sum;
      float retained_sum = 0.0f;
      int64_t retained = 0;
      for (; retained < static_cast<int64_t>(candidates.size()); ++retained) {
        // Match the PyTorch fallback: remove an item only when probability
        // mass preceding it is strictly greater than top-p.
        if (top_p < 1.0f && retained_sum > nucleus_limit) {
          break;
        }
        retained_sum += std::exp(candidates[retained].logit * inv_temp - row_max);
      }

      const float target = uniforms[b] * retained_sum;
      float cumsum = 0.0f;
      int32_t selected = candidates[retained - 1].index;
      for (int64_t i = 0; i < retained; ++i) {
        cumsum += std::exp(candidates[i].logit * inv_temp - row_max);
        if (cumsum >= target) {
          selected = candidates[i].index;
          break;
        }
      }
      output[b] = selected;
    }
  });
}

}  // namespace

at::Tensor sample_logits_cpu(at::Tensor& logits, at::Tensor& temperatures) {
  RECORD_FUNCTION("sgl-kernel::sample_logits_cpu", std::vector<c10::IValue>({logits, temperatures}));

  CHECK_INPUT(logits);
  CHECK_INPUT(temperatures);
  CHECK_DIM(2, logits);
  TORCH_CHECK(
      temperatures.dim() == 1 || (temperatures.dim() == 2 && temperatures.size(1) == 1),
      "temperatures must be shaped [batch] or [batch, 1]");
  CHECK_EQ(logits.size(0), temperatures.size(0));
  TORCH_CHECK(temperatures.scalar_type() == at::kFloat, "temperatures must be float32");
  TORCH_CHECK(
      logits.scalar_type() == at::kFloat || logits.scalar_type() == at::kBFloat16,
      "logits must be float32 or bfloat16");

  const auto batch = logits.size(0);
  const auto vocab = logits.size(1);
  auto output = at::empty({batch}, logits.options().dtype(at::kInt));
  auto uniforms = at::rand({batch}, logits.options().dtype(at::kFloat));

  AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16, logits.scalar_type(), "sample_logits_cpu", [&] {
#if defined(CPU_CAPABILITY_SVE)
    if constexpr (std::is_same_v<scalar_t, float> || std::is_same_v<scalar_t, at::BFloat16>) {
      sample_logits_kernel_sve<scalar_t>(
          logits.data_ptr<scalar_t>(),
          temperatures.data_ptr<float>(),
          uniforms.data_ptr<float>(),
          output.data_ptr<int32_t>(),
          batch,
          vocab);
    } else {
      sample_logits_kernel<scalar_t>(
          logits.data_ptr<scalar_t>(),
          temperatures.data_ptr<float>(),
          uniforms.data_ptr<float>(),
          output.data_ptr<int32_t>(),
          batch,
          vocab);
    }
#else
    sample_logits_kernel<scalar_t>(
        logits.data_ptr<scalar_t>(),
        temperatures.data_ptr<float>(),
        uniforms.data_ptr<float>(),
        output.data_ptr<int32_t>(),
        batch,
        vocab);
#endif
  });

  return output;
}

at::Tensor
sample_top_k_top_p_logits_cpu(at::Tensor& logits, at::Tensor& temperatures, at::Tensor& top_ks, at::Tensor& top_ps) {
  RECORD_FUNCTION(
      "sgl-kernel::sample_top_k_top_p_logits_cpu", std::vector<c10::IValue>({logits, temperatures, top_ks, top_ps}));

  CHECK_INPUT(logits);
  CHECK_INPUT(temperatures);
  CHECK_INPUT(top_ks);
  CHECK_INPUT(top_ps);
  CHECK_DIM(2, logits);
  TORCH_CHECK(
      temperatures.dim() == 1 || (temperatures.dim() == 2 && temperatures.size(1) == 1),
      "temperatures must be shaped [batch] or [batch, 1]");
  TORCH_CHECK(
      top_ks.dim() == 1 || (top_ks.dim() == 2 && top_ks.size(1) == 1), "top_ks must be shaped [batch] or [batch, 1]");
  TORCH_CHECK(
      top_ps.dim() == 1 || (top_ps.dim() == 2 && top_ps.size(1) == 1), "top_ps must be shaped [batch] or [batch, 1]");
  CHECK_EQ(logits.size(0), temperatures.size(0));
  CHECK_EQ(logits.size(0), top_ks.size(0));
  CHECK_EQ(logits.size(0), top_ps.size(0));
  TORCH_CHECK(temperatures.scalar_type() == at::kFloat, "temperatures must be float32");
  TORCH_CHECK(top_ks.scalar_type() == at::kInt, "top_ks must be int32");
  TORCH_CHECK(top_ps.scalar_type() == at::kFloat, "top_ps must be float32");
  TORCH_CHECK(
      logits.scalar_type() == at::kFloat || logits.scalar_type() == at::kBFloat16,
      "logits must be float32 or bfloat16");

  const auto batch = logits.size(0);
  const auto vocab = logits.size(1);
  auto output = at::empty({batch}, logits.options().dtype(at::kInt));
  auto uniforms = at::rand({batch}, logits.options().dtype(at::kFloat));

  AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16, logits.scalar_type(), "sample_top_k_top_p_logits_cpu", [&] {
    sample_top_k_top_p_logits_kernel<scalar_t>(
        logits.data_ptr<scalar_t>(),
        temperatures.data_ptr<float>(),
        top_ks.data_ptr<int32_t>(),
        top_ps.data_ptr<float>(),
        uniforms.data_ptr<float>(),
        output.data_ptr<int32_t>(),
        batch,
        vocab);
  });

  return output;
}
