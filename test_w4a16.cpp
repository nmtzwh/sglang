#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <algorithm>

template <typename act_dtype, typename out_dtype, int64_t block_n>
void _da16w4_linear_impl_pure_cpp(
    const act_dtype* __restrict__ input,
    const uint8_t* __restrict__ weight,
    const float* __restrict__ weight_scales,
    const int8_t* __restrict__ weight_qzeros,
    const float* __restrict__ bias,
    out_dtype* __restrict__ output,
    float* __restrict__ output_temp,
    int64_t M, int64_t N, int64_t K, int64_t lda, int64_t num_groups) {

  int64_t block_m = (M <= 48) ? M : (M < 64 ? 32 : (M < 96 ? 64 : 128));
  int64_t Mc = (M + block_m - 1) / block_m;
  int64_t Nc = N / block_n;
  int64_t num_blocks = Mc * Nc;
  int64_t group_size = (K + num_groups - 1) / num_groups;
  int64_t _block_k = group_size; // simplified block_k
  int64_t Kc = K / _block_k;
  int64_t block_per_group = group_size / _block_k;

  for (int64_t i = 0; i < num_blocks; ++i) {
      int64_t mc = 0;
      int64_t nc = i; // simplifies logic for small M just iterating Nc
      int64_t mc_end = Mc;

      for (int mci = mc; mci < mc_end; ++mci) {
        int64_t m_size = std::min((int64_t)block_m, M - mci * block_m);
        float* C_tmp = output_temp;
        if (bias) {
           for(int m=0; m<m_size; ++m) {
              for(int n=0; n<block_n; ++n) C_tmp[m * block_n + n] = bias[nc * block_n + n];
           }
        } else {
           for(int m=0; m<m_size; ++m) {
              for(int n=0; n<block_n; ++n) C_tmp[m * block_n + n] = 0;
           }
        }
        for (int kci = 0; kci < Kc; ++kci) {
           // We mimic the exact unpacking offset
           const uint8_t* B_block = weight + (nc * Kc + kci) * (block_n * (_block_k / 2 + sizeof(int32_t)));
           const float* scales_block = weight_scales + nc * block_n * num_groups + kci / block_per_group * block_n;
           const int8_t* qzeros_block = weight_qzeros + nc * block_n * num_groups + kci / block_per_group * block_n;
           const act_dtype* A_block = input + mci * block_m * lda + kci * _block_k;

           for (int m = 0; m < m_size; ++m) {
              for (int n = 0; n < block_n; ++n) {
                 float sum = 0;
                 int8_t zp = qzeros_block[n];
                 for (int k = 0; k < _block_k; ++k) {
                    float a_val = static_cast<float>(A_block[m * lda + k]);
                    int k_out = k / 4;
                    int k_in = k % 4;
                    int n_out = (n / 16) * 8 + (n % 8);
                    int n_in = (n / 8) % 2;
                    uint8_t packed = B_block[k_out * (block_n * 2) + n_out * 4 + k_in];
                    int8_t b_val;
                    if (n_in == 0) b_val = (packed & 0xf) - zp;
                    else b_val = (packed >> 4) - zp;
                    sum += a_val * static_cast<float>(b_val);
                 }
                 C_tmp[m * block_n + n] += sum * scales_block[n];
              }
           }
        }
        for(int m=0; m<m_size; ++m) {
           for(int n=0; n<block_n; ++n) {
              output[(mci * block_m + m) * N + nc * block_n + n] = static_cast<out_dtype>(C_tmp[m * block_n + n]);
           }
        }
      }
  }
}

int main() {
    int64_t M = 2, N = 32, K = 32, block_n = 32, num_groups = 1;

    std::vector<float> input(M * K);
    std::vector<uint8_t> weight(N * K / 2 + N * sizeof(int32_t)); // Include space for compensation
    std::vector<float> weight_scales(N * num_groups, 1.0f);
    std::vector<int8_t> weight_qzeros(N * num_groups, 8);
    std::vector<float> output(M * N, 0.0f);
    std::vector<float> output_temp(128 * 256, 0.0f);
    std::vector<float> bias(N, 0.0f);

    std::vector<int8_t> ref_w(N * K, 0);

    for (int i = 0; i < M * K; ++i) input[i] = (rand() % 100) / 100.0f;
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            int k_out = k / 4;
            int k_in = k % 4;
            int n_out = (n / 16) * 8 + (n % 8);
            int n_in = (n / 8) % 2;
            int idx = k_out * (block_n * 2) + n_out * 4 + k_in;
            int val = rand() % 16;
            ref_w[k * N + n] = val - 8;
            if (n_in == 0) {
               weight[idx] = (weight[idx] & 0xf0) | (val & 0xf);
            } else {
               weight[idx] = (weight[idx] & 0x0f) | ((val & 0xf) << 4);
            }
        }
    }

    _da16w4_linear_impl_pure_cpp<float, float, 32>(
      input.data(), weight.data(), weight_scales.data(), weight_qzeros.data(), bias.data(),
      output.data(), output_temp.data(), M, N, K, K, num_groups);

    // Check vs expected
    for (int m = 0; m < M; ++m) {
       for (int n = 0; n < N; ++n) {
          float expected = 0;
          for (int k = 0; k < K; ++k) {
             expected += input[m * K + k] * ref_w[k * N + n];
          }
          if (std::abs(expected - output[m * N + n]) > 1e-4) {
             std::cout << "Mismatch at " << m << "," << n << ": expected " << expected << " got " << output[m * N + n] << "\n";
             return 1;
          }
       }
    }
    std::cout << "Test passed!\n";
    return 0;
}
