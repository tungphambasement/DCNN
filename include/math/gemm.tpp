#pragma once

#include "gemm.hpp"

namespace tmath {
constexpr int BLOCK_SIZE = 64; // Tunable block size for cache efficiency

#ifdef __AVX2__
inline void sgemm_kernel_avx2(const float *A, const float *B, float *C, const int M, const int N,
                              const int K, const int i, const int j, const int k, const int i_max,
                              const int j_max, const int k_max) {
  for (int ii = i; ii < i_max; ++ii) {
    int jj = j;
    // Process 8 columns at a time with AVX2
    for (; jj + 7 < j_max; jj += 8) {
      // Initialize accumulator for 8 output elements to zero
      __m256 c_vec = _mm256_setzero_ps();
      // Accumulate over K dimension
      for (int kk = k; kk < k_max; ++kk) {
        // Broadcast A[ii, kk] to all 8 lanes
        __m256 a_vec = _mm256_set1_ps(A[ii * K + kk]);
        // Load 8 consecutive elements from B[kk, jj:jj+8]
        __m256 b_vec = _mm256_loadu_ps(&B[kk * N + jj]);
        // Fused multiply-add: c_vec += a_vec * b_vec
        c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
      }
      // Add the computed inner-block product to the existing C values.
      // The original code was loading `C` at the start, this is the correct way
      // to handle the accumulation from different `k` blocks.
      __m256 c_current = _mm256_loadu_ps(&C[ii * N + jj]);
      c_vec = _mm256_add_ps(c_vec, c_current);
      // Store result back to C
      _mm256_storeu_ps(&C[ii * N + jj], c_vec);
    }

    // Handle remaining columns (< 8) with scalar code
    for (; jj < j_max; ++jj) {
      float sum = 0.0f; // Initialize sum to zero for the block
      for (int kk = k; kk < k_max; ++kk) {
        sum += A[ii * K + kk] * B[kk * N + jj];
      }
      C[ii * N + jj] += sum;
    }
  }
}
#endif

void sgemm(const float *A, const float *B, float *C, const int M, const int N, const int K) {
#ifdef __AVX2__
  int M_blocks = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
  int N_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
  utils::parallel_for_2d(M_blocks, N_blocks, [&](int block_i, int block_j) {
    int i = block_i * BLOCK_SIZE;
    int j = block_j * BLOCK_SIZE;
    for (int k = 0; k < K; k += BLOCK_SIZE) {
      int i_max = std::min(i + BLOCK_SIZE, M);
      int j_max = std::min(j + BLOCK_SIZE, N);
      int k_max = std::min(k + BLOCK_SIZE, K);

      sgemm_kernel_avx2(A, B, C, M, N, K, i, j, k, i_max, j_max, k_max);
    }
  });
#else
  for (int i = 0; i < M; i += BLOCK_SIZE) {
    for (int j = 0; j < N; j += BLOCK_SIZE) {
      for (int k = 0; k < K; k += BLOCK_SIZE) {
        int i_max = std::min(i + BLOCK_SIZE, M);
        int j_max = std::min(j + BLOCK_SIZE, N);
        int k_max = std::min(k + BLOCK_SIZE, K);
        for (int ii = i; ii < i_max; ++ii) {
          for (int jj = j; jj < j_max; ++jj) {
            float sum = C[ii * N + jj];
            for (int kk = k; kk < k_max; ++kk) {
              sum += A[ii * K + kk] * B[kk * N + jj];
            }
            C[ii * N + jj] = sum;
          }
        }
      }
    }
  }
#endif
}

void old_gemm(const float *A, const float *B, float *C, const int M, const int N, const int K) {
  float *B_T = (float *)malloc(sizeof(float) * K * N);
  utils::transpose_2d(B, B_T, K, N);
  utils::parallel_for_2d(M, N, [&](int i, int j) {
    float sum = utils::simd_dot_product(&A[i * K], &B_T[j * K], K);
    C[i * N + j] = sum;
  });
  free(B_T);
}

} // namespace tmath