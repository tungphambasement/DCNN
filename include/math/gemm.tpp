#pragma once

#include "gemm.hpp"
#include <algorithm>

#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace tmath {
constexpr int BLOCK_SIZE = 64; // Tunable block size for cache efficiency

#ifdef __AVX2__
// AVX2 optimized kernel for inner block computation
inline void sgemm_kernel_avx2(const float *A, const float *B, float *C, const int M, const int N,
                              const int K, const int i, const int j, const int k, const int i_max,
                              const int j_max, const int k_max) {
  for (int ii = i; ii < i_max; ++ii) {
    int jj = j;

    // Process 8 columns at a time with AVX2
    for (; jj + 7 < j_max; jj += 8) {
      // Load or initialize accumulator for 8 output elements
      __m256 c_vec = _mm256_loadu_ps(&C[ii * N + jj]);

      // Accumulate over K dimension
      for (int kk = k; kk < k_max; ++kk) {
        // Broadcast A[ii, kk] to all 8 lanes
        __m256 a_vec = _mm256_set1_ps(A[ii * K + kk]);

        // Load 8 consecutive elements from B[kk, jj:jj+8]
        __m256 b_vec = _mm256_loadu_ps(&B[kk * N + jj]);

        // Fused multiply-add: c_vec += a_vec * b_vec
        c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
      }

      // Store result back to C
      _mm256_storeu_ps(&C[ii * N + jj], c_vec);
    }

    // Handle remaining columns (< 8) with scalar code
    for (; jj < j_max; ++jj) {
      float sum = C[ii * N + jj];
      for (int kk = k; kk < k_max; ++kk) {
        sum += A[ii * K + kk] * B[kk * N + jj];
      }
      C[ii * N + jj] = sum;
    }
  }
}
#endif

void sgemm(const float *A, const float *B, float *C, const int M, const int N, const int K) {
#ifdef __AVX2__
  // Blocked matrix multiplication with AVX2 optimization
  for (int i = 0; i < M; i += BLOCK_SIZE) {
    for (int j = 0; j < N; j += BLOCK_SIZE) {
      for (int k = 0; k < K; k += BLOCK_SIZE) {
        int i_max = std::min(i + BLOCK_SIZE, M);
        int j_max = std::min(j + BLOCK_SIZE, N);
        int k_max = std::min(k + BLOCK_SIZE, K);

        sgemm_kernel_avx2(A, B, C, M, N, K, i, j, k, i_max, j_max, k_max);
      }
    }
  }
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

} // namespace tmath