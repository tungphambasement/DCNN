#include "math/sgemm.hpp"
#include "threading/thread_handler.hpp"
#include <cstring>

namespace tmath {
constexpr size_t DEFAULT_BLOCK_SIZE = 32;

#ifdef __AVX2__
inline bool is_aligned_32(const void *ptr) { return (reinterpret_cast<uintptr_t>(ptr) & 31) == 0; }

inline void sgemm_kernel_avx2_nn(const float *A, const float *B, float *C, const size_t M,
                                 const size_t N, const size_t K, const size_t i, const size_t j,
                                 const size_t k, const size_t i_max, const size_t j_max,
                                 const size_t k_max) {
  size_t ii = i;
  for (; ii + 3 < i_max; ii += 4) {
    size_t jj = j;
    for (; jj + 7 < j_max; jj += 8) {
      __m256 c_vec_0 = _mm256_setzero_ps();
      __m256 c_vec_1 = _mm256_setzero_ps();
      __m256 c_vec_2 = _mm256_setzero_ps();
      __m256 c_vec_3 = _mm256_setzero_ps();

      for (size_t kk = k; kk < k_max; ++kk) {
        __m256 b_vec = _mm256_loadu_ps(&B[kk * N + jj]);
        __m256 a_vec_0 = _mm256_broadcast_ss(&A[(ii + 0) * K + kk]);
        __m256 a_vec_1 = _mm256_broadcast_ss(&A[(ii + 1) * K + kk]);
        __m256 a_vec_2 = _mm256_broadcast_ss(&A[(ii + 2) * K + kk]);
        __m256 a_vec_3 = _mm256_broadcast_ss(&A[(ii + 3) * K + kk]);
        c_vec_0 = _mm256_fmadd_ps(a_vec_0, b_vec, c_vec_0);
        c_vec_1 = _mm256_fmadd_ps(a_vec_1, b_vec, c_vec_1);
        c_vec_2 = _mm256_fmadd_ps(a_vec_2, b_vec, c_vec_2);
        c_vec_3 = _mm256_fmadd_ps(a_vec_3, b_vec, c_vec_3);
      }
      _mm256_storeu_ps(&C[(ii + 0) * N + jj],
                       _mm256_add_ps(_mm256_loadu_ps(&C[(ii + 0) * N + jj]), c_vec_0));
      _mm256_storeu_ps(&C[(ii + 1) * N + jj],
                       _mm256_add_ps(_mm256_loadu_ps(&C[(ii + 1) * N + jj]), c_vec_1));
      _mm256_storeu_ps(&C[(ii + 2) * N + jj],
                       _mm256_add_ps(_mm256_loadu_ps(&C[(ii + 2) * N + jj]), c_vec_2));
      _mm256_storeu_ps(&C[(ii + 3) * N + jj],
                       _mm256_add_ps(_mm256_loadu_ps(&C[(ii + 3) * N + jj]), c_vec_3));
    }
    for (; jj < j_max; ++jj) {
      float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
      for (size_t kk = k; kk < k_max; ++kk) {
        float b_val = B[kk * N + jj];
        sum0 += A[(ii + 0) * K + kk] * b_val;
        sum1 += A[(ii + 1) * K + kk] * b_val;
        sum2 += A[(ii + 2) * K + kk] * b_val;
        sum3 += A[(ii + 3) * K + kk] * b_val;
      }
      C[(ii + 0) * N + jj] += sum0;
      C[(ii + 1) * N + jj] += sum1;
      C[(ii + 2) * N + jj] += sum2;
      C[(ii + 3) * N + jj] += sum3;
    }
  }
  for (; ii < i_max; ++ii) {
    size_t jj = j;
    for (; jj + 7 < j_max; jj += 8) {
      __m256 c_vec = _mm256_setzero_ps();
      for (size_t kk = k; kk < k_max; ++kk) {
        __m256 a_vec = _mm256_broadcast_ss(&A[ii * K + kk]);
        __m256 b_vec = _mm256_loadu_ps(&B[kk * N + jj]);
        c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
      }
      _mm256_storeu_ps(&C[ii * N + jj], _mm256_add_ps(_mm256_loadu_ps(&C[ii * N + jj]), c_vec));
    }
    for (; jj < j_max; ++jj) {
      float sum = 0.0f;
      for (size_t kk = k; kk < k_max; ++kk) {
        sum += A[ii * K + kk] * B[kk * N + jj];
      }
      C[ii * N + jj] += sum;
    }
  }
}

inline void sgemm_kernel_avx2_nn_aligned(const float *A, const float *B, float *C, const size_t M,
                                         const size_t N, const size_t K, const size_t i,
                                         const size_t j, const size_t k, const size_t i_max,
                                         const size_t j_max, const size_t k_max) {
  size_t ii = i;
  for (; ii + 3 < i_max; ii += 4) {
    size_t jj = j;
    for (; jj + 7 < j_max; jj += 8) {
      __m256 c_vec_0 = _mm256_setzero_ps();
      __m256 c_vec_1 = _mm256_setzero_ps();
      __m256 c_vec_2 = _mm256_setzero_ps();
      __m256 c_vec_3 = _mm256_setzero_ps();

      for (size_t kk = k; kk < k_max; ++kk) {
        __m256 b_vec = _mm256_load_ps(&B[kk * N + jj]);
        __m256 a_vec_0 = _mm256_broadcast_ss(&A[(ii + 0) * K + kk]);
        __m256 a_vec_1 = _mm256_broadcast_ss(&A[(ii + 1) * K + kk]);
        __m256 a_vec_2 = _mm256_broadcast_ss(&A[(ii + 2) * K + kk]);
        __m256 a_vec_3 = _mm256_broadcast_ss(&A[(ii + 3) * K + kk]);
        c_vec_0 = _mm256_fmadd_ps(a_vec_0, b_vec, c_vec_0);
        c_vec_1 = _mm256_fmadd_ps(a_vec_1, b_vec, c_vec_1);
        c_vec_2 = _mm256_fmadd_ps(a_vec_2, b_vec, c_vec_2);
        c_vec_3 = _mm256_fmadd_ps(a_vec_3, b_vec, c_vec_3);
      }
      _mm256_store_ps(&C[(ii + 0) * N + jj],
                      _mm256_add_ps(_mm256_load_ps(&C[(ii + 0) * N + jj]), c_vec_0));
      _mm256_store_ps(&C[(ii + 1) * N + jj],
                      _mm256_add_ps(_mm256_load_ps(&C[(ii + 1) * N + jj]), c_vec_1));
      _mm256_store_ps(&C[(ii + 2) * N + jj],
                      _mm256_add_ps(_mm256_load_ps(&C[(ii + 2) * N + jj]), c_vec_2));
      _mm256_store_ps(&C[(ii + 3) * N + jj],
                      _mm256_add_ps(_mm256_load_ps(&C[(ii + 3) * N + jj]), c_vec_3));
    }
    for (; jj < j_max; ++jj) {
      float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
      for (size_t kk = k; kk < k_max; ++kk) {
        float b_val = B[kk * N + jj];
        sum0 += A[(ii + 0) * K + kk] * b_val;
        sum1 += A[(ii + 1) * K + kk] * b_val;
        sum2 += A[(ii + 2) * K + kk] * b_val;
        sum3 += A[(ii + 3) * K + kk] * b_val;
      }
      C[(ii + 0) * N + jj] += sum0;
      C[(ii + 1) * N + jj] += sum1;
      C[(ii + 2) * N + jj] += sum2;
      C[(ii + 3) * N + jj] += sum3;
    }
  }
  for (; ii < i_max; ++ii) {
    size_t jj = j;
    for (; jj + 7 < j_max; jj += 8) {
      __m256 c_vec = _mm256_setzero_ps();
      for (size_t kk = k; kk < k_max; ++kk) {
        __m256 a_vec = _mm256_broadcast_ss(&A[ii * K + kk]);
        __m256 b_vec = _mm256_load_ps(&B[kk * N + jj]);
        c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
      }
      _mm256_store_ps(&C[ii * N + jj], _mm256_add_ps(_mm256_load_ps(&C[ii * N + jj]), c_vec));
    }
    for (; jj < j_max; ++jj) {
      float sum = 0.0f;
      for (size_t kk = k; kk < k_max; ++kk) {
        sum += A[ii * K + kk] * B[kk * N + jj];
      }
      C[ii * N + jj] += sum;
    }
  }
}

inline void sgemm_kernel_avx2_nt(const float *A, const float *B, float *C, const size_t M,
                                 const size_t N, const size_t K, const size_t i, const size_t j,
                                 const size_t k, const size_t i_max, const size_t j_max,
                                 const size_t k_max) {
  for (size_t ii = i; ii < i_max; ++ii) {
    size_t jj = j;
    for (; jj + 3 < j_max; jj += 4) {
      __m256 sum0 = _mm256_setzero_ps();
      __m256 sum1 = _mm256_setzero_ps();
      __m256 sum2 = _mm256_setzero_ps();
      __m256 sum3 = _mm256_setzero_ps();

      size_t kk = k;
      for (; kk + 7 < k_max; kk += 8) {
        __m256 a_vec = _mm256_loadu_ps(&A[ii * K + kk]);
        __m256 b0_vec = _mm256_loadu_ps(&B[(jj + 0) * K + kk]);
        __m256 b1_vec = _mm256_loadu_ps(&B[(jj + 1) * K + kk]);
        __m256 b2_vec = _mm256_loadu_ps(&B[(jj + 2) * K + kk]);
        __m256 b3_vec = _mm256_loadu_ps(&B[(jj + 3) * K + kk]);

        sum0 = _mm256_fmadd_ps(a_vec, b0_vec, sum0);
        sum1 = _mm256_fmadd_ps(a_vec, b1_vec, sum1);
        sum2 = _mm256_fmadd_ps(a_vec, b2_vec, sum2);
        sum3 = _mm256_fmadd_ps(a_vec, b3_vec, sum3);
      }

      auto horizontal_sum = [](const __m256 &vec) -> float {
        __m128 vlow = _mm256_castps256_ps128(vec);
        __m128 vhigh = _mm256_extractf128_ps(vec, 1);
        vlow = _mm_add_ps(vlow, vhigh);
        vlow = _mm_hadd_ps(vlow, vlow);
        vlow = _mm_hadd_ps(vlow, vlow);
        return _mm_cvtss_f32(vlow);
      };

      float partial_sum0 = horizontal_sum(sum0);
      float partial_sum1 = horizontal_sum(sum1);
      float partial_sum2 = horizontal_sum(sum2);
      float partial_sum3 = horizontal_sum(sum3);

      for (; kk < k_max; ++kk) {
        float a_val = A[ii * K + kk];
        partial_sum0 += a_val * B[(jj + 0) * K + kk];
        partial_sum1 += a_val * B[(jj + 1) * K + kk];
        partial_sum2 += a_val * B[(jj + 2) * K + kk];
        partial_sum3 += a_val * B[(jj + 3) * K + kk];
      }

      if (k == 0) {
        C[ii * N + jj + 0] = partial_sum0;
        C[ii * N + jj + 1] = partial_sum1;
        C[ii * N + jj + 2] = partial_sum2;
        C[ii * N + jj + 3] = partial_sum3;
      } else {
        C[ii * N + jj + 0] += partial_sum0;
        C[ii * N + jj + 1] += partial_sum1;
        C[ii * N + jj + 2] += partial_sum2;
        C[ii * N + jj + 3] += partial_sum3;
      }
    }

    for (; jj < j_max; ++jj) {
      __m256 sum_vec = _mm256_setzero_ps();
      size_t kk = k;
      for (; kk + 7 < k_max; kk += 8) {
        __m256 a_vec = _mm256_loadu_ps(&A[ii * K + kk]);
        __m256 b_vec = _mm256_loadu_ps(&B[jj * K + kk]);
        sum_vec = _mm256_fmadd_ps(a_vec, b_vec, sum_vec);
      }

      __m128 vlow = _mm256_castps256_ps128(sum_vec);
      __m128 vhigh = _mm256_extractf128_ps(sum_vec, 1);
      vlow = _mm_add_ps(vlow, vhigh);
      vlow = _mm_hadd_ps(vlow, vlow);
      vlow = _mm_hadd_ps(vlow, vlow);
      float sum = _mm_cvtss_f32(vlow);

      for (; kk < k_max; ++kk) {
        sum += A[ii * K + kk] * B[jj * K + kk];
      }

      if (k == 0) {
        C[ii * N + jj] = sum;
      } else {
        C[ii * N + jj] += sum;
      }
    }
  }
}

inline void sgemm_kernel_avx2_nt_aligned(const float *A, const float *B, float *C, const size_t M,
                                         const size_t N, const size_t K, const size_t i,
                                         const size_t j, const size_t k, const size_t i_max,
                                         const size_t j_max, const size_t k_max) {
  for (size_t ii = i; ii < i_max; ++ii) {
    size_t jj = j;
    for (; jj + 3 < j_max; jj += 4) {
      __m256 sum0 = _mm256_setzero_ps();
      __m256 sum1 = _mm256_setzero_ps();
      __m256 sum2 = _mm256_setzero_ps();
      __m256 sum3 = _mm256_setzero_ps();

      size_t kk = k;
      for (; kk + 7 < k_max; kk += 8) {
        __m256 a_vec = _mm256_load_ps(&A[ii * K + kk]);
        __m256 b0_vec = _mm256_load_ps(&B[(jj + 0) * K + kk]);
        __m256 b1_vec = _mm256_load_ps(&B[(jj + 1) * K + kk]);
        __m256 b2_vec = _mm256_load_ps(&B[(jj + 2) * K + kk]);
        __m256 b3_vec = _mm256_load_ps(&B[(jj + 3) * K + kk]);

        sum0 = _mm256_fmadd_ps(a_vec, b0_vec, sum0);
        sum1 = _mm256_fmadd_ps(a_vec, b1_vec, sum1);
        sum2 = _mm256_fmadd_ps(a_vec, b2_vec, sum2);
        sum3 = _mm256_fmadd_ps(a_vec, b3_vec, sum3);
      }

      auto horizontal_sum = [](const __m256 &vec) -> float {
        __m128 vlow = _mm256_castps256_ps128(vec);
        __m128 vhigh = _mm256_extractf128_ps(vec, 1);
        vlow = _mm_add_ps(vlow, vhigh);
        vlow = _mm_hadd_ps(vlow, vlow);
        vlow = _mm_hadd_ps(vlow, vlow);
        return _mm_cvtss_f32(vlow);
      };

      float partial_sum0 = horizontal_sum(sum0);
      float partial_sum1 = horizontal_sum(sum1);
      float partial_sum2 = horizontal_sum(sum2);
      float partial_sum3 = horizontal_sum(sum3);

      for (; kk < k_max; ++kk) {
        float a_val = A[ii * K + kk];
        partial_sum0 += a_val * B[(jj + 0) * K + kk];
        partial_sum1 += a_val * B[(jj + 1) * K + kk];
        partial_sum2 += a_val * B[(jj + 2) * K + kk];
        partial_sum3 += a_val * B[(jj + 3) * K + kk];
      }

      if (k == 0) {
        C[ii * N + jj + 0] = partial_sum0;
        C[ii * N + jj + 1] = partial_sum1;
        C[ii * N + jj + 2] = partial_sum2;
        C[ii * N + jj + 3] = partial_sum3;
      } else {
        C[ii * N + jj + 0] += partial_sum0;
        C[ii * N + jj + 1] += partial_sum1;
        C[ii * N + jj + 2] += partial_sum2;
        C[ii * N + jj + 3] += partial_sum3;
      }
    }

    for (; jj < j_max; ++jj) {
      __m256 sum_vec = _mm256_setzero_ps();
      size_t kk = k;
      for (; kk + 7 < k_max; kk += 8) {
        __m256 a_vec = _mm256_load_ps(&A[ii * K + kk]);
        __m256 b_vec = _mm256_load_ps(&B[jj * K + kk]);
        sum_vec = _mm256_fmadd_ps(a_vec, b_vec, sum_vec);
      }

      __m128 vlow = _mm256_castps256_ps128(sum_vec);
      __m128 vhigh = _mm256_extractf128_ps(sum_vec, 1);
      vlow = _mm_add_ps(vlow, vhigh);
      vlow = _mm_hadd_ps(vlow, vlow);
      vlow = _mm_hadd_ps(vlow, vlow);
      float sum = _mm_cvtss_f32(vlow);

      for (; kk < k_max; ++kk) {
        sum += A[ii * K + kk] * B[jj * K + kk];
      }

      if (k == 0) {
        C[ii * N + jj] = sum;
      } else {
        C[ii * N + jj] += sum;
      }
    }
  }
}

inline void sgemm_kernel_avx2_tn(const float *A, const float *B, float *C, const size_t M,
                                 const size_t N, const size_t K, const size_t i, const size_t j,
                                 const size_t k, const size_t i_max, const size_t j_max,
                                 const size_t k_max) {
  size_t ii = i;
  for (; ii + 3 < i_max; ii += 4) {
    size_t jj = j;
    for (; jj + 7 < j_max; jj += 8) {
      __m256 c_vec_0 = _mm256_setzero_ps();
      __m256 c_vec_1 = _mm256_setzero_ps();
      __m256 c_vec_2 = _mm256_setzero_ps();
      __m256 c_vec_3 = _mm256_setzero_ps();

      for (size_t kk = k; kk < k_max; ++kk) {
        __m256 b_vec = _mm256_loadu_ps(&B[kk * N + jj]);

        __m256 a_vec_0 = _mm256_broadcast_ss(&A[kk * M + ii + 0]);
        __m256 a_vec_1 = _mm256_broadcast_ss(&A[kk * M + ii + 1]);
        __m256 a_vec_2 = _mm256_broadcast_ss(&A[kk * M + ii + 2]);
        __m256 a_vec_3 = _mm256_broadcast_ss(&A[kk * M + ii + 3]);

        c_vec_0 = _mm256_fmadd_ps(a_vec_0, b_vec, c_vec_0);
        c_vec_1 = _mm256_fmadd_ps(a_vec_1, b_vec, c_vec_1);
        c_vec_2 = _mm256_fmadd_ps(a_vec_2, b_vec, c_vec_2);
        c_vec_3 = _mm256_fmadd_ps(a_vec_3, b_vec, c_vec_3);
      }

      _mm256_storeu_ps(&C[(ii + 0) * N + jj],
                       _mm256_add_ps(_mm256_loadu_ps(&C[(ii + 0) * N + jj]), c_vec_0));
      _mm256_storeu_ps(&C[(ii + 1) * N + jj],
                       _mm256_add_ps(_mm256_loadu_ps(&C[(ii + 1) * N + jj]), c_vec_1));
      _mm256_storeu_ps(&C[(ii + 2) * N + jj],
                       _mm256_add_ps(_mm256_loadu_ps(&C[(ii + 2) * N + jj]), c_vec_2));
      _mm256_storeu_ps(&C[(ii + 3) * N + jj],
                       _mm256_add_ps(_mm256_loadu_ps(&C[(ii + 3) * N + jj]), c_vec_3));
    }
    for (; jj < j_max; ++jj) {
      float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
      for (size_t kk = k; kk < k_max; ++kk) {
        float b_val = B[kk * N + jj];
        sum0 += A[kk * M + ii + 0] * b_val;
        sum1 += A[kk * M + ii + 1] * b_val;
        sum2 += A[kk * M + ii + 2] * b_val;
        sum3 += A[kk * M + ii + 3] * b_val;
      }
      C[(ii + 0) * N + jj] += sum0;
      C[(ii + 1) * N + jj] += sum1;
      C[(ii + 2) * N + jj] += sum2;
      C[(ii + 3) * N + jj] += sum3;
    }
  }

  for (; ii < i_max; ++ii) {
    size_t jj = j;
    for (; jj + 7 < j_max; jj += 8) {
      __m256 c_vec = _mm256_setzero_ps();
      for (size_t kk = k; kk < k_max; ++kk) {
        __m256 a_vec = _mm256_broadcast_ss(&A[kk * M + ii]);
        __m256 b_vec = _mm256_loadu_ps(&B[kk * N + jj]);
        c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
      }
      _mm256_storeu_ps(&C[ii * N + jj], _mm256_add_ps(_mm256_loadu_ps(&C[ii * N + jj]), c_vec));
    }

    for (; jj < j_max; ++jj) {
      float sum = 0.0f;
      for (size_t kk = k; kk < k_max; ++kk) {
        sum += A[kk * M + ii] * B[kk * N + jj];
      }
      C[ii * N + jj] += sum;
    }
  }
}

inline void sgemm_kernel_avx2_tn_aligned(const float *A, const float *B, float *C, const size_t M,
                                         const size_t N, const size_t K, const size_t i,
                                         const size_t j, const size_t k, const size_t i_max,
                                         const size_t j_max, const size_t k_max) {
  size_t ii = i;
  for (; ii + 3 < i_max; ii += 4) {
    size_t jj = j;
    for (; jj + 7 < j_max; jj += 8) {
      __m256 c_vec_0 = _mm256_setzero_ps();
      __m256 c_vec_1 = _mm256_setzero_ps();
      __m256 c_vec_2 = _mm256_setzero_ps();
      __m256 c_vec_3 = _mm256_setzero_ps();

      for (size_t kk = k; kk < k_max; ++kk) {
        __m256 b_vec = _mm256_load_ps(&B[kk * N + jj]);
        __m256 a_vec_0 = _mm256_broadcast_ss(&A[kk * M + ii + 0]);
        __m256 a_vec_1 = _mm256_broadcast_ss(&A[kk * M + ii + 1]);
        __m256 a_vec_2 = _mm256_broadcast_ss(&A[kk * M + ii + 2]);
        __m256 a_vec_3 = _mm256_broadcast_ss(&A[kk * M + ii + 3]);
        c_vec_0 = _mm256_fmadd_ps(a_vec_0, b_vec, c_vec_0);
        c_vec_1 = _mm256_fmadd_ps(a_vec_1, b_vec, c_vec_1);
        c_vec_2 = _mm256_fmadd_ps(a_vec_2, b_vec, c_vec_2);
        c_vec_3 = _mm256_fmadd_ps(a_vec_3, b_vec, c_vec_3);
      }
      _mm256_store_ps(&C[(ii + 0) * N + jj],
                      _mm256_add_ps(_mm256_load_ps(&C[(ii + 0) * N + jj]), c_vec_0));
      _mm256_store_ps(&C[(ii + 1) * N + jj],
                      _mm256_add_ps(_mm256_load_ps(&C[(ii + 1) * N + jj]), c_vec_1));
      _mm256_store_ps(&C[(ii + 2) * N + jj],
                      _mm256_add_ps(_mm256_load_ps(&C[(ii + 2) * N + jj]), c_vec_2));
      _mm256_store_ps(&C[(ii + 3) * N + jj],
                      _mm256_add_ps(_mm256_load_ps(&C[(ii + 3) * N + jj]), c_vec_3));
    }
    for (; jj < j_max; ++jj) {
      float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
      for (size_t kk = k; kk < k_max; ++kk) {
        float b_val = B[kk * N + jj];
        sum0 += A[kk * M + ii + 0] * b_val;
        sum1 += A[kk * M + ii + 1] * b_val;
        sum2 += A[kk * M + ii + 2] * b_val;
        sum3 += A[kk * M + ii + 3] * b_val;
      }
      C[(ii + 0) * N + jj] += sum0;
      C[(ii + 1) * N + jj] += sum1;
      C[(ii + 2) * N + jj] += sum2;
      C[(ii + 3) * N + jj] += sum3;
    }
  }
  for (; ii < i_max; ++ii) {
    size_t jj = j;
    for (; jj + 7 < j_max; jj += 8) {
      __m256 c_vec = _mm256_setzero_ps();
      for (size_t kk = k; kk < k_max; ++kk) {
        __m256 a_vec = _mm256_broadcast_ss(&A[kk * M + ii]);
        __m256 b_vec = _mm256_load_ps(&B[kk * N + jj]);
        c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
      }
      _mm256_store_ps(&C[ii * N + jj], _mm256_add_ps(_mm256_load_ps(&C[ii * N + jj]), c_vec));
    }
    for (; jj < j_max; ++jj) {
      float sum = 0.0f;
      for (size_t kk = k; kk < k_max; ++kk) {
        sum += A[kk * M + ii] * B[kk * N + jj];
      }
      C[ii * N + jj] += sum;
    }
  }
}
#endif

void sgemm(const float *A, const float *B, float *C, const size_t M, const size_t N, const size_t K,
           const bool trans_A, const bool trans_B) {
#ifdef __AVX2__
  bool all_aligned = is_aligned_32(A) && is_aligned_32(B) && is_aligned_32(C);

  size_t M_BLOCK_SIZE, N_BLOCK_SIZE, K_BLOCK_SIZE;

  if (!trans_A && !trans_B) {
    M_BLOCK_SIZE = DEFAULT_BLOCK_SIZE;
    N_BLOCK_SIZE = DEFAULT_BLOCK_SIZE;
    K_BLOCK_SIZE = DEFAULT_BLOCK_SIZE * 2;
    size_t M_blocks = (M + M_BLOCK_SIZE - 1) / M_BLOCK_SIZE;
    size_t N_blocks = (N + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE;

    if (all_aligned) {
      tthreads::parallel_for_2d(
          M_blocks, N_blocks,
          [&](size_t block_i, size_t block_j) {
            size_t i = block_i * M_BLOCK_SIZE;
            size_t j = block_j * N_BLOCK_SIZE;
            for (size_t k = 0; k < K; k += K_BLOCK_SIZE) {
              size_t i_max = std::min(i + M_BLOCK_SIZE, M);
              size_t j_max = std::min(j + N_BLOCK_SIZE, N);
              size_t k_max = std::min(k + K_BLOCK_SIZE, K);
              sgemm_kernel_avx2_nn_aligned(A, B, C, M, N, K, i, j, k, i_max, j_max, k_max);
            }
          },
          tthreads::SchedulePolicy::Auto);
    } else {
      tthreads::parallel_for_2d(
          M_blocks, N_blocks,
          [&](size_t block_i, size_t block_j) {
            size_t i = block_i * M_BLOCK_SIZE;
            size_t j = block_j * N_BLOCK_SIZE;
            for (size_t k = 0; k < K; k += K_BLOCK_SIZE) {
              size_t i_max = std::min(i + M_BLOCK_SIZE, M);
              size_t j_max = std::min(j + N_BLOCK_SIZE, N);
              size_t k_max = std::min(k + K_BLOCK_SIZE, K);
              sgemm_kernel_avx2_nn(A, B, C, M, N, K, i, j, k, i_max, j_max, k_max);
            }
          },
          tthreads::SchedulePolicy::Auto);
    }
  } else if (!trans_A && trans_B) {
    M_BLOCK_SIZE = DEFAULT_BLOCK_SIZE / 2;
    N_BLOCK_SIZE = DEFAULT_BLOCK_SIZE / 2;
    K_BLOCK_SIZE = DEFAULT_BLOCK_SIZE * 16;

    size_t M_blocks = (M + M_BLOCK_SIZE - 1) / M_BLOCK_SIZE;
    size_t N_blocks = (N + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE;

    if (all_aligned) {
      tthreads::parallel_for_2d(
          M_blocks, N_blocks,
          [&](size_t block_i, size_t block_j) {
            size_t i = block_i * M_BLOCK_SIZE;
            size_t j = block_j * N_BLOCK_SIZE;
            size_t i_max = std::min(i + M_BLOCK_SIZE, M);
            size_t j_max = std::min(j + N_BLOCK_SIZE, N);
            for (size_t k = 0; k < K; k += K_BLOCK_SIZE) {
              size_t k_max = std::min(k + K_BLOCK_SIZE, K);
              sgemm_kernel_avx2_nt_aligned(A, B, C, M, N, K, i, j, k, i_max, j_max, k_max);
            }
          },
          tthreads::SchedulePolicy::Auto);
    } else {
      tthreads::parallel_for_2d(
          M_blocks, N_blocks,
          [&](size_t block_i, size_t block_j) {
            size_t i = block_i * M_BLOCK_SIZE;
            size_t j = block_j * N_BLOCK_SIZE;
            size_t i_max = std::min(i + M_BLOCK_SIZE, M);
            size_t j_max = std::min(j + N_BLOCK_SIZE, N);
            for (size_t k = 0; k < K; k += K_BLOCK_SIZE) {
              size_t k_max = std::min(k + K_BLOCK_SIZE, K);
              sgemm_kernel_avx2_nt(A, B, C, M, N, K, i, j, k, i_max, j_max, k_max);
            }
          },
          tthreads::SchedulePolicy::Auto);
    }
  } else if (trans_A && !trans_B) {
    M_BLOCK_SIZE = DEFAULT_BLOCK_SIZE;
    N_BLOCK_SIZE = DEFAULT_BLOCK_SIZE;
    K_BLOCK_SIZE = DEFAULT_BLOCK_SIZE;
    size_t M_blocks = (M + M_BLOCK_SIZE - 1) / M_BLOCK_SIZE;
    size_t N_blocks = (N + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE;

    if (all_aligned) {
      tthreads::parallel_for_2d(
          M_blocks, N_blocks,
          [&](size_t block_i, size_t block_j) {
            size_t i = block_i * M_BLOCK_SIZE;
            size_t j = block_j * N_BLOCK_SIZE;
            for (size_t k = 0; k < K; k += K_BLOCK_SIZE) {
              size_t i_max = std::min(i + M_BLOCK_SIZE, M);
              size_t j_max = std::min(j + N_BLOCK_SIZE, N);
              size_t k_max = std::min(k + K_BLOCK_SIZE, K);
              sgemm_kernel_avx2_tn_aligned(A, B, C, M, N, K, i, j, k, i_max, j_max, k_max);
            }
          },
          tthreads::SchedulePolicy::Auto);
    } else {
      tthreads::parallel_for_2d(
          M_blocks, N_blocks,
          [&](size_t block_i, size_t block_j) {
            size_t i = block_i * M_BLOCK_SIZE;
            size_t j = block_j * N_BLOCK_SIZE;
            for (size_t k = 0; k < K; k += K_BLOCK_SIZE) {
              size_t i_max = std::min(i + M_BLOCK_SIZE, M);
              size_t j_max = std::min(j + N_BLOCK_SIZE, N);
              size_t k_max = std::min(k + K_BLOCK_SIZE, K);
              sgemm_kernel_avx2_tn(A, B, C, M, N, K, i, j, k, i_max, j_max, k_max);
            }
          },
          tthreads::SchedulePolicy::Auto);
    }
  } else {

    float *D = (float *)aligned_alloc(32, sizeof(float) * N * M);
    sgemm(B, A, D, N, M, K, false, false);
    utils::transpose_2d(D, C, N, M);
    free(D);
  }
#else
  const size_t BLOCK_SIZE = 32;

  for (size_t i = 0; i < M; i += BLOCK_SIZE) {
    for (size_t j = 0; j < N; j += BLOCK_SIZE) {
      for (size_t k = 0; k < K; k += BLOCK_SIZE) {
        size_t i_max = std::min(i + BLOCK_SIZE, M);
        size_t j_max = std::min(j + BLOCK_SIZE, N);
        size_t k_max = std::min(k + BLOCK_SIZE, K);
        for (size_t ii = i; ii < i_max; ++ii) {
          for (size_t jj = j; jj < j_max; ++jj) {
            float sum = C[ii * N + jj];
            for (size_t kk = k; kk < k_max; ++kk) {
              float a_val = trans_A ? A[kk * M + ii] : A[ii * K + kk];
              float b_val = trans_B ? B[jj * K + kk] : B[kk * N + jj];
              sum += a_val * b_val;
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
