#include "math/gemm.hpp"
#include "tensor/tensor.hpp"
#include "utils/ops.hpp"
#include <chrono>
#include <iostream>
#include <utils/mkl_utils.hpp>

using namespace tmath;
using namespace utils;

constexpr int N = 64;
constexpr int C = 128;
constexpr int H = 128;
constexpr int W = 128;

int main() {

#ifdef USE_TBB
  tbb::task_arena arena(tbb::task_arena::constraints{}.set_max_concurrency(8));

  std::cout << "TBB max threads limited to: " << arena.max_concurrency() << std::endl;
  arena.execute([&] {
#endif
    Matrix<float> a(N, C * H * W);
    Matrix<float> b(C * H * W, C);
    Matrix<float> c1(N, C);
    Matrix<float> c2(N, C);
    Matrix<float> c3(N, C);

    a.fill_random_normal(0.5f, 0.25f);
    b.fill_random_normal(0.0f, 1.0f);
    c1.fill(0.0f);
    c2.fill(0.0f);
    c3.fill(0.0f);

    auto current_start = std::chrono::high_resolution_clock::now();
    sgemm(a.data(), b.data(), c1.data(), N, C, C * H * W);
    auto current_end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> current_duration = current_end - current_start;
    std::cout << "SGEMM completed in " << current_duration.count() << " ms\n";

    auto old_start = std::chrono::high_resolution_clock::now();
    old_gemm(a.data(), b.data(), c2.data(), N, C, C * H * W);
    auto old_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> old_duration = old_end - old_start;
    std::cout << "Old SGEMM completed in " << old_duration.count() << " ms\n";

    // check if match using relative error
    bool match = true;
    float max_abs_diff = 0.0f;
    float max_rel_diff = 0.0f;
    const float rtol = 1e-5f; // relative tolerance
    const float atol = 1e-6f; // absolute tolerance for near-zero values

    for (size_t i = 0; i < N; ++i) {
      for (size_t j = 0; j < C; ++j) {
        float abs_diff = std::abs(c1(i, j) - c2(i, j));
        float magnitude = std::max(std::abs(c1(i, j)), std::abs(c2(i, j)));
        float rel_diff = magnitude > 1e-6f ? abs_diff / magnitude : abs_diff;

        // Use relative tolerance for large values, absolute for small values
        float tolerance = atol + rtol * magnitude;

        if (abs_diff > tolerance) {
          match = false;
          max_abs_diff = std::max(max_abs_diff, abs_diff);
          max_rel_diff = std::max(max_rel_diff, rel_diff);

          if (i == 0 && j < 10) { // Only print first few mismatches
            std::cout << "Mismatch at (" << i << ", " << j << "): " << c1(i, j) << " vs "
                      << c2(i, j) << ", abs_diff: " << abs_diff << ", rel_diff: " << rel_diff
                      << std::endl;
          }
        }
      }
    }

    std::cout << "Max absolute difference: " << max_abs_diff << std::endl;
    std::cout << "Max relative difference: " << max_rel_diff << std::endl;

    if (match) {
      std::cout << "✓ Results match within tolerance (rtol=" << rtol << ", atol=" << atol << ")"
                << std::endl;
    } else {
      std::cout << "✗ Results exceed tolerance!" << std::endl;
      std::cout << "Note: For K=" << (C * H * W)
                << " accumulations, this level of error is expected" << std::endl;
    }
#ifdef USE_MKL
    auto mkl_start = std::chrono::high_resolution_clock::now();
    utils::mkl::gemm('N', 'N', N, C, C * H * W, 1.0f, a.data(), C * H * W, b.data(), C, 0.0f,
                     c.data(), C);
    auto mkl_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> mkl_duration = mkl_end - mkl_start;
    std::cout << "MKL SGEMM completed in " << mkl_duration.count() << " ms\n";
#endif

#ifdef USE_TBB
  });
#endif

  return 0;
}