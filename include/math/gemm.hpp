#pragma once

#include <cstdint>
#include <cstdlib>
#include <memory>

namespace tmath {
void sgemm(const float *A, const float *B, float *C, const int M, const int N, const int K);
void dgemm(const double *A, const double *B, double *C, const int M, const int N, const int K);

} // namespace tmath