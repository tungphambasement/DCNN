/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "matrix/matrix.hpp"
#include "nn/activations.hpp"
#include "nn/layers.hpp"
#include "nn/loss.hpp"
#include "nn/optimizers.hpp"
#include "nn/sequential.hpp"
#include "tensor/tensor.hpp"

namespace tnn {
/**
 * Template instantiations for commonly used types. Uncomment as needed.
 */

// Sequential model instantiations
template class Sequential<float>;
// template class Sequential<double>;

// Sequential builder instantiations
template class SequentialBuilder<float>;
// template class SequentialBuilder<double>;

} // namespace tnn