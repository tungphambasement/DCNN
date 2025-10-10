/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/conv2d_layer.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

#include "math/gemm.hpp"
#include "threading/thread_handler.hpp"
#include "utils/ops.hpp"

#ifdef USE_MKL
#include "utils/mkl_utils.hpp"
#endif

#ifdef USE_TBB
#include <tbb/blocked_range2d.h>
#include <tbb/parallel_for.h>
#endif

namespace tnn {

template <typename T>
Conv2DLayer<T>::Conv2DLayer(size_t in_channels, size_t out_channels, size_t kernel_h,
                            size_t kernel_w, size_t stride_h, size_t stride_w, size_t pad_h,
                            size_t pad_w, bool use_bias, const std::string &name)
    : ParameterizedLayer<T>(name), in_channels_(in_channels), out_channels_(out_channels),
      kernel_h_(kernel_h), kernel_w_(kernel_w), stride_h_(stride_h), stride_w_(stride_w),
      pad_h_(pad_h), pad_w_(pad_w), use_bias_(use_bias), micro_batch_im2col_matrices_() {
  weights_ = Tensor<T>(out_channels, in_channels, kernel_h, kernel_w);
  weight_gradients_ = Tensor<T>(out_channels, in_channels, kernel_h, kernel_w);

  if (use_bias_) {
    bias_ = Tensor<T>(out_channels, 1, 1, 1);
    bias_gradients_ = Tensor<T>(out_channels, 1, 1, 1);
  }

  T fan_in = static_cast<T>(in_channels * kernel_h * kernel_w);
  T fan_out = static_cast<T>(out_channels * kernel_h * kernel_w);
  T std_dev = std::sqrt(T(2.0) / (fan_in + fan_out));
  weights_.fill_random_normal(T(0), std_dev);
}

template <typename T>
Tensor<T> Conv2DLayer<T>::forward(const Tensor<T> &input, size_t micro_batch_id) {
  if (input.channels() != in_channels_) {
    std::cerr << "Input shape: " << input.channels() << " channels, expected: " << in_channels_
              << " channels" << std::endl;
    throw std::invalid_argument("Input channel size mismatch in Conv2DLayer");
  }

  micro_batch_input_shapes_[micro_batch_id] = {input.batch_size(), input.channels(), input.height(),
                                               input.width()};

  const size_t batch_size = input.batch_size();
  const size_t input_h = input.height();
  const size_t input_w = input.width();

  const size_t output_h = (input_h + 2 * pad_h_ - kernel_h_) / stride_h_ + 1;
  const size_t output_w = (input_w + 2 * pad_w_ - kernel_w_) / stride_w_ + 1;

  auto im2col_start = std::chrono::high_resolution_clock::now();
  Matrix<T> col_matrix = im2col(input, kernel_h_, kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_);
  auto im2col_end = std::chrono::high_resolution_clock::now();
  if (this->enable_profiling_) {
    float im2col_duration =
        std::chrono::duration<float, std::milli>(im2col_end - im2col_start).count();
    this->perf_timers_["im2col"] += im2col_duration;
  }

  Tensor<T> output(batch_size, out_channels_, output_h, output_w, nullptr);

  size_t kernel_size = in_channels_ * kernel_h_ * kernel_w_;
  size_t output_size = batch_size * output_h * output_w;

  T *output_flat = (T *)aligned_alloc(64, sizeof(T) * out_channels_ * output_size);
  compute_conv_forward(col_matrix.data(), weights_.data(), output_flat, output_size, kernel_size,
                       out_channels_);

  micro_batch_im2col_matrices_[micro_batch_id] = std::move(col_matrix);

  utils::cnhw_to_nchw(output_flat, output.data(), batch_size, out_channels_, output_h, output_w);

  free(output_flat);

  if (use_bias_) {
    add_bias_to_output(output.data(), bias_.data(), batch_size, output_h, output_w, out_channels_);
  }

  return output;
}

template <typename T>
Tensor<T> Conv2DLayer<T>::backward(const Tensor<T> &gradient, size_t micro_batch_id) {
  auto it_input_shape = micro_batch_input_shapes_.find(micro_batch_id);
  auto it_im2col = micro_batch_im2col_matrices_.find(micro_batch_id);

  if (it_input_shape == micro_batch_input_shapes_.end()) {
    throw std::runtime_error("No cached input shape found for micro-batch ID: " +
                             std::to_string(micro_batch_id));
  }

  if (it_im2col == micro_batch_im2col_matrices_.end()) {
    throw std::runtime_error("No cached im2col matrix found for micro-batch ID: " +
                             std::to_string(micro_batch_id));
  }

  const auto &input_shape = it_input_shape->second;
  const Matrix<T> &cached_im2col_matrix = it_im2col->second;

  const size_t batch_size = input_shape[0];
  const size_t input_h = input_shape[2];
  const size_t input_w = input_shape[3];
  const size_t output_h = gradient.height();
  const size_t output_w = gradient.width();

  size_t kernel_size = in_channels_ * kernel_h_ * kernel_w_;
  size_t output_size = batch_size * output_h * output_w;

  T *gradient_flat = (T *)aligned_alloc(64, sizeof(T) * out_channels_ * output_size);

  utils::nchw_to_cnhw(gradient.data(), gradient_flat, batch_size, out_channels_, output_h,
                      output_w);

  compute_weight_gradients(cached_im2col_matrix.data(), gradient_flat, weight_gradients_.data(),
                           output_size, kernel_size, out_channels_);

  if (use_bias_) {
    compute_bias_gradients(gradient.data(), bias_gradients_.data(), batch_size, output_h, output_w,
                           out_channels_);
  }

  Matrix<T> col_grad_matrix(kernel_size, output_size);
  compute_input_gradients(gradient_flat, weights_.data(), col_grad_matrix.data(), output_size,
                          kernel_size, out_channels_);

  Tensor<T> grad_input = col2im(col_grad_matrix, batch_size, in_channels_, input_h, input_w,
                                kernel_h_, kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_);

  free(gradient_flat);
  return grad_input;
}

template <typename T>
void Conv2DLayer<T>::compute_conv_forward(const T *col_data, const T *weight_data, T *output_data,
                                          const size_t output_size, const size_t kernel_size,
                                          const size_t out_channels) const {
  auto conv_start = std::chrono::high_resolution_clock::now();
#ifdef USE_MKL
  utils::mkl::conv_forward_gemm(
      weight_data, col_data, output_data, static_cast<MKL_INT>(out_channels),
      static_cast<MKL_INT>(kernel_size), static_cast<MKL_INT>(output_size));
#else
  utils::avx2_set_scalar(output_data, T(0), out_channels * output_size);
  tmath::sgemm(weight_data, col_data, output_data, out_channels, output_size, kernel_size);
#endif
  auto conv_end = std::chrono::high_resolution_clock::now();
  if (this->enable_profiling_) {
    float conv_duration = std::chrono::duration<float, std::milli>(conv_end - conv_start).count();
    this->perf_timers_["conv_forward"] += conv_duration;
  }
}

template <typename T>
void Conv2DLayer<T>::compute_weight_gradients(const T *col_data, const T *gradient_data,
                                              T *weight_grad_data, const size_t output_size,
                                              const size_t kernel_size,
                                              const size_t out_channels) const {
  auto wg_start = std::chrono::high_resolution_clock::now();
#ifdef USE_MKL
  utils::mkl::conv_weight_grad_gemm(
      gradient_data, col_data, weight_grad_data, static_cast<MKL_INT>(out_channels),
      static_cast<MKL_INT>(kernel_size), static_cast<MKL_INT>(output_size));
#else
  tmath::sgemm(gradient_data, col_data, weight_grad_data, out_channels, kernel_size, output_size,
               false, true);
#endif
  auto wg_end = std::chrono::high_resolution_clock::now();
  if (this->enable_profiling_) {
    float wg_duration = std::chrono::duration<float, std::milli>(wg_end - wg_start).count();
    this->perf_timers_["weight_gradients"] += wg_duration;
  }
}

template <typename T>
void Conv2DLayer<T>::compute_input_gradients(const T *gradient_data, const T *weight_data,
                                             T *col_grad_data, const size_t output_size,
                                             const size_t kernel_size,
                                             const size_t out_channels) const {
  auto ig_start = std::chrono::high_resolution_clock::now();
#ifdef USE_MKL
  utils::mkl::conv_input_grad_gemm(
      weight_data, gradient_data, col_grad_data, static_cast<MKL_INT>(out_channels),
      static_cast<MKL_INT>(kernel_size), static_cast<MKL_INT>(output_size));
#else
  utils::avx2_set_scalar(col_grad_data, T(0), kernel_size * output_size);
  tmath::sgemm(weight_data, gradient_data, col_grad_data, kernel_size, output_size, out_channels,
               true, false);
#endif
  auto ig_end = std::chrono::high_resolution_clock::now();
  if (this->enable_profiling_) {
    float ig_duration = std::chrono::duration<float, std::milli>(ig_end - ig_start).count();
    this->perf_timers_["input_gradients"] += ig_duration;
  }
}

template <typename T>
void Conv2DLayer<T>::compute_bias_gradients(const T *gradient_data, T *bias_grad_data,
                                            const size_t batch_size, const size_t output_h,
                                            const size_t output_w,
                                            const size_t out_channels) const {
  const size_t N_stride = out_channels * output_h * output_w;
  const size_t C_stride = output_h * output_w;

  tthreads::parallel_for<size_t>(0, out_channels, [&](size_t oc) {
    T grad_sum = T(0);
    for (size_t n = 0; n < batch_size; ++n) {
      std::accumulate(gradient_data + n * N_stride + oc * C_stride,
                      gradient_data + n * N_stride + (oc + 1) * C_stride, grad_sum);
    }
    bias_grad_data[oc] += grad_sum;
  });
}

template <typename T>
void Conv2DLayer<T>::add_bias_to_output(T *output_data, const T *bias_data, const size_t batch_size,
                                        const size_t output_h, const size_t output_w,
                                        const size_t out_channels) const {
  tthreads::parallel_for_2d(batch_size, out_channels, [&](size_t n, size_t oc) {
    utils::avx2_add_scalar(
        output_data + (n * out_channels + oc) * output_h * output_w, bias_data[oc],
        output_data + (n * out_channels + oc) * output_h * output_w, output_h * output_w);
  });
}

template <typename T> std::string Conv2DLayer<T>::type() const { return "conv2d"; }

template <typename T> LayerConfig Conv2DLayer<T>::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.parameters["in_channels"] = in_channels_;
  config.parameters["out_channels"] = out_channels_;
  config.parameters["kernel_h"] = kernel_h_;
  config.parameters["kernel_w"] = kernel_w_;
  config.parameters["stride_h"] = stride_h_;
  config.parameters["stride_w"] = stride_w_;
  config.parameters["pad_h"] = pad_h_;
  config.parameters["pad_w"] = pad_w_;
  config.parameters["use_bias"] = use_bias_;
  config.parameters["optimized"] = std::string("native");
  return config;
}

template <typename T> std::unique_ptr<Layer<T>> Conv2DLayer<T>::clone() const {
  return std::make_unique<Conv2DLayer<T>>(in_channels_, out_channels_, kernel_h_, kernel_w_,
                                          stride_h_, stride_w_, pad_h_, pad_w_, use_bias_,
                                          this->name_);
}

template <typename T>
std::vector<size_t>
Conv2DLayer<T>::compute_output_shape(const std::vector<size_t> &input_shape) const {
  if (input_shape.size() != 4) {
    throw std::invalid_argument("Conv2DLayer expects 4D input");
  }

  size_t output_h = (input_shape[2] + 2 * pad_h_ - kernel_h_) / stride_h_ + 1;
  size_t output_w = (input_shape[3] + 2 * pad_w_ - kernel_w_) / stride_w_ + 1;

  return {input_shape[0], out_channels_, output_h, output_w};
}

template <typename T> void Conv2DLayer<T>::collect_parameters(std::vector<Tensor<T> *> &params) {
  params.push_back(&weights_);
  if (use_bias_) {
    params.push_back(&bias_);
  }
}

template <typename T> void Conv2DLayer<T>::collect_gradients(std::vector<Tensor<T> *> &grads) {
  grads.push_back(&weight_gradients_);
  if (use_bias_) {
    grads.push_back(&bias_gradients_);
  }
}

template <typename T> void Conv2DLayer<T>::clear_gradients() {
  weight_gradients_.fill(T(0));
  if (use_bias_) {
    bias_gradients_.fill(T(0));
  }
}

template <typename T>
uint64_t Conv2DLayer<T>::forward_flops(const std::vector<size_t> &input_shape) const {
  assert(input_shape.size() == 4 && "Input shape must be 4D");
  size_t batch_size = input_shape[0];
  size_t input_h = input_shape[2];
  size_t input_w = input_shape[3];
  size_t output_h = (input_h + 2 * pad_h_ - kernel_h_) / stride_h_ + 1;
  size_t output_w = (input_w + 2 * pad_w_ - kernel_w_) / stride_w_ + 1;
  size_t output_size = batch_size * output_h * output_w;
  size_t kernel_size = in_channels_ * kernel_h_ * kernel_w_;

  // Main convolution computation: 2 FLOPs per MAC (multiply-add)
  uint64_t conv_flops = 2ULL * out_channels_ * kernel_size * output_size;

  // Bias addition: 1 FLOP per output element
  uint64_t bias_flops = use_bias_ ? (batch_size * out_channels_ * output_h * output_w) : 0;

  return conv_flops + bias_flops;
}

template <typename T>
uint64_t Conv2DLayer<T>::backward_flops(const std::vector<size_t> &input_shape) const {
  assert(input_shape.size() == 4 && "Input shape must be 4D");
  size_t batch_size = input_shape[0];
  size_t input_h = input_shape[2];
  size_t input_w = input_shape[3];
  size_t output_h = (input_h + 2 * pad_h_ - kernel_h_) / stride_h_ + 1;
  size_t output_w = (input_w + 2 * pad_w_ - kernel_w_) / stride_w_ + 1;
  size_t output_size = batch_size * output_h * output_w;
  size_t kernel_size = in_channels_ * kernel_h_ * kernel_w_;

  // weight gradients: gradient × im2col_input^T (2 FLOPs per MAC)
  uint64_t weight_grad_flops = 2ULL * out_channels_ * kernel_size * output_size;

  // input gradients: weights^T × gradient (2 FLOPs per MAC)
  uint64_t input_grad_flops = 2ULL * out_channels_ * kernel_size * output_size;

  // bias gradients: reduction across batch and spatial dimensions (1 FLOP per add)
  uint64_t bias_grad_flops = use_bias_ ? (batch_size * out_channels_ * output_h * output_w) : 0;

  return weight_grad_flops + input_grad_flops + bias_grad_flops;
}

template <typename T>
uint64_t Conv2DLayer<T>::forward_complexity(const std::vector<size_t> &input_shape) {
  return static_cast<uint64_t>(
      std::min(forward_flops(input_shape), static_cast<uint64_t>(UINT32_MAX)));
}

template <typename T>
uint64_t Conv2DLayer<T>::backward_complexity(const std::vector<size_t> &input_shape) {
  return static_cast<uint64_t>(
      std::min(backward_flops(input_shape), static_cast<uint64_t>(UINT32_MAX)));
}

// Explicit template instantiations
template class Conv2DLayer<float>;

} // namespace tnn
