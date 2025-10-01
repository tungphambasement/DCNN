# Intel MKL Integration Guide

This document explains how to build and use the DCNN library with Intel MKL support for optimized GEMM operations in convolutional layers.

## Prerequisites

### 1. Install Intel MKL

Choose one of the following installation methods:

#### Option A: Intel oneAPI Base Toolkit (Recommended)
```bash
# Download and install Intel oneAPI Base Toolkit
wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/b4adec4f-e154-40c6-8b48-8c71a71e5821/l_BaseKit_p_2024.0.1.46_offline.sh
chmod +x l_BaseKit_p_2024.0.1.46_offline.sh
sudo ./l_BaseKit_p_2024.0.1.46_offline.sh

# Source the environment (add this to your .bashrc)
source /opt/intel/oneapi/setvars.sh
```

#### Option B: Package Manager
```bash
# Ubuntu/Debian
sudo apt install libmkl-dev libmkl-rt

# CentOS/RHEL/Fedora
sudo yum install intel-mkl-devel
# or
sudo dnf install intel-mkl-devel
```

### 2. Set Environment Variables

If using Intel oneAPI, make sure to source the environment:
```bash
source /opt/intel/oneapi/setvars.sh
```

Or set MKLROOT manually:
```bash
export MKLROOT=/opt/intel/oneapi/mkl/latest
```

## Building with MKL Support

### 1. Configure CMake with MKL
```bash
mkdir build && cd build

# Configure with MKL enabled
cmake -DENABLE_MKL=ON -DCMAKE_BUILD_TYPE=Release ..

# Or with additional options
cmake -DENABLE_MKL=ON -DENABLE_TBB=ON -DCMAKE_BUILD_TYPE=Release ..
```

### 2. Build the Project
```bash
make -j$(nproc)
```

### 3. Verify MKL Integration
Check the build output for messages like:
```
-- Finding Intel MKL
-- Found Intel MKL manually
-- MKL Include: /opt/intel/oneapi/mkl/latest/include
-- MKL Libraries: [list of MKL libraries]
-- Intel MKL enabled
```

## Performance Benefits

When using Intel MKL, the convolution operations will use highly optimized BLAS routines instead of custom SIMD code:

### Forward Pass
- **Without MKL**: Custom SIMD dot products with manual transposition
- **With MKL**: Optimized SGEMM/DGEMM operations

### Backward Pass
- **Weight Gradients**: Uses `cblas_sgemm` with transpose operations
- **Input Gradients**: Uses `cblas_sgemm` for efficient gradient computation

### Expected Performance Gains
- **Large kernels (7x7, 11x11)**: 2-5x speedup
- **Standard kernels (3x3, 5x5)**: 1.5-3x speedup
- **Deep networks**: Cumulative speedup across all layers

## Usage Examples

### Basic Training with MKL
```cpp
#include "nn/layers/conv2d_layer.hpp"

// Create a Conv2D layer - MKL will be used automatically if enabled
auto conv_layer = std::make_unique<Conv2DLayer<float>>(
    3,      // in_channels
    64,     // out_channels
    3, 3,   // kernel_h, kernel_w
    1, 1,   // stride_h, stride_w
    1, 1,   // pad_h, pad_w
    true    // use_bias
);

// Forward and backward passes will automatically use MKL GEMM
Tensor<float> output = conv_layer->forward(input, 0);
Tensor<float> grad_input = conv_layer->backward(grad_output, 0);
```

### MKL Configuration at Runtime
```cpp
#ifdef USE_MKL
#include "utils/mkl_utils.hpp"

// Initialize MKL for optimal performance
utils::mkl::initialize();

// Set number of threads (optional)
utils::mkl::set_num_threads(8);

// Your training loop here...

// Cleanup when done
utils::mkl::finalize();
#endif
```

## Troubleshooting

### Common Issues

1. **MKL not found during cmake**
   ```bash
   # Ensure MKLROOT is set
   export MKLROOT=/opt/intel/oneapi/mkl/latest
   
   # Or specify the path manually
   cmake -DENABLE_MKL=ON -DMKL_ROOT=/path/to/mkl ..
   ```

2. **Runtime linking errors**
   ```bash
   # Add MKL lib path to LD_LIBRARY_PATH
   export LD_LIBRARY_PATH=$MKLROOT/lib/intel64:$LD_LIBRARY_PATH
   
   # Or source the Intel environment
   source /opt/intel/oneapi/setvars.sh
   ```

3. **Performance not improved**
   - Check that `USE_MKL` is defined during compilation
   - Verify MKL is using multiple threads: `utils::mkl::get_num_threads()`
   - Ensure input tensors are large enough to benefit from BLAS optimization

### Verification

To verify MKL is being used, you can:

1. **Check compile definitions**:
   ```bash
   # Look for USE_MKL in the compile commands
   grep "USE_MKL" compile_commands.json
   ```

2. **Runtime verification**:
   ```cpp
   #ifdef USE_MKL
   std::cout << "MKL enabled, threads: " << utils::mkl::get_num_threads() << std::endl;
   #else
   std::cout << "Using fallback SIMD implementation" << std::endl;
   #endif
   ```

3. **Performance benchmark**:
   ```bash
   # Run benchmarks with and without MKL
   ./bin/compression_test
   ```

## Build Configurations

### Release Build with MKL
```bash
cmake -DENABLE_MKL=ON -DCMAKE_BUILD_TYPE=Release -DENABLE_TBB=ON ..
make -j$(nproc)
```

### Debug Build with MKL
```bash
cmake -DENABLE_MKL=ON -DCMAKE_BUILD_TYPE=Debug -DENABLE_DEBUG=ON ..
make -j$(nproc)
```

### Build without MKL (fallback)
```bash
cmake -DENABLE_MKL=OFF -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

The code will automatically fall back to the custom SIMD implementation when MKL is not available.