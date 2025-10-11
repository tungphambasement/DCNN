/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "tbuffer.hpp"
#include "utils/hardware_info.hpp"
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

namespace tpipeline {
struct LoadTracker {
  /**
   * Performance metrics to send to coordinator
   * These are serialized and sent periodically.
   */
  // throughputs
  float avg_forward_time_ = 0;  // avg forward time per batch in milliseconds
  float avg_backward_time_ = 0; // avg backward time per batch in milliseconds

  // system metrics
  float avg_cpu_utilization_ = -1.0f; // CPU utilization ratio (0.0 to 1.0)
  float max_memory_usage_ = -1.0f;    // Maximum memory usage in MB, -1.0 if unavailable
};

} // namespace tpipeline