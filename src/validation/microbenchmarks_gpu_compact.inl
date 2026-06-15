GPUPrimitivesMicrobenchmark::GPUPrimitivesMicrobenchmark(
    const MicrobenchmarkConfig &config)
    : config_(config), rng_(config.random_seed) {
  optimization::AcceleratedGpuPrimitives::GPUConfig gpu_cfg{};
  if (!config_.gpu_batch_sizes.empty()) {
    gpu_cfg.min_batch_size = *std::min_element(config_.gpu_batch_sizes.begin(),
                                               config_.gpu_batch_sizes.end());
    gpu_cfg.max_batch_size = *std::max_element(config_.gpu_batch_sizes.begin(),
                                               config_.gpu_batch_sizes.end());
    gpu_cfg.optimal_batch_size = config_.gpu_batch_sizes.front();
  }
  gpu_cfg.num_streams =
      static_cast<int>(std::max<std::size_t>(1, config_.num_threads));
  gpu_cfg.enable_async_operations = true;
  gpu_cfg.enable_one_kernel_pipeline = true;
  gpu_primitives_ =
      std::make_unique<optimization::AcceleratedGpuPrimitives>(gpu_cfg);
}

std::vector<MicrobenchmarkResult>
GPUPrimitivesMicrobenchmark::runGpuBenchmarks() {
  std::vector<MicrobenchmarkResult> out;

  const std::size_t n_points =
      std::max<std::size_t>(64, config_.streaming_ph_points / 8);
  for (const std::size_t batch_size : config_.gpu_batch_sizes) {
    auto test_data = generateTestData(batch_size, n_points);
    auto buffers = allocateGpuBuffers(batch_size, n_points);

    std::vector<double> latencies_us;
    const auto bench_start = std::chrono::steady_clock::now();
    for (std::size_t iter = 0; iter < config_.gpu_iterations; ++iter) {
      const auto t0 = std::chrono::steady_clock::now();
      for (std::size_t b = 0; b < batch_size; ++b) {
        std::vector<float> flat(n_points * kPointDim, 0.0f);
        for (std::size_t i = 0; i < n_points; ++i) {
          for (std::size_t d = 0; d < kPointDim; ++d) {
            flat[i * kPointDim + d] = test_data[b][i][d];
          }
        }
        cudaMemcpy(buffers[b], flat.data(), flat.size() * sizeof(float),
                   cudaMemcpyHostToDevice);
        gpu_primitives_->computeDistanceMatrixBatch(buffers[b], n_points,
                                                    kPointDim);
      }
      cudaDeviceSynchronize();
      const auto t1 = std::chrono::steady_clock::now();
      latencies_us.push_back(
          std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    const auto bench_end = std::chrono::steady_clock::now();
    deallocateGpuBuffers(buffers);

    auto result = makeResult("gpu_distance_batch_n" + std::to_string(n_points) +
                                 "_b" + std::to_string(batch_size),
                             std::move(latencies_us), bench_start, bench_end);
    result.metrics["batch_size"] = static_cast<double>(batch_size);
    result.metrics["n_points"] = static_cast<double>(n_points);
    result.metrics["iterations"] = static_cast<double>(config_.gpu_iterations);
    out.push_back(std::move(result));
  }

  std::vector<uint32_t> reduction_column = makeRandomColumn(rng_, 2048);
  std::vector<double> reduction_latencies;
  const auto reduce_start = std::chrono::steady_clock::now();
  for (std::size_t i = 0; i < config_.gpu_iterations; ++i) {
    auto col = reduction_column;
    const auto t0 = std::chrono::steady_clock::now();
    gpu_primitives_->reduceColumnGpu(col.data(), col.size());
    const auto t1 = std::chrono::steady_clock::now();
    reduction_latencies.push_back(
        std::chrono::duration<double, std::micro>(t1 - t0).count());
  }
  const auto reduce_end = std::chrono::steady_clock::now();
  out.push_back(makeResult("gpu_column_reduction",
                           std::move(reduction_latencies), reduce_start,
                           reduce_end));

  return out;
}

bool GPUPrimitivesMicrobenchmark::validateGpuPerformance(
    const std::vector<MicrobenchmarkResult> &results) {
  if (results.empty()) {
    return false;
  }
  for (const auto &result : results) {
    if (!result.success || result.sample_count == 0) {
      return false;
    }
    if (result.p95_latency > config_.p95_latency_threshold_ms * 1000.0) {
      return false;
    }
  }
  return true;
}

std::vector<std::vector<std::vector<float>>>
GPUPrimitivesMicrobenchmark::generateTestData(std::size_t batch_size,
                                              std::size_t num_points) {
  std::vector<std::vector<std::vector<float>>> out(batch_size);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  for (std::size_t b = 0; b < batch_size; ++b) {
    out[b].assign(num_points, std::vector<float>(kPointDim, 0.0f));
    for (std::size_t i = 0; i < num_points; ++i) {
      for (std::size_t d = 0; d < kPointDim; ++d) {
        out[b][i][d] = dist(rng_);
      }
    }
  }
  return out;
}

std::vector<float *>
GPUPrimitivesMicrobenchmark::allocateGpuBuffers(std::size_t batch_size,
                                                std::size_t num_points) {
  std::vector<float *> buffers(batch_size, nullptr);
  const std::size_t bytes = num_points * kPointDim * sizeof(float);
  for (std::size_t i = 0; i < batch_size; ++i) {
    cudaMalloc(&buffers[i], bytes);
  }
  return buffers;
}

void GPUPrimitivesMicrobenchmark::deallocateGpuBuffers(
    const std::vector<float *> &buffers) {
  for (float *ptr : buffers) {
    cudaFree(ptr);
  }
}

CompactSummaryMicrobenchmark::CompactSummaryMicrobenchmark(
    const MicrobenchmarkConfig &config)
    : config_(config), rng_(config.random_seed) {
  summaries_ = std::make_unique<optimization::AcceleratedCompactSummaries>(
      optimization::AcceleratedCompactSummaries::SummaryConfig{});
}

MicrobenchmarkResult
CompactSummaryMicrobenchmark::runCompactSummaryBenchmark() {
  auto windows =
      generateTestWindows(config_.compact_summary_windows, config_.window_size);
  std::vector<double> latencies_us;
  const auto start = std::chrono::steady_clock::now();
  for (const auto &window : windows) {
    const auto t0 = std::chrono::steady_clock::now();
    (void)summaries_->computeSummary(window, makeContract("compact_summary"));
    const auto t1 = std::chrono::steady_clock::now();
    latencies_us.push_back(
        std::chrono::duration<double, std::micro>(t1 - t0).count());
  }
  const auto end = std::chrono::steady_clock::now();
  auto result =
      makeResult("compact_summary", std::move(latencies_us), start, end);
  result.metrics["window_size"] = static_cast<double>(config_.window_size);
  result.metrics["window_count"] =
      static_cast<double>(config_.compact_summary_windows);
  return result;
}

bool CompactSummaryMicrobenchmark::validatePerformance(
    const MicrobenchmarkResult &result) {
  return result.success && result.sample_count > 0 &&
         result.mean_latency <= config_.mean_latency_threshold_ms * 1000.0;
}

std::vector<std::vector<std::vector<float>>>
CompactSummaryMicrobenchmark::generateTestWindows(std::size_t count,
                                                  std::size_t window_size) {
  std::vector<std::vector<std::vector<float>>> windows(count);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  for (auto &window : windows) {
    window.assign(window_size, std::vector<float>(kPointDim, 0.0f));
    for (auto &point : window) {
      for (float &v : point) {
        v = dist(rng_);
      }
    }
  }
  return windows;
}

optimization::AcceleratedCompactSummaries::CompactSummary
CompactSummaryMicrobenchmark::computeSummaryReference(
    const std::vector<std::vector<float>> &points) {
  optimization::AcceleratedCompactSummaries::CompactSummary summary{};
  summary.num_points = static_cast<uint32_t>(points.size());
  summary.persistence_entropy = 0.0f;
  summary.computation_time_us = 0;
  return summary;
}
