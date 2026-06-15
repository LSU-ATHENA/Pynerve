StreamingPHMicrobenchmark::StreamingPHMicrobenchmark(
    const MicrobenchmarkConfig &config)
    : config_(config), rng_(config.random_seed) {
  streaming_ph_ = std::make_unique<optimization::AcceleratedStreamingPh>(
      optimization::AcceleratedStreamingPh::StreamingConfig{});
}

MicrobenchmarkResult StreamingPHMicrobenchmark::runStreamingPhBenchmark() {
  const auto points = generateTimeSeries(config_.streaming_ph_points);
  const auto start = std::chrono::steady_clock::now();
  const auto t0 = std::chrono::steady_clock::now();
  (void)streaming_ph_->computeStreamingPh(points);
  const auto t1 = std::chrono::steady_clock::now();
  const auto end = t1;
  return makeResult(
      "streaming_ph",
      {std::chrono::duration<double, std::micro>(t1 - t0).count()}, start, end);
}

std::vector<double> StreamingPHMicrobenchmark::testAdversarialBursts() {
  auto points = generateTimeSeries(config_.streaming_ph_points);
  points = injectAdversarialBursts(points, config_.adversarial_bursts);
  std::vector<double> latencies_us;
  for (std::size_t i = 0; i < config_.adversarial_bursts; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    (void)streaming_ph_->computeStreamingPh(points);
    const auto t1 = std::chrono::steady_clock::now();
    latencies_us.push_back(
        std::chrono::duration<double, std::micro>(t1 - t0).count());
  }
  return latencies_us;
}

double StreamingPHMicrobenchmark::compareUpdateVsExact() {
  const auto points = generateTimeSeries(
      std::max<std::size_t>(32, config_.streaming_ph_points / 10));
  const auto exact = computeExactPh(points);
  const auto approx = streaming_ph_->compute(points);
  if (exact.empty() || approx.empty()) {
    return 0.0;
  }
  const std::size_t n = std::min(exact.size(), approx.size());
  double err = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    err += std::abs(static_cast<double>(exact[i].second) -
                    static_cast<double>(approx[i].second));
  }
  return err / static_cast<double>(n);
}

std::vector<std::vector<float>>
StreamingPHMicrobenchmark::generateTimeSeries(std::size_t num_points) {
  std::vector<std::vector<float>> points(num_points,
                                         std::vector<float>(kPointDim, 0.0f));
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  for (auto &point : points) {
    for (float &v : point) {
      v = dist(rng_);
    }
  }
  return points;
}

std::vector<std::vector<float>>
StreamingPHMicrobenchmark::injectAdversarialBursts(
    const std::vector<std::vector<float>> &base_series,
    std::size_t num_bursts) {
  auto out = base_series;
  if (out.empty()) {
    return out;
  }
  std::uniform_int_distribution<std::size_t> idx_dist(0, out.size() - 1);
  std::uniform_real_distribution<float> bump(8.0f, 16.0f);
  for (std::size_t b = 0; b < num_bursts; ++b) {
    auto &point = out[idx_dist(rng_)];
    for (float &v : point) {
      v += bump(rng_);
    }
  }
  return out;
}

std::vector<std::pair<float, float>> StreamingPHMicrobenchmark::computeExactPh(
    const std::vector<std::vector<float>> &points) {
  std::vector<std::pair<float, float>> pairs;
  pairs.reserve(points.size() * (points.size() - 1) / 2);
  for (std::size_t i = 0; i < points.size(); ++i) {
    for (std::size_t j = i + 1; j < points.size(); ++j) {
      float sq = 0.0f;
      for (std::size_t d = 0; d < kPointDim; ++d) {
        const float delta = points[i][d] - points[j][d];
        sq += delta * delta;
      }
      pairs.push_back({0.0f, std::sqrt(sq)});
    }
  }
  return pairs;
}

IncrementalLaplacianMicrobenchmark::IncrementalLaplacianMicrobenchmark(
    const MicrobenchmarkConfig &config)
    : config_(config), rng_(config.random_seed) {
  incremental_laplacian_ =
      std::make_unique<optimization::AcceleratedIncrementalLaplacian>(
          optimization::AcceleratedIncrementalLaplacian::LaplacianConfig{});
}

MicrobenchmarkResult
IncrementalLaplacianMicrobenchmark::runIncrementalLaplacianBenchmark() {
  auto matrix = generateTestMatrix(config_.laplacian_matrix_size);
  auto weights = generateTestWeights(config_.laplacian_matrix_size);
  auto updates =
      generateIncrementalUpdates(matrix, config_.incremental_updates);

  std::vector<double> latencies_us;
  optimization::AcceleratedIncrementalLaplacian::EigenpairResult warm_start{};
  const auto start = std::chrono::steady_clock::now();
  for (std::size_t i = 0; i < updates.size(); ++i) {
    matrix[i % matrix.size()] = updates[i];
    incremental_laplacian_->updateMatrixStructure(matrix);
    incremental_laplacian_->updateMatrixValues(weights);

    const auto t0 = std::chrono::steady_clock::now();
    warm_start = incremental_laplacian_->computeIncrementalLaplacian(
        matrix, weights, makeContract("incremental_laplacian"), warm_start);
    const auto t1 = std::chrono::steady_clock::now();
    latencies_us.push_back(
        std::chrono::duration<double, std::micro>(t1 - t0).count());
  }
  const auto end = std::chrono::steady_clock::now();
  return makeResult("incremental_laplacian", std::move(latencies_us), start,
                    end);
}

bool IncrementalLaplacianMicrobenchmark::validateResiduals(
    const std::vector<double> &residuals) {
  if (residuals.empty()) {
    return false;
  }
  const double max_residual =
      *std::max_element(residuals.begin(), residuals.end());
  return max_residual <= config_.residual_threshold;
}

std::vector<std::vector<uint32_t>>
IncrementalLaplacianMicrobenchmark::generateTestMatrix(std::size_t size) {
  std::vector<std::vector<uint32_t>> matrix(size);
  std::uniform_int_distribution<uint32_t> node(0,
                                               static_cast<uint32_t>(size - 1));
  for (auto &row : matrix) {
    const std::size_t degree = 1 + (rng_() % 8);
    row.reserve(degree);
    for (std::size_t j = 0; j < degree; ++j) {
      row.push_back(node(rng_));
    }
    std::sort(row.begin(), row.end());
    row.erase(std::unique(row.begin(), row.end()), row.end());
  }
  return matrix;
}

std::vector<double>
IncrementalLaplacianMicrobenchmark::generateTestWeights(std::size_t size) {
  std::vector<double> weights(size, 1.0);
  std::uniform_real_distribution<double> dist(0.2, 2.0);
  for (double &w : weights) {
    w = dist(rng_);
  }
  return weights;
}

std::vector<std::vector<uint32_t>>
IncrementalLaplacianMicrobenchmark::generateIncrementalUpdates(
    const std::vector<std::vector<uint32_t>> &base_matrix,
    std::size_t num_updates) {
  if (base_matrix.empty()) {
    return {};
  }
  std::vector<std::vector<uint32_t>> updates;
  updates.reserve(num_updates);
  std::uniform_int_distribution<std::size_t> row_dist(0,
                                                      base_matrix.size() - 1);
  std::uniform_int_distribution<uint32_t> node_dist(
      0, static_cast<uint32_t>(base_matrix.size() - 1));
  for (std::size_t i = 0; i < num_updates; ++i) {
    auto row = base_matrix[row_dist(rng_)];
    row.push_back(node_dist(rng_));
    std::sort(row.begin(), row.end());
    row.erase(std::unique(row.begin(), row.end()), row.end());
    updates.push_back(std::move(row));
  }
  return updates;
}
