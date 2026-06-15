AugmentationBenchmark benchmarkAugmentation(int num_samples, int feature_dim,
                                            int num_augs) {
  if (num_samples <= 0 || feature_dim <= 0 || num_augs <= 0) {
    throw std::invalid_argument("augmentation benchmark dimensions are invalid");
  }
  std::size_t total_values = 0;
  if (!checkedProduct(static_cast<std::size_t>(num_samples),
                      static_cast<std::size_t>(feature_dim), total_values)) {
    throw std::length_error("augmentation benchmark input size overflows");
  }

  AugmentationBenchmark bench;
  bench.num_samples = num_samples;
  bench.feature_dim = feature_dim;
  bench.num_augmentations = num_augs;

  GPUDataAugmentation::AugmentationConfig config;
  config.augmentations = {
      GPUDataAugmentation::AugmentationType::GAUSSIAN_NOISE,
      GPUDataAugmentation::AugmentationType::GEOMETRIC_TRANSFORM};

  GPUDataAugmentation augmenter(config);

  std::vector<float> data(total_values);
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = static_cast<float>((i * 19) % 257) / 257.0f;
  }

  auto start_cpu = std::chrono::high_resolution_clock::now();
  std::vector<float> cpu_augmented = data;
  std::mt19937 gen(42);
  std::normal_distribution<float> noise(0.0f, config.gaussian_std);
  for (float &value : cpu_augmented) {
    value += noise(gen);
  }
  const float scale = 1.0f + config.scale_range;
  const float cos_a = std::cos(config.rotation_range);
  const float sin_a = std::sin(config.rotation_range);
  if (feature_dim >= 2) {
    for (int sample = 0; sample < num_samples; ++sample) {
      const size_t base = static_cast<size_t>(sample) * feature_dim;
      const float x = cpu_augmented[base];
      const float y = cpu_augmented[base + 1];
      cpu_augmented[base] = scale * (x * cos_a - y * sin_a);
      cpu_augmented[base + 1] = scale * (x * sin_a + y * cos_a);
      for (int f = 2; f < feature_dim; ++f) {
        cpu_augmented[base + f] *= scale;
      }
    }
  }
  auto end_cpu = std::chrono::high_resolution_clock::now();
  bench.cpu_time_ms =
      std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();

  auto start = std::chrono::high_resolution_clock::now();
  auto augmented = augmenter.augment(data, num_samples, feature_dim);
  auto end = std::chrono::high_resolution_clock::now();
  bench.gpu_time_ms =
      std::chrono::duration<double, std::milli>(end - start).count();

  if (augmented.empty() || augmented.size() != data.size() ||
      cpu_augmented.empty()) {
    throw std::runtime_error("augmentation benchmark produced empty output");
  }
  auto ratio = [](double cpu_ms, double gpu_ms) {
    if (!std::isfinite(cpu_ms) || !std::isfinite(gpu_ms) || cpu_ms < 0.0 ||
        gpu_ms <= 0.0) {
      return 1.0;
    }
    return cpu_ms / gpu_ms;
  };
  bench.speedup = ratio(bench.cpu_time_ms, bench.gpu_time_ms);

  return bench;
}
