std::vector<float>
computeCpuTopologicalFeatures(const std::vector<float> &diagrams,
                              int batch_size, int max_pairs, int num_features) {
  std::vector<float> features(batch_size * num_features, 0.0f);
  for (int batch = 0; batch < batch_size; ++batch) {
    const float *diagram = diagrams.data() + batch * max_pairs * 2;
    for (int feature_idx = 0; feature_idx < num_features; ++feature_idx) {
      float value = 0.0f;
      switch (feature_idx % 8) {
      case 0:
        for (int i = 0; i < max_pairs; ++i)
          value += diagram[2 * i + 1] - diagram[2 * i];
        break;
      case 1:
        for (int i = 0; i < max_pairs; ++i)
          value = std::max(value, diagram[2 * i + 1] - diagram[2 * i]);
        break;
      case 2:
        for (int i = 0; i < max_pairs; ++i)
          value += diagram[2 * i + 1] > diagram[2 * i] ? 1.0f : 0.0f;
        break;
      case 3:
      case 4: {
        float sum = 0.0f;
        int count = 0;
        for (int i = 0; i < max_pairs; ++i) {
          if (diagram[2 * i + 1] > diagram[2 * i]) {
            sum += diagram[2 * i + (feature_idx == 3 ? 0 : 1)];
            ++count;
          }
        }
        value = count > 0 ? sum / static_cast<float>(count) : 0.0f;
        break;
      }
      case 5: {
        float total = 0.0f;
        for (int i = 0; i < max_pairs; ++i)
          total += diagram[2 * i + 1] - diagram[2 * i];
        for (int i = 0; i < max_pairs; ++i) {
          const float p =
              (diagram[2 * i + 1] - diagram[2 * i]) / (total + 1e-8f);
          if (p > 0.0f)
            value -= p * std::log(p);
        }
        break;
      }
      case 6:
        for (int i = 0; i < max_pairs; ++i)
          value += std::isinf(diagram[2 * i + 1]) ? 1.0f : 0.0f;
        break;
      default:
        for (int i = 0; i < max_pairs; ++i) {
          const float persistence = diagram[2 * i + 1] - diagram[2 * i];
          value += (persistence > 0.1f && persistence < 1.0f) ? 1.0f : 0.0f;
        }
        break;
      }
      features[batch * num_features + feature_idx] = value;
    }
  }
  return features;
}

std::vector<float> cpuDenseEncoder(std::vector<float> layer_input,
                                   const std::vector<int> &layers,
                                   int batch_size) {
  for (size_t layer = 0; layer + 1 < layers.size(); ++layer) {
    const int in_dim = layers[layer];
    const int out_dim = layers[layer + 1];
    std::vector<float> layer_output(batch_size * out_dim, 0.0f);
    for (int batch = 0; batch < batch_size; ++batch) {
      for (int out_idx = 0; out_idx < out_dim; ++out_idx) {
        float sum = deterministicEncoderBias(static_cast<int>(layer), out_idx);
        for (int in_idx = 0; in_idx < in_dim; ++in_idx) {
          sum += layer_input[batch * in_dim + in_idx] *
                 deterministicEncoderWeight(static_cast<int>(layer), in_idx,
                                            out_idx);
        }
        layer_output[batch * out_dim + out_idx] = std::max(sum, 0.0f);
      }
    }
    layer_input.swap(layer_output);
  }
  return layer_input;
}

EncoderGPUBenchmark benchmarkGPUEncoder(int batch_size, int num_features) {
  if (batch_size <= 0 || num_features <= 0) {
    throw std::invalid_argument("encoder benchmark dimensions must be positive");
  }
  EncoderGPUBenchmark bench;
  bench.batch_size = batch_size;
  bench.num_features = num_features;

  int max_pairs = 100;
  std::vector<float> diagrams(batch_size * max_pairs * 2);
  for (int sample = 0; sample < batch_size; ++sample) {
    for (int pair = 0; pair < max_pairs; ++pair) {
      const size_t offset = static_cast<size_t>(sample * max_pairs + pair) * 2;
      const float birth =
          static_cast<float>((sample * 31 + pair * 17) % 101) / 101.0f;
      diagrams[offset] = birth;
      diagrams[offset + 1] =
          birth + 0.02f + static_cast<float>((pair * 13) % 37) / 370.0f;
    }
  }

  std::vector<int> layers = {8, 64, num_features};
  auto start_cpu = std::chrono::high_resolution_clock::now();
  auto cpu_features = computeCpuTopologicalFeatures(diagrams, batch_size,
                                                    max_pairs, layers.front());
  auto cpu_output =
      cpuDenseEncoder(std::move(cpu_features), layers, batch_size);
  auto end_cpu = std::chrono::high_resolution_clock::now();
  bench.cpu_time_ms =
      std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();
  volatile float cpu_sink = cpu_output.empty() ? 0.0f : cpu_output.back();
  (void)cpu_sink;

  GPUTopologicalEncoder encoder(8, layers);
  std::vector<float> gpu_output;

  auto start_gpu = std::chrono::high_resolution_clock::now();
  encoder.encode(diagrams, gpu_output, batch_size, max_pairs);
  auto end_gpu = std::chrono::high_resolution_clock::now();
  bench.gpu_time_ms =
      std::chrono::duration<double, std::milli>(end_gpu - start_gpu).count();

  auto start_fused = std::chrono::high_resolution_clock::now();
  encoder.encodeFused(diagrams, gpu_output, batch_size, max_pairs);
  auto end_fused = std::chrono::high_resolution_clock::now();
  bench.fused_time_ms =
      std::chrono::duration<double, std::milli>(end_fused - start_fused)
          .count();

  bench.speedup_gpu =
      finiteBenchmarkSpeedup(bench.cpu_time_ms, bench.gpu_time_ms);
  bench.speedup_fused =
      finiteBenchmarkSpeedup(bench.cpu_time_ms, bench.fused_time_ms);

  return bench;
}
