struct EigensolverBenchmark {
  double cpu_time_ms;
  double gpu_time_ms;
  double speedup;
  int matrix_size;
  bool mixed_precision;
};

EigensolverBenchmark benchmarkEigensolver(int matrix_size,
                                          bool mixed_precision) {
  if (matrix_size <= 0) {
    throw std::invalid_argument("matrix_size must be positive");
  }

  EigensolverBenchmark bench;
  bench.matrix_size = matrix_size;
  bench.mixed_precision = mixed_precision;

  const size_t matrix_entries =
      checkedPositiveSquareElements(matrix_size, "eigensolver benchmark matrix size overflows");
  const size_t n = static_cast<size_t>(matrix_size);
  std::vector<float> matrix(matrix_entries);
  std::mt19937 rng(0x31415926U);
  std::uniform_real_distribution<float> unit(0.0f, 1.0f);
  for (int i = 0; i < matrix_size; ++i) {
    for (int j = i; j < matrix_size; ++j) {
      float val = unit(rng);
      const size_t ij = static_cast<size_t>(i) * n + static_cast<size_t>(j);
      const size_t ji = static_cast<size_t>(j) * n + static_cast<size_t>(i);
      matrix[ij] = val;
      matrix[ji] = val;
    }
    matrix[static_cast<size_t>(i) * n + static_cast<size_t>(i)] += matrix_size;
  }

  auto start_cpu = std::chrono::high_resolution_clock::now();
  std::vector<float> cpu_eigenvalues(n);
  std::vector<float> cpu_eigenvectors(matrix_entries);

  std::vector<float> b(n, 1.0f);
  std::vector<float> b_next(n);

  for (int iter = 0; iter < 100; ++iter) {
    for (int i = 0; i < matrix_size; ++i) {
      float sum = 0.0f;
      for (int j = 0; j < matrix_size; ++j) {
        sum += matrix[static_cast<size_t>(i) * n + static_cast<size_t>(j)] *
               b[static_cast<size_t>(j)];
      }
      b_next[static_cast<size_t>(i)] = sum;
    }

    float norm = 0.0f;
    for (int i = 0; i < matrix_size; ++i) {
      norm += b_next[static_cast<size_t>(i)] * b_next[static_cast<size_t>(i)];
    }
    norm = std::sqrt(norm);

    for (int i = 0; i < matrix_size; ++i) {
      b[static_cast<size_t>(i)] = b_next[static_cast<size_t>(i)] / norm;
    }

    if (iter == 99) {
      float lambda = 0.0f;
      for (int i = 0; i < matrix_size; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < matrix_size; ++j) {
          sum += matrix[static_cast<size_t>(i) * n + static_cast<size_t>(j)] *
                 b[static_cast<size_t>(j)];
        }
        lambda += b[static_cast<size_t>(i)] * sum;
      }
      cpu_eigenvalues[0] = lambda;
      for (int i = 0; i < matrix_size; ++i) {
        cpu_eigenvectors[static_cast<size_t>(i) * n] = b[static_cast<size_t>(i)];
      }
    }
  }

  auto end_cpu = std::chrono::high_resolution_clock::now();
  bench.cpu_time_ms =
      std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();

  GPUEigensolver solver(matrix_size, mixed_precision);
  std::vector<float> gpu_eigenvalues;
  std::vector<float> gpu_eigenvectors;

  auto start_gpu = std::chrono::high_resolution_clock::now();
  solver.solve(matrix, gpu_eigenvalues, gpu_eigenvectors);
  auto end_gpu = std::chrono::high_resolution_clock::now();
  bench.gpu_time_ms =
      std::chrono::duration<double, std::milli>(end_gpu - start_gpu).count();

  bench.speedup = finiteBenchmarkSpeedup(bench.cpu_time_ms, bench.gpu_time_ms);

  return bench;
}
