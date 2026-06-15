/**
 * @brief Benchmark cohomology computation
 */
struct CohomologyBenchmark {
  double cpu_time_ms;
  double gpu_time_ms;
  double speedup;
  int num_simplices;
  int max_dim;
  int cup_products_computed;
};

CohomologyBenchmark benchmarkCohomology(int num_simplices, int max_dim) {
  if (num_simplices <= 0 || max_dim < 0) {
    throw std::invalid_argument("cohomology benchmark sizes are invalid");
  }

  CohomologyBenchmark bench;
  bench.num_simplices = num_simplices;
  bench.max_dim = max_dim;

  GPUCohomologyComputer computer(num_simplices, max_dim);

  // Generate a deterministic benchmark workload.
  std::vector<std::vector<int>> simplices;
  std::vector<std::vector<int>> coboundaries;

  for (int i = 0; i < num_simplices; ++i) {
    simplices.push_back({i});
    coboundaries.push_back({(i + 1) % num_simplices});
  }

  auto start_cpu = std::chrono::high_resolution_clock::now();
  std::vector<int> cpu_basis;
  std::vector<int> used_pivots(std::max(0, num_simplices), -1);
  for (int col = 0; col < num_simplices; ++col) {
    int pivot = -1;
    for (int row = num_simplices - 1; row >= 0; --row) {
      const auto &row_coboundaries = coboundaries[static_cast<size_t>(col)];
      if (std::find(row_coboundaries.begin(), row_coboundaries.end(), row) !=
          row_coboundaries.end()) {
        pivot = row;
        break;
      }
    }
    if (pivot >= 0 && used_pivots[static_cast<size_t>(pivot)] < 0) {
      used_pivots[static_cast<size_t>(pivot)] = col;
      cpu_basis.push_back(col);
    }
  }
  int cpu_products = 0;
  for (size_t i = 0; i < cpu_basis.size() && i < 10; ++i) {
    for (size_t j = i; j < cpu_basis.size() && j < 10; ++j) {
      for (int candidate = 0; candidate < num_simplices; ++candidate) {
        const auto &first = coboundaries[static_cast<size_t>(cpu_basis[i])];
        const auto &second = coboundaries[static_cast<size_t>(cpu_basis[j])];
        const bool contains_first =
            candidate == cpu_basis[i] ||
            std::find(first.begin(), first.end(), candidate) != first.end();
        const bool contains_second =
            candidate == cpu_basis[j] ||
            std::find(second.begin(), second.end(), candidate) != second.end();
        if (contains_first && contains_second) {
          ++cpu_products;
          break;
        }
      }
    }
  }
  auto end_cpu = std::chrono::high_resolution_clock::now();
  bench.cpu_time_ms =
      std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();

  // GPU benchmark
  auto start_gpu = std::chrono::high_resolution_clock::now();
  auto basis = computer.computeCohomologyBasis(simplices, coboundaries);

  // Compute some cup products
  bench.cup_products_computed = 0;
  for (size_t i = 0; i < basis.size() && i < 10; ++i) {
    for (size_t j = i; j < basis.size() && j < 10; ++j) {
      auto product = computer.cupProduct(basis[i], basis[j]);
      bench.cup_products_computed += product.num_simplices;
      bench.cup_products_computed++;
    }
  }

  auto end_gpu = std::chrono::high_resolution_clock::now();
  bench.gpu_time_ms =
      std::chrono::duration<double, std::milli>(end_gpu - start_gpu).count();

  auto ratio = [](double cpu_ms, double gpu_ms) {
    if (!std::isfinite(cpu_ms) || !std::isfinite(gpu_ms) || cpu_ms < 0.0 ||
        gpu_ms <= 0.0) {
      return 1.0;
    }
    return cpu_ms / gpu_ms;
  };
  bench.speedup = ratio(bench.cpu_time_ms, bench.gpu_time_ms);
  bench.cup_products_computed =
      std::max(bench.cup_products_computed, cpu_products);

  return bench;
}
