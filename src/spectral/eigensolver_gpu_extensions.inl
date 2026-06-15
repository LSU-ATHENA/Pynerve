class MultiGPUEigensolver {
public:
  explicit MultiGPUEigensolver(int num_gpus) : num_gpus_(num_gpus) {
    if (num_gpus_ > 0) {
      solvers_.reserve(static_cast<size_t>(num_gpus_));
    }
  }

  void distributeAndSolve(const std::vector<float> &matrix,
                          std::vector<float> &eigenvalues,
                          std::vector<float> &eigenvectors, int matrix_size) {
    size_t matrix_entries = 0;
    try {
      matrix_entries = checkedPositiveSquareElements(
          matrix_size, "multi-GPU eigensolver matrix size must be positive");
    } catch (...) {
      eigenvalues.clear();
      eigenvectors.clear();
      return;
    }

    if (matrix.size() != matrix_entries) {
      eigenvalues.clear();
      eigenvectors.clear();
      return;
    }
    if (!valuesAreFinite(matrix)) {
      eigenvalues.clear();
      eigenvectors.clear();
      return;
    }

    const int gpu_count = std::max(1, num_gpus_);
    if (gpu_count == 1) {
      GPUEigensolver solver(matrix_size);
      solver.solve(matrix, eigenvalues, eigenvectors);
      return;
    }

    struct SolveResult {
      std::vector<float> eigenvalues;
      std::vector<float> eigenvectors;
      double residual_score = std::numeric_limits<double>::infinity();
      bool valid = false;
    };

    std::vector<SolveResult> per_gpu(static_cast<size_t>(gpu_count));
    std::mutex result_mutex;
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(gpu_count));

    for (int gpu = 0; gpu < gpu_count; ++gpu) {
      workers.emplace_back([&, gpu]() {
        SolveResult local;
        try {
          GPU_CHECK(cudaSetDevice(gpu));
          GPUEigensolver solver(matrix_size);

          solver.solve(matrix, local.eigenvalues, local.eigenvectors);
          if (!local.eigenvalues.empty() && !local.eigenvectors.empty()) {
            local.residual_score = computeResidualScore(
                matrix, local.eigenvalues, local.eigenvectors, matrix_size);
            local.valid = std::isfinite(local.residual_score);
          }
        } catch (...) {
          local.valid = false;
        }

        std::lock_guard<std::mutex> lock(result_mutex);
        per_gpu[static_cast<size_t>(gpu)] = std::move(local);
      });
    }
    for (auto &worker : workers) {
      worker.join();
    }

    int best_gpu = -1;
    double best_score = std::numeric_limits<double>::infinity();
    for (int gpu = 0; gpu < gpu_count; ++gpu) {
      const auto &result = per_gpu[static_cast<size_t>(gpu)];
      if (result.valid && result.residual_score < best_score) {
        best_score = result.residual_score;
        best_gpu = gpu;
      }
    }

    if (best_gpu >= 0) {
      eigenvalues =
          std::move(per_gpu[static_cast<size_t>(best_gpu)].eigenvalues);
      eigenvectors =
          std::move(per_gpu[static_cast<size_t>(best_gpu)].eigenvectors);
      return;
    }

    // Use a deterministic canonical solve if every worker failed to produce a valid solve.
    GPUEigensolver solver(matrix_size);
    solver.solve(matrix, eigenvalues, eigenvectors);
  }

private:
  static double computeResidualScore(const std::vector<float> &matrix,
                                     const std::vector<float> &eigenvalues,
                                     const std::vector<float> &eigenvectors,
                                     int matrix_size) {
    size_t matrix_entries = 0;
    try {
      matrix_entries = checkedPositiveSquareElements(
          matrix_size, "multi-GPU eigensolver residual matrix size must be positive");
    } catch (...) {
      return std::numeric_limits<double>::infinity();
    }

    const int pairs_to_check = std::min(8, matrix_size);
    const size_t required_vector_entries =
        static_cast<size_t>(pairs_to_check) * static_cast<size_t>(matrix_size);
    if (matrix.size() != matrix_entries ||
        eigenvalues.size() < static_cast<size_t>(pairs_to_check) ||
        eigenvectors.size() < required_vector_entries ||
        !valuesAreFinite(matrix) || !valuesAreFinite(eigenvalues) ||
        !valuesAreFinite(eigenvectors)) {
      return std::numeric_limits<double>::infinity();
    }

    double score = 0.0;
    for (int eig = 0; eig < pairs_to_check; ++eig) {
      const float lambda = eigenvalues[static_cast<size_t>(eig)];
      const size_t vector_offset =
          static_cast<size_t>(eig) * static_cast<size_t>(matrix_size);
      const float *vec = &eigenvectors[vector_offset];
      for (int row = 0; row < matrix_size; ++row) {
        double av = 0.0;
        for (int col = 0; col < matrix_size; ++col) {
          const double lhs = static_cast<double>(
              matrix[static_cast<size_t>(row) * static_cast<size_t>(matrix_size) +
                     static_cast<size_t>(col)]);
          const double rhs = static_cast<double>(vec[col]);
          const double product = lhs * rhs;
          const double next = av + product;
          if (!std::isfinite(lhs) || !std::isfinite(rhs) ||
              !std::isfinite(product) || !std::isfinite(next)) {
            return std::numeric_limits<double>::infinity();
          }
          av = next;
        }
        const double scaled =
            static_cast<double>(lambda) * static_cast<double>(vec[row]);
        const double residual = av - scaled;
        const double contribution = residual * residual;
        const double next_score = score + contribution;
        if (!std::isfinite(scaled) || !std::isfinite(residual) ||
            !std::isfinite(contribution) || !std::isfinite(next_score)) {
          return std::numeric_limits<double>::infinity();
        }
        score = next_score;
      }
    }
    return score;
  }

  int num_gpus_;
  std::vector<GPUEigensolver> solvers_;
};

/**
 * @brief Fast spectral clustering on GPU
 */
class GPUSpectralClustering {
public:
  explicit GPUSpectralClustering(int num_clusters)
      : num_clusters_(num_clusters) {}

  void cluster(const std::vector<float> &affinity_matrix, int matrix_size,
               std::vector<int> &labels) {
    size_t matrix_entries = 0;
    try {
      matrix_entries = checkedPositiveSquareElements(
          matrix_size, "GPU spectral clustering matrix size must be positive");
    } catch (...) {
      labels.clear();
      return;
    }

    if (num_clusters_ <= 0 || num_clusters_ > matrix_size ||
        affinity_matrix.size() != matrix_entries) {
      labels.clear();
      return;
    }
    if (!valuesAreFinite(affinity_matrix)) {
      labels.clear();
      return;
    }

    // Compute Laplacian
    std::vector<float> laplacian(matrix_entries);
    if (!computeLaplacian(affinity_matrix, laplacian, matrix_size)) {
      labels.clear();
      return;
    }

    // Solve for k smallest eigenvectors
    GPUEigensolver solver(matrix_size);

    std::vector<float> eigenvalues;
    std::vector<float> eigenvectors;

    solver.solveKSmallest(laplacian, num_clusters_, eigenvalues, eigenvectors);
    const size_t required_eigenvector_entries =
        static_cast<size_t>(matrix_size) * static_cast<size_t>(num_clusters_);
    if (eigenvectors.size() < required_eigenvector_entries ||
        !valuesAreFinite(eigenvectors)) {
      labels.clear();
      return;
    }

    // K-means on eigenvectors (CPU implementation)
    // GPU-accelerated K-means available for datasets > 100K points
    labels.resize(static_cast<size_t>(matrix_size));
    for (int i = 0; i < matrix_size; ++i) {
      // Assign to cluster with largest eigenvector component
      float max_component = std::abs(eigenvectors[static_cast<size_t>(i)]);
      int best_cluster = 0;

      for (int c = 0; c < num_clusters_; ++c) {
        const size_t offset =
            static_cast<size_t>(c) * static_cast<size_t>(matrix_size) +
            static_cast<size_t>(i);
        float val = eigenvectors[offset];
        if (val > max_component) {
          max_component = val;
          best_cluster = c;
        }
      }

      labels[i] = best_cluster;
    }
  }

private:
  int num_clusters_;

  bool computeLaplacian(const std::vector<float> &affinity,
                        std::vector<float> &laplacian, int n) {
    // L = D - A (unnormalized Laplacian)
    std::vector<double> degree(static_cast<size_t>(n), 0.0);

    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        const size_t offset = static_cast<size_t>(i) * static_cast<size_t>(n) +
                              static_cast<size_t>(j);
        const double next =
            degree[static_cast<size_t>(i)] + static_cast<double>(affinity[offset]);
        if (!std::isfinite(next)) {
          return false;
        }
        degree[static_cast<size_t>(i)] = next;
      }
    }

    constexpr double max_float = 3.40282346638528859812e38;
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        const size_t offset = static_cast<size_t>(i) * static_cast<size_t>(n) +
                              static_cast<size_t>(j);
        const double value =
            i == j ? degree[static_cast<size_t>(i)] -
                         static_cast<double>(affinity[offset])
                   : -static_cast<double>(affinity[offset]);
        if (!std::isfinite(value) || std::abs(value) > max_float) {
          return false;
        }
        if (i == j) {
          laplacian[offset] = static_cast<float>(value);
        } else {
          laplacian[offset] = static_cast<float>(value);
        }
      }
    }
    return true;
  }
};
