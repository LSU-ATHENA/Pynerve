class GPUTopologicalEncoder {
public:
  GPUTopologicalEncoder(int topo_features, const std::vector<int> &mlp_layers)
      : topo_features_(topo_features), mlp_layers_(mlp_layers) {
    if (topo_features_ <= 0 || mlp_layers_.size() < 2 ||
        mlp_layers_.front() != topo_features_) {
      throw std::invalid_argument("encoder layers must start with topo_features");
    }
    for (int dim : mlp_layers_) {
      if (dim <= 0) {
        throw std::invalid_argument("encoder layer dimensions must be positive");
      }
    }

    try {
      for (size_t i = 0; i + 1 < mlp_layers_.size(); ++i) {
        const int in_dim = mlp_layers_[i];
        const int out_dim = mlp_layers_[i + 1];
        const size_t weight_count = checkedMulSize(static_cast<size_t>(in_dim),
                                                   static_cast<size_t>(out_dim),
                                                   "encoder layer weight count");
        float *d_weights = nullptr;
        float *d_bias = nullptr;
        allocateDevice(&d_weights, weight_count, "allocate encoder layer weights");
        allocateDevice(&d_bias, static_cast<size_t>(out_dim), "allocate encoder layer bias");

        std::vector<float> weights(weight_count);
        std::vector<float> bias(static_cast<size_t>(out_dim));
        for (int in_idx = 0; in_idx < in_dim; ++in_idx) {
          for (int out_idx = 0; out_idx < out_dim; ++out_idx) {
            weights[static_cast<size_t>(in_idx) * out_dim + out_idx] =
                deterministicEncoderWeight(static_cast<int>(i), in_idx, out_idx);
          }
        }
        for (int out_idx = 0; out_idx < out_dim; ++out_idx) {
          bias[static_cast<size_t>(out_idx)] = deterministicEncoderBias(static_cast<int>(i), out_idx);
        }
        copyToDevice(d_weights, weights.data(), weights.size(), "copy encoder layer weights");
        copyToDevice(d_bias, bias.data(), bias.size(), "copy encoder layer bias");
        d_mlp_weights_.push_back(d_weights);
        d_mlp_bias_.push_back(d_bias);
      }
    } catch (...) {
      cleanup();
      throw;
    }
  }

  ~GPUTopologicalEncoder() { cleanup(); }

  void encode(const std::vector<float> &diagrams, std::vector<float> &output,
              int batch_size, int max_pairs) {
    validateEncodeInput(diagrams, batch_size, max_pairs);
    float *d_diagrams = nullptr;
    float *d_features = nullptr;
    float *d_output = nullptr;
    auto free_runtime = [&]() {
      if (d_diagrams) cudaFree(d_diagrams);
      if (d_features) cudaFree(d_features);
      if (d_output) cudaFree(d_output);
    };

    try {
      allocateDevice(&d_diagrams, diagrams.size(), "allocate encoder diagrams");
      int workspace_dim = topo_features_;
      for (int dim : mlp_layers_) {
        workspace_dim = std::max(workspace_dim, dim);
      }
      const size_t workspace_count = checkedMulSize(static_cast<size_t>(batch_size),
                                                    static_cast<size_t>(workspace_dim),
                                                    "encoder workspace count");
      allocateDevice(&d_features, workspace_count, "allocate encoder features");
      allocateDevice(&d_output, workspace_count, "allocate encoder output workspace");
      copyToDevice(d_diagrams, diagrams.data(), diagrams.size(), "copy encoder diagrams");

      dim3 grid1(batch_size, 1);
      dim3 block1(FEATURE_EXTRACTION_BLOCK_SIZE);
      topologicalFeatureExtractionKernel<<<grid1, block1>>>(
          d_diagrams, d_features, batch_size, max_pairs, topo_features_);
      checkCuda(cudaPeekAtLastError(), "launch encoder feature extraction");
      checkCuda(cudaDeviceSynchronize(), "synchronize encoder feature extraction");
      validateDeviceFinite(
          d_features,
          checkedMulSize(static_cast<size_t>(batch_size), static_cast<size_t>(topo_features_),
                         "encoder feature output count"),
          "encoder extracted features");

      float *layer_input = d_features;
      float *layer_output = d_output;
      for (size_t i = 0; i + 1 < mlp_layers_.size(); ++i) {
        const int in_dim = mlp_layers_[i];
        const int out_dim = mlp_layers_[i + 1];
        dim3 grid2((out_dim + GRID_ROUNDING_OFFSET) / GRID_DIVISOR, batch_size);
        dim3 block2(BLOCK_SIZE);
        fusedMLPLayerKernel<<<grid2, block2>>>(layer_input, d_mlp_weights_[i], d_mlp_bias_[i],
                                               layer_output, batch_size, in_dim, out_dim, 0);
        checkCuda(cudaPeekAtLastError(), "launch encoder MLP layer");
        checkCuda(cudaDeviceSynchronize(), "synchronize encoder MLP layer");
        validateDeviceFinite(
            layer_output,
            checkedMulSize(static_cast<size_t>(batch_size), static_cast<size_t>(out_dim),
                           "encoder layer output count"),
            "encoder layer output");
        std::swap(layer_input, layer_output);
      }

      output.resize(checkedMulSize(static_cast<size_t>(batch_size),
                                   static_cast<size_t>(mlp_layers_.back()),
                                   "encoder output count"));
      copyToHost(output.data(), layer_input, output.size(), "copy encoder output");
      requireFiniteValues(output, "encoder output");
      free_runtime();
    } catch (...) {
      free_runtime();
      throw;
    }
  }

  void encodeFused(const std::vector<float> &diagrams,
                   std::vector<float> &output, int batch_size, int max_pairs) {
    validateEncodeInput(diagrams, batch_size, max_pairs);
    if (mlp_layers_.size() != 2) {
      encode(diagrams, output, batch_size, max_pairs);
      return;
    }

    float *d_diagrams = nullptr;
    float *d_output = nullptr;
    auto free_runtime = [&]() {
      if (d_diagrams) cudaFree(d_diagrams);
      if (d_output) cudaFree(d_output);
    };

    try {
      allocateDevice(&d_diagrams, diagrams.size(), "allocate fused encoder diagrams");
      const size_t output_count = checkedMulSize(static_cast<size_t>(batch_size),
                                                 static_cast<size_t>(mlp_layers_.back()),
                                                 "fused encoder output count");
      allocateDevice(&d_output, output_count, "allocate fused encoder output");
      copyToDevice(d_diagrams, diagrams.data(), diagrams.size(), "copy fused encoder diagrams");

      const size_t smem_size = checkedMulSize(static_cast<size_t>(topo_features_), sizeof(float),
                                              "fused encoder shared memory bytes");
      fusedTopologicalEncoderKernel<<<batch_size, BLOCK_SIZE, smem_size>>>(
          d_diagrams, d_mlp_weights_[0], d_mlp_bias_[0], d_output, batch_size,
          max_pairs, topo_features_, mlp_layers_.back());
      checkCuda(cudaPeekAtLastError(), "launch fused encoder kernel");
      checkCuda(cudaDeviceSynchronize(), "synchronize fused encoder kernel");

      output.resize(output_count);
      copyToHost(output.data(), d_output, output.size(), "copy fused encoder output");
      requireFiniteValues(output, "fused encoder output");
      free_runtime();
    } catch (...) {
      free_runtime();
      throw;
    }
  }

private:
  void validateEncodeInput(const std::vector<float> &diagrams, int batch_size,
                           int max_pairs) const {
    if (batch_size <= 0 || max_pairs <= 0) {
      throw std::invalid_argument("batch_size and max_pairs must be positive");
    }
    const size_t expected = checkedMulSize(
        checkedMulSize(static_cast<size_t>(batch_size), static_cast<size_t>(max_pairs),
                       "encoder diagram pair count"),
        2U, "encoder diagram value count");
    checkedIntSize(expected, "encoder diagram value count");
    if (diagrams.size() != expected) {
      throw std::invalid_argument("diagram buffer shape does not match batch and pair counts");
    }
    requireValidDiagramValues(diagrams, "encoder diagram input");
  }

  void cleanup() noexcept {
    for (auto *ptr : d_mlp_weights_) {
      if (ptr)
        cudaFree(ptr);
    }
    for (auto *ptr : d_mlp_bias_) {
      if (ptr)
        cudaFree(ptr);
    }
    d_mlp_weights_.clear();
    d_mlp_bias_.clear();
  }

  int topo_features_;
  std::vector<int> mlp_layers_;
  std::vector<float *> d_mlp_weights_;
  std::vector<float *> d_mlp_bias_;
};

class UnifiedMemoryEncoder {
public:
  explicit UnifiedMemoryEncoder(size_t buffer_size) {
    if (buffer_size == 0) {
      throw std::invalid_argument("buffer_size must be positive");
    }
    buffer_size_ = buffer_size;
    checkCuda(cudaMallocManaged(&buffer_, checkedMulSize(buffer_size_, sizeof(float),
                                                        "unified encoder buffer bytes")),
              "allocate unified encoder buffer");
  }

  ~UnifiedMemoryEncoder() {
    if (buffer_)
      cudaFree(buffer_);
  }

  float *getBuffer() { return buffer_; }

  void prefetchToGPU() {
    int active_device = 0;
    checkCuda(cudaGetDevice(&active_device), "get active CUDA device");
#if CUDART_VERSION >= 12000
    cudaMemLocation location{};
    location.type = cudaMemLocationTypeDevice;
    location.id = active_device;
    checkCuda(cudaMemPrefetchAsync(buffer_, checkedMulSize(buffer_size_, sizeof(float),
                                                          "unified encoder prefetch bytes"),
                                   location, 0u, nullptr),
              "prefetch unified encoder buffer");
#else
    checkCuda(cudaMemPrefetchAsync(buffer_, checkedMulSize(buffer_size_, sizeof(float),
                                                          "unified encoder prefetch bytes"),
                                   active_device, nullptr),
              "prefetch unified encoder buffer");
#endif
  }

private:
  float *buffer_ = nullptr;
  size_t buffer_size_ = 0;
};
