class GPUDiagramGNNLayer
{
public:
    enum class AggregationType
    {
        MEAN,
        SUM,
        MAX,
        ATTENTION
    };

    GPUDiagramGNNLayer(int max_pairs, int feature_dim, AggregationType agg_type)
        : max_pairs_(max_pairs)
        , feature_dim_(feature_dim)
        , agg_type_(agg_type)
    {
        if (max_pairs_ <= 0 || feature_dim_ <= 0)
        {
            throw std::invalid_argument("diagram GNN dimensions must be positive");
        }
        pair_matrix_count_ =
            checkedMulSize(static_cast<size_t>(max_pairs_), static_cast<size_t>(max_pairs_),
                           "diagram pair matrix count");
        feature_count_capacity_ =
            checkedMulSize(static_cast<size_t>(max_pairs_), static_cast<size_t>(feature_dim_),
                           "diagram feature capacity");
        checkedIntSize(pair_matrix_count_, "diagram pair matrix count");
        checkedIntSize(feature_count_capacity_, "diagram feature capacity");
        try
        {
            allocateDevice(&d_adjacency_, pair_matrix_count_, "allocate diagram adjacency");
            allocateDevice(&d_messages_, feature_count_capacity_, "allocate diagram messages");
            allocateDevice(&d_attention_weights_, pair_matrix_count_,
                           "allocate diagram attention weights");
            if (agg_type == AggregationType::ATTENTION)
            {
                allocateDevice(&d_query_, feature_count_capacity_, "allocate diagram query");
                allocateDevice(&d_key_, feature_count_capacity_, "allocate diagram key");
                allocateDevice(&d_value_, feature_count_capacity_, "allocate diagram value");
            }
        }
        catch (...)
        {
            cleanup();
            throw;
        }
    }

    ~GPUDiagramGNNLayer() { cleanup(); }

    void computeAdjacency(const std::vector<float> &diagrams, float bandwidth)
    {
        const int num_pairs = validateDiagramBuffer(diagrams);
        if (!std::isfinite(bandwidth) || bandwidth <= 0.0f)
        {
            throw std::invalid_argument("bandwidth must be finite and positive");
        }
        const size_t active_matrix_count =
            checkedMulSize(static_cast<size_t>(num_pairs), static_cast<size_t>(num_pairs),
                           "diagram active adjacency count");

        float *d_diagrams = nullptr;
        try
        {
            allocateDevice(&d_diagrams, diagrams.size(), "allocate diagram adjacency input");
            copyToDevice(d_diagrams, diagrams.data(), diagrams.size(),
                         "copy diagram adjacency input");
            dim3 blocks((num_pairs + DIAGRAM_TILE_MASK) / DIAGRAM_TILE_DIM,
                        (num_pairs + DIAGRAM_TILE_MASK) / DIAGRAM_TILE_DIM);
            dim3 threads(DIAGRAM_TILE_DIM, DIAGRAM_TILE_DIM);
            diagramDistanceKernel<<<blocks, threads>>>(d_diagrams, d_adjacency_, num_pairs,
                                                       bandwidth);
            checkCuda(cudaPeekAtLastError(), "launch diagram distance kernel");
            checkCuda(cudaDeviceSynchronize(), "synchronize diagram distance kernel");
            validateDeviceFinite(d_adjacency_, active_matrix_count, "diagram adjacency");
            cudaFree(d_diagrams);
        }
        catch (...)
        {
            if (d_diagrams)
                cudaFree(d_diagrams);
            throw;
        }
    }

    std::vector<float> forward(const std::vector<float> &node_features,
                               const std::vector<float> &diagrams)
    {
        const int num_pairs = validateDiagramBuffer(diagrams);
        const size_t feature_count =
            checkedMulSize(static_cast<size_t>(num_pairs), static_cast<size_t>(feature_dim_),
                           "diagram forward feature count");
        if (node_features.size() != feature_count)
        {
            throw std::invalid_argument("node_features must match num_pairs * feature_dim");
        }
        requireFiniteValues(node_features, "diagram node features");
        const size_t active_matrix_count =
            checkedMulSize(static_cast<size_t>(num_pairs), static_cast<size_t>(num_pairs),
                           "diagram forward adjacency count");

        float *d_features = nullptr;
        float *d_diagrams = nullptr;
        auto free_runtime = [&]() {
            if (d_features)
                cudaFree(d_features);
            if (d_diagrams)
                cudaFree(d_diagrams);
        };

        try
        {
            allocateDevice(&d_features, node_features.size(), "allocate diagram features");
            allocateDevice(&d_diagrams, diagrams.size(), "allocate diagram data");
            copyToDevice(d_features, node_features.data(), node_features.size(),
                         "copy diagram features");
            copyToDevice(d_diagrams, diagrams.data(), diagrams.size(), "copy diagram data");

            dim3 distance_blocks((num_pairs + DIAGRAM_TILE_MASK) / DIAGRAM_TILE_DIM,
                                 (num_pairs + DIAGRAM_TILE_MASK) / DIAGRAM_TILE_DIM);
            dim3 distance_threads(DIAGRAM_TILE_DIM, DIAGRAM_TILE_DIM);
            diagramDistanceKernel<<<distance_blocks, distance_threads>>>(
                d_diagrams, d_adjacency_, num_pairs, DEFAULT_BANDWIDTH);
            checkCuda(cudaPeekAtLastError(), "launch diagram forward distance kernel");
            checkCuda(cudaDeviceSynchronize(), "synchronize diagram forward distance kernel");
            validateDeviceFinite(d_adjacency_, active_matrix_count, "diagram forward adjacency");

            if (agg_type_ == AggregationType::ATTENTION)
            {
                copyDeviceToDevice(d_query_, d_features, feature_count, "copy diagram query");
                copyDeviceToDevice(d_key_, d_features, feature_count, "copy diagram key");
                copyDeviceToDevice(d_value_, d_features, feature_count, "copy diagram value");

                int threads_per_pair = 1;
                while (threads_per_pair < feature_dim_ && threads_per_pair < BLOCK_SIZE)
                {
                    threads_per_pair <<= 1;
                }
                dim3 blocks(num_pairs);
                dim3 threads(threads_per_pair);
                const size_t smem_size =
                    checkedMulSize(static_cast<size_t>(num_pairs + threads_per_pair), sizeof(float),
                                   "diagram attention shared memory bytes");
                diagramAttentionKernel<float><<<blocks, threads, smem_size>>>(
                    d_query_, d_key_, d_value_, d_diagrams, d_attention_weights_, d_messages_,
                    num_pairs, feature_dim_, 1.0f);
                checkCuda(cudaPeekAtLastError(), "launch diagram attention kernel");
            }
            else
            {
                dim3 blocks(num_pairs);
                dim3 threads(std::min(feature_dim_, BLOCK_SIZE));
                const size_t smem_size =
                    checkedMulSize(static_cast<size_t>(num_pairs), sizeof(float),
                                   "diagram message shared memory bytes");
                diagramMessagePassingKernel<float><<<blocks, threads, smem_size>>>(
                    d_features, d_adjacency_, d_messages_, num_pairs, feature_dim_, false);
                checkCuda(cudaPeekAtLastError(), "launch diagram message passing kernel");
            }
            checkCuda(cudaDeviceSynchronize(), "synchronize diagram GNN forward");

            std::vector<float> output(feature_count);
            copyToHost(output.data(), d_messages_, output.size(), "copy diagram GNN output");
            requireFiniteValues(output, "diagram GNN output");
            free_runtime();
            return output;
        }
        catch (...)
        {
            free_runtime();
            throw;
        }
    }

    std::vector<float> getAttentionWeights() const
    {
        std::vector<float> weights(pair_matrix_count_);
        copyToHost(weights.data(), d_attention_weights_, weights.size(),
                   "copy diagram attention weights");
        return weights;
    }

private:
    int validateDiagramBuffer(const std::vector<float> &diagrams) const
    {
        if (diagrams.empty() || diagrams.size() % DIAGRAM_FIELDS_PER_PAIR != 0)
        {
            throw std::invalid_argument("diagram buffer must contain complete persistence pairs");
        }
        const int num_pairs =
            checkedIntSize(diagrams.size() / DIAGRAM_FIELDS_PER_PAIR, "diagram pair count");
        if (num_pairs <= 0 || num_pairs > max_pairs_)
        {
            throw std::length_error("diagram pair count exceeds configured capacity");
        }
        requireFiniteDiagramValues(diagrams, "diagram buffer");
        return num_pairs;
    }

    void copyDeviceToDevice(float *dst, const float *src, size_t count, const char *label) const
    {
        if (count == 0)
        {
            return;
        }
        checkCuda(cudaMemcpy(dst, src, checkedMulSize(count, sizeof(float), label),
                             cudaMemcpyDeviceToDevice),
                  label);
    }

    void cleanup() noexcept
    {
        if (d_adjacency_)
            cudaFree(d_adjacency_);
        if (d_messages_)
            cudaFree(d_messages_);
        if (d_attention_weights_)
            cudaFree(d_attention_weights_);
        if (d_query_)
            cudaFree(d_query_);
        if (d_key_)
            cudaFree(d_key_);
        if (d_value_)
            cudaFree(d_value_);
        d_adjacency_ = nullptr;
        d_messages_ = nullptr;
        d_attention_weights_ = nullptr;
        d_query_ = nullptr;
        d_key_ = nullptr;
        d_value_ = nullptr;
    }

    int max_pairs_ = 0;
    int feature_dim_ = 0;
    AggregationType agg_type_;
    size_t pair_matrix_count_ = 0;
    size_t feature_count_capacity_ = 0;

    float *d_adjacency_ = nullptr;
    float *d_messages_ = nullptr;
    float *d_attention_weights_ = nullptr;
    float *d_query_ = nullptr;
    float *d_key_ = nullptr;
    float *d_value_ = nullptr;
};

/**
 * @brief Multi-scale diagram convolution
 */
class GPUMultiScaleDiagramConv
{
public:
    GPUMultiScaleDiagramConv(int max_pairs, int feature_dim, const std::vector<float> &scales)
        : max_pairs_(max_pairs)
        , feature_dim_(feature_dim)
        , scales_(scales)
    {
        if (max_pairs_ <= 0 || feature_dim_ <= 0)
        {
            throw std::invalid_argument("multiscale diagram dimensions must be positive");
        }
        if (scales_.empty())
        {
            throw std::invalid_argument("at least one scale is required");
        }
        for (float scale : scales_)
        {
            if (!std::isfinite(scale) || scale <= 0.0f)
            {
                throw std::invalid_argument("diagram scales must be finite and positive");
            }
        }
        pair_matrix_count_ =
            checkedMulSize(static_cast<size_t>(max_pairs_), static_cast<size_t>(max_pairs_),
                           "multiscale diagram matrix count");
        feature_count_capacity_ =
            checkedMulSize(static_cast<size_t>(max_pairs_), static_cast<size_t>(feature_dim_),
                           "multiscale diagram feature capacity");
        try
        {
            for (size_t i = 0; i < scales_.size(); ++i)
            {
                float *d_adj = nullptr;
                allocateDevice(&d_adj, pair_matrix_count_, "allocate multiscale diagram adjacency");
                adjacency_per_scale_.push_back(d_adj);
            }
            allocateDevice(&d_combined_, feature_count_capacity_,
                           "allocate multiscale combined output");
        }
        catch (...)
        {
            cleanup();
            throw;
        }
    }

    ~GPUMultiScaleDiagramConv() { cleanup(); }

    std::vector<float> forward(const std::vector<float> &diagrams,
                               const std::vector<float> &features)
    {
        const int num_pairs = validateDiagramBuffer(diagrams);
        const size_t total_values =
            checkedMulSize(static_cast<size_t>(num_pairs), static_cast<size_t>(feature_dim_),
                           "multiscale diagram total values");
        if (features.size() != total_values)
        {
            throw std::invalid_argument(
                "feature vector size does not match num_pairs * feature_dim");
        }
        requireFiniteValues(features, "multiscale diagram features");

        float *d_diagrams = nullptr;
        float *d_features = nullptr;
        float *d_scale_messages = nullptr;
        auto free_runtime = [&]() {
            if (d_scale_messages)
                cudaFree(d_scale_messages);
            if (d_features)
                cudaFree(d_features);
            if (d_diagrams)
                cudaFree(d_diagrams);
        };

        try
        {
            allocateDevice(&d_diagrams, diagrams.size(), "allocate multiscale diagrams");
            allocateDevice(&d_features, features.size(), "allocate multiscale features");
            allocateDevice(&d_scale_messages, total_values, "allocate multiscale messages");
            copyToDevice(d_diagrams, diagrams.data(), diagrams.size(), "copy multiscale diagrams");
            copyToDevice(d_features, features.data(), features.size(), "copy multiscale features");
            checkCuda(cudaMemset(
                          d_combined_, 0,
                          checkedMulSize(total_values, sizeof(float), "multiscale combined bytes")),
                      "clear multiscale combined output");

            dim3 distance_blocks((num_pairs + DIAGRAM_TILE_MASK) / DIAGRAM_TILE_DIM,
                                 (num_pairs + DIAGRAM_TILE_MASK) / DIAGRAM_TILE_DIM);
            dim3 distance_threads(DIAGRAM_TILE_DIM, DIAGRAM_TILE_DIM);
            const int message_blocks = num_pairs;
            const int message_threads = std::min(feature_dim_, BLOCK_SIZE);
            const size_t message_smem =
                checkedMulSize(static_cast<size_t>(num_pairs), sizeof(float),
                               "multiscale message shared memory bytes");
            const int total_values_int =
                checkedIntSize(total_values, "multiscale total value count");
            const int reduce_blocks =
                static_cast<int>((total_values + BLOCK_SIZE - 1) / BLOCK_SIZE);
            const float scale_weight = 1.0f / static_cast<float>(scales_.size());
            for (size_t i = 0; i < scales_.size(); ++i)
            {
                const float bandwidth = fmaxf(scales_[i], EPSILON);
                diagramDistanceKernel<<<distance_blocks, distance_threads>>>(
                    d_diagrams, adjacency_per_scale_[i], num_pairs, bandwidth);
                checkCuda(cudaPeekAtLastError(), "launch multiscale distance kernel");
                diagramMessagePassingKernel<float>
                    <<<message_blocks, message_threads, message_smem>>>(
                        d_features, adjacency_per_scale_[i], d_scale_messages, num_pairs,
                        feature_dim_, false);
                checkCuda(cudaPeekAtLastError(), "launch multiscale message kernel");
                accumulateScaleKernel<<<reduce_blocks, BLOCK_SIZE>>>(
                    d_scale_messages, d_combined_, total_values_int, scale_weight);
                checkCuda(cudaPeekAtLastError(), "launch multiscale accumulate kernel");
                checkCuda(cudaDeviceSynchronize(), "synchronize multiscale diagram kernels");
            }

            std::vector<float> output(total_values);
            copyToHost(output.data(), d_combined_, output.size(), "copy multiscale output");
            requireFiniteValues(output, "multiscale diagram output");
            free_runtime();
            return output;
        }
        catch (...)
        {
            free_runtime();
            throw;
        }
    }

private:
    int validateDiagramBuffer(const std::vector<float> &diagrams) const
    {
        if (diagrams.empty() || diagrams.size() % DIAGRAM_FIELDS_PER_PAIR != 0)
        {
            throw std::invalid_argument("diagram buffer must contain complete persistence pairs");
        }
        const int num_pairs = checkedIntSize(diagrams.size() / DIAGRAM_FIELDS_PER_PAIR,
                                             "multiscale diagram pair count");
        if (num_pairs <= 0 || num_pairs > max_pairs_)
        {
            throw std::length_error("diagram pair count exceeds configured capacity");
        }
        requireFiniteDiagramValues(diagrams, "multiscale diagram buffer");
        return num_pairs;
    }

    void cleanup() noexcept
    {
        for (auto *adj : adjacency_per_scale_)
        {
            if (adj)
                cudaFree(adj);
        }
        adjacency_per_scale_.clear();
        if (d_combined_)
            cudaFree(d_combined_);
        d_combined_ = nullptr;
    }

    int max_pairs_ = 0;
    int feature_dim_ = 0;
    std::vector<float> scales_;
    std::vector<float *> adjacency_per_scale_;
    float *d_combined_ = nullptr;
    size_t pair_matrix_count_ = 0;
    size_t feature_count_capacity_ = 0;
};
