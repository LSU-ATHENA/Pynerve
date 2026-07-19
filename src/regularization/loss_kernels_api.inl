namespace
{

bool checkedProduct(std::size_t lhs, std::size_t rhs, std::size_t &out) noexcept
{
    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs)
    {
        return false;
    }
    out = lhs * rhs;
    return true;
}

template <typename T>
std::size_t checkedBytes(std::size_t count, const char *label)
{
    std::size_t bytes = 0;
    if (!checkedProduct(count, sizeof(T), bytes))
    {
        throw std::length_error(label);
    }
    return bytes;
}

int checkedIntSize(std::size_t value, const char *label)
{
    if (value > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::length_error(label);
    }
    return static_cast<int>(value);
}

std::size_t checkedPairValues(std::size_t pair_count, const char *label)
{
    std::size_t values = 0;
    if (!checkedProduct(pair_count, 2U, values))
    {
        throw std::length_error(label);
    }
    checkedIntSize(pair_count, label);
    return values;
}

template <typename T>
void allocateDevice(T **ptr, std::size_t count, const char *label)
{
    if (count == 0)
    {
        *ptr = nullptr;
        return;
    }
    GPU_CHECK(cudaMalloc(ptr, checkedBytes<T>(count, label)));
}

template <typename T>
void copyToDevice(T *dst, const std::vector<T> &src, const char *label)
{
    if (src.empty())
    {
        return;
    }
    GPU_CHECK(
        cudaMemcpy(dst, src.data(), checkedBytes<T>(src.size(), label), cudaMemcpyHostToDevice));
}

void requireFiniteValues(const std::vector<float> &values, const char *label)
{
    for (float value : values)
    {
        if (!std::isfinite(value))
        {
            throw std::invalid_argument(std::string(label) + " must be finite");
        }
    }
}

void requireNonnegativeFiniteValues(const std::vector<float> &values, const char *label)
{
    for (float value : values)
    {
        if (!std::isfinite(value) || value < 0.0f)
        {
            throw std::invalid_argument(std::string(label) + " must be finite and non-negative");
        }
    }
}

void requireFiniteDiagram(const std::vector<std::pair<float, float>> &diagram, const char *label)
{
    for (const auto &pair : diagram)
    {
        if (!std::isfinite(pair.first) || !std::isfinite(pair.second) || pair.second < pair.first)
        {
            throw std::invalid_argument(std::string(label) +
                                        " pairs must be finite with death >= birth");
        }
    }
}

float checkedFloatProduct(float lhs, float rhs, const char *label)
{
    const double product = static_cast<double>(lhs) * static_cast<double>(rhs);
    if (!std::isfinite(product) ||
        std::abs(product) > static_cast<double>(std::numeric_limits<float>::max()))
    {
        throw std::overflow_error(std::string(label) + " is not finite");
    }
    return static_cast<float>(product);
}

float readDeviceLoss(float *d_loss_buffer)
{
    float loss = 0.0f;
    GPU_CHECK(cudaMemcpy(&loss, d_loss_buffer, sizeof(float), cudaMemcpyDeviceToHost));
    if (!std::isfinite(loss))
    {
        throw std::runtime_error("GPU topology loss produced non-finite output");
    }
    return loss;
}

void resetDeviceLoss(float *d_loss_buffer)
{
    GPU_CHECK(cudaMemset(d_loss_buffer, 0, sizeof(float)));
}

} // namespace

class GPUTopologyLoss
{
public:
    struct LossConfig
    {
        float lambda_betti = 0.1f;
        float lambda_persistence = 0.05f;
        float lambda_entropy = 0.01f;
        float lambda_smoothness = 0.1f;
        int p_norm = 2;
    };

    explicit GPUTopologyLoss(const LossConfig &config)
        : config_(config)
        , d_loss_buffer_(nullptr)
    {
        if (config_.p_norm <= 0 || !std::isfinite(config_.lambda_betti) ||
            !std::isfinite(config_.lambda_persistence) || !std::isfinite(config_.lambda_entropy) ||
            !std::isfinite(config_.lambda_smoothness) || config_.lambda_betti < 0.0f ||
            config_.lambda_persistence < 0.0f || config_.lambda_entropy < 0.0f ||
            config_.lambda_smoothness < 0.0f)
        {
            throw std::invalid_argument("topology loss config must be finite and non-negative");
        }
        allocateDevice(&d_loss_buffer_, 1, "loss buffer");
    }

    ~GPUTopologyLoss() { cudaFree(d_loss_buffer_); }

    float bettiMSELoss(const std::vector<float> &target, const std::vector<float> &prediction)
    {
        if (target.size() != prediction.size())
        {
            throw std::invalid_argument("target and prediction must have matching lengths");
        }
        if (target.empty())
        {
            return 0.0f;
        }
        requireFiniteValues(target, "Betti target");
        requireFiniteValues(prediction, "Betti prediction");
        const int n = checkedIntSize(target.size(), "betti target size exceeds CUDA limits");

        float *d_target = nullptr;
        float *d_pred = nullptr;
        try
        {
            allocateDevice(&d_target, target.size(), "betti target");
            allocateDevice(&d_pred, prediction.size(), "betti prediction");
            copyToDevice(d_target, target, "betti target");
            copyToDevice(d_pred, prediction, "betti prediction");
            resetDeviceLoss(d_loss_buffer_);

            bettiCurveMSELossKernel<<<1, BLOCK_SIZE>>>(d_target, d_pred, d_loss_buffer_, n);
            GPU_CHECK(cudaPeekAtLastError());
            const float loss = readDeviceLoss(d_loss_buffer_);
            cudaFree(d_target);
            cudaFree(d_pred);
            return checkedFloatProduct(loss, config_.lambda_betti, "Betti loss");
        }
        catch (...)
        {
            cudaFree(d_target);
            cudaFree(d_pred);
            throw;
        }
    }

    float wassersteinLoss(const std::vector<std::pair<float, float>> &target_diagram,
                          const std::vector<std::pair<float, float>> &pred_diagram)
    {
        requireFiniteDiagram(target_diagram, "target diagram");
        requireFiniteDiagram(pred_diagram, "predicted diagram");
        if (target_diagram.empty() || pred_diagram.empty())
        {
            float unmatched = 0.0f;
            for (const auto &pair : target_diagram)
            {
                const float next = unmatched + std::fabs(pair.second - pair.first);
                if (!std::isfinite(next))
                {
                    throw std::overflow_error("unmatched target diagram loss is not finite");
                }
                unmatched = next;
            }
            for (const auto &pair : pred_diagram)
            {
                const float next = unmatched + std::fabs(pair.second - pair.first);
                if (!std::isfinite(next))
                {
                    throw std::overflow_error("unmatched predicted diagram loss is not finite");
                }
                unmatched = next;
            }
            return unmatched;
        }
        const int n_target =
            checkedIntSize(target_diagram.size(), "target diagram size exceeds CUDA limits");
        const int n_pred =
            checkedIntSize(pred_diagram.size(), "predicted diagram size exceeds CUDA limits");
        const std::size_t target_values =
            checkedPairValues(target_diagram.size(), "target diagram values overflow");
        const std::size_t pred_values =
            checkedPairValues(pred_diagram.size(), "predicted diagram values overflow");

        std::vector<float> flat_target(target_values);
        std::vector<float> flat_pred(pred_values);

        for (int i = 0; i < n_target; ++i)
        {
            flat_target[static_cast<std::size_t>(i) * 2U] =
                target_diagram[static_cast<std::size_t>(i)].first;
            flat_target[static_cast<std::size_t>(i) * 2U + 1U] =
                target_diagram[static_cast<std::size_t>(i)].second;
        }

        for (int i = 0; i < n_pred; ++i)
        {
            flat_pred[static_cast<std::size_t>(i) * 2U] =
                pred_diagram[static_cast<std::size_t>(i)].first;
            flat_pred[static_cast<std::size_t>(i) * 2U + 1U] =
                pred_diagram[static_cast<std::size_t>(i)].second;
        }

        float *d_target = nullptr;
        float *d_pred = nullptr;
        try
        {
            allocateDevice(&d_target, flat_target.size(), "target diagram");
            allocateDevice(&d_pred, flat_pred.size(), "predicted diagram");
            copyToDevice(d_target, flat_target, "target diagram");
            copyToDevice(d_pred, flat_pred, "predicted diagram");
            resetDeviceLoss(d_loss_buffer_);

            wassersteinLossKernel<<<n_pred, BLOCK_SIZE>>>(d_target, d_pred, d_loss_buffer_,
                                                          n_target, n_pred, config_.p_norm);
            GPU_CHECK(cudaPeekAtLastError());
            const float loss = readDeviceLoss(d_loss_buffer_);
            cudaFree(d_target);
            cudaFree(d_pred);
            return loss;
        }
        catch (...)
        {
            cudaFree(d_target);
            cudaFree(d_pred);
            throw;
        }
    }

    float persistenceEntropy(const std::vector<float> &persistence_values)
    {
        if (persistence_values.empty())
        {
            return 0.0f;
        }
        requireNonnegativeFiniteValues(persistence_values, "persistence values");
        const int n = checkedIntSize(persistence_values.size(),
                                     "persistence value count exceeds CUDA limits");

        float *d_persistence = nullptr;
        try
        {
            allocateDevice(&d_persistence, persistence_values.size(), "persistence values");
            copyToDevice(d_persistence, persistence_values, "persistence values");
            resetDeviceLoss(d_loss_buffer_);

            const int threads = BLOCK_SIZE;
            const size_t smem_size = checkedBytes<float>(static_cast<std::size_t>(threads),
                                                         "persistence entropy shared memory");
            persistenceEntropyKernel<<<1, threads, smem_size>>>(d_persistence, n, d_loss_buffer_);
            GPU_CHECK(cudaPeekAtLastError());
            const float entropy = readDeviceLoss(d_loss_buffer_);
            cudaFree(d_persistence);
            return checkedFloatProduct(entropy, config_.lambda_entropy, "persistence entropy loss");
        }
        catch (...)
        {
            cudaFree(d_persistence);
            throw;
        }
    }

    float smoothnessLoss(const std::vector<float> &features, const std::vector<int> &adjacency,
                         int num_nodes, int feature_dim)
    {
        if (num_nodes <= 0 || feature_dim <= 0)
        {
            return 0.0f;
        }
        if (feature_dim > BLOCK_SIZE)
        {
            throw std::invalid_argument("feature_dim exceeds CUDA block size");
        }
        std::size_t feature_values = 0;
        std::size_t adjacency_values = 0;
        if (!checkedProduct(static_cast<std::size_t>(num_nodes),
                            static_cast<std::size_t>(feature_dim), feature_values) ||
            !checkedProduct(static_cast<std::size_t>(num_nodes),
                            static_cast<std::size_t>(num_nodes), adjacency_values))
        {
            throw std::length_error("smoothness input size overflows");
        }
        checkedIntSize(feature_values, "smoothness feature count exceeds CUDA limits");
        if (features.size() != feature_values || adjacency.size() != adjacency_values)
        {
            throw std::invalid_argument("smoothness inputs have inconsistent shapes");
        }
        requireFiniteValues(features, "smoothness features");

        float *d_features = nullptr;
        int *d_adjacency = nullptr;
        try
        {
            allocateDevice(&d_features, features.size(), "smoothness features");
            allocateDevice(&d_adjacency, adjacency.size(), "smoothness adjacency");
            copyToDevice(d_features, features, "smoothness features");
            copyToDevice(d_adjacency, adjacency, "smoothness adjacency");
            resetDeviceLoss(d_loss_buffer_);

            topologySmoothnessKernel<<<num_nodes, feature_dim>>>(d_features, d_adjacency, num_nodes,
                                                                 feature_dim, d_loss_buffer_);
            GPU_CHECK(cudaPeekAtLastError());
            const float loss = readDeviceLoss(d_loss_buffer_);
            cudaFree(d_features);
            cudaFree(d_adjacency);
            return checkedFloatProduct(loss, config_.lambda_smoothness, "smoothness loss");
        }
        catch (...)
        {
            cudaFree(d_features);
            cudaFree(d_adjacency);
            throw;
        }
    }

    float combinedLoss(const std::vector<float> &betti_target, const std::vector<float> &betti_pred,
                       const std::vector<float> &persistence_values)
    {
        if (betti_target.size() != betti_pred.size())
        {
            throw std::invalid_argument("Betti target and prediction must have matching lengths");
        }
        if (betti_target.empty() && persistence_values.empty())
        {
            return 0.0f;
        }
        requireFiniteValues(betti_target, "Betti target");
        requireFiniteValues(betti_pred, "Betti prediction");
        requireNonnegativeFiniteValues(persistence_values, "persistence values");
        const int num_dims =
            checkedIntSize(betti_target.size(), "Betti dimension count exceeds CUDA limits");
        const int num_persistence = checkedIntSize(persistence_values.size(),
                                                   "persistence value count exceeds CUDA limits");

        float *d_betti_target = nullptr;
        float *d_betti_pred = nullptr;
        float *d_persistence = nullptr;
        try
        {
            allocateDevice(&d_betti_target, betti_target.size(), "Betti target");
            allocateDevice(&d_betti_pred, betti_pred.size(), "Betti prediction");
            allocateDevice(&d_persistence, persistence_values.size(), "persistence values");
            copyToDevice(d_betti_target, betti_target, "Betti target");
            copyToDevice(d_betti_pred, betti_pred, "Betti prediction");
            copyToDevice(d_persistence, persistence_values, "persistence values");
            resetDeviceLoss(d_loss_buffer_);

            combinedTopologyLossKernel<<<1, BLOCK_SIZE, sizeof(float)>>>(
                d_betti_target, d_betti_pred, d_persistence, num_dims, num_persistence,
                config_.lambda_betti, config_.lambda_persistence, d_loss_buffer_);
            GPU_CHECK(cudaPeekAtLastError());
            const float loss = readDeviceLoss(d_loss_buffer_);
            cudaFree(d_betti_target);
            cudaFree(d_betti_pred);
            cudaFree(d_persistence);
            return loss;
        }
        catch (...)
        {
            cudaFree(d_betti_target);
            cudaFree(d_betti_pred);
            cudaFree(d_persistence);
            throw;
        }
    }

private:
    LossConfig config_;
    float *d_loss_buffer_;
};
