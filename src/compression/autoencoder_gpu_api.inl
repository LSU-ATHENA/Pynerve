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
        throw std::length_error(std::string(label) + " byte count overflows");
    }
    return bytes;
}

std::size_t checkedElementCount(int lhs, int rhs, const char *label)
{
    if (lhs <= 0 || rhs <= 0)
    {
        throw std::invalid_argument(std::string(label) + " dimensions must be positive");
    }
    std::size_t values = 0;
    if (!checkedProduct(static_cast<std::size_t>(lhs), static_cast<std::size_t>(rhs), values))
    {
        throw std::length_error(std::string(label) + " element count overflows");
    }
    if (values > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::length_error(std::string(label) + " exceeds CUDA kernel int range");
    }
    return values;
}

int checkedIntElements(std::size_t values, const char *label)
{
    if (values > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::length_error(std::string(label) + " exceeds CUDA kernel int range");
    }
    return static_cast<int>(values);
}

int checkedGridBlocks(std::size_t values, const char *label)
{
    const std::size_t blocks = (values / static_cast<std::size_t>(BLOCK_SIZE)) +
                               ((values % static_cast<std::size_t>(BLOCK_SIZE)) != 0 ? 1U : 0U);
    return checkedIntElements(blocks, label);
}

void checkCublasStatus(cublasStatus_t status, const char *label)
{
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        throw std::runtime_error(std::string(label) + " failed");
    }
}

bool valuesAreFinite(const std::vector<float> &values)
{
    return std::all_of(values.begin(), values.end(),
                       [](float value) { return std::isfinite(value); });
}

void requireFiniteInput(const std::vector<float> &values, const char *label)
{
    if (!valuesAreFinite(values))
    {
        throw std::invalid_argument(label);
    }
}

void requireFiniteOutput(const std::vector<float> &values, const char *label)
{
    if (!valuesAreFinite(values))
    {
        throw std::runtime_error(label);
    }
}

template <typename T>
void allocateDevice(T **ptr, std::size_t count, const char *label)
{
    GPU_CHECK(cudaMalloc(ptr, checkedBytes<T>(count, label)));
}

template <typename T>
void copyToDevice(T *dst, const std::vector<T> &src, const char *label)
{
    GPU_CHECK(
        cudaMemcpy(dst, src.data(), checkedBytes<T>(src.size(), label), cudaMemcpyHostToDevice));
}

template <typename T>
void copyFromDevice(std::vector<T> &dst, const T *src, const char *label)
{
    GPU_CHECK(
        cudaMemcpy(dst.data(), src, checkedBytes<T>(dst.size(), label), cudaMemcpyDeviceToHost));
}

void validateDeviceFinite(const float *src, std::size_t count, const char *label)
{
    std::vector<float> values(count);
    copyFromDevice(values, src, label);
    requireFiniteOutput(values, label);
}

} // namespace

class BenchmarkAutoencoder
{
public:
    struct Config
    {
        std::vector<int> encoder_layers;
        std::vector<int> decoder_layers;
        int batch_size = 32;
    };

    explicit BenchmarkAutoencoder(const Config &config)
        : config_(config)
        , cublas_handle_(nullptr)
    {
        validateConfig();
        try
        {
            checkCublasStatus(cublasCreate(&cublas_handle_), "cublasCreate");
            allocateWeights();
            randomInitialize();
        }
        catch (...)
        {
            releaseDeviceState();
            throw;
        }
    }

    ~BenchmarkAutoencoder() { releaseDeviceState(); }

    std::vector<float> encode(const std::vector<float> &input_batch)
    {
        const std::size_t expected_values = checkedElementCount(
            config_.batch_size, config_.encoder_layers.front(), "encoder input");
        if (input_batch.size() != expected_values)
        {
            throw std::invalid_argument("encoder input must match batch_size * input_dim");
        }
        requireFiniteInput(input_batch, "encoder input must be finite");
        const int batch_size = config_.batch_size;

        float *d_input = nullptr;
        float *current = nullptr;
        float *d_output = nullptr;
        try
        {
            allocateDevice(&d_input, input_batch.size(), "encoder input");
            copyToDevice(d_input, input_batch, "encoder input");
            current = d_input;

            for (size_t i = 0; i < d_encoder_weights_.size(); ++i)
            {
                int in_dim = config_.encoder_layers[i];
                int out_dim = config_.encoder_layers[i + 1];
                const std::size_t output_values =
                    checkedElementCount(batch_size, out_dim, "encoder layer output");
                allocateDevice(&d_output, output_values, "encoder layer output");

                float alpha = 1.0f, beta = 0.0f;
                checkCublasStatus(cublasSgemm(cublas_handle_, CUBLAS_OP_T, CUBLAS_OP_N, out_dim,
                                              batch_size, in_dim, &alpha, d_encoder_weights_[i],
                                              in_dim, current, in_dim, &beta, d_output, out_dim),
                                  "encoder cublasSgemm");
                validateDeviceFinite(d_output, output_values,
                                     "encoder affine layer produced non-finite values");

                if (d_encoder_bias_[i] != nullptr)
                {
                    const int n = checkedIntElements(output_values, "encoder bias kernel size");
                    const int blocks = checkedGridBlocks(output_values, "encoder bias grid");
                    addBiasKernel<float>
                        <<<blocks, BLOCK_SIZE>>>(d_output, d_encoder_bias_[i], batch_size, out_dim);
                    GPU_CHECK(cudaPeekAtLastError());
                    validateDeviceFinite(d_output, output_values,
                                         "encoder bias layer produced non-finite values");
                }

                if (i < d_encoder_weights_.size() - 1)
                {
                    const int n = checkedIntElements(output_values, "encoder relu kernel size");
                    const int blocks = checkedGridBlocks(output_values, "encoder relu grid");
                    reluKernel<float><<<blocks, BLOCK_SIZE>>>(d_output, n);
                    GPU_CHECK(cudaPeekAtLastError());
                    validateDeviceFinite(d_output, output_values,
                                         "encoder activation produced non-finite values");
                }

                if (current != d_input)
                {
                    cudaFree(current);
                }
                current = d_output;
                d_output = nullptr;
            }

            const std::size_t latent_values =
                checkedElementCount(batch_size, config_.encoder_layers.back(), "encoder latent");
            std::vector<float> latent(latent_values);
            copyFromDevice(latent, current, "encoder latent");
            requireFiniteOutput(latent, "encoder produced non-finite latent values");
            cudaFree(current);
            current = nullptr;
            cudaFree(d_input);
            d_input = nullptr;
            return latent;
        }
        catch (...)
        {
            if (current != nullptr && current != d_input)
            {
                cudaFree(current);
            }
            cudaFree(d_output);
            cudaFree(d_input);
            throw;
        }
    }

    std::vector<float> decode(const std::vector<float> &latent_batch)
    {
        const std::size_t expected_values = checkedElementCount(
            config_.batch_size, config_.decoder_layers.front(), "decoder input");
        if (latent_batch.size() != expected_values)
        {
            throw std::invalid_argument("decoder input must match batch_size * latent_dim");
        }
        requireFiniteInput(latent_batch, "decoder input must be finite");
        const int batch_size = config_.batch_size;

        float *d_latent = nullptr;
        float *current = nullptr;
        float *d_output = nullptr;
        try
        {
            allocateDevice(&d_latent, latent_batch.size(), "decoder input");
            copyToDevice(d_latent, latent_batch, "decoder input");
            current = d_latent;

            for (size_t i = 0; i < d_decoder_weights_.size(); ++i)
            {
                int in_dim = config_.decoder_layers[i];
                int out_dim = config_.decoder_layers[i + 1];
                const std::size_t output_values =
                    checkedElementCount(batch_size, out_dim, "decoder layer output");
                allocateDevice(&d_output, output_values, "decoder layer output");

                float alpha = 1.0f, beta = 0.0f;
                checkCublasStatus(cublasSgemm(cublas_handle_, CUBLAS_OP_T, CUBLAS_OP_N, out_dim,
                                              batch_size, in_dim, &alpha, d_decoder_weights_[i],
                                              in_dim, current, in_dim, &beta, d_output, out_dim),
                                  "decoder cublasSgemm");
                validateDeviceFinite(d_output, output_values,
                                     "decoder affine layer produced non-finite values");

                const int n = checkedIntElements(output_values, "decoder activation kernel size");
                const int blocks = checkedGridBlocks(output_values, "decoder activation grid");
                if (i == d_decoder_weights_.size() - 1)
                {
                    sigmoidKernel<float><<<blocks, BLOCK_SIZE>>>(d_output, n);
                }
                else
                {
                    reluKernel<float><<<blocks, BLOCK_SIZE>>>(d_output, n);
                }
                GPU_CHECK(cudaPeekAtLastError());
                validateDeviceFinite(d_output, output_values,
                                     "decoder activation produced non-finite values");

                if (current != d_latent)
                {
                    cudaFree(current);
                }
                current = d_output;
                d_output = nullptr;
            }

            const std::size_t output_values =
                checkedElementCount(batch_size, config_.decoder_layers.back(), "decoder output");
            std::vector<float> output(output_values);
            copyFromDevice(output, current, "decoder output");
            requireFiniteOutput(output, "decoder produced non-finite output values");
            cudaFree(current);
            current = nullptr;
            cudaFree(d_latent);
            d_latent = nullptr;
            return output;
        }
        catch (...)
        {
            if (current != nullptr && current != d_latent)
            {
                cudaFree(current);
            }
            cudaFree(d_output);
            cudaFree(d_latent);
            throw;
        }
    }

private:
    void validateConfig() const
    {
        if (config_.batch_size <= 0 || config_.encoder_layers.size() < 2 ||
            config_.decoder_layers.size() < 2)
        {
            throw std::invalid_argument(
                "autoencoder benchmark requires positive batch and layer sizes");
        }
        for (int dim : config_.encoder_layers)
        {
            if (dim <= 0)
            {
                throw std::invalid_argument("encoder layer dimensions must be positive");
            }
        }
        for (int dim : config_.decoder_layers)
        {
            if (dim <= 0)
            {
                throw std::invalid_argument("decoder layer dimensions must be positive");
            }
        }
        if (config_.encoder_layers.back() != config_.decoder_layers.front())
        {
            throw std::invalid_argument(
                "encoder output dimension must match decoder input dimension");
        }
    }

    void allocateWeights()
    {
        for (size_t i = 0; i + 1 < config_.encoder_layers.size(); ++i)
        {
            int rows = config_.encoder_layers[i];
            int cols = config_.encoder_layers[i + 1];
            const std::size_t weight_values = checkedElementCount(rows, cols, "encoder weights");

            float *d_w = nullptr;
            allocateDevice(&d_w, weight_values, "encoder weights");
            d_encoder_weights_.push_back(d_w);

            float *d_b = nullptr;
            allocateDevice(&d_b, static_cast<std::size_t>(cols), "encoder bias");
            GPU_CHECK(cudaMemset(
                d_b, 0, checkedBytes<float>(static_cast<std::size_t>(cols), "encoder bias")));
            d_encoder_bias_.push_back(d_b);
        }

        for (size_t i = 0; i + 1 < config_.decoder_layers.size(); ++i)
        {
            int rows = config_.decoder_layers[i];
            int cols = config_.decoder_layers[i + 1];
            const std::size_t weight_values = checkedElementCount(rows, cols, "decoder weights");

            float *d_w = nullptr;
            allocateDevice(&d_w, weight_values, "decoder weights");
            d_decoder_weights_.push_back(d_w);

            float *d_b = nullptr;
            allocateDevice(&d_b, static_cast<std::size_t>(cols), "decoder bias");
            GPU_CHECK(cudaMemset(
                d_b, 0, checkedBytes<float>(static_cast<std::size_t>(cols), "decoder bias")));
            d_decoder_bias_.push_back(d_b);
        }
    }

    void randomInitialize()
    {
        std::mt19937 gen(42);
        std::normal_distribution<float> dist(0.0f, 0.1f);

        for (size_t i = 0; i < d_encoder_weights_.size(); ++i)
        {
            int rows = config_.encoder_layers[i];
            int cols = config_.encoder_layers[i + 1];
            const std::size_t weight_values = checkedElementCount(rows, cols, "encoder weights");
            std::vector<float> weights(weight_values);
            for (auto &w : weights)
                w = dist(gen);

            copyToDevice(d_encoder_weights_[i], weights, "encoder weights");
        }

        for (size_t i = 0; i < d_decoder_weights_.size(); ++i)
        {
            int rows = config_.decoder_layers[i];
            int cols = config_.decoder_layers[i + 1];
            const std::size_t weight_values = checkedElementCount(rows, cols, "decoder weights");
            std::vector<float> weights(weight_values);
            for (auto &w : weights)
                w = dist(gen);

            copyToDevice(d_decoder_weights_[i], weights, "decoder weights");
        }
    }

    void releaseDeviceState() noexcept
    {
        if (cublas_handle_ != nullptr)
        {
            cublasDestroy(cublas_handle_);
            cublas_handle_ = nullptr;
        }
        for (auto *ptr : d_encoder_weights_)
            cudaFree(ptr);
        for (auto *ptr : d_decoder_weights_)
            cudaFree(ptr);
        for (auto *ptr : d_encoder_bias_)
            cudaFree(ptr);
        for (auto *ptr : d_decoder_bias_)
            cudaFree(ptr);
        d_encoder_weights_.clear();
        d_decoder_weights_.clear();
        d_encoder_bias_.clear();
        d_decoder_bias_.clear();
    }

    Config config_;
    cublasHandle_t cublas_handle_;

    std::vector<float *> d_encoder_weights_;
    std::vector<float *> d_decoder_weights_;
    std::vector<float *> d_encoder_bias_;
    std::vector<float *> d_decoder_bias_;
};
