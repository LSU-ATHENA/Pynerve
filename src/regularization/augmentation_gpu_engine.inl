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

int checkedGridBlocks(std::size_t values, const char *label)
{
    const std::size_t blocks = (values / static_cast<std::size_t>(BLOCK_SIZE)) +
                               ((values % static_cast<std::size_t>(BLOCK_SIZE)) != 0 ? 1U : 0U);
    return checkedIntSize(blocks, label);
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

void freeTwo(float *a, float *b) noexcept
{
    cudaFree(a);
    cudaFree(b);
}

void freeThree(float *a, float *b, float *c) noexcept
{
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
}

bool valuesAreFinite(const std::vector<float> &values)
{
    return std::all_of(values.begin(), values.end(),
                       [](float value) { return std::isfinite(value); });
}

bool valuesAreFiniteAndNonNegative(const std::vector<float> &values)
{
    return std::all_of(values.begin(), values.end(),
                       [](float value) { return std::isfinite(value) && value >= 0.0f; });
}

void requireFiniteInput(const std::vector<float> &values, const char *label)
{
    if (!valuesAreFinite(values))
    {
        throw std::invalid_argument(label);
    }
}

void requireFiniteNonNegativeInput(const std::vector<float> &values, const char *label)
{
    if (!valuesAreFiniteAndNonNegative(values))
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

} // namespace

class GPUDataAugmentation
{
public:
    enum class AugmentationType
    {
        GAUSSIAN_NOISE,
        UNIFORM_NOISE,
        GEOMETRIC_TRANSFORM,
        PERSISTENCE_AWARE_NOISE,
        MIXUP,
        CUTOUT
    };

    struct AugmentationConfig
    {
        std::vector<AugmentationType> augmentations;
        float gaussian_std = 0.1f;
        float uniform_range = 0.1f;
        float rotation_range = 0.2f; // radians
        float scale_range = 0.1f;
        float persistence_scale = 0.5f;
        float mixup_alpha = 0.4f;
        int cutout_size = 10;
        int num_augmentations = 1;
        unsigned int deterministic_seed = 42U;
    };

    explicit GPUDataAugmentation(const AugmentationConfig &config)
        : config_(config)
    {
        validateConfig();
    }

    ~GPUDataAugmentation() = default;

    std::vector<float> augment(const std::vector<float> &data, int num_samples, int feature_dim)
    {
        if (num_samples <= 0 || feature_dim <= 0)
        {
            return {};
        }
        std::size_t total_values = 0;
        if (!checkedProduct(static_cast<std::size_t>(num_samples),
                            static_cast<std::size_t>(feature_dim), total_values))
        {
            throw std::length_error("augmentation element count overflows");
        }
        checkedIntSize(total_values, "augmentation element count exceeds CUDA limits");
        if (data.size() != total_values)
        {
            throw std::invalid_argument("augmentation data shape does not match dimensions");
        }
        requireFiniteInput(data, "augmentation data must be finite");
        if (config_.augmentations.empty())
        {
            return data;
        }
        const int total_elements = static_cast<int>(total_values);

        float *d_input = nullptr;
        float *d_output = nullptr;
        try
        {
            allocateDevice(&d_input, total_values, "augmentation input");
            allocateDevice(&d_output, total_values, "augmentation output");
            copyToDevice(d_input, data, "augmentation input");

            unsigned int seed = config_.deterministic_seed;
            const int blocks = checkedGridBlocks(total_values, "augmentation element grid");
            const int sample_blocks = checkedGridBlocks(static_cast<std::size_t>(num_samples),
                                                        "augmentation sample grid");

            for (auto &aug_type : config_.augmentations)
            {
                switch (aug_type)
                {
                    case AugmentationType::GAUSSIAN_NOISE:
                        gaussianNoiseKernel<<<blocks, BLOCK_SIZE>>>(
                            d_input, d_output, total_elements, 0.0f, config_.gaussian_std, seed++);
                        GPU_CHECK(cudaPeekAtLastError());
                        break;

                    case AugmentationType::UNIFORM_NOISE:
                        uniformNoiseKernel<<<blocks, BLOCK_SIZE>>>(
                            d_input, d_output, total_elements, -config_.uniform_range,
                            config_.uniform_range, seed++);
                        GPU_CHECK(cudaPeekAtLastError());
                        break;

                    case AugmentationType::GEOMETRIC_TRANSFORM:
                        geometricTransformKernel<<<blocks, BLOCK_SIZE>>>(
                            d_input, d_output, num_samples, feature_dim, config_.rotation_range,
                            1.0f + config_.scale_range);
                        GPU_CHECK(cudaPeekAtLastError());
                        break;

                    case AugmentationType::CUTOUT:
                        cutoutKernel<<<sample_blocks, BLOCK_SIZE>>>(
                            d_input, d_output, num_samples, feature_dim, config_.cutout_size, 0.0f,
                            seed++);
                        GPU_CHECK(cudaPeekAtLastError());
                        break;

                    default:
                        GPU_CHECK(cudaMemcpy(d_output, d_input,
                                             checkedBytes<float>(total_values, "augmentation copy"),
                                             cudaMemcpyDeviceToDevice));
                        break;
                }

                std::swap(d_input, d_output);
            }

            std::vector<float> augmented(total_values);
            copyFromDevice(augmented, d_input, "augmentation output");
            requireFiniteOutput(augmented, "augmentation produced non-finite output");
            freeTwo(d_input, d_output);
            return augmented;
        }
        catch (...)
        {
            freeTwo(d_input, d_output);
            throw;
        }
    }

    std::vector<float> augmentWithPersistence(const std::vector<float> &data,
                                              const std::vector<float> &persistence_values)
    {
        if (data.size() != persistence_values.size())
        {
            throw std::invalid_argument("persistence values must match data length");
        }
        if (data.empty())
        {
            return {};
        }
        requireFiniteInput(data, "persistence augmentation data must be finite");
        requireFiniteNonNegativeInput(persistence_values,
                                      "persistence values must be finite and non-negative");
        const int n =
            checkedIntSize(data.size(), "persistence augmentation size exceeds CUDA limits");

        float *d_input = nullptr;
        float *d_persistence = nullptr;
        float *d_output = nullptr;
        try
        {
            allocateDevice(&d_input, data.size(), "persistence augmentation input");
            allocateDevice(&d_persistence, persistence_values.size(), "persistence values");
            allocateDevice(&d_output, data.size(), "persistence augmentation output");
            copyToDevice(d_input, data, "persistence augmentation input");
            copyToDevice(d_persistence, persistence_values, "persistence values");

            unsigned int seed = config_.deterministic_seed;
            const int blocks = checkedGridBlocks(data.size(), "persistence augmentation grid");

            persistenceAwareNoiseKernel<<<blocks, BLOCK_SIZE>>>(d_input, d_persistence, d_output, n,
                                                                config_.gaussian_std,
                                                                config_.persistence_scale, seed);
            GPU_CHECK(cudaPeekAtLastError());

            std::vector<float> augmented(data.size());
            copyFromDevice(augmented, d_output, "persistence augmentation output");
            requireFiniteOutput(augmented, "persistence augmentation produced non-finite output");
            freeThree(d_input, d_persistence, d_output);
            return augmented;
        }
        catch (...)
        {
            freeThree(d_input, d_persistence, d_output);
            throw;
        }
    }

    std::vector<float> mixup(const std::vector<float> &data1, const std::vector<float> &data2)
    {
        if (data1.size() != data2.size())
        {
            throw std::invalid_argument("mixup inputs must have matching lengths");
        }
        if (data1.empty())
        {
            return {};
        }
        requireFiniteInput(data1, "mixup input 1 must be finite");
        requireFiniteInput(data2, "mixup input 2 must be finite");
        const int n = checkedIntSize(data1.size(), "mixup size exceeds CUDA limits");

        float *d_in1 = nullptr;
        float *d_in2 = nullptr;
        float *d_out = nullptr;
        try
        {
            allocateDevice(&d_in1, data1.size(), "mixup input 1");
            allocateDevice(&d_in2, data2.size(), "mixup input 2");
            allocateDevice(&d_out, data1.size(), "mixup output");
            copyToDevice(d_in1, data1, "mixup input 1");
            copyToDevice(d_in2, data2, "mixup input 2");

            unsigned int seed = config_.deterministic_seed;
            const int blocks = checkedGridBlocks(data1.size(), "mixup grid");

            mixupKernel<<<blocks, BLOCK_SIZE>>>(d_in1, d_in2, d_out, n, config_.mixup_alpha, seed);
            GPU_CHECK(cudaPeekAtLastError());

            std::vector<float> mixed(data1.size());
            copyFromDevice(mixed, d_out, "mixup output");
            requireFiniteOutput(mixed, "mixup produced non-finite output");
            freeThree(d_in1, d_in2, d_out);
            return mixed;
        }
        catch (...)
        {
            freeThree(d_in1, d_in2, d_out);
            throw;
        }
    }

private:
    void validateConfig() const
    {
        if (!std::isfinite(config_.gaussian_std) || config_.gaussian_std < 0.0f ||
            !std::isfinite(config_.uniform_range) || config_.uniform_range < 0.0f ||
            !std::isfinite(config_.rotation_range) || !std::isfinite(config_.scale_range) ||
            !std::isfinite(1.0f + config_.scale_range) ||
            !std::isfinite(config_.persistence_scale) || config_.persistence_scale < 0.0f ||
            !std::isfinite(config_.mixup_alpha) || config_.mixup_alpha < 0.0f ||
            config_.cutout_size < 0)
        {
            throw std::invalid_argument("augmentation configuration contains invalid values");
        }
    }

    AugmentationConfig config_;
};
