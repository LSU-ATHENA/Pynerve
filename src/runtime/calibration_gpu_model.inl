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
std::size_t checkedDeviceBytes(std::size_t count, const char *label)
{
    std::size_t bytes = 0;
    if (!checkedProduct(count, sizeof(T), bytes))
    {
        throw std::length_error(std::string(label) + " byte count overflows");
    }
    return bytes;
}

int checkedIntSize(std::size_t value, const char *label)
{
    if (value > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::length_error(std::string(label) + " exceeds int range");
    }
    return static_cast<int>(value);
}

int checkedGridBlocks(int count, const char *label)
{
    if (count < 0)
    {
        throw std::invalid_argument(std::string(label) + " must be non-negative");
    }
    const std::size_t blocks =
        (static_cast<std::size_t>(count) + static_cast<std::size_t>(BLOCK_SIZE) - 1U) /
        static_cast<std::size_t>(BLOCK_SIZE);
    return checkedIntSize(blocks, label);
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

bool valuesAreFinite(const std::vector<double> &values)
{
    return std::all_of(values.begin(), values.end(),
                       [](double value) { return std::isfinite(value); });
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

void requireFiniteOutput(const std::vector<double> &values, const char *label)
{
    if (!valuesAreFinite(values))
    {
        throw std::runtime_error(label);
    }
}

template <typename T>
void allocateDeviceArray(T **ptr, std::size_t count, const char *label)
{
    GPU_CHECK(cudaMalloc(ptr, checkedDeviceBytes<T>(count, label)));
}

template <typename T>
void copyHostToDevice(T *dst, const std::vector<T> &src, const char *label)
{
    GPU_CHECK(cudaMemcpy(dst, src.data(), checkedDeviceBytes<T>(src.size(), label),
                         cudaMemcpyHostToDevice));
}

template <typename T>
void copyDeviceToHost(std::vector<T> &dst, const T *src, const char *label)
{
    GPU_CHECK(cudaMemcpy(dst.data(), src, checkedDeviceBytes<T>(dst.size(), label),
                         cudaMemcpyDeviceToHost));
}

void freeCalibrationScratch(double *d_pred, double *d_obs, int *d_bids) noexcept
{
    cudaFree(d_pred);
    cudaFree(d_obs);
    cudaFree(d_bids);
}

void freeLinearScratch(float *d_features, float *d_targets, float *d_weights,
                       float *d_bias) noexcept
{
    cudaFree(d_features);
    cudaFree(d_targets);
    cudaFree(d_weights);
    cudaFree(d_bias);
}

} // namespace

class GPUCalibrationModel
{
public:
    struct Sample
    {
        std::string hardware_fingerprint;
        std::string problem_bucket;
        std::string algorithm;
        double predicted_time_ms;
        double observed_time_ms;
        double confidence;
        double timestamp;
    };

    struct BucketStats
    {
        double predicted_time_avg = 0.0;
        double observed_time_avg = 0.0;
        double error_avg = 0.0;
        double confidence = 0.0;
        double error_bound = 0.0;
        int sample_count = 0;
    };

    explicit GPUCalibrationModel(int max_buckets = 1000)
        : max_buckets_(max_buckets)
        , cublas_handle_(nullptr)
        , d_bucket_predicted_sum_(nullptr)
        , d_bucket_observed_sum_(nullptr)
        , d_bucket_error_sum_(nullptr)
        , d_bucket_counts_(nullptr)
        , d_confidence_(nullptr)
        , d_error_bounds_(nullptr)
    {
        if (max_buckets_ <= 0)
        {
            throw std::invalid_argument("max_buckets must be positive");
        }

        try
        {
            checkCublasStatus(cublasCreate(&cublas_handle_), "cublasCreate");
            allocateDeviceArray(&d_bucket_predicted_sum_, static_cast<std::size_t>(max_buckets_),
                                "bucket predicted sum");
            allocateDeviceArray(&d_bucket_observed_sum_, static_cast<std::size_t>(max_buckets_),
                                "bucket observed sum");
            allocateDeviceArray(&d_bucket_error_sum_, static_cast<std::size_t>(max_buckets_),
                                "bucket error sum");
            allocateDeviceArray(&d_bucket_counts_, static_cast<std::size_t>(max_buckets_),
                                "bucket counts");
            allocateDeviceArray(&d_confidence_, static_cast<std::size_t>(max_buckets_),
                                "bucket confidence");
            allocateDeviceArray(&d_error_bounds_, static_cast<std::size_t>(max_buckets_),
                                "bucket error bounds");
            resetBuckets();
        }
        catch (...)
        {
            releaseDeviceState();
            throw;
        }
    }

    ~GPUCalibrationModel() { releaseDeviceState(); }

    void addSamples(const std::vector<Sample> &samples)
    {
        if (samples.empty())
            return;

        const int n = checkedIntSize(samples.size(), "calibration sample count");
        std::vector<double> predicted;
        std::vector<double> observed;
        std::vector<int> bucket_ids;
        predicted.reserve(samples.size());
        observed.reserve(samples.size());
        bucket_ids.reserve(samples.size());

        for (const auto &s : samples)
        {
            if (!std::isfinite(s.predicted_time_ms) || s.predicted_time_ms < 0.0 ||
                !std::isfinite(s.observed_time_ms) || s.observed_time_ms < 0.0 ||
                !std::isfinite(s.confidence) || s.confidence < 0.0 || s.confidence > 1.0 ||
                !std::isfinite(s.timestamp))
            {
                throw std::invalid_argument("calibration samples contain invalid numeric values");
            }
            int bucket = getBucketId(s.hardware_fingerprint, s.problem_bucket, s.algorithm);

            predicted.push_back(s.predicted_time_ms);
            observed.push_back(s.observed_time_ms);
            bucket_ids.push_back(bucket);
        }

        double *d_pred = nullptr;
        double *d_obs = nullptr;
        int *d_bids = nullptr;

        try
        {
            allocateDeviceArray(&d_pred, predicted.size(), "calibration predicted samples");
            allocateDeviceArray(&d_obs, observed.size(), "calibration observed samples");
            allocateDeviceArray(&d_bids, bucket_ids.size(), "calibration bucket ids");

            copyHostToDevice(d_pred, predicted, "calibration predicted samples");
            copyHostToDevice(d_obs, observed, "calibration observed samples");
            copyHostToDevice(d_bids, bucket_ids, "calibration bucket ids");

            int blocks = checkedGridBlocks(n, "calibration sample grid");
            aggregateBucketsKernel<<<blocks, BLOCK_SIZE>>>(
                d_pred, d_obs, d_bids, d_bucket_predicted_sum_, d_bucket_observed_sum_,
                d_bucket_error_sum_, d_bucket_counts_, n, max_buckets_);
            GPU_CHECK(cudaPeekAtLastError());

            int conf_blocks = checkedGridBlocks(max_buckets_, "calibration confidence grid");
            computeConfidenceKernel<<<conf_blocks, BLOCK_SIZE>>>(
                d_bucket_error_sum_, d_bucket_counts_, d_confidence_, d_error_bounds_, max_buckets_,
                20, 0.8);
            GPU_CHECK(cudaPeekAtLastError());
            GPU_CHECK(cudaDeviceSynchronize());

            std::vector<double> predicted_sums(static_cast<std::size_t>(max_buckets_));
            std::vector<double> observed_sums(static_cast<std::size_t>(max_buckets_));
            std::vector<double> error_sums(static_cast<std::size_t>(max_buckets_));
            std::vector<double> confidence(static_cast<std::size_t>(max_buckets_));
            std::vector<double> error_bounds(static_cast<std::size_t>(max_buckets_));
            copyDeviceToHost(predicted_sums, d_bucket_predicted_sum_,
                             "copy calibration predicted sums");
            copyDeviceToHost(observed_sums, d_bucket_observed_sum_,
                             "copy calibration observed sums");
            copyDeviceToHost(error_sums, d_bucket_error_sum_, "copy calibration error sums");
            copyDeviceToHost(confidence, d_confidence_, "copy calibration confidence");
            copyDeviceToHost(error_bounds, d_error_bounds_, "copy calibration error bounds");
            requireFiniteOutput(predicted_sums, "calibration predicted sums are non-finite");
            requireFiniteOutput(observed_sums, "calibration observed sums are non-finite");
            requireFiniteOutput(error_sums, "calibration error sums are non-finite");
            requireFiniteOutput(confidence, "calibration confidence is non-finite");
            requireFiniteOutput(error_bounds, "calibration error bounds are non-finite");
        }
        catch (...)
        {
            freeCalibrationScratch(d_pred, d_obs, d_bids);
            throw;
        }

        freeCalibrationScratch(d_pred, d_obs, d_bids);
    }

    BucketStats queryBucket(const std::string &hardware, const std::string &problem,
                            const std::string &algorithm)
    {
        int bucket = getBucketId(hardware, problem, algorithm);
        if (bucket < 0 || bucket >= max_buckets_)
        {
            return {};
        }

        BucketStats stats;
        int count = 0;

        GPU_CHECK(
            cudaMemcpy(&count, &d_bucket_counts_[bucket], sizeof(int), cudaMemcpyDeviceToHost));
        GPU_CHECK(cudaMemcpy(&stats.predicted_time_avg, &d_bucket_predicted_sum_[bucket],
                             sizeof(double), cudaMemcpyDeviceToHost));
        GPU_CHECK(cudaMemcpy(&stats.observed_time_avg, &d_bucket_observed_sum_[bucket],
                             sizeof(double), cudaMemcpyDeviceToHost));
        GPU_CHECK(cudaMemcpy(&stats.error_avg, &d_bucket_error_sum_[bucket], sizeof(double),
                             cudaMemcpyDeviceToHost));
        GPU_CHECK(cudaMemcpy(&stats.confidence, &d_confidence_[bucket], sizeof(double),
                             cudaMemcpyDeviceToHost));
        GPU_CHECK(cudaMemcpy(&stats.error_bound, &d_error_bounds_[bucket], sizeof(double),
                             cudaMemcpyDeviceToHost));

        stats.sample_count = count;
        if (count > 0)
        {
            stats.predicted_time_avg /= count;
            stats.observed_time_avg /= count;
            stats.error_avg /= count;
        }
        if (!std::isfinite(stats.predicted_time_avg) || !std::isfinite(stats.observed_time_avg) ||
            !std::isfinite(stats.error_avg) || !std::isfinite(stats.confidence) ||
            !std::isfinite(stats.error_bound))
        {
            throw std::runtime_error("calibration bucket contains non-finite values");
        }

        return stats;
    }

    void fitLinearModel(const std::vector<std::vector<float>> &features,
                        const std::vector<float> &targets, int num_epochs = 100,
                        float learning_rate = 0.01f)
    {
        if (features.empty() || targets.empty())
            return;
        if (features.size() != targets.size())
        {
            throw std::invalid_argument("feature rows and targets must have matching lengths");
        }
        if (num_epochs <= 0 || learning_rate <= 0.0f || !std::isfinite(learning_rate))
        {
            throw std::invalid_argument("training parameters must be positive and finite");
        }

        const int num_samples = checkedIntSize(features.size(), "linear model sample count");
        const int num_features = checkedIntSize(features[0].size(), "linear model feature count");
        if (num_features == 0)
        {
            throw std::invalid_argument("feature vectors must not be empty");
        }
        for (const auto &row : features)
        {
            if (row.size() != static_cast<std::size_t>(num_features))
            {
                throw std::invalid_argument("feature rows must have a consistent width");
            }
            requireFiniteInput(row, "linear model features must be finite");
        }
        requireFiniteInput(targets, "linear model targets must be finite");

        std::size_t flat_count = 0;
        if (!checkedProduct(features.size(), static_cast<std::size_t>(num_features), flat_count))
        {
            throw std::length_error("linear model feature matrix size overflows");
        }
        std::vector<float> flat_features;
        flat_features.reserve(flat_count);
        for (const auto &f : features)
        {
            flat_features.insert(flat_features.end(), f.begin(), f.end());
        }

        float *d_features = nullptr;
        float *d_targets = nullptr;
        float *d_weights = nullptr;
        float *d_bias = nullptr;

        std::vector<float> h_weights(static_cast<std::size_t>(num_features), 0.0f);
        try
        {
            allocateDeviceArray(&d_features, flat_features.size(), "linear model features");
            allocateDeviceArray(&d_targets, targets.size(), "linear model targets");
            allocateDeviceArray(&d_weights, h_weights.size(), "linear model weights");
            allocateDeviceArray(&d_bias, 1, "linear model bias");

            copyHostToDevice(d_features, flat_features, "linear model features");
            copyHostToDevice(d_targets, targets, "linear model targets");
            copyHostToDevice(d_weights, h_weights, "linear model weights");
            GPU_CHECK(cudaMemset(d_bias, 0, sizeof(float)));

            const int blocks = checkedGridBlocks(num_features, "linear model feature grid");
            for (int epoch = 0; epoch < num_epochs; ++epoch)
            {
                modelUpdateKernel<<<blocks, BLOCK_SIZE>>>(d_features, d_targets, d_weights, d_bias,
                                                          num_samples, num_features, learning_rate,
                                                          0.001f);
                GPU_CHECK(cudaPeekAtLastError());
            }

            GPU_CHECK(
                cudaMemcpy(h_weights.data(), d_weights,
                           checkedDeviceBytes<float>(h_weights.size(), "linear model weights"),
                           cudaMemcpyDeviceToHost));
            float h_bias = 0.0f;
            GPU_CHECK(cudaMemcpy(&h_bias, d_bias, sizeof(float), cudaMemcpyDeviceToHost));
            requireFiniteOutput(h_weights, "linear model weights are non-finite");
            if (!std::isfinite(h_bias))
            {
                throw std::runtime_error("linear model bias is non-finite");
            }
        }
        catch (...)
        {
            freeLinearScratch(d_features, d_targets, d_weights, d_bias);
            throw;
        }

        freeLinearScratch(d_features, d_targets, d_weights, d_bias);
    }

    void resetBuckets()
    {
        const auto bucket_count = static_cast<std::size_t>(max_buckets_);
        GPU_CHECK(cudaMemset(d_bucket_predicted_sum_, 0,
                             checkedDeviceBytes<double>(bucket_count, "bucket predicted sum")));
        GPU_CHECK(cudaMemset(d_bucket_observed_sum_, 0,
                             checkedDeviceBytes<double>(bucket_count, "bucket observed sum")));
        GPU_CHECK(cudaMemset(d_bucket_error_sum_, 0,
                             checkedDeviceBytes<double>(bucket_count, "bucket error sum")));
        GPU_CHECK(cudaMemset(d_bucket_counts_, 0,
                             checkedDeviceBytes<int>(bucket_count, "bucket counts")));
        GPU_CHECK(cudaMemset(d_confidence_, 0,
                             checkedDeviceBytes<double>(bucket_count, "bucket confidence")));
        GPU_CHECK(cudaMemset(d_error_bounds_, 0,
                             checkedDeviceBytes<double>(bucket_count, "bucket error bounds")));
    }

private:
    int getBucketId(const std::string &hw, const std::string &prob, const std::string &alg)
    {
        std::hash<std::string> hasher;
        size_t h = hasher(hw) ^ hasher(prob) ^ hasher(alg);
        return static_cast<int>(h % static_cast<std::size_t>(max_buckets_));
    }

    void releaseDeviceState() noexcept
    {
        if (cublas_handle_ != nullptr)
        {
            cublasDestroy(cublas_handle_);
            cublas_handle_ = nullptr;
        }
        cudaFree(d_bucket_predicted_sum_);
        cudaFree(d_bucket_observed_sum_);
        cudaFree(d_bucket_error_sum_);
        cudaFree(d_bucket_counts_);
        cudaFree(d_confidence_);
        cudaFree(d_error_bounds_);
        d_bucket_predicted_sum_ = nullptr;
        d_bucket_observed_sum_ = nullptr;
        d_bucket_error_sum_ = nullptr;
        d_bucket_counts_ = nullptr;
        d_confidence_ = nullptr;
        d_error_bounds_ = nullptr;
    }

    int max_buckets_;
    cublasHandle_t cublas_handle_;

    double *d_bucket_predicted_sum_;
    double *d_bucket_observed_sum_;
    double *d_bucket_error_sum_;
    int *d_bucket_counts_;
    double *d_confidence_;
    double *d_error_bounds_;
};
