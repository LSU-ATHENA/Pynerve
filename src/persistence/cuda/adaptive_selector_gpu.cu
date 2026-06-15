#include "nerve/persistence/cuda/gpu_adaptive_selector.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace nerve
{
namespace persistence
{
namespace gpu
{
namespace
{

constexpr int BLOCK_SIZE = 256;
constexpr float NORMALIZATION_EPSILON = 1e-6f;

const char *cublasStatusName(cublasStatus_t status) noexcept
{
    switch (status)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";
#ifdef CUBLAS_STATUS_LICENSE_ERROR
        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR";
#endif
        default:
            return "CUBLAS_STATUS_UNKNOWN";
    }
}

void checkCuda(cudaError_t status, const char *context)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(status));
    }
}

void checkCublas(cublasStatus_t status, const char *context)
{
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        throw std::runtime_error(std::string(context) + ": " + cublasStatusName(status));
    }
}

std::size_t checkedMul(std::size_t a, std::size_t b, const char *label)
{
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a)
    {
        throw std::length_error(std::string(label) + " exceeds size_t limits");
    }
    return a * b;
}

std::size_t checkedElements(int a, int b, const char *label)
{
    if (a <= 0 || b <= 0)
    {
        throw std::invalid_argument(std::string(label) + " dimensions must be positive");
    }
    return checkedMul(static_cast<std::size_t>(a), static_cast<std::size_t>(b), label);
}

std::size_t checkedBytes(std::size_t count, std::size_t element_size, const char *label)
{
    return checkedMul(count, element_size, label);
}

std::size_t checkedFloatBytes(std::size_t count, const char *label)
{
    return checkedBytes(count, sizeof(float), label);
}

std::size_t checkedIntBytes(std::size_t count, const char *label)
{
    return checkedBytes(count, sizeof(int), label);
}

int checkedIntSize(std::size_t value, const char *label)
{
    if (value > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::length_error(std::string(label) + " exceeds CUDA int range");
    }
    return static_cast<int>(value);
}

int checkedGridBlocks(std::size_t item_count, const char *label)
{
    if (item_count == 0)
    {
        return 0;
    }
    const std::size_t blocks = (item_count + static_cast<std::size_t>(BLOCK_SIZE) - 1) /
                               static_cast<std::size_t>(BLOCK_SIZE);
    if (blocks > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::length_error(std::string(label) + " exceeds CUDA grid range");
    }
    return static_cast<int>(blocks);
}

void allocateFloat(float **ptr, std::size_t count, const char *label)
{
    checkCuda(cudaMalloc(reinterpret_cast<void **>(ptr), checkedFloatBytes(count, label)), label);
}

void allocateInt(int **ptr, std::size_t count, const char *label)
{
    checkCuda(cudaMalloc(reinterpret_cast<void **>(ptr), checkedIntBytes(count, label)), label);
}

} // namespace

__device__ float atomicMaxFloat(float *address, float value)
{
    int *address_as_int = reinterpret_cast<int *>(address);
    int old = *address_as_int;
    while (value > __int_as_float(old))
    {
        const int assumed = old;
        old = atomicCAS(address_as_int, assumed, __float_as_int(value));
        if (old == assumed)
            break;
    }
    return __int_as_float(old);
}

__global__ __launch_bounds__(BLOCK_SIZE) void extractFeaturesKernel(
    const float *__restrict__ persistence_pairs, const int *__restrict__ pair_dims, int num_pairs,
    float *__restrict__ features, int max_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pairs)
        return;

    float birth = persistence_pairs[idx * 2];
    float death = persistence_pairs[idx * 2 + 1];
    float persistence = death - birth;
    int dim = pair_dims[idx];

    if (dim <= max_dim && dim >= 0)
    {
        int feat_offset = dim * 4;
        atomicAdd(&features[feat_offset], persistence);
        atomicAdd(&features[feat_offset + 1], 1.0f);
        atomicAdd(&features[feat_offset + 2], persistence * persistence);
        atomicMaxFloat(&features[feat_offset + 3], persistence);
    }
}

__global__ __launch_bounds__(BLOCK_SIZE) void normalizeFeaturesKernel(
    float *__restrict__ features, int num_features, const float *__restrict__ mean,
    const float *__restrict__ std)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_features)
        return;

    if (std[idx] > NORMALIZATION_EPSILON)
    {
        features[idx] = (features[idx] - mean[idx]) / std[idx];
    }
}

__global__ __launch_bounds__(BLOCK_SIZE) void addBiasReluKernel(float *__restrict__ values,
                                                                const float *__restrict__ bias,
                                                                int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count)
        return;
    values[idx] = fmaxf(0.0f, values[idx] + bias[idx]);
}

__global__ __launch_bounds__(BLOCK_SIZE) void addBiasKernel(float *__restrict__ values,
                                                            const float *__restrict__ bias,
                                                            int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count)
        return;
    values[idx] += bias[idx];
}

__global__ __launch_bounds__(1) void softmaxKernel(float *__restrict__ scores, int num_classes)
{
    float max_score = -INFINITY;
    for (int i = 0; i < num_classes; ++i)
    {
        max_score = fmaxf(max_score, scores[i]);
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < num_classes; ++i)
    {
        scores[i] = expf(scores[i] - max_score);
        sum_exp += scores[i];
    }

    for (int i = 0; i < num_classes; ++i)
    {
        scores[i] /= sum_exp;
    }
}

class GPUAdaptiveSelector::Impl
{
public:
    explicit Impl(const SelectorConfig &config)
        : config_(config)
    {
        validateConfig();
        try
        {
            checkCublas(cublasCreate(&cublas_handle_), "create adaptive selector cuBLAS handle");

            allocateFloat(&d_w1_,
                          checkedElements(config_.feature_dim, config_.hidden_dim,
                                          "adaptive selector w1 count"),
                          "allocate adaptive selector w1");
            allocateFloat(&d_w2_,
                          checkedElements(config_.hidden_dim, config_.num_algorithms,
                                          "adaptive selector w2 count"),
                          "allocate adaptive selector w2");
            allocateFloat(&d_b1_, static_cast<std::size_t>(config_.hidden_dim),
                          "allocate adaptive selector b1");
            allocateFloat(&d_b2_, static_cast<std::size_t>(config_.num_algorithms),
                          "allocate adaptive selector b2");
            allocateFloat(&d_feature_mean_, static_cast<std::size_t>(config_.feature_dim),
                          "allocate adaptive selector feature mean");
            allocateFloat(&d_feature_std_, static_cast<std::size_t>(config_.feature_dim),
                          "allocate adaptive selector feature std");
            allocateFloat(&d_hidden_, static_cast<std::size_t>(config_.hidden_dim),
                          "allocate adaptive selector hidden");
            allocateFloat(&d_scores_, static_cast<std::size_t>(config_.num_algorithms),
                          "allocate adaptive selector scores");
            allocateFloat(&d_features_, static_cast<std::size_t>(config_.feature_dim),
                          "allocate adaptive selector features");

            checkCuda(cudaMemset(d_features_, 0,
                                 checkedFloatBytes(static_cast<std::size_t>(config_.feature_dim),
                                                   "adaptive selector feature bytes")),
                      "initialize adaptive selector features");
            initializeWeights();
        }
        catch (...)
        {
            cleanup();
            throw;
        }
    }

    ~Impl() { cleanup(); }

    void registerAlgorithm(const AlgorithmInfo &info) { algorithms_.push_back(info); }

    void
    extractFeaturesFromPersistence(const std::vector<std::pair<float, float>> &persistence_pairs,
                                   const std::vector<int> &dimensions)
    {
        checkCuda(cudaMemset(d_features_, 0,
                             checkedFloatBytes(static_cast<std::size_t>(config_.feature_dim),
                                               "adaptive selector feature reset bytes")),
                  "reset adaptive selector features");
        if (persistence_pairs.empty())
            return;
        if (dimensions.size() != persistence_pairs.size())
        {
            throw std::invalid_argument("dimension count must match persistence pair count");
        }

        const int num_pairs =
            checkedIntSize(persistence_pairs.size(), "adaptive selector pair count");
        std::vector<float> flat_pairs;
        flat_pairs.reserve(checkedMul(persistence_pairs.size(), static_cast<std::size_t>(2),
                                      "adaptive selector flat pair count"));
        for (const auto &p : persistence_pairs)
        {
            if (!std::isfinite(p.first) || !std::isfinite(p.second))
            {
                throw std::invalid_argument("persistence pair values must be finite");
            }
            flat_pairs.push_back(p.first);
            flat_pairs.push_back(p.second);
        }

        float *d_pairs = nullptr;
        int *d_dims = nullptr;
        try
        {
            allocateFloat(&d_pairs, flat_pairs.size(), "allocate adaptive selector pairs");
            allocateInt(&d_dims, persistence_pairs.size(), "allocate adaptive selector dims");

            checkCuda(cudaMemcpy(
                          d_pairs, flat_pairs.data(),
                          checkedFloatBytes(flat_pairs.size(), "adaptive selector pair copy bytes"),
                          cudaMemcpyHostToDevice),
                      "copy adaptive selector pairs");
            checkCuda(
                cudaMemcpy(d_dims, dimensions.data(),
                           checkedIntBytes(dimensions.size(), "adaptive selector dim copy bytes"),
                           cudaMemcpyHostToDevice),
                "copy adaptive selector dimensions");

            const int blocks =
                checkedGridBlocks(persistence_pairs.size(), "adaptive selector feature grid");
            const int max_dim = (config_.feature_dim / 4) - 1;
            extractFeaturesKernel<<<blocks, BLOCK_SIZE>>>(d_pairs, d_dims, num_pairs, d_features_,
                                                          max_dim);
            checkCuda(cudaPeekAtLastError(), "launch adaptive selector feature extraction");

            const int feature_blocks =
                checkedGridBlocks(static_cast<std::size_t>(config_.feature_dim),
                                  "adaptive selector normalization grid");
            normalizeFeaturesKernel<<<feature_blocks, BLOCK_SIZE>>>(
                d_features_, config_.feature_dim, d_feature_mean_, d_feature_std_);
            checkCuda(cudaPeekAtLastError(), "launch adaptive selector feature normalization");
            checkCuda(cudaDeviceSynchronize(), "synchronize adaptive selector feature extraction");

            cudaFree(d_pairs);
            cudaFree(d_dims);
        }
        catch (...)
        {
            if (d_pairs)
                cudaFree(d_pairs);
            if (d_dims)
                cudaFree(d_dims);
            throw;
        }
    }

    Prediction predict()
    {
        float alpha = 1.0f, beta = 0.0f;

        checkCublas(cublasSgemm(cublas_handle_, CUBLAS_OP_T, CUBLAS_OP_N, config_.hidden_dim, 1,
                                config_.feature_dim, &alpha, d_w1_, config_.feature_dim,
                                d_features_, config_.feature_dim, &beta, d_hidden_,
                                config_.hidden_dim),
                    "run adaptive selector hidden GEMM");

        const int hidden_blocks = checkedGridBlocks(static_cast<std::size_t>(config_.hidden_dim),
                                                    "adaptive selector hidden grid");
        addBiasReluKernel<<<hidden_blocks, BLOCK_SIZE>>>(d_hidden_, d_b1_, config_.hidden_dim);
        checkCuda(cudaPeekAtLastError(), "launch adaptive selector hidden activation");

        checkCublas(cublasSgemm(cublas_handle_, CUBLAS_OP_T, CUBLAS_OP_N, config_.num_algorithms, 1,
                                config_.hidden_dim, &alpha, d_w2_, config_.hidden_dim, d_hidden_,
                                config_.hidden_dim, &beta, d_scores_, config_.num_algorithms),
                    "run adaptive selector score GEMM");

        const int score_blocks = checkedGridBlocks(static_cast<std::size_t>(config_.num_algorithms),
                                                   "adaptive selector score grid");
        addBiasKernel<<<score_blocks, BLOCK_SIZE>>>(d_scores_, d_b2_, config_.num_algorithms);
        checkCuda(cudaPeekAtLastError(), "launch adaptive selector score bias");

        softmaxKernel<<<1, 1>>>(d_scores_, config_.num_algorithms);
        checkCuda(cudaPeekAtLastError(), "launch adaptive selector softmax");

        std::vector<float> scores(static_cast<std::size_t>(config_.num_algorithms));
        checkCuda(cudaMemcpy(scores.data(), d_scores_,
                             checkedFloatBytes(scores.size(), "adaptive selector score copy bytes"),
                             cudaMemcpyDeviceToHost),
                  "copy adaptive selector scores");

        int best_idx = 0;
        float best_score = scores[0];
        for (std::size_t i = 1; i < scores.size(); ++i)
        {
            if (scores[i] > best_score)
            {
                best_score = scores[i];
                best_idx = static_cast<int>(i);
            }
        }

        Prediction pred;
        pred.selected_algorithm = best_idx;
        pred.confidence = best_score;
        pred.algorithm_scores = scores;
        return pred;
    }

    std::vector<Prediction> batchPredict(const std::vector<std::vector<float>> &feature_batches)
    {
        std::vector<Prediction> predictions;
        predictions.reserve(feature_batches.size());

        for (const auto &features : feature_batches)
        {
            if (features.size() != static_cast<std::size_t>(config_.feature_dim))
            {
                throw std::invalid_argument("feature batch width must match selector feature_dim");
            }
            checkCuda(cudaMemcpy(d_features_, features.data(),
                                 checkedFloatBytes(features.size(),
                                                   "adaptive selector batch feature copy bytes"),
                                 cudaMemcpyHostToDevice),
                      "copy adaptive selector batch features");
            predictions.push_back(predict());
        }

        return predictions;
    }

    void trainStep(const std::vector<float> &features, int correct_algorithm, float learning_rate)
    {
        if (features.size() != static_cast<std::size_t>(config_.feature_dim))
        {
            throw std::invalid_argument("feature width must match selector feature_dim");
        }
        if (correct_algorithm < 0 || correct_algorithm >= config_.num_algorithms)
        {
            throw std::out_of_range("correct algorithm index is outside selector output range");
        }
        if (!std::isfinite(learning_rate) || learning_rate <= 0.0f)
        {
            throw std::invalid_argument("learning rate must be finite and positive");
        }

        checkCuda(cudaMemcpy(d_features_, features.data(),
                             checkedFloatBytes(features.size(),
                                               "adaptive selector train feature copy bytes"),
                             cudaMemcpyHostToDevice),
                  "copy adaptive selector train features");
        Prediction pred = predict();

        std::vector<float> grad_logits = pred.algorithm_scores;
        grad_logits[static_cast<std::size_t>(correct_algorithm)] -= 1.0f;

        std::vector<float> hidden(static_cast<std::size_t>(config_.hidden_dim));
        std::vector<float> w2(checkedElements(config_.hidden_dim, config_.num_algorithms,
                                              "adaptive selector train w2 count"));
        std::vector<float> b2(static_cast<std::size_t>(config_.num_algorithms));
        checkCuda(
            cudaMemcpy(hidden.data(), d_hidden_,
                       checkedFloatBytes(hidden.size(), "adaptive selector hidden copy bytes"),
                       cudaMemcpyDeviceToHost),
            "copy adaptive selector hidden activations");
        checkCuda(cudaMemcpy(w2.data(), d_w2_,
                             checkedFloatBytes(w2.size(), "adaptive selector w2 copy bytes"),
                             cudaMemcpyDeviceToHost),
                  "copy adaptive selector w2");
        checkCuda(cudaMemcpy(b2.data(), d_b2_,
                             checkedFloatBytes(b2.size(), "adaptive selector b2 copy bytes"),
                             cudaMemcpyDeviceToHost),
                  "copy adaptive selector b2");

        for (int algo = 0; algo < config_.num_algorithms; ++algo)
        {
            const float grad = grad_logits[static_cast<std::size_t>(algo)];
            b2[static_cast<std::size_t>(algo)] -= learning_rate * grad;
            float *row = &w2[static_cast<std::size_t>(algo) * config_.hidden_dim];
            for (int h = 0; h < config_.hidden_dim; ++h)
            {
                row[h] -= learning_rate * grad * hidden[static_cast<std::size_t>(h)];
            }
        }

        checkCuda(cudaMemcpy(d_w2_, w2.data(),
                             checkedFloatBytes(w2.size(), "adaptive selector w2 update bytes"),
                             cudaMemcpyHostToDevice),
                  "update adaptive selector w2");
        checkCuda(cudaMemcpy(d_b2_, b2.data(),
                             checkedFloatBytes(b2.size(), "adaptive selector b2 update bytes"),
                             cudaMemcpyHostToDevice),
                  "update adaptive selector b2");
    }

private:
    void validateConfig() const
    {
        if (config_.num_algorithms <= 0 || config_.feature_dim <= 0 || config_.hidden_dim <= 0)
        {
            throw std::invalid_argument("selector dimensions must be positive");
        }
        if (config_.feature_dim < 4 || config_.feature_dim % 4 != 0)
        {
            throw std::invalid_argument("selector feature_dim must be a positive multiple of four");
        }
    }

    void initializeWeights()
    {
        std::mt19937 gen(42);
        std::normal_distribution<float> dist(0.0f, 0.1f);

        std::vector<float> w1(checkedElements(config_.feature_dim, config_.hidden_dim,
                                              "adaptive selector initial w1 count"));
        for (auto &w : w1)
            w = dist(gen);
        checkCuda(cudaMemcpy(d_w1_, w1.data(),
                             checkedFloatBytes(w1.size(), "adaptive selector initial w1 bytes"),
                             cudaMemcpyHostToDevice),
                  "initialize adaptive selector w1");

        std::vector<float> w2(checkedElements(config_.hidden_dim, config_.num_algorithms,
                                              "adaptive selector initial w2 count"));
        for (auto &w : w2)
            w = dist(gen);
        checkCuda(cudaMemcpy(d_w2_, w2.data(),
                             checkedFloatBytes(w2.size(), "adaptive selector initial w2 bytes"),
                             cudaMemcpyHostToDevice),
                  "initialize adaptive selector w2");

        checkCuda(cudaMemset(d_b1_, 0,
                             checkedFloatBytes(static_cast<std::size_t>(config_.hidden_dim),
                                               "adaptive selector b1 bytes")),
                  "initialize adaptive selector b1");
        checkCuda(cudaMemset(d_b2_, 0,
                             checkedFloatBytes(static_cast<std::size_t>(config_.num_algorithms),
                                               "adaptive selector b2 bytes")),
                  "initialize adaptive selector b2");

        std::vector<float> feature_mean(static_cast<std::size_t>(config_.feature_dim), 0.0f);
        std::vector<float> feature_std(static_cast<std::size_t>(config_.feature_dim), 1.0f);
        checkCuda(cudaMemcpy(d_feature_mean_, feature_mean.data(),
                             checkedFloatBytes(feature_mean.size(),
                                               "adaptive selector feature mean bytes"),
                             cudaMemcpyHostToDevice),
                  "initialize adaptive selector feature mean");
        checkCuda(
            cudaMemcpy(d_feature_std_, feature_std.data(),
                       checkedFloatBytes(feature_std.size(), "adaptive selector feature std bytes"),
                       cudaMemcpyHostToDevice),
            "initialize adaptive selector feature std");
    }

    void cleanup() noexcept
    {
        if (cublas_handle_)
            cublasDestroy(cublas_handle_);
        if (d_w1_)
            cudaFree(d_w1_);
        if (d_w2_)
            cudaFree(d_w2_);
        if (d_b1_)
            cudaFree(d_b1_);
        if (d_b2_)
            cudaFree(d_b2_);
        if (d_feature_mean_)
            cudaFree(d_feature_mean_);
        if (d_feature_std_)
            cudaFree(d_feature_std_);
        if (d_hidden_)
            cudaFree(d_hidden_);
        if (d_scores_)
            cudaFree(d_scores_);
        if (d_features_)
            cudaFree(d_features_);
        cublas_handle_ = nullptr;
        d_w1_ = nullptr;
        d_w2_ = nullptr;
        d_b1_ = nullptr;
        d_b2_ = nullptr;
        d_feature_mean_ = nullptr;
        d_feature_std_ = nullptr;
        d_hidden_ = nullptr;
        d_scores_ = nullptr;
        d_features_ = nullptr;
    }

    SelectorConfig config_;
    cublasHandle_t cublas_handle_ = nullptr;
    std::vector<AlgorithmInfo> algorithms_;

    float *d_w1_ = nullptr;
    float *d_w2_ = nullptr;
    float *d_b1_ = nullptr;
    float *d_b2_ = nullptr;
    float *d_feature_mean_ = nullptr;
    float *d_feature_std_ = nullptr;
    float *d_hidden_ = nullptr;
    float *d_scores_ = nullptr;
    float *d_features_ = nullptr;
};

GPUAdaptiveSelector::GPUAdaptiveSelector(const SelectorConfig &config)
    : impl_(std::make_unique<Impl>(config))
{}

GPUAdaptiveSelector::~GPUAdaptiveSelector() = default;

void GPUAdaptiveSelector::registerAlgorithm(const AlgorithmInfo &info)
{
    impl_->registerAlgorithm(info);
}

void GPUAdaptiveSelector::extractFeaturesFromPersistence(
    const std::vector<std::pair<float, float>> &persistence_pairs,
    const std::vector<int> &dimensions)
{
    impl_->extractFeaturesFromPersistence(persistence_pairs, dimensions);
}

Prediction GPUAdaptiveSelector::predict()
{
    return impl_->predict();
}

std::vector<Prediction>
GPUAdaptiveSelector::batchPredict(const std::vector<std::vector<float>> &feature_batches)
{
    return impl_->batchPredict(feature_batches);
}

void GPUAdaptiveSelector::trainStep(const std::vector<float> &features, int correct_algorithm,
                                    float learning_rate)
{
    impl_->trainStep(features, correct_algorithm, learning_rate);
}

// Benchmark support is split out to keep this translation unit within
// repository file-size caps without changing public behavior.
#include "detail/adaptive_selector_benchmark.inl"

} // namespace gpu
} // namespace persistence
} // namespace nerve
