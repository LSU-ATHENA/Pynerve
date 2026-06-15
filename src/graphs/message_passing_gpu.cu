#include "nerve/graphs/gpu_gnn.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace nerve
{
namespace graphs
{
namespace gpu
{

constexpr int BLOCK_SIZE = 256;

void throwCudaLaunchError(cudaError_t status, const char *operation)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
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

int checkedIntSize(std::size_t value, const char *label)
{
    if (value > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::length_error(std::string(label) + " exceeds CUDA int range");
    }
    return static_cast<int>(value);
}

int checkedGridBlocks(std::size_t count, const char *label)
{
    if (count == 0)
    {
        return 0;
    }
    const std::size_t blocks =
        (count + static_cast<std::size_t>(BLOCK_SIZE) - 1) / static_cast<std::size_t>(BLOCK_SIZE);
    if (blocks > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::length_error(std::string(label) + " exceeds CUDA grid range");
    }
    return static_cast<int>(blocks);
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
    *ptr = nullptr;
    if (count == 0)
    {
        return;
    }
    throwCudaLaunchError(
        cudaMalloc(reinterpret_cast<void **>(ptr), checkedMul(count, sizeof(T), label)), label);
}

template <typename T>
void copyToDevice(T *dst, const T *src, std::size_t count, const char *label)
{
    if (count == 0)
    {
        return;
    }
    throwCudaLaunchError(
        cudaMemcpy(dst, src, checkedMul(count, sizeof(T), label), cudaMemcpyHostToDevice), label);
}

template <typename T>
void copyToHost(T *dst, const T *src, std::size_t count, const char *label)
{
    if (count == 0)
    {
        return;
    }
    throwCudaLaunchError(
        cudaMemcpy(dst, src, checkedMul(count, sizeof(T), label), cudaMemcpyDeviceToHost), label);
}

void validateCSR(const std::vector<int> &row_ptr, const std::vector<int> &col_idx, int num_nodes)
{
    if (row_ptr.size() != static_cast<std::size_t>(num_nodes) + 1)
    {
        throw std::invalid_argument("row_ptr must have num_nodes + 1 entries");
    }
    if (col_idx.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::length_error("edge count exceeds CUDA int range");
    }
    if (row_ptr.empty() || row_ptr.front() != 0 ||
        row_ptr.back() != static_cast<int>(col_idx.size()))
    {
        throw std::invalid_argument("row_ptr must start at zero and end at edge count");
    }
    for (std::size_t i = 1; i < row_ptr.size(); ++i)
    {
        if (row_ptr[i - 1] > row_ptr[i] || row_ptr[i] < 0)
        {
            throw std::invalid_argument("row_ptr must be monotonic and non-negative");
        }
    }
    for (int col : col_idx)
    {
        if (col < 0 || col >= num_nodes)
        {
            throw std::out_of_range("col_idx contains a node outside graph bounds");
        }
    }
}

template <typename T>
__global__ void __launch_bounds__(256)
    messagePassingKernel(const T *__restrict__ node_features,
                         const int *__restrict__ adjacency_row_ptr,
                         const int *__restrict__ adjacency_col_idx,
                         const T *__restrict__ edge_features, T *__restrict__ messages,
                         int num_nodes, int feature_dim, bool aggregate_mean)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = num_nodes * feature_dim;
    if (idx >= total)
    {
        return;
    }

    const int node_idx = idx / feature_dim;
    const int feature_idx = idx % feature_dim;
    const int row_start = adjacency_row_ptr[node_idx];
    const int row_end = adjacency_row_ptr[node_idx + 1];
    T sum = 0.0f;
    for (int edge = row_start; edge < row_end; ++edge)
    {
        const int neighbor = adjacency_col_idx[edge];
        const T edge_weight = edge_features != nullptr ? edge_features[edge] : 1.0f;
        sum += edge_weight * node_features[neighbor * feature_dim + feature_idx];
    }
    const int degree = row_end - row_start;
    if (aggregate_mean && degree > 0)
    {
        sum /= static_cast<T>(degree);
    }
    messages[idx] = sum;
}

template <typename T>
__global__ void __launch_bounds__(256)
    attentionMessagePassingKernel(const T *__restrict__ query, const T *__restrict__ key,
                                  const T *__restrict__ value,
                                  const int *__restrict__ adjacency_row_ptr,
                                  const int *__restrict__ adjacency_col_idx,
                                  const int *__restrict__ edge_types,
                                  T *__restrict__ output_messages, int num_nodes, int feature_dim,
                                  int num_edge_types, float temperature)
{
    (void)num_edge_types;
    int node_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int tid = threadIdx.x;

    if (node_idx >= num_nodes)
        return;

    int head_dim = feature_dim / gridDim.y;
    int head_offset = head_idx * head_dim;

    int row_start = adjacency_row_ptr[node_idx];
    int row_end = adjacency_row_ptr[node_idx + 1];

    for (int d = tid; d < head_dim; d += blockDim.x)
    {
        T max_score = -INFINITY;
        for (int i = 0; i < (row_end - row_start); ++i)
        {
            int neighbor = adjacency_col_idx[row_start + i];
            T score = 0.0f;
            for (int f = 0; f < head_dim; ++f)
            {
                score += query[node_idx * feature_dim + head_offset + f] *
                         key[neighbor * feature_dim + head_offset + f];
            }
            score /= sqrtf(static_cast<T>(head_dim));
            score /= temperature;
            if (edge_types != nullptr)
            {
                score += static_cast<T>(edge_types[row_start + i]) * 0.1f;
            }
            max_score = max(max_score, score);
        }
        T sum_exp = 0.0f;
        T out_val = 0.0f;
        for (int i = 0; i < (row_end - row_start); ++i)
        {
            int neighbor = adjacency_col_idx[row_start + i];
            T score = 0.0f;
            for (int f = 0; f < head_dim; ++f)
            {
                score += query[node_idx * feature_dim + head_offset + f] *
                         key[neighbor * feature_dim + head_offset + f];
            }
            score /= sqrtf(static_cast<T>(head_dim));
            score /= temperature;
            if (edge_types != nullptr)
            {
                score += static_cast<T>(edge_types[row_start + i]) * 0.1f;
            }
            const T weight = expf(score - max_score);
            sum_exp += weight;
            out_val += weight * value[neighbor * feature_dim + head_offset + d];
        }
        output_messages[node_idx * feature_dim + head_offset + d] =
            sum_exp > 0 ? out_val / sum_exp : 0.0f;
    }
}

class GPUMessagePassing
{
public:
    enum class AggregationType
    {
        MEAN,
        SUM,
        MAX,
        ATTENTION
    };

    GPUMessagePassing(int num_nodes, int feature_dim, AggregationType agg_type)
        : num_nodes_(num_nodes)
        , num_edges_(0)
        , feature_dim_(feature_dim)
        , agg_type_(agg_type)
    {
        if (num_nodes_ <= 0 || feature_dim_ <= 0)
        {
            throw std::invalid_argument("message passing dimensions must be positive");
        }
        node_feature_count_ =
            checkedMul(static_cast<std::size_t>(num_nodes_), static_cast<std::size_t>(feature_dim_),
                       "message passing feature count");
        checkedIntSize(node_feature_count_, "message passing feature count");
        try
        {
            allocateDevice(&d_messages_, node_feature_count_, "allocate message buffer");
            throwCudaLaunchError(
                cudaMemset(d_messages_, 0,
                           checkedMul(node_feature_count_, sizeof(float), "message buffer bytes")),
                "initialize message buffer");
        }
        catch (...)
        {
            cleanup();
            throw;
        }
    }

    ~GPUMessagePassing() { cleanup(); }

    void setGraph(const std::vector<int> &row_ptr, const std::vector<int> &col_idx,
                  const std::vector<float> &edge_weights = {})
    {
        validateCSR(row_ptr, col_idx, num_nodes_);
        if (!edge_weights.empty() && edge_weights.size() != col_idx.size())
        {
            throw std::invalid_argument("edge_weights must match col_idx");
        }
        requireFiniteInput(edge_weights, "edge_weights must be finite");
        releaseGraphBuffers();

        try
        {
            allocateDevice(&d_row_ptr_, row_ptr.size(), "allocate message row ptr");
            allocateDevice(&d_col_idx_, col_idx.size(), "allocate message col idx");
            copyToDevice(d_row_ptr_, row_ptr.data(), row_ptr.size(), "copy message row ptr");
            copyToDevice(d_col_idx_, col_idx.data(), col_idx.size(), "copy message col idx");

            if (!edge_weights.empty())
            {
                allocateDevice(&d_edge_weights_, edge_weights.size(),
                               "allocate message edge weights");
                copyToDevice(d_edge_weights_, edge_weights.data(), edge_weights.size(),
                             "copy message edge weights");
            }
            num_edges_ = static_cast<int>(col_idx.size());
        }
        catch (...)
        {
            releaseGraphBuffers();
            throw;
        }
    }

    std::vector<float> aggregate(const std::vector<float> &node_features, int num_iterations = 1)
    {
        if (node_features.size() != node_feature_count_)
        {
            throw std::invalid_argument("node_features must match num_nodes * feature_dim");
        }
        if (num_iterations <= 0)
        {
            throw std::invalid_argument("num_iterations must be positive");
        }
        if (d_row_ptr_ == nullptr)
        {
            throw std::logic_error("graph must be set before aggregate");
        }
        requireFiniteInput(node_features, "node_features must be finite");
        float *d_features = nullptr;
        try
        {
            allocateDevice(&d_features, node_features.size(), "allocate message input features");
            copyToDevice(d_features, node_features.data(), node_features.size(),
                         "copy message input features");
            throwCudaLaunchError(
                cudaMemset(d_messages_, 0,
                           checkedMul(node_feature_count_, sizeof(float), "message reset bytes")),
                "reset message buffer");

            const int blocks = checkedGridBlocks(node_feature_count_, "message passing grid");
            const bool aggregate_mean = (agg_type_ == AggregationType::MEAN);

            for (int iter = 0; iter < num_iterations; ++iter)
            {
                messagePassingKernel<float>
                    <<<blocks, BLOCK_SIZE>>>(d_features, d_row_ptr_, d_col_idx_, d_edge_weights_,
                                             d_messages_, num_nodes_, feature_dim_, aggregate_mean);
                throwCudaLaunchError(cudaPeekAtLastError(), "messagePassingKernel launch failed");
                throwCudaLaunchError(cudaDeviceSynchronize(),
                                     "messagePassingKernel execution failed");

                throwCudaLaunchError(
                    cudaMemcpy(d_features, d_messages_,
                               checkedMul(node_feature_count_, sizeof(float), "message D2D bytes"),
                               cudaMemcpyDeviceToDevice),
                    "copy message iteration output");

                if (iter < num_iterations - 1)
                {
                    throwCudaLaunchError(cudaMemset(d_messages_, 0,
                                                    checkedMul(node_feature_count_, sizeof(float),
                                                               "message reset bytes")),
                                         "reset message buffer");
                }
            }

            std::vector<float> result(node_feature_count_);
            copyToHost(result.data(), d_messages_, result.size(), "copy message output");
            requireFiniteOutput(result, "message passing produced non-finite output");
            cudaFree(d_features);
            return result;
        }
        catch (...)
        {
            if (d_features)
                cudaFree(d_features);
            throw;
        }
    }

    std::vector<float> attentionAggregate(const std::vector<float> &query,
                                          const std::vector<float> &key,
                                          const std::vector<float> &value, int num_heads = 1,
                                          float temperature = 1.0f)
    {
        if (query.size() != node_feature_count_ || key.size() != query.size() ||
            value.size() != query.size())
        {
            throw std::invalid_argument("attention inputs must match num_nodes * feature_dim");
        }
        if (num_heads <= 0 || feature_dim_ % num_heads != 0 || !std::isfinite(temperature) ||
            temperature <= 0.0f)
        {
            throw std::invalid_argument("attention parameters are invalid");
        }
        if (d_row_ptr_ == nullptr)
        {
            throw std::logic_error("graph must be set before attentionAggregate");
        }
        requireFiniteInput(query, "attention query must be finite");
        requireFiniteInput(key, "attention key must be finite");
        requireFiniteInput(value, "attention value must be finite");

        float *d_q = nullptr, *d_k = nullptr, *d_v = nullptr;
        auto free_runtime = [&]() {
            if (d_q)
                cudaFree(d_q);
            if (d_k)
                cudaFree(d_k);
            if (d_v)
                cudaFree(d_v);
        };
        try
        {
            allocateDevice(&d_q, query.size(), "allocate attention query");
            allocateDevice(&d_k, key.size(), "allocate attention key");
            allocateDevice(&d_v, value.size(), "allocate attention value");
            copyToDevice(d_q, query.data(), query.size(), "copy attention query");
            copyToDevice(d_k, key.data(), key.size(), "copy attention key");
            copyToDevice(d_v, value.data(), value.size(), "copy attention value");

            dim3 blocks(num_nodes_, num_heads);
            dim3 threads(BLOCK_SIZE);
            attentionMessagePassingKernel<float>
                <<<blocks, threads>>>(d_q, d_k, d_v, d_row_ptr_, d_col_idx_, nullptr, d_messages_,
                                      num_nodes_, feature_dim_, 0, temperature);
            throwCudaLaunchError(cudaPeekAtLastError(),
                                 "attentionMessagePassingKernel launch failed");
            throwCudaLaunchError(cudaDeviceSynchronize(),
                                 "attentionMessagePassingKernel execution failed");

            std::vector<float> result(node_feature_count_);
            copyToHost(result.data(), d_messages_, result.size(), "copy attention output");
            requireFiniteOutput(result, "attention message passing produced non-finite output");
            free_runtime();
            return result;
        }
        catch (...)
        {
            free_runtime();
            throw;
        }
    }

private:
    void releaseGraphBuffers() noexcept
    {
        if (d_row_ptr_)
            cudaFree(d_row_ptr_);
        if (d_col_idx_)
            cudaFree(d_col_idx_);
        if (d_edge_weights_)
            cudaFree(d_edge_weights_);
        d_row_ptr_ = nullptr;
        d_col_idx_ = nullptr;
        d_edge_weights_ = nullptr;
        num_edges_ = 0;
    }

    void cleanup() noexcept
    {
        if (d_messages_)
            cudaFree(d_messages_);
        releaseGraphBuffers();
        d_messages_ = nullptr;
    }

    int num_nodes_ = 0;
    int num_edges_ = 0;
    int feature_dim_ = 0;
    AggregationType agg_type_;
    std::size_t node_feature_count_ = 0;

    float *d_messages_ = nullptr;
    int *d_row_ptr_ = nullptr;
    int *d_col_idx_ = nullptr;
    float *d_edge_weights_ = nullptr;
};

MessagePassingBenchmark benchmarkMessagePassing(int num_nodes, int num_edges, int feature_dim)
{
    if (num_nodes <= 0 || num_edges < 0 || feature_dim <= 0)
    {
        throw std::invalid_argument("message passing benchmark dimensions are invalid");
    }
    MessagePassingBenchmark bench;
    bench.num_nodes = num_nodes;
    bench.num_edges = num_edges;
    bench.feature_dim = feature_dim;

    const size_t feature_count =
        checkedMul(static_cast<size_t>(num_nodes), static_cast<size_t>(feature_dim),
                   "message benchmark feature count");
    checkedIntSize(feature_count, "message benchmark feature count");
    std::vector<int> row_ptr(static_cast<size_t>(num_nodes) + 1);
    std::vector<int> col_idx;
    col_idx.reserve(static_cast<size_t>(num_edges));
    std::vector<float> features(feature_count);

    for (int i = 0; i <= num_nodes; ++i)
    {
        row_ptr[static_cast<size_t>(i)] =
            static_cast<int>((static_cast<long long>(i) * num_edges) / num_nodes);
    }
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> node_dist(0, num_nodes - 1);
    std::uniform_real_distribution<float> feature_dist(0.0f, 1.0f);
    for (int i = 0; i < num_edges; ++i)
    {
        col_idx.push_back(node_dist(gen));
    }
    for (auto &f : features)
    {
        f = feature_dist(gen);
    }

    GPUMessagePassing gpu_mp(num_nodes, feature_dim, GPUMessagePassing::AggregationType::MEAN);
    gpu_mp.setGraph(row_ptr, col_idx);

    auto start_gpu = std::chrono::high_resolution_clock::now();
    auto gpu_result = gpu_mp.aggregate(features, 1);
    auto end_gpu = std::chrono::high_resolution_clock::now();
    (void)gpu_result;
    bench.gpu_time_ms = std::chrono::duration<double, std::milli>(end_gpu - start_gpu).count();

    bench.cpu_time_ms = 0.0;
    bench.gpu_fp16_time_ms = 0.0;
    bench.speedup = 1.0;

    return bench;
}

} // namespace gpu
} // namespace graphs
} // namespace nerve
