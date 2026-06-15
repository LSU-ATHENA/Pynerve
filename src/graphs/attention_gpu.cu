#include "nerve/graphs/gpu_gnn.hpp"

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
namespace graphs
{
namespace gpu
{

constexpr int ATTENTION_BLOCK_SIZE = 256;
constexpr int BENCHMARK_EDGES_PER_NODE = 10;

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

int checkedIntSize(std::size_t value, const char *label)
{
    if (value > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::length_error(std::string(label) + " exceeds CUDA int range");
    }
    return static_cast<int>(value);
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
    checkCuda(cudaMalloc(reinterpret_cast<void **>(ptr), checkedMul(count, sizeof(T), label)),
              label);
}

template <typename T>
void copyToDevice(T *dst, const T *src, std::size_t count, const char *label)
{
    if (count == 0)
    {
        return;
    }
    checkCuda(cudaMemcpy(dst, src, checkedMul(count, sizeof(T), label), cudaMemcpyHostToDevice),
              label);
}

template <typename T>
void copyToHost(T *dst, const T *src, std::size_t count, const char *label)
{
    if (count == 0)
    {
        return;
    }
    checkCuda(cudaMemcpy(dst, src, checkedMul(count, sizeof(T), label), cudaMemcpyDeviceToHost),
              label);
}

void validateCSR(const std::vector<int> &row_ptr, const std::vector<int> &col_idx, int num_nodes)
{
    if (row_ptr.size() != static_cast<std::size_t>(num_nodes) + 1)
    {
        throw std::invalid_argument("row_ptr must have num_nodes + 1 entries");
    }
    checkedIntSize(col_idx.size(), "attention edge count");
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

__global__ void __launch_bounds__(256)
    multiHeadAttentionKernel(const float *__restrict__ query, const float *__restrict__ key,
                             const float *__restrict__ value,
                             const int *__restrict__ adjacency_row_ptr,
                             const int *__restrict__ adjacency_col_idx,
                             float *__restrict__ attention_weights, float *__restrict__ output,
                             int num_nodes, int feature_dim, int num_heads, float scale_factor)
{
    const int node_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int tid = threadIdx.x;
    if (node_idx >= num_nodes || head_idx >= num_heads)
    {
        return;
    }

    const int head_dim = feature_dim / num_heads;
    const int head_offset = head_idx * head_dim;
    const int row_start = adjacency_row_ptr[node_idx];
    const int row_end = adjacency_row_ptr[node_idx + 1];
    const int degree = row_end - row_start;

    extern __shared__ float shared_mem[];
    float *s_query = shared_mem;
    float *s_reduce = shared_mem + head_dim;

    for (int d = tid; d < head_dim; d += blockDim.x)
    {
        s_query[d] = query[node_idx * feature_dim + head_offset + d];
    }
    __syncthreads();

    if (degree <= 0)
    {
        for (int d = tid; d < head_dim; d += blockDim.x)
        {
            output[node_idx * feature_dim + head_offset + d] = 0.0f;
        }
        return;
    }

    float local_max = -INFINITY;
    for (int edge_off = tid; edge_off < degree; edge_off += blockDim.x)
    {
        const int neighbor = adjacency_col_idx[row_start + edge_off];
        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d)
        {
            score += s_query[d] * key[neighbor * feature_dim + head_offset + d];
        }
        score *= scale_factor;
        const size_t attn_idx = (static_cast<size_t>(node_idx) * num_nodes * num_heads) +
                                (static_cast<size_t>(head_idx) * num_nodes) + neighbor;
        attention_weights[attn_idx] = score;
        local_max = fmaxf(local_max, score);
    }

    s_reduce[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if (tid < stride)
        {
            s_reduce[tid] = fmaxf(s_reduce[tid], s_reduce[tid + stride]);
        }
        __syncthreads();
    }
    const float max_score = s_reduce[0];

    float local_sum = 0.0f;
    for (int edge_off = tid; edge_off < degree; edge_off += blockDim.x)
    {
        const int neighbor = adjacency_col_idx[row_start + edge_off];
        const size_t attn_idx = (static_cast<size_t>(node_idx) * num_nodes * num_heads) +
                                (static_cast<size_t>(head_idx) * num_nodes) + neighbor;
        local_sum += expf(attention_weights[attn_idx] - max_score);
    }

    s_reduce[tid] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if (tid < stride)
        {
            s_reduce[tid] += s_reduce[tid + stride];
        }
        __syncthreads();
    }
    const float inv_sum = (s_reduce[0] > 0.0f) ? (1.0f / s_reduce[0]) : 0.0f;

    for (int d = tid; d < head_dim; d += blockDim.x)
    {
        float out_val = 0.0f;
        for (int edge_off = 0; edge_off < degree; ++edge_off)
        {
            const int neighbor = adjacency_col_idx[row_start + edge_off];
            const size_t attn_idx = (static_cast<size_t>(node_idx) * num_nodes * num_heads) +
                                    (static_cast<size_t>(head_idx) * num_nodes) + neighbor;
            const float weight = expf(attention_weights[attn_idx] - max_score) * inv_sum;
            out_val += weight * value[neighbor * feature_dim + head_offset + d];
        }
        output[node_idx * feature_dim + head_offset + d] = out_val;
    }
}

class GPUMultiHeadAttention
{
public:
    GPUMultiHeadAttention(int num_nodes, int feature_dim, int num_heads)
        : num_nodes_(num_nodes)
        , feature_dim_(feature_dim)
        , num_heads_(num_heads)
    {
        if (num_nodes_ <= 0 || feature_dim_ <= 0 || num_heads_ <= 0 ||
            feature_dim_ % num_heads_ != 0)
        {
            throw std::invalid_argument("attention dimensions are invalid");
        }
        head_dim_ = feature_dim_ / num_heads_;
        scale_factor_ = 1.0f / sqrtf(static_cast<float>(head_dim_));
        feature_count_ = checkedMul(static_cast<size_t>(num_nodes_),
                                    static_cast<size_t>(feature_dim_), "attention feature count");
        weight_count_ = checkedMul(static_cast<size_t>(feature_dim_),
                                   static_cast<size_t>(feature_dim_), "attention weight count");
        attention_count_ =
            checkedMul(checkedMul(static_cast<size_t>(num_nodes_), static_cast<size_t>(num_nodes_),
                                  "attention dense node pair count"),
                       static_cast<size_t>(num_heads_), "attention dense weight count");
        checkedIntSize(feature_count_, "attention feature count");
        checkedIntSize(attention_count_, "attention dense weight count");
        try
        {
            allocateDevice(&d_query_, feature_count_, "allocate attention query");
            allocateDevice(&d_key_, feature_count_, "allocate attention key");
            allocateDevice(&d_value_, feature_count_, "allocate attention value");
            allocateDevice(&d_output_, feature_count_, "allocate attention output");
            allocateDevice(&d_attention_weights_, attention_count_, "allocate attention weights");
            allocateDevice(&d_w_q_, weight_count_, "allocate attention Wq");
            allocateDevice(&d_w_k_, weight_count_, "allocate attention Wk");
            allocateDevice(&d_w_v_, weight_count_, "allocate attention Wv");
            checkCuda(cudaMemset(d_attention_weights_, 0,
                                 checkedMul(attention_count_, sizeof(float),
                                            "initialize attention weights")),
                      "initialize attention weights");
            checkCublas(cublasCreate(&cublas_handle_), "create attention cuBLAS handle");
            checkCublas(cublasSetMathMode(cublas_handle_, CUBLAS_TENSOR_OP_MATH),
                        "set attention cuBLAS math mode");
            initializeWeights();
        }
        catch (...)
        {
            cleanup();
            throw;
        }
    }

    ~GPUMultiHeadAttention() { cleanup(); }

    void setGraph(const std::vector<int> &row_ptr, const std::vector<int> &col_idx)
    {
        validateCSR(row_ptr, col_idx, num_nodes_);
        releaseGraphBuffers();
        try
        {
            allocateDevice(&d_row_ptr_, row_ptr.size(), "allocate attention row ptr");
            allocateDevice(&d_col_idx_, col_idx.size(), "allocate attention col idx");
            copyToDevice(d_row_ptr_, row_ptr.data(), row_ptr.size(), "copy attention row ptr");
            copyToDevice(d_col_idx_, col_idx.data(), col_idx.size(), "copy attention col idx");
            graph_set_ = true;
        }
        catch (...)
        {
            releaseGraphBuffers();
            throw;
        }
    }

    std::vector<float> forward(const std::vector<float> &input_features)
    {
        if (input_features.size() != feature_count_)
        {
            throw std::invalid_argument("input_features must match num_nodes * feature_dim");
        }
        if (!graph_set_)
        {
            throw std::logic_error("graph must be set before forward");
        }
        requireFiniteInput(input_features, "attention input_features must be finite");
        float *d_input = nullptr;
        try
        {
            allocateDevice(&d_input, input_features.size(), "allocate attention input");
            copyToDevice(d_input, input_features.data(), input_features.size(),
                         "copy attention input");

            const float alpha = 1.0f;
            const float beta = 0.0f;
            checkCublas(cublasSgemm(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N, feature_dim_,
                                    num_nodes_, feature_dim_, &alpha, d_w_q_, feature_dim_, d_input,
                                    feature_dim_, &beta, d_query_, feature_dim_),
                        "run attention query projection");
            checkCublas(cublasSgemm(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N, feature_dim_,
                                    num_nodes_, feature_dim_, &alpha, d_w_k_, feature_dim_, d_input,
                                    feature_dim_, &beta, d_key_, feature_dim_),
                        "run attention key projection");
            checkCublas(cublasSgemm(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N, feature_dim_,
                                    num_nodes_, feature_dim_, &alpha, d_w_v_, feature_dim_, d_input,
                                    feature_dim_, &beta, d_value_, feature_dim_),
                        "run attention value projection");
            cudaFree(d_input);
            d_input = nullptr;

            dim3 blocks(num_nodes_, num_heads_);
            dim3 threads(ATTENTION_BLOCK_SIZE);
            const size_t shared_bytes =
                checkedMul(static_cast<size_t>(head_dim_ + ATTENTION_BLOCK_SIZE), sizeof(float),
                           "attention shared memory bytes");
            checkCuda(
                cudaMemset(d_attention_weights_, 0,
                           checkedMul(attention_count_, sizeof(float), "reset attention weights")),
                "reset attention weights");
            multiHeadAttentionKernel<<<blocks, threads, shared_bytes>>>(
                d_query_, d_key_, d_value_, d_row_ptr_, d_col_idx_, d_attention_weights_, d_output_,
                num_nodes_, feature_dim_, num_heads_, scale_factor_);
            checkCuda(cudaPeekAtLastError(), "launch multi-head attention kernel");
            checkCuda(cudaDeviceSynchronize(), "synchronize multi-head attention kernel");

            std::vector<float> output(feature_count_);
            copyToHost(output.data(), d_output_, output.size(), "copy attention output");
            requireFiniteOutput(output, "multi-head attention produced non-finite output");
            return output;
        }
        catch (...)
        {
            if (d_input)
                cudaFree(d_input);
            throw;
        }
    }

    std::vector<float> getAttentionWeights() const
    {
        std::vector<float> weights(attention_count_);
        copyToHost(weights.data(), d_attention_weights_, weights.size(), "copy attention weights");
        requireFiniteOutput(weights, "multi-head attention produced non-finite weights");
        return weights;
    }

private:
    void initializeWeights()
    {
        std::mt19937 gen(42);
        std::normal_distribution<float> dist(0.0f, sqrtf(2.0f / (feature_dim_ + feature_dim_)));

        std::vector<float> w_q(weight_count_);
        std::vector<float> w_k(weight_count_);
        std::vector<float> w_v(weight_count_);
        for (auto &w : w_q)
            w = dist(gen);
        for (auto &w : w_k)
            w = dist(gen);
        for (auto &w : w_v)
            w = dist(gen);

        copyToDevice(d_w_q_, w_q.data(), w_q.size(), "copy attention Wq");
        copyToDevice(d_w_k_, w_k.data(), w_k.size(), "copy attention Wk");
        copyToDevice(d_w_v_, w_v.data(), w_v.size(), "copy attention Wv");
    }

    void releaseGraphBuffers() noexcept
    {
        if (d_row_ptr_)
            cudaFree(d_row_ptr_);
        if (d_col_idx_)
            cudaFree(d_col_idx_);
        d_row_ptr_ = nullptr;
        d_col_idx_ = nullptr;
        graph_set_ = false;
    }

    void cleanup() noexcept
    {
        if (d_query_)
            cudaFree(d_query_);
        if (d_key_)
            cudaFree(d_key_);
        if (d_value_)
            cudaFree(d_value_);
        if (d_output_)
            cudaFree(d_output_);
        if (d_attention_weights_)
            cudaFree(d_attention_weights_);
        if (d_w_q_)
            cudaFree(d_w_q_);
        if (d_w_k_)
            cudaFree(d_w_k_);
        if (d_w_v_)
            cudaFree(d_w_v_);
        releaseGraphBuffers();
        if (cublas_handle_)
            cublasDestroy(cublas_handle_);
        d_query_ = nullptr;
        d_key_ = nullptr;
        d_value_ = nullptr;
        d_output_ = nullptr;
        d_attention_weights_ = nullptr;
        d_w_q_ = nullptr;
        d_w_k_ = nullptr;
        d_w_v_ = nullptr;
        cublas_handle_ = nullptr;
    }

    int num_nodes_ = 0;
    int feature_dim_ = 0;
    int num_heads_ = 0;
    int head_dim_ = 0;
    float scale_factor_ = 1.0f;
    bool graph_set_ = false;
    size_t feature_count_ = 0;
    size_t weight_count_ = 0;
    size_t attention_count_ = 0;

    float *d_query_ = nullptr;
    float *d_key_ = nullptr;
    float *d_value_ = nullptr;
    float *d_output_ = nullptr;
    float *d_attention_weights_ = nullptr;
    float *d_w_q_ = nullptr;
    float *d_w_k_ = nullptr;
    float *d_w_v_ = nullptr;
    int *d_row_ptr_ = nullptr;
    int *d_col_idx_ = nullptr;
    cublasHandle_t cublas_handle_ = nullptr;
};

AttentionBenchmark benchmarkMultiHeadAttention(int num_nodes, int feature_dim, int num_heads)
{
    if (num_nodes <= 0 || feature_dim <= 0 || num_heads <= 0 || feature_dim % num_heads != 0)
    {
        throw std::invalid_argument("attention benchmark dimensions are invalid");
    }
    AttentionBenchmark bench{};
    bench.num_nodes = num_nodes;
    bench.feature_dim = feature_dim;
    bench.num_heads = num_heads;

    const size_t feature_count =
        checkedMul(static_cast<size_t>(num_nodes), static_cast<size_t>(feature_dim),
                   "attention benchmark feature count");
    checkedIntSize(feature_count, "attention benchmark feature count");
    const int edge_count = checkedIntSize(checkedMul(static_cast<size_t>(num_nodes),
                                                     static_cast<size_t>(BENCHMARK_EDGES_PER_NODE),
                                                     "attention benchmark edge count"),
                                          "attention benchmark edge count");

    std::vector<float> features(feature_count);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> feature_dist(0.0f, 1.0f);
    std::uniform_int_distribution<int> node_dist(0, num_nodes - 1);
    for (auto &f : features)
    {
        f = feature_dist(gen);
    }

    std::vector<int> row_ptr(static_cast<size_t>(num_nodes) + 1);
    std::vector<int> col_idx;
    col_idx.reserve(static_cast<size_t>(edge_count));
    for (int i = 0; i <= num_nodes; ++i)
    {
        row_ptr[static_cast<size_t>(i)] = i * BENCHMARK_EDGES_PER_NODE;
    }
    for (int i = 0; i < edge_count; ++i)
    {
        col_idx.push_back(node_dist(gen));
    }

    GPUMultiHeadAttention attn(num_nodes, feature_dim, num_heads);
    attn.setGraph(row_ptr, col_idx);

    auto start_gpu = std::chrono::high_resolution_clock::now();
    auto output = attn.forward(features);
    auto end_gpu = std::chrono::high_resolution_clock::now();
    (void)output;

    bench.gpu_time_ms = std::chrono::duration<double, std::milli>(end_gpu - start_gpu).count();
    bench.cpu_time_ms = 0.0;
    bench.tensor_core_time_ms = 0.0;
    bench.speedup = 1.0;
    return bench;
}

} // namespace gpu
} // namespace graphs
} // namespace nerve
