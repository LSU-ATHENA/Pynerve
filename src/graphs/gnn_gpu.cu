#include <cooperative_groups.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace cg = cooperative_groups;

namespace nerve
{
namespace graphs
{
namespace gpu
{

constexpr int BLOCK_SIZE = 256;
constexpr int WARP_SIZE = 32;

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
        throw std::invalid_argument("adjacency_row_ptr must have num_nodes + 1 entries");
    }
    checkedIntSize(col_idx.size(), "GNN edge count");
    if (row_ptr.empty() || row_ptr.front() != 0 ||
        row_ptr.back() != static_cast<int>(col_idx.size()))
    {
        throw std::invalid_argument("adjacency_row_ptr must start at zero and end at edge count");
    }
    for (std::size_t i = 1; i < row_ptr.size(); ++i)
    {
        if (row_ptr[i - 1] > row_ptr[i] || row_ptr[i] < 0)
        {
            throw std::invalid_argument("adjacency_row_ptr must be monotonic and non-negative");
        }
    }
    for (int col : col_idx)
    {
        if (col < 0 || col >= num_nodes)
        {
            throw std::out_of_range("adjacency_col_idx contains a node outside graph bounds");
        }
    }
}

template <typename T>
__global__ void __launch_bounds__(256)
    graphConvolutionKernel(const T *__restrict__ node_features,
                           const int *__restrict__ adjacency_row_ptr,
                           const int *__restrict__ adjacency_col_idx,
                           const T *__restrict__ edge_weights, T *__restrict__ output_features,
                           int num_nodes, int feature_dim, bool use_edge_weights)
{
    const int node_idx = blockIdx.x;
    const int feature_idx = blockIdx.y * blockDim.x + threadIdx.x;

    if (node_idx >= num_nodes || feature_idx >= feature_dim)
        return;
    const int row_start = adjacency_row_ptr[node_idx];
    const int row_end = adjacency_row_ptr[node_idx + 1];
    const int degree = row_end - row_start;

    T sum = 0.0f;
    for (int edge = row_start; edge < row_end; ++edge)
    {
        const int neighbor = adjacency_col_idx[edge];
        const T weight = use_edge_weights ? edge_weights[edge] : 1.0f;
        sum += weight * node_features[neighbor * feature_dim + feature_idx];
    }
    if (degree > 0)
    {
        sum /= static_cast<T>(degree);
    }
    sum += node_features[node_idx * feature_dim + feature_idx];
    output_features[node_idx * feature_dim + feature_idx] = sum;
}

template <typename T>
__global__ void __launch_bounds__(32)
    graphAttentionKernel(const T *__restrict__ query, const T *__restrict__ key,
                         const T *__restrict__ value, const int *__restrict__ adjacency_row_ptr,
                         const int *__restrict__ adjacency_col_idx,
                         T *__restrict__ edge_attention_scores, T *__restrict__ output,
                         int num_nodes, int num_edges, int feature_dim, int num_heads,
                         float attention_dropout)
{
    const int node_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    if (node_idx >= num_nodes || head_idx >= num_heads)
        return;

    const int head_dim = feature_dim / num_heads;
    const int head_offset = head_idx * head_dim;
    const int row_start = adjacency_row_ptr[node_idx];
    const int row_end = adjacency_row_ptr[node_idx + 1];
    const int degree = row_end - row_start;
    T *head_scores = edge_attention_scores + static_cast<size_t>(head_idx) * num_edges;

    if (threadIdx.x == 0)
    {
        T max_score = -INFINITY;
        for (int edge = row_start; edge < row_end; ++edge)
        {
            const int neighbor = adjacency_col_idx[edge];
            T score = 0.0f;
            for (int feat = 0; feat < head_dim; ++feat)
            {
                score += query[node_idx * feature_dim + head_offset + feat] *
                         key[neighbor * feature_dim + head_offset + feat];
            }
            head_scores[edge] = score;
            max_score = max(max_score, score);
        }
        T sum_exp = 0.0f;
        for (int edge = row_start; edge < row_end; ++edge)
        {
            const T stabilized = expf(head_scores[edge] - max_score);
            head_scores[edge] = stabilized;
            sum_exp += stabilized;
        }
        const T inv_sum = sum_exp > 0 ? 1.0f / sum_exp : 0.0f;
        const T keep_scale = attention_dropout > 0 ? (1.0f - attention_dropout) : 1.0f;
        for (int edge = row_start; edge < row_end; ++edge)
        {
            head_scores[edge] *= inv_sum;
            head_scores[edge] *= keep_scale;
        }
        if (degree == 0)
        {
            for (int feat = 0; feat < head_dim; ++feat)
            {
                output[node_idx * feature_dim + head_offset + feat] = 0.0f;
            }
        }
    }
    __syncthreads();

    for (int feat = threadIdx.x; feat < head_dim; feat += blockDim.x)
    {
        T out_val = 0.0f;
        for (int edge = row_start; edge < row_end; ++edge)
        {
            const int neighbor = adjacency_col_idx[edge];
            out_val += head_scores[edge] * value[neighbor * feature_dim + head_offset + feat];
        }
        output[node_idx * feature_dim + head_offset + feat] = out_val;
    }
}

class GPUGraphNeuralLayer
{
public:
    GPUGraphNeuralLayer(int num_nodes, int input_dim, int output_dim, int num_heads = 1)
        : num_nodes_(num_nodes)
        , input_dim_(input_dim)
        , output_dim_(output_dim)
        , num_heads_(num_heads)
    {
        if (num_nodes_ <= 0 || input_dim_ <= 0 || output_dim_ <= 0 || num_heads_ <= 0)
        {
            throw std::invalid_argument("GNN dimensions must be positive");
        }
        if (output_dim_ % num_heads_ != 0)
        {
            throw std::invalid_argument("output_dim must be divisible by num_heads");
        }
        input_feature_count_ =
            checkedMul(static_cast<std::size_t>(num_nodes_), static_cast<std::size_t>(input_dim_),
                       "GNN input feature count");
        output_feature_count_ =
            checkedMul(static_cast<std::size_t>(num_nodes_), static_cast<std::size_t>(output_dim_),
                       "GNN output feature count");
        weight_count_ = checkedMul(static_cast<std::size_t>(input_dim_),
                                   static_cast<std::size_t>(output_dim_), "GNN weight count");
        checkedIntSize(input_feature_count_, "GNN input feature count");
        checkedIntSize(output_feature_count_, "GNN output feature count");
        try
        {
            allocateDevice(&d_node_features_, input_feature_count_, "allocate GNN node features");
            allocateDevice(&d_aggregated_features_, input_feature_count_,
                           "allocate GNN aggregated features");
            allocateDevice(&d_output_features_, output_feature_count_,
                           "allocate GNN output features");
            allocateDevice(&d_weights_, weight_count_, "allocate GNN weights");
            checkCublas(cublasCreate(&cublas_handle_), "create GNN cuBLAS handle");
        }
        catch (...)
        {
            cleanup();
            throw;
        }
    }

    ~GPUGraphNeuralLayer() { cleanup(); }

    void setGraphStructure(const std::vector<int> &adjacency_row_ptr,
                           const std::vector<int> &adjacency_col_idx,
                           const std::vector<float> &edge_weights)
    {
        validateCSR(adjacency_row_ptr, adjacency_col_idx, num_nodes_);
        if (!edge_weights.empty() && edge_weights.size() != adjacency_col_idx.size())
        {
            throw std::invalid_argument("edge_weights must match adjacency_col_idx");
        }
        requireFiniteInput(edge_weights, "GNN edge_weights must be finite");
        releaseGraphBuffers();
        try
        {
            allocateDevice(&d_adj_row_ptr_, adjacency_row_ptr.size(), "allocate GNN row ptr");
            allocateDevice(&d_adj_col_idx_, adjacency_col_idx.size(), "allocate GNN col idx");
            copyToDevice(d_adj_row_ptr_, adjacency_row_ptr.data(), adjacency_row_ptr.size(),
                         "copy GNN row ptr");
            copyToDevice(d_adj_col_idx_, adjacency_col_idx.data(), adjacency_col_idx.size(),
                         "copy GNN col idx");
            if (!edge_weights.empty())
            {
                allocateDevice(&d_edge_weights_, edge_weights.size(), "allocate GNN edge weights");
                copyToDevice(d_edge_weights_, edge_weights.data(), edge_weights.size(),
                             "copy GNN edge weights");
                use_edge_weights_ = true;
            }
            num_edges_ = static_cast<int>(adjacency_col_idx.size());
            if (num_edges_ > 0)
            {
                const size_t score_count =
                    checkedMul(static_cast<size_t>(num_heads_), static_cast<size_t>(num_edges_),
                               "GNN attention score count");
                allocateDevice(&d_attention_scores_, score_count, "allocate GNN attention scores");
            }
            graph_set_ = true;
        }
        catch (...)
        {
            releaseGraphBuffers();
            throw;
        }
    }

    void setWeights(const std::vector<float> &weights)
    {
        if (weights.size() != weight_count_)
        {
            throw std::invalid_argument("weights must match input_dim * output_dim");
        }
        requireFiniteInput(weights, "GNN weights must be finite");
        copyToDevice(d_weights_, weights.data(), weights.size(), "copy GNN weights");
    }

    std::vector<float> forward(const std::vector<float> &node_features)
    {
        if (node_features.size() != input_feature_count_)
        {
            throw std::invalid_argument("node_features must match num_nodes * input_dim");
        }
        if (!graph_set_)
        {
            throw std::logic_error("graph structure must be set before forward");
        }
        requireFiniteInput(node_features, "GNN node_features must be finite");
        copyToDevice(d_node_features_, node_features.data(), node_features.size(),
                     "copy GNN node features");

        dim3 blocks(num_nodes_, checkedGridBlocks(static_cast<size_t>(input_dim_),
                                                  "GNN convolution feature grid"));
        dim3 threads(BLOCK_SIZE);
        graphConvolutionKernel<float><<<blocks, threads>>>(
            d_node_features_, d_adj_row_ptr_, d_adj_col_idx_, d_edge_weights_,
            d_aggregated_features_, num_nodes_, input_dim_, use_edge_weights_);
        checkCuda(cudaPeekAtLastError(), "launch GNN graph convolution");
        checkCuda(cudaDeviceSynchronize(), "synchronize GNN graph convolution");

        float alpha = 1.0f, beta = 0.0f;
        checkCublas(cublasSgemm(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N, output_dim_, num_nodes_,
                                input_dim_, &alpha, d_weights_, output_dim_, d_aggregated_features_,
                                input_dim_, &beta, d_output_features_, output_dim_),
                    "run GNN dense projection");

        std::vector<float> output(output_feature_count_);
        copyToHost(output.data(), d_output_features_, output.size(), "copy GNN output");
        requireFiniteOutput(output, "GNN forward produced non-finite output");
        return output;
    }

    std::vector<float> attentionForward(const std::vector<float> &query,
                                        const std::vector<float> &key,
                                        const std::vector<float> &value, float dropout = 0.0f)
    {
        if (query.size() != output_feature_count_ || key.size() != output_feature_count_ ||
            value.size() != output_feature_count_)
        {
            throw std::invalid_argument(
                "attentionForward expects query/key/value to match num_nodes * output_dim");
        }
        if (!std::isfinite(dropout) || dropout < 0.0f || dropout >= 1.0f)
        {
            throw std::invalid_argument("dropout must be finite and in [0, 1)");
        }
        if (!graph_set_)
        {
            throw std::logic_error("graph structure must be set before attentionForward");
        }
        requireFiniteInput(query, "GNN attention query must be finite");
        requireFiniteInput(key, "GNN attention key must be finite");
        requireFiniteInput(value, "GNN attention value must be finite");

        float *d_query = nullptr;
        float *d_key = nullptr;
        float *d_value = nullptr;
        float *d_out = nullptr;
        auto free_runtime = [&]() {
            if (d_query)
                cudaFree(d_query);
            if (d_key)
                cudaFree(d_key);
            if (d_value)
                cudaFree(d_value);
            if (d_out)
                cudaFree(d_out);
        };
        try
        {
            allocateDevice(&d_query, output_feature_count_, "allocate GNN attention query");
            allocateDevice(&d_key, output_feature_count_, "allocate GNN attention key");
            allocateDevice(&d_value, output_feature_count_, "allocate GNN attention value");
            allocateDevice(&d_out, output_feature_count_, "allocate GNN attention output");
            copyToDevice(d_query, query.data(), query.size(), "copy GNN attention query");
            copyToDevice(d_key, key.data(), key.size(), "copy GNN attention key");
            copyToDevice(d_value, value.data(), value.size(), "copy GNN attention value");

            std::vector<float> output(output_feature_count_, 0.0f);
            if (num_edges_ == 0)
            {
                checkCuda(cudaMemset(d_out, 0,
                                     checkedMul(output_feature_count_, sizeof(float),
                                                "GNN zero-edge output bytes")),
                          "clear GNN zero-edge attention output");
                copyToHost(output.data(), d_out, output.size(), "copy GNN zero-edge output");
                requireFiniteOutput(output, "GNN attention produced non-finite output");
                free_runtime();
                return output;
            }

            if (d_attention_scores_ == nullptr)
            {
                const size_t score_count =
                    checkedMul(static_cast<size_t>(num_heads_), static_cast<size_t>(num_edges_),
                               "GNN attention score count");
                allocateDevice(&d_attention_scores_, score_count, "allocate GNN attention scores");
            }

            dim3 blocks(num_nodes_, num_heads_);
            dim3 threads(WARP_SIZE);
            graphAttentionKernel<float><<<blocks, threads>>>(
                d_query, d_key, d_value, d_adj_row_ptr_, d_adj_col_idx_, d_attention_scores_, d_out,
                num_nodes_, num_edges_, output_dim_, num_heads_, dropout);
            checkCuda(cudaPeekAtLastError(), "launch GNN attention");
            checkCuda(cudaDeviceSynchronize(), "synchronize GNN attention");
            copyToHost(output.data(), d_out, output.size(), "copy GNN attention output");
            requireFiniteOutput(output, "GNN attention produced non-finite output");
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
    void releaseGraphBuffers() noexcept
    {
        if (d_adj_row_ptr_)
            cudaFree(d_adj_row_ptr_);
        if (d_adj_col_idx_)
            cudaFree(d_adj_col_idx_);
        if (d_edge_weights_)
            cudaFree(d_edge_weights_);
        if (d_attention_scores_)
            cudaFree(d_attention_scores_);
        d_adj_row_ptr_ = nullptr;
        d_adj_col_idx_ = nullptr;
        d_edge_weights_ = nullptr;
        d_attention_scores_ = nullptr;
        num_edges_ = 0;
        use_edge_weights_ = false;
        graph_set_ = false;
    }

    void cleanup() noexcept
    {
        releaseGraphBuffers();
        if (d_node_features_)
            cudaFree(d_node_features_);
        if (d_aggregated_features_)
            cudaFree(d_aggregated_features_);
        if (d_output_features_)
            cudaFree(d_output_features_);
        if (d_weights_)
            cudaFree(d_weights_);
        if (cublas_handle_)
            cublasDestroy(cublas_handle_);
        d_node_features_ = nullptr;
        d_aggregated_features_ = nullptr;
        d_output_features_ = nullptr;
        d_weights_ = nullptr;
        cublas_handle_ = nullptr;
    }

    int num_nodes_ = 0;
    int input_dim_ = 0;
    int output_dim_ = 0;
    int num_heads_ = 0;
    int num_edges_ = 0;
    bool use_edge_weights_ = false;
    bool graph_set_ = false;
    std::size_t input_feature_count_ = 0;
    std::size_t output_feature_count_ = 0;
    std::size_t weight_count_ = 0;

    float *d_node_features_ = nullptr;
    float *d_aggregated_features_ = nullptr;
    float *d_output_features_ = nullptr;
    float *d_weights_ = nullptr;
    float *d_attention_scores_ = nullptr;

    int *d_adj_row_ptr_ = nullptr;
    int *d_adj_col_idx_ = nullptr;
    float *d_edge_weights_ = nullptr;
    cublasHandle_t cublas_handle_ = nullptr;
};

} // namespace gpu
} // namespace graphs
} // namespace nerve
