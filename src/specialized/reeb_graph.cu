#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nerve
{
namespace specialized
{
namespace gpu
{

namespace
{
constexpr int REEB_BLOCK_SIZE = 256;
constexpr int kMaxAdjacencyPerVertex = 10;

void checkCuda(cudaError_t status, const char *context)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(status));
    }
}

size_t checkedMulSize(size_t a, size_t b, const char *label)
{
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a)
    {
        throw std::length_error(std::string(label) + " exceeds size_t limits");
    }
    return a * b;
}

int checkedIntSize(size_t value, const char *label)
{
    if (value > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        throw std::length_error(std::string(label) + " exceeds CUDA int range");
    }
    return static_cast<int>(value);
}

int checkedGridBlocks(size_t count, const char *label)
{
    if (count == 0)
    {
        return 0;
    }
    const size_t blocks =
        (count + static_cast<size_t>(REEB_BLOCK_SIZE) - 1) / static_cast<size_t>(REEB_BLOCK_SIZE);
    if (blocks > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        throw std::length_error(std::string(label) + " exceeds CUDA grid range");
    }
    return static_cast<int>(blocks);
}

template <typename T>
void allocateDevice(T **ptr, size_t count, const char *label)
{
    *ptr = nullptr;
    if (count == 0)
    {
        return;
    }
    checkCuda(cudaMalloc(reinterpret_cast<void **>(ptr), checkedMulSize(count, sizeof(T), label)),
              label);
}

template <typename T>
void copyToDevice(T *dst, const T *src, size_t count, const char *label)
{
    if (count == 0)
    {
        return;
    }
    checkCuda(cudaMemcpy(dst, src, checkedMulSize(count, sizeof(T), label), cudaMemcpyHostToDevice),
              label);
}

template <typename T>
void copyToHost(T *dst, const T *src, size_t count, const char *label)
{
    if (count == 0)
    {
        return;
    }
    checkCuda(cudaMemcpy(dst, src, checkedMulSize(count, sizeof(T), label), cudaMemcpyDeviceToHost),
              label);
}
} // namespace

struct ReebNode
{
    int vertex_id;
    float function_value;
    int up_degree;
    int down_degree;
    int component_label;
};

struct ReebArc
{
    int from_node;
    int to_node;
    float persistence;
};

__global__ __launch_bounds__(REEB_BLOCK_SIZE) void classifyVerticesKernel(
    const float *__restrict__ function_values, const int *__restrict__ adjacency_list,
    const int *__restrict__ adjacency_offsets, int num_vertices, int *__restrict__ vertex_types,
    int *__restrict__ component_labels)
{
    (void)component_labels;
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices)
    {
        return;
    }

    const float value = function_values[v];
    const int start = adjacency_offsets[v];
    const int end = adjacency_offsets[v + 1];

    int num_lower = 0;
    int num_higher = 0;
    for (int i = start; i < end; ++i)
    {
        const int neighbor = adjacency_list[i];
        const float neighbor_value = function_values[neighbor];
        num_lower += static_cast<int>(neighbor_value < value);
        num_higher += static_cast<int>(neighbor_value > value);
    }

    if (num_lower == 0 && num_higher > 0)
    {
        vertex_types[v] = 1;
        return;
    }
    if (num_higher == 0 && num_lower > 0)
    {
        vertex_types[v] = 2;
        return;
    }
    if (num_lower > 0 && num_higher > 0)
    {
        if (num_lower == 1 && num_higher >= 2)
        {
            vertex_types[v] = 3;
        }
        else if (num_lower >= 2 && num_higher == 1)
        {
            vertex_types[v] = 4;
        }
        else
        {
            vertex_types[v] = 3;
        }
        return;
    }
    vertex_types[v] = 0;
}

__global__ __launch_bounds__(REEB_BLOCK_SIZE) void connectedComponentsKernel(
    const int *__restrict__ adjacency_list, const int *__restrict__ adjacency_offsets,
    int num_vertices, int *__restrict__ labels)
{
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices)
    {
        return;
    }

    const int current = labels[v];
    const int start = adjacency_offsets[v];
    const int end = adjacency_offsets[v + 1];
    for (int i = start; i < end; ++i)
    {
        const int neighbor = adjacency_list[i];
        const int neighbor_label = labels[neighbor];
        if (neighbor_label < current)
        {
            atomicMin(&labels[v], neighbor_label);
        }
    }
}

__global__ __launch_bounds__(REEB_BLOCK_SIZE) void constructArcsKernel(
    const int *__restrict__ critical_points, const int *__restrict__ vertex_types, int num_critical,
    const int *__restrict__ adjacency_list, const int *__restrict__ adjacency_offsets,
    const float *__restrict__ function_values, ReebArc *__restrict__ arcs,
    int *__restrict__ arc_count, int max_arcs)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_critical)
    {
        return;
    }

    const int start_point = critical_points[idx];
    const float start_value = function_values[start_point];

    int current = start_point;
    int previous = -1;
    while (true)
    {
        const int begin = adjacency_offsets[current];
        const int finish = adjacency_offsets[current + 1];

        int next = -1;
        float next_value = start_value;
        for (int i = begin; i < finish; ++i)
        {
            const int neighbor = adjacency_list[i];
            if (neighbor == previous)
            {
                continue;
            }
            const float candidate = function_values[neighbor];
            if (candidate > next_value)
            {
                next_value = candidate;
                next = neighbor;
            }
        }

        if (next == -1)
        {
            break;
        }

        previous = current;
        current = next;
        const int type = vertex_types[current];
        const bool is_critical = (type == 1 || type == 2 || type == 3 || type == 4);
        if (is_critical || current == previous)
        {
            const int out_idx = atomicAdd(arc_count, 1);
            if (out_idx < max_arcs)
            {
                arcs[out_idx].from_node = start_point;
                arcs[out_idx].to_node = current;
                arcs[out_idx].persistence = next_value - start_value;
            }
            break;
        }
    }
}

class GPUReebGraphConstructor
{
public:
    explicit GPUReebGraphConstructor(int max_vertices)
        : max_vertices_(max_vertices)
    {
        if (max_vertices_ <= 0)
        {
            throw std::invalid_argument("max_vertices must be positive");
        }
        try
        {
            const size_t vertex_count = static_cast<size_t>(max_vertices_);
            adjacency_capacity_ =
                checkedMulSize(vertex_count, kMaxAdjacencyPerVertex, "Reeb adjacency capacity");
            allocateDevice(&d_function_values_, vertex_count, "allocate Reeb function values");
            allocateDevice(&d_adjacency_list_, adjacency_capacity_, "allocate Reeb adjacency");
            allocateDevice(&d_adjacency_offsets_, vertex_count + 1, "allocate Reeb offsets");
            allocateDevice(&d_vertex_types_, vertex_count, "allocate Reeb vertex types");
            allocateDevice(&d_component_labels_, vertex_count, "allocate Reeb component labels");
            allocateDevice(&d_arcs_, vertex_count, "allocate Reeb arcs");
            allocateDevice(&d_arc_count_, 1, "allocate Reeb arc count");
        }
        catch (...)
        {
            cleanup();
            throw;
        }
    }

    ~GPUReebGraphConstructor() { cleanup(); }

    std::pair<std::vector<ReebNode>, std::vector<ReebArc>>
    constructReebGraph(const std::vector<float> &function_values,
                       const std::vector<std::vector<int>> &adjacency)
    {
        const int n = checkedIntSize(function_values.size(), "Reeb vertex count");
        if (n <= 0 || n > max_vertices_ || adjacency.size() != function_values.size())
        {
            throw std::invalid_argument("Reeb graph input sizes are invalid");
        }
        for (float value : function_values)
        {
            if (!std::isfinite(value))
            {
                throw std::invalid_argument("Reeb function values must be finite");
            }
        }
        validateAdjacency(adjacency, n);
        copyToDevice(d_function_values_, function_values.data(), function_values.size(),
                     "copy Reeb function values");

        buildAdjacencyGPU(adjacency, n);
        classifyVertices(n);
        computeConnectedComponents(n);

        const std::vector<int> critical_points = extractCriticalPoints(n);
        const std::vector<ReebArc> arcs = constructArcs(critical_points);
        const std::vector<ReebNode> nodes = buildNodes(critical_points, function_values);
        return {nodes, arcs};
    }

    std::pair<std::vector<ReebNode>, std::vector<ReebArc>>
    simplifyReebGraph(const std::vector<ReebNode> &nodes, const std::vector<ReebArc> &arcs,
                      float persistence_threshold)
    {
        if (!std::isfinite(persistence_threshold))
        {
            throw std::invalid_argument("persistence threshold must be finite");
        }
        std::vector<ReebArc> kept_arcs;
        kept_arcs.reserve(arcs.size());
        const int node_count = checkedIntSize(nodes.size(), "Reeb node count");
        for (const ReebArc &arc : arcs)
        {
            if (arc.persistence >= persistence_threshold && arc.from_node >= 0 &&
                arc.from_node < node_count && arc.to_node >= 0 && arc.to_node < node_count)
            {
                kept_arcs.push_back(arc);
            }
        }

        std::vector<bool> used(nodes.size(), false);
        for (const ReebArc &arc : kept_arcs)
        {
            used[static_cast<size_t>(arc.from_node)] = true;
            used[static_cast<size_t>(arc.to_node)] = true;
        }

        std::vector<ReebNode> compact_nodes;
        compact_nodes.reserve(nodes.size());
        std::vector<int> remap(nodes.size(), -1);
        for (size_t i = 0; i < nodes.size(); ++i)
        {
            if (used[i])
            {
                remap[i] = checkedIntSize(compact_nodes.size(), "Reeb compact node index");
                compact_nodes.push_back(nodes[i]);
            }
        }

        for (ReebArc &arc : kept_arcs)
        {
            arc.from_node = remap[static_cast<size_t>(arc.from_node)];
            arc.to_node = remap[static_cast<size_t>(arc.to_node)];
        }
        return {compact_nodes, kept_arcs};
    }

    std::vector<ReebArc> computeMergeTree(const std::vector<float> &function_values,
                                          bool compute_join_tree = false)
    {
        std::vector<ReebArc> merge_tree;
        const int n = checkedIntSize(function_values.size(), "Reeb merge tree vertex count");
        if (n <= 1)
        {
            return merge_tree;
        }

        std::vector<int> order(static_cast<size_t>(n));
        std::iota(order.begin(), order.end(), 0);
        if (compute_join_tree)
        {
            std::sort(order.begin(), order.end(), [&](int a, int b) {
                return function_values[static_cast<size_t>(a)] <
                       function_values[static_cast<size_t>(b)];
            });
        }
        else
        {
            std::sort(order.begin(), order.end(), [&](int a, int b) {
                return function_values[static_cast<size_t>(a)] >
                       function_values[static_cast<size_t>(b)];
            });
        }

        merge_tree.reserve(static_cast<size_t>(n - 1));
        for (int i = 1; i < n; ++i)
        {
            ReebArc arc{};
            arc.from_node = order[static_cast<size_t>(i - 1)];
            arc.to_node = order[static_cast<size_t>(i)];
            arc.persistence = std::abs(function_values[static_cast<size_t>(arc.to_node)] -
                                       function_values[static_cast<size_t>(arc.from_node)]);
            merge_tree.push_back(arc);
        }
        return merge_tree;
    }

private:
    void validateAdjacency(const std::vector<std::vector<int>> &adjacency, int n) const
    {
        size_t total_edges = 0;
        for (const auto &neighbors : adjacency)
        {
            if (neighbors.size() > std::numeric_limits<size_t>::max() - total_edges)
            {
                throw std::length_error("Reeb adjacency edge count exceeds size_t limits");
            }
            total_edges += neighbors.size();
            for (int neighbor : neighbors)
            {
                if (neighbor < 0 || neighbor >= n)
                {
                    throw std::out_of_range("Reeb adjacency contains invalid vertex id");
                }
            }
        }
        if (total_edges > adjacency_capacity_)
        {
            throw std::length_error("Reeb adjacency exceeds configured GPU capacity");
        }
        checkedIntSize(total_edges, "Reeb adjacency edge count");
    }

    void buildAdjacencyGPU(const std::vector<std::vector<int>> &adjacency, int n)
    {
        std::vector<int> offsets(static_cast<size_t>(n) + 1, 0);
        std::vector<int> flat;
        flat.reserve(std::min(adjacency_capacity_, static_cast<size_t>(n) * 4));

        offsets[0] = 0;
        for (int i = 0; i < n; ++i)
        {
            const auto &neighbors = adjacency[static_cast<size_t>(i)];
            flat.insert(flat.end(), neighbors.begin(), neighbors.end());
            offsets[static_cast<size_t>(i + 1)] =
                checkedIntSize(flat.size(), "Reeb adjacency offset");
        }

        copyToDevice(d_adjacency_offsets_, offsets.data(), offsets.size(), "copy Reeb offsets");
        copyToDevice(d_adjacency_list_, flat.data(), flat.size(), "copy Reeb adjacency");
    }

    void classifyVertices(int n)
    {
        const int blocks = checkedGridBlocks(static_cast<size_t>(n), "Reeb classify grid");
        classifyVerticesKernel<<<blocks, REEB_BLOCK_SIZE>>>(d_function_values_, d_adjacency_list_,
                                                            d_adjacency_offsets_, n,
                                                            d_vertex_types_, d_component_labels_);
        checkCuda(cudaPeekAtLastError(), "launch Reeb classify kernel");
        checkCuda(cudaDeviceSynchronize(), "synchronize Reeb classify kernel");
    }

    void computeConnectedComponents(int n)
    {
        std::vector<int> labels(static_cast<size_t>(n));
        std::iota(labels.begin(), labels.end(), 0);
        copyToDevice(d_component_labels_, labels.data(), labels.size(), "copy Reeb labels");

        const int blocks = checkedGridBlocks(static_cast<size_t>(n), "Reeb component grid");
        for (int iter = 0; iter < 10; ++iter)
        {
            connectedComponentsKernel<<<blocks, REEB_BLOCK_SIZE>>>(
                d_adjacency_list_, d_adjacency_offsets_, n, d_component_labels_);
            checkCuda(cudaPeekAtLastError(), "launch Reeb component kernel");
        }
        checkCuda(cudaDeviceSynchronize(), "synchronize Reeb component kernels");
    }

    std::vector<int> extractCriticalPoints(int n)
    {
        std::vector<int> types(static_cast<size_t>(n));
        copyToHost(types.data(), d_vertex_types_, types.size(), "copy Reeb vertex types");

        std::vector<int> critical;
        critical.reserve(static_cast<size_t>(n / 2));
        for (int i = 0; i < n; ++i)
        {
            if (types[static_cast<size_t>(i)] != 0)
            {
                critical.push_back(i);
            }
        }
        return critical;
    }

    std::vector<ReebArc> constructArcs(const std::vector<int> &critical_points)
    {
        if (critical_points.empty())
        {
            return {};
        }
        checkedIntSize(critical_points.size(), "Reeb critical point count");

        int *d_critical = nullptr;
        try
        {
            allocateDevice(&d_critical, critical_points.size(), "allocate Reeb critical points");
            copyToDevice(d_critical, critical_points.data(), critical_points.size(),
                         "copy Reeb critical points");

            checkCuda(cudaMemset(d_arc_count_, 0, sizeof(int)), "reset Reeb arc count");
            const int blocks = checkedGridBlocks(critical_points.size(), "Reeb arc grid");
            constructArcsKernel<<<blocks, REEB_BLOCK_SIZE>>>(
                d_critical, d_vertex_types_, static_cast<int>(critical_points.size()),
                d_adjacency_list_, d_adjacency_offsets_, d_function_values_, d_arcs_, d_arc_count_,
                max_vertices_);
            checkCuda(cudaPeekAtLastError(), "launch Reeb arc kernel");
            checkCuda(cudaDeviceSynchronize(), "synchronize Reeb arc kernel");

            int arc_count = 0;
            copyToHost(&arc_count, d_arc_count_, 1, "copy Reeb arc count");
            arc_count = std::clamp(arc_count, 0, max_vertices_);

            std::vector<ReebArc> arcs(static_cast<size_t>(arc_count));
            copyToHost(arcs.data(), d_arcs_, arcs.size(), "copy Reeb arcs");
            cudaFree(d_critical);
            return arcs;
        }
        catch (...)
        {
            if (d_critical)
                cudaFree(d_critical);
            throw;
        }
    }

    static std::vector<ReebNode> buildNodes(const std::vector<int> &critical_points,
                                            const std::vector<float> &function_values)
    {
        std::vector<ReebNode> nodes;
        nodes.reserve(critical_points.size());
        for (const int v : critical_points)
        {
            ReebNode node{};
            node.vertex_id = v;
            node.function_value = function_values[static_cast<size_t>(v)];
            node.up_degree = 0;
            node.down_degree = 0;
            node.component_label = 0;
            nodes.push_back(node);
        }
        return nodes;
    }

    void cleanup() noexcept
    {
        if (d_function_values_)
            cudaFree(d_function_values_);
        if (d_adjacency_list_)
            cudaFree(d_adjacency_list_);
        if (d_adjacency_offsets_)
            cudaFree(d_adjacency_offsets_);
        if (d_vertex_types_)
            cudaFree(d_vertex_types_);
        if (d_component_labels_)
            cudaFree(d_component_labels_);
        if (d_arcs_)
            cudaFree(d_arcs_);
        if (d_arc_count_)
            cudaFree(d_arc_count_);
        d_function_values_ = nullptr;
        d_adjacency_list_ = nullptr;
        d_adjacency_offsets_ = nullptr;
        d_vertex_types_ = nullptr;
        d_component_labels_ = nullptr;
        d_arcs_ = nullptr;
        d_arc_count_ = nullptr;
    }

    int max_vertices_ = 0;
    size_t adjacency_capacity_ = 0;
    float *d_function_values_ = nullptr;
    int *d_adjacency_list_ = nullptr;
    int *d_adjacency_offsets_ = nullptr;
    int *d_vertex_types_ = nullptr;
    int *d_component_labels_ = nullptr;
    ReebArc *d_arcs_ = nullptr;
    int *d_arc_count_ = nullptr;
};

} // namespace gpu
} // namespace specialized
} // namespace nerve
