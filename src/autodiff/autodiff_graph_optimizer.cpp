#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

#include "nerve/core.hpp"

namespace nerve::autodiff::graph
{

struct Node
{
    enum class Type
    {
        INPUT,
        OUTPUT,
        ADD,
        MUL,
        DIV,
        POW,
        EXP,
        LOG,
        SIN,
        COS,
        MATMUL,
        REDUCE_SUM,
        REDUCE_MEAN,
        PERSISTENCE,
        CUSTOM
    };

    int id;
    Type type;
    std::vector<int> inputs;
    std::vector<int> outputs;

    bool fused = false;
    int fused_parent = -1;

    std::string name;
    std::vector<size_t> shape;

    Node(int _id, Type _type)
        : id(_id)
        , type(_type)
    {}
};

class ComputationGraph
{
public:
    std::vector<std::unique_ptr<Node>> nodes;
    std::unordered_map<int, int> id_to_index;

    int addNode(Node::Type type, const std::vector<int> &inputs = {})
    {
        int id = nodes.size();
        auto node = std::make_unique<Node>(id, type);
        node->inputs = inputs;

        for (int inp : inputs)
        {
            nodes[id_to_index[inp]]->outputs.push_back(id);
        }

        nodes.push_back(std::move(node));
        id_to_index[id] = id;
        return id;
    }

    Node *getNode(int id)
    {
        auto it = id_to_index.find(id);
        if (it != id_to_index.end())
        {
            return nodes[it->second].get();
        }
        return nullptr;
    }

    const Node *getNode(int id) const
    {
        auto it = id_to_index.find(id);
        if (it != id_to_index.end())
        {
            return nodes[it->second].get();
        }
        return nullptr;
    }
};

class GraphOptimizer
{
public:
    void fuseOperations(ComputationGraph &graph)
    {
        std::unordered_set<int> visited;

        for (auto &node : graph.nodes)
        {
            if (node->fused || visited.count(node->id))
                continue;

            if (canFuseChain(node.get(), graph))
            {
                fuseChain(graph, node.get(), visited);
            }
        }
    }

    void eliminateCommonSubexpressions(ComputationGraph &graph)
    {
        std::unordered_map<std::string, int> expr_map;

        std::vector<int> to_remove;

        for (auto &node : graph.nodes)
        {
            std::string sig = getNodeSignature(node.get());

            auto it = expr_map.find(sig);
            if (it != expr_map.end())
            {
                int original_id = it->second;
                redirectOutputs(graph, node->id, original_id);
                to_remove.push_back(node->id);
            }
            else
            {
                expr_map[sig] = node->id;
            }
        }

        for (int id : to_remove)
        {
            if (auto *node = graph.getNode(id))
            {
                node->type = Node::Type::CUSTOM;
                node->name = "dead";
            }
        }
    }

    void eliminateDeadCode(ComputationGraph &graph)
    {
        std::unordered_set<int> reachable;
        std::queue<int> to_visit;

        for (auto &node : graph.nodes)
        {
            if (node->type == Node::Type::OUTPUT)
            {
                to_visit.push(node->id);
                reachable.insert(node->id);
            }
        }

        while (!to_visit.empty())
        {
            int id = to_visit.front();
            to_visit.pop();

            const Node *node = graph.getNode(id);
            if (!node)
                continue;

            for (int inp : node->inputs)
            {
                if (!reachable.count(inp))
                {
                    reachable.insert(inp);
                    to_visit.push(inp);
                }
            }
        }

        for (auto &node : graph.nodes)
        {
            if (!reachable.count(node->id))
            {
                node->type = Node::Type::CUSTOM;
                node->name = "dead";
            }
        }
    }

    std::vector<int> topologicalSort(const ComputationGraph &graph)
    {
        std::vector<int> result;
        std::unordered_map<int, int> in_degree;

        for (const auto &node : graph.nodes)
        {
            in_degree[node->id] = node->inputs.size();
        }

        std::queue<int> ready;
        for (const auto &[id, degree] : in_degree)
        {
            if (degree == 0)
            {
                ready.push(id);
            }
        }

        while (!ready.empty())
        {
            int id = ready.front();
            ready.pop();
            result.push_back(id);

            const Node *node = graph.getNode(id);
            if (!node)
                continue;

            for (int out_id : node->outputs)
            {
                in_degree[out_id]--;
                if (in_degree[out_id] == 0)
                {
                    ready.push(out_id);
                }
            }
        }

        return result;
    }

    void optimize(ComputationGraph &graph)
    {
        eliminateCommonSubexpressions(graph);
        eliminateDeadCode(graph);
        fuseOperations(graph);
    }

private:
    bool canFuseChain(const Node *node, const ComputationGraph &graph)
    {
        if (node->outputs.size() != 1)
            return false;

        int child_id = node->outputs[0];
        const Node *child = graph.getNode(child_id);

        if (!child || child->inputs.size() != 1)
            return false;
        if (child->inputs[0] != node->id)
            return false;

        return isElementWise(node->type) && isElementWise(child->type);
    }

    bool isElementWise(Node::Type type)
    {
        return type == Node::Type::ADD || type == Node::Type::MUL || type == Node::Type::DIV ||
               type == Node::Type::POW || type == Node::Type::EXP || type == Node::Type::LOG ||
               type == Node::Type::SIN || type == Node::Type::COS;
    }

    void fuseChain(ComputationGraph &graph, Node *start_node, std::unordered_set<int> &visited)
    {
        std::vector<int> chain;
        chain.push_back(start_node->id);

        int current = start_node->id;
        while (true)
        {
            visited.insert(current);
            Node *node = graph.getNode(current);
            if (!node || node->outputs.size() != 1)
            {
                break;
            }

            const int next = node->outputs[0];
            Node *next_node = graph.getNode(next);
            if (!next_node || next_node->inputs.size() != 1 || next_node->inputs[0] != current ||
                !isElementWise(next_node->type))
            {
                break;
            }

            chain.push_back(next);
            current = next;
        }

        if (chain.size() < 2)
        {
            return;
        }

        const int group_id = chain.front();
        const int tail_id = chain.back();
        Node *group_node = graph.getNode(group_id);
        Node *tail_node = graph.getNode(tail_id);
        if (!group_node || !tail_node)
        {
            return;
        }

        std::vector<int> tail_outputs = tail_node->outputs;
        group_node->outputs = tail_outputs;
        for (int out_id : tail_outputs)
        {
            Node *out_node = graph.getNode(out_id);
            if (!out_node)
            {
                continue;
            }
            for (int &inp : out_node->inputs)
            {
                if (inp == tail_id)
                {
                    inp = group_id;
                }
            }
        }

        for (size_t idx = 1; idx < chain.size(); ++idx)
        {
            Node *fused_node = graph.getNode(chain[idx]);
            if (!fused_node)
            {
                continue;
            }
            fused_node->fused = true;
            fused_node->fused_parent = group_id;
            fused_node->type = Node::Type::CUSTOM;
            fused_node->name = "fused";
            fused_node->inputs.clear();
            fused_node->outputs.clear();
        }
    }

    std::string getNodeSignature(const Node *node)
    {
        std::string sig = std::to_string(static_cast<int>(node->type)) + ":";
        for (int inp : node->inputs)
        {
            sig += std::to_string(inp) + ",";
        }
        return sig;
    }

    void redirectOutputs(ComputationGraph &graph, int from_id, int to_id)
    {
        Node *from_node = graph.getNode(from_id);
        Node *to_node = graph.getNode(to_id);

        if (!from_node || !to_node)
            return;

        for (int out_id : from_node->outputs)
        {
            Node *out_node = graph.getNode(out_id);
            if (out_node)
            {
                for (int &inp : out_node->inputs)
                {
                    if (inp == from_id)
                    {
                        inp = to_id;
                    }
                }
            }
            to_node->outputs.push_back(out_id);
        }

        from_node->outputs.clear();
    }
};

class FusedKernelLauncher
{
private:
    static constexpr int FUSION_BLOCK_SIZE = 256;
#ifdef __CUDACC__
    using FusedCudaStream = cudaStream_t;
#else
    using FusedCudaStream = void *;
#endif
public:
    void launchFusedElementWise(const std::vector<int> &op_types,
                                const std::vector<float *> &inputs, float *output, int n,
                                FusedCudaStream stream)
    {
#ifdef __CUDACC__
        int threads = FUSION_BLOCK_SIZE;
        int blocks = (n + threads - 1) / threads;

        fusedElementWiseKernel<<<blocks, threads, 0, stream>>>(op_types.data(), inputs.data(),
                                                               output, n, op_types.size());
        if (const cudaError_t launch_status = cudaPeekAtLastError(); launch_status != cudaSuccess)
        {
            throw std::runtime_error(std::string("fusedElementWiseKernel launch failed: ") +
                                     cudaGetErrorString(launch_status));
        }
#else
        (void)stream;
        for (size_t op = 0; op < op_types.size(); ++op)
        {
            for (int i = 0; i < n; ++i)
            {
                applyElementWiseOp(op_types[op], inputs[op], output, i);
            }
        }
#endif
    }

#ifdef __CUDACC__
    __global__ void fusedElementWiseKernel(const int *op_types, const float *const *inputs,
                                           float *output, int n, int num_ops)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n)
            return;

        float val = inputs[0][idx];
        for (int i = 1; i < num_ops; ++i)
        {
            float other = inputs[i][idx];
            switch (op_types[i])
            {
                case 0:
                    val = val + other;
                    break;
                case 1:
                    val = val * other;
                    break;
                case 2:
                    val = fmaxf(val, other);
                    break;
                case 3:
                    val = fminf(val, other);
                    break;
                default:
                    break;
            }
        }
        output[idx] = val;
    }
#endif

    void applyElementWiseOp(int op_type, const float *input, float *output, int idx)
    {
        switch (op_type)
        {
            case 0:
                output[idx] = output[idx] + input[idx];
                break;
            case 1:
                output[idx] = output[idx] * input[idx];
                break;
            case 2:
                output[idx] = std::fmax(output[idx], input[idx]);
                break;
            case 3:
                output[idx] = std::fmin(output[idx], input[idx]);
                break;
            default:
                break;
        }
    }
};

} // namespace nerve::autodiff::graph
