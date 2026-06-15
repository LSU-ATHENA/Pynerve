#include "nerve/autodiff/autodiff.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace nerve::autodiff
{
namespace
{

std::string tensorKey(const Tensor &tensor)
{
    std::string key = "shape";
    for (Size dim : tensor.shape())
    {
        key += ":" + std::to_string(dim);
    }
    key += "|data";
    for (double value : tensor.data())
    {
        if (!std::isfinite(value))
        {
            throw std::invalid_argument("computational graph tensor contains a non-finite value");
        }
        key += ":" + std::to_string(value);
    }
    return key;
}

bool sameTensor(const Tensor &left, const Tensor &right)
{
    return left.shape() == right.shape() && left.data() == right.data();
}

Size findNode(const std::vector<Tensor> &nodes, const Tensor &tensor)
{
    for (Size i = 0; i < nodes.size(); ++i)
    {
        if (sameTensor(nodes[i], tensor))
        {
            return i;
        }
    }
    return nodes.size();
}

Size ensureNode(std::vector<Tensor> &nodes, const Tensor &tensor)
{
    const Size index = findNode(nodes, tensor);
    if (index != nodes.size())
    {
        return index;
    }
    nodes.push_back(tensor);
    return nodes.size() - 1;
}

std::vector<double> expandGradient(const Tensor &gradient, Size size)
{
    if (gradient.size() == size)
    {
        return gradient.data();
    }
    if (gradient.size() == 1)
    {
        return std::vector<double>(size, gradient.data().front());
    }
    throw std::invalid_argument("edge gradient shape is not broadcastable to source tensor");
}

} // namespace

void ComputationalGraph::addNode(const Tensor &tensor)
{
    ensureNode(nodes_, tensor);
}

void ComputationalGraph::addEdge(const Tensor &from, const Tensor &to, const Tensor &grad_tensor)
{
    ensureNode(nodes_, from);
    ensureNode(nodes_, to);
    expandGradient(grad_tensor, from.size());
    edges_.emplace_back(from, to);
    edge_gradients_.push_back(grad_tensor);
}

void ComputationalGraph::clear()
{
    nodes_.clear();
    edges_.clear();
    edge_gradients_.clear();
}

void ComputationalGraph::backward()
{
    if (nodes_.empty())
    {
        return;
    }

    std::vector<std::vector<double>> gradients;
    gradients.reserve(nodes_.size());
    std::vector<bool> has_outgoing(nodes_.size(), false);
    for (const auto &node : nodes_)
    {
        gradients.emplace_back(node.size(), 0.0);
    }

    for (const auto &[from, to] : edges_)
    {
        const Size from_index = findNode(nodes_, from);
        const Size to_index = findNode(nodes_, to);
        if (from_index == nodes_.size() || to_index == nodes_.size())
        {
            throw std::runtime_error("graph edge references a missing node");
        }
        has_outgoing[from_index] = true;
    }

    bool seeded_terminal = false;
    for (Size i = 0; i < nodes_.size(); ++i)
    {
        if (!has_outgoing[i])
        {
            std::fill(gradients[i].begin(), gradients[i].end(), 1.0);
            seeded_terminal = true;
        }
    }
    if (!seeded_terminal)
    {
        std::fill(gradients.back().begin(), gradients.back().end(), 1.0);
    }

    for (Size edge_index = edges_.size(); edge_index > 0; --edge_index)
    {
        const auto &[from, to] = edges_[edge_index - 1];
        const Size from_index = findNode(nodes_, from);
        const Size to_index = findNode(nodes_, to);
        const auto local = expandGradient(edge_gradients_[edge_index - 1], from.size());
        const double downstream =
            gradients[to_index].empty()
                ? 0.0
                : std::accumulate(gradients[to_index].begin(), gradients[to_index].end(), 0.0) /
                      static_cast<double>(gradients[to_index].size());
        for (Size i = 0; i < gradients[from_index].size(); ++i)
        {
            gradients[from_index][i] += downstream * local[i];
        }
    }

    for (Size i = 0; i < nodes_.size(); ++i)
    {
        nodes_[i].setGrad(Tensor(gradients[i], nodes_[i].shape()));
    }
}

void ComputationalGraph::zeroGrad()
{
    for (Tensor &node : nodes_)
    {
        node.zeroGrad();
    }
}

std::vector<Tensor> ComputationalGraph::getParameters() const
{
    return nodes_;
}

std::vector<std::pair<Tensor, Tensor>> ComputationalGraph::getEdges() const
{
    return edges_;
}

std::vector<Tensor> ComputationalGraph::getEdgeGradients() const
{
    return edge_gradients_;
}

void ComputationalGraph::optimize()
{
    std::vector<Tensor> compact_nodes;
    compact_nodes.reserve(nodes_.size());
    for (const auto &node : nodes_)
    {
        ensureNode(compact_nodes, node);
    }

    std::unordered_map<std::string, Size> first_edge;
    std::vector<std::pair<Tensor, Tensor>> compact_edges;
    std::vector<Tensor> compact_gradients;
    for (Size i = 0; i < edges_.size(); ++i)
    {
        const auto &[from, to] = edges_[i];
        const Size from_index = findNode(compact_nodes, from);
        const Size to_index = findNode(compact_nodes, to);
        if (from_index == compact_nodes.size() || to_index == compact_nodes.size())
        {
            continue;
        }

        const std::string key =
            tensorKey(from) + "->" + tensorKey(to) + ":" + tensorKey(edge_gradients_[i]);
        if (first_edge.emplace(key, compact_edges.size()).second)
        {
            compact_edges.emplace_back(compact_nodes[from_index], compact_nodes[to_index]);
            compact_gradients.push_back(edge_gradients_[i]);
        }
    }

    nodes_ = std::move(compact_nodes);
    edges_ = std::move(compact_edges);
    edge_gradients_ = std::move(compact_gradients);
}

} // namespace nerve::autodiff
