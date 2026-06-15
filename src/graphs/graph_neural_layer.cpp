
#include "nerve/graphs/graph.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

namespace nerve::graphs
{

GraphNeuralLayer::GraphNeuralLayer(const Graph &graph, Size input_dim, Size output_dim)
    : graph_(&graph)
    , input_dim_(input_dim)
    , output_dim_(output_dim)
    , adjacency_matrix_(graph.getAdjacencyMatrix())
{
    initializeWeights();
}

std::vector<std::vector<double>>
GraphNeuralLayer::forward(const std::vector<std::vector<double>> &input) const
{
    auto messages = computeMessagePassing(input);
    return graphConvolution(messages, weights_);
}

std::vector<std::vector<double>>
GraphNeuralLayer::backward(const std::vector<std::vector<double>> &grad_output) const
{
    if (graph_ == nullptr || grad_output.empty())
    {
        return {};
    }

    std::vector<std::vector<double>> grad_messages(grad_output.size(),
                                                   std::vector<double>(input_dim_, 0.0));
    for (Size i = 0; i < grad_output.size(); ++i)
    {
        for (Size input_d = 0; input_d < input_dim_; ++input_d)
        {
            if (input_d >= weights_.size())
            {
                continue;
            }
            const Size width = std::min(grad_output[i].size(), weights_[input_d].size());
            for (Size output_d = 0; output_d < width; ++output_d)
            {
                grad_messages[i][input_d] += grad_output[i][output_d] * weights_[input_d][output_d];
            }
        }
    }

    std::vector<std::vector<double>> grad_input(grad_messages.size(),
                                                std::vector<double>(input_dim_, 0.0));
    for (Size u = 0; u < grad_messages.size(); ++u)
    {
        if (u >= graph_->numVertices())
        {
            for (Size d = 0; d < input_dim_; ++d)
            {
                grad_input[u][d] += grad_messages[u][d];
            }
            continue;
        }
        const auto nbrs = graph_->getNeighbors(static_cast<Index>(u));
        for (Size d = 0; d < input_dim_; ++d)
        {
            Size contributors = 1;
            for (Index v : nbrs)
            {
                if (v >= 0 && static_cast<Size>(v) < grad_input.size())
                {
                    ++contributors;
                }
            }

            const double share = grad_messages[u][d] / static_cast<double>(contributors);
            grad_input[u][d] += share;
            for (Index v : nbrs)
            {
                if (v >= 0 && static_cast<Size>(v) < grad_input.size())
                {
                    grad_input[static_cast<Size>(v)][d] += share;
                }
            }
        }
    }
    return grad_input;
}

std::vector<std::vector<double>>
GraphNeuralLayer::graphConvolution(const std::vector<std::vector<double>> &input,
                                   const std::vector<std::vector<double>> &weights) const
{
    if (input.empty() || weights.empty())
    {
        return {};
    }
    std::vector<std::vector<double>> out(input.size(), std::vector<double>(weights[0].size(), 0.0));
    for (Size i = 0; i < input.size(); ++i)
    {
        for (Size j = 0; j < weights[0].size(); ++j)
        {
            for (Size k = 0; k < std::min(input[i].size(), weights.size()); ++k)
            {
                out[i][j] += input[i][k] * weights[k][j];
            }
        }
    }
    return out;
}

std::vector<std::vector<double>>
GraphNeuralLayer::graphAttention(const std::vector<std::vector<double>> &input,
                                 const std::vector<std::vector<double>> &queries,
                                 const std::vector<std::vector<double>> &keys,
                                 const std::vector<std::vector<double>> &values) const
{
    const Size n = input.size();
    if (graph_ == nullptr || n == 0 || queries.size() != n || keys.size() != n ||
        values.size() != n)
    {
        return {};
    }

    const Size output_dim = values[0].size();
    std::vector<std::vector<double>> output(n, std::vector<double>(output_dim, 0.0));

    /*
     * Scaled dot-product attention over the 1-hop neighborhood:
     * for each vertex i, score i and all neighbors j via q_i*k_j/sqrt(d_k),
     * normalize with softmax, then aggregate the value vectors v_j.
     */
    for (Size i = 0; i < n; ++i)
    {
        std::vector<Index> neighbors = graph_->getNeighbors(static_cast<Index>(i));
        neighbors.push_back(static_cast<Index>(i));

        std::vector<double> logits;
        logits.reserve(neighbors.size());
        for (Index j : neighbors)
        {
            if (j < 0 || static_cast<Size>(j) >= n)
            {
                logits.push_back(-std::numeric_limits<double>::infinity());
                continue;
            }
            const auto &q = queries[i];
            const auto &k = keys[static_cast<Size>(j)];
            const Size dk = std::min(q.size(), k.size());
            if (dk == 0)
            {
                logits.push_back(0.0);
                continue;
            }
            double dot = 0.0;
            for (Size d = 0; d < dk; ++d)
            {
                dot += q[d] * k[d];
            }
            logits.push_back(dot / std::sqrt(static_cast<double>(dk)));
        }

        const double max_logit = *std::max_element(logits.begin(), logits.end());
        double denom = 0.0;
        std::vector<double> weights_local(logits.size(), 0.0);
        for (Size t = 0; t < logits.size(); ++t)
        {
            weights_local[t] = std::exp(logits[t] - max_logit);
            denom += weights_local[t];
        }
        if (denom <= 0.0)
        {
            continue;
        }

        for (Size t = 0; t < neighbors.size(); ++t)
        {
            const Index j = neighbors[t];
            if (j < 0 || static_cast<Size>(j) >= n)
            {
                continue;
            }
            const double alpha = weights_local[t] / denom;
            const auto &v = values[static_cast<Size>(j)];
            const Size dim = std::min(output_dim, v.size());
            for (Size d = 0; d < dim; ++d)
            {
                output[i][d] += alpha * v[d];
            }
        }
    }
    return output;
}

void GraphNeuralLayer::initializeWeights()
{
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(-0.1, 0.1);
    weights_.assign(input_dim_, std::vector<double>(output_dim_, 0.0));
    for (auto &row : weights_)
    {
        for (double &w : row)
        {
            w = dist(rng);
        }
    }
}

std::vector<std::vector<double>>
GraphNeuralLayer::computeMessagePassing(const std::vector<std::vector<double>> &input) const
{
    if (graph_ == nullptr || input.empty())
    {
        return {};
    }
    std::vector<std::vector<double>> out = input;
    for (Size u = 0; u < input.size(); ++u)
    {
        if (u >= graph_->numVertices())
        {
            continue;
        }
        const auto nbrs = graph_->getNeighbors(static_cast<Index>(u));
        if (nbrs.empty())
        {
            continue;
        }
        for (Size d = 0; d < out[u].size(); ++d)
        {
            double acc = input[u][d];
            Size contributors = 1;
            for (Index v : nbrs)
            {
                if (v < 0 || static_cast<Size>(v) >= input.size() ||
                    d >= input[static_cast<Size>(v)].size())
                {
                    continue;
                }
                acc += input[static_cast<Size>(v)][d];
                ++contributors;
            }
            out[u][d] = acc / static_cast<double>(contributors);
        }
    }
    return out;
}

} // namespace nerve::graphs
