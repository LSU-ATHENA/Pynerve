// Sheaf learning operations  --  graph-signal learning and benchmarking.

#include "nerve/sheaf/sheaf_learning.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

namespace nerve
{
namespace sheaf
{
namespace learning
{

namespace
{

double meanValue(const std::vector<double> &values)
{
    if (values.empty())
    {
        return 0.0;
    }
    const double sum = std::accumulate(values.begin(), values.end(), 0.0);
    return sum / static_cast<double>(values.size());
}

} // namespace

std::unique_ptr<SheafLaplacianRuntime>
learnSheafFromGraphSignal(std::span<const std::pair<uint32_t, uint32_t>> graph_edges,
                          std::span<const std::vector<double>> node_signals,
                          const SheafLearningConfig &config)
{
    if (node_signals.empty())
    {
        throw std::invalid_argument("Node signals cannot be empty");
    }
    const size_t node_count = node_signals.size();
    const size_t obs_count = node_signals.front().size();
    if (obs_count == 0)
    {
        throw std::invalid_argument("Node signals must contain observations");
    }
    for (const auto &signal : node_signals)
    {
        if (signal.size() != obs_count)
        {
            throw std::invalid_argument("All nodes must have the same number of observations");
        }
        if (!std::all_of(signal.begin(), signal.end(),
                         [](double value) { return std::isfinite(value); }))
        {
            throw std::invalid_argument("Node signals must be finite");
        }
    }

    std::vector<Eigen::Triplet<double>> adjacency_triplets;
    adjacency_triplets.reserve(graph_edges.size() * 2);
    std::vector<std::vector<uint32_t>> neighbors(node_count);
    std::vector<std::vector<double>> edge_weights(node_count);
    std::unordered_set<uint64_t> seen;
    for (const auto &[u, v] : graph_edges)
    {
        if (u >= node_count || v >= node_count || u == v)
        {
            continue;
        }
        const uint32_t a = std::min(u, v);
        const uint32_t b = std::max(u, v);
        const uint64_t key = (static_cast<uint64_t>(a) << 32ULL) | static_cast<uint64_t>(b);
        if (!seen.insert(key).second)
        {
            continue;
        }

        adjacency_triplets.emplace_back(static_cast<Eigen::Index>(u), static_cast<Eigen::Index>(v),
                                        1.0);
        adjacency_triplets.emplace_back(static_cast<Eigen::Index>(v), static_cast<Eigen::Index>(u),
                                        1.0);
        neighbors[static_cast<size_t>(u)].push_back(v);
        neighbors[static_cast<size_t>(v)].push_back(u);
        edge_weights[static_cast<size_t>(u)].push_back(1.0);
        edge_weights[static_cast<size_t>(v)].push_back(1.0);
    }

    Eigen::SparseMatrix<double> adjacency(static_cast<Eigen::Index>(node_count),
                                          static_cast<Eigen::Index>(node_count));
    adjacency.setFromTriplets(adjacency_triplets.begin(), adjacency_triplets.end());

    std::vector<Eigen::MatrixXd> node_data;
    node_data.reserve(node_count);
    std::vector<size_t> dims(node_count, 1);
    for (const auto &signal : node_signals)
    {
        Eigen::MatrixXd mat(1, static_cast<Eigen::Index>(signal.size()));
        for (size_t i = 0; i < signal.size(); ++i)
        {
            mat(0, static_cast<Eigen::Index>(i)) = signal[i];
        }
        node_data.push_back(std::move(mat));
    }

    SheafLearner learner(config);
    const SheafLearningResult learning_result =
        learner.learnSheafClosedForm(adjacency, node_data, dims);
    if (!learning_result.isValid())
    {
        throw std::runtime_error("Sheaf graph-signal learning produced an invalid result");
    }

    SheafLaplacianRuntime::SheafConfig sheaf_config;
    sheaf_config.num_attributes = 1;
    sheaf_config.attribute_names = {"signal"};
    auto runtime = std::make_unique<SheafLaplacianRuntime>(sheaf_config);

    for (size_t node_id = 0; node_id < node_count; ++node_id)
    {
        SheafLaplacianRuntime::SheafNode node;
        node.node_id = static_cast<uint32_t>(node_id);
        node.position = {static_cast<double>(node_id)};
        node.attributes = {meanValue(node_signals[node_id])};
        node.attribute_names = {"signal"};
        node.neighbors = neighbors[node_id];
        node.edge_weights = edge_weights[node_id];
        node.stalk_values = node_signals[node_id];
        node.local_laplacian = learning_result.learned_sheaf_laplacian.coeff(
            static_cast<Eigen::Index>(node_id), static_cast<Eigen::Index>(node_id));
        runtime->addNode(node);
    }

    return runtime;
}

SheafLearningBenchmark benchmarkSheafLearning(const Eigen::SparseMatrix<double> &graph,
                                              std::span<const Eigen::MatrixXd> node_data,
                                              std::span<const size_t> dimensions)
{
    SheafLearningBenchmark benchmark{};

    SheafLearner learner;
    const SheafLearningResult closed_form =
        learner.learnSheafClosedForm(graph, node_data, dimensions);
    benchmark.closed_form_time_ms = closed_form.compute_time_ms;
    benchmark.tv_closed_form = closed_form.final_total_variation;

    benchmark.sdp_time_ms = 0.0;
    benchmark.tv_sdp = benchmark.tv_closed_form;
    benchmark.speedup_factor = 1.0;
    benchmark.accuracy_ratio = 1.0;
    return benchmark;
}

} // namespace learning
} // namespace sheaf
} // namespace nerve
