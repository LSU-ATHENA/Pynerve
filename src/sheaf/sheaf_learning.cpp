// Sheaf learning implementation for closed-form edge-wise restriction maps.
// This unit provides concrete algorithms for map estimation, variation scoring,
// and block-Laplacian construction from learned maps.

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

constexpr double kRidgeRegularization = 1e-9;
constexpr double kSparseTolerance = 1e-13;

struct DirectedEdge
{
    Eigen::Index src;
    Eigen::Index dst;
    double weight;
};

[[nodiscard]] std::vector<DirectedEdge>
collectDirectedEdges(const Eigen::SparseMatrix<double> &adjacency)
{
    std::vector<DirectedEdge> edges;
    edges.reserve(static_cast<size_t>(adjacency.nonZeros()));
    for (Eigen::Index col = 0; col < adjacency.outerSize(); ++col)
    {
        for (Eigen::SparseMatrix<double>::InnerIterator it(adjacency, col); it; ++it)
        {
            if (it.row() == it.col())
            {
                continue;
            }
            const double w = it.value();
            if (!(w > 0.0) || !std::isfinite(w))
            {
                continue;
            }
            edges.push_back(DirectedEdge{it.row(), it.col(), w});
        }
    }
    return edges;
}

void validateLearningConfig(const SheafLearningConfig &config)
{
    if (!std::isfinite(config.learning_rate) || config.learning_rate <= 0.0)
    {
        throw std::invalid_argument("Sheaf learning rate must be finite and positive");
    }
    if (config.max_iterations <= 0)
    {
        throw std::invalid_argument("Sheaf learning max_iterations must be positive");
    }
    if (!std::isfinite(config.convergence_tolerance) || config.convergence_tolerance <= 0.0)
    {
        throw std::invalid_argument(
            "Sheaf learning convergence tolerance must be finite and positive");
    }
}

bool hasOnlyFiniteValues(const Eigen::SparseMatrix<double> &matrix)
{
    for (Eigen::Index outer = 0; outer < matrix.outerSize(); ++outer)
    {
        for (Eigen::SparseMatrix<double>::InnerIterator it(matrix, outer); it; ++it)
        {
            if (!std::isfinite(it.value()))
            {
                return false;
            }
        }
    }
    return true;
}

bool hasOnlyFiniteMatrices(std::span<const Eigen::MatrixXd> matrices)
{
    return std::all_of(matrices.begin(), matrices.end(),
                       [](const Eigen::MatrixXd &matrix) { return matrix.allFinite(); });
}

bool hasCompatibleObservationData(std::span<const Eigen::MatrixXd> node_data)
{
    if (node_data.empty() || node_data.front().cols() <= 0 || node_data.front().rows() <= 0)
    {
        return false;
    }
    const Eigen::Index observations = node_data.front().cols();
    return std::all_of(node_data.begin(), node_data.end(),
                       [observations](const Eigen::MatrixXd &matrix) {
                           return matrix.rows() > 0 && matrix.cols() == observations;
                       });
}

[[nodiscard]] std::vector<size_t> buildStalkOffsets(std::span<const size_t> dims)
{
    std::vector<size_t> offsets;
    offsets.reserve(dims.size() + 1);
    offsets.push_back(0);
    for (const size_t d : dims)
    {
        offsets.push_back(offsets.back() + d);
    }
    return offsets;
}

} // namespace

bool SheafLearningResult::isValid() const noexcept
{
    if (!converged || iterations_used <= 0 || !std::isfinite(compute_time_ms) ||
        compute_time_ms < 0.0)
    {
        return false;
    }
    if (!std::isfinite(final_total_variation) || final_total_variation < 0.0)
    {
        return false;
    }
    if (learned_sheaf_laplacian.rows() != learned_sheaf_laplacian.cols())
    {
        return false;
    }
    for (int outer = 0; outer < learned_sheaf_laplacian.outerSize(); ++outer)
    {
        for (Eigen::SparseMatrix<double>::InnerIterator it(learned_sheaf_laplacian, outer); it;
             ++it)
        {
            if (!std::isfinite(it.value()))
            {
                return false;
            }
        }
    }
    return std::all_of(restriction_maps.begin(), restriction_maps.end(),
                       [](const Eigen::MatrixXd &map) { return map.allFinite(); });
}

double SheafLearningResult::getDirichletEnergy() const
{
    if (learned_sheaf_laplacian.rows() == 0)
    {
        return 0.0;
    }
    const Eigen::VectorXd ones = Eigen::VectorXd::Ones(learned_sheaf_laplacian.rows());
    return 0.5 * ones.dot(learned_sheaf_laplacian * ones);
}

SheafLearner::SheafLearner(const SheafLearningConfig &config)
    : config_(config)
{
    validateLearningConfig(config_);
}

void SheafLearner::setConfig(const SheafLearningConfig &config)
{
    validateLearningConfig(config);
    config_ = config;
}

SheafLearningResult
SheafLearner::learnSheafClosedForm(const Eigen::SparseMatrix<double> &graph_adjacency,
                                   std::span<const Eigen::MatrixXd> node_data,
                                   std::span<const size_t> node_dimensions)
{
    return learnSheafLocal(graph_adjacency, node_data, node_dimensions);
}

SheafLearningResult
SheafLearner::learnSheafLocal(const Eigen::SparseMatrix<double> &graph_adjacency,
                              std::span<const Eigen::MatrixXd> node_data,
                              std::span<const size_t> node_dimensions)
{
    const auto start = std::chrono::steady_clock::now();
    if (!validateInputs(graph_adjacency, node_data, node_dimensions))
    {
        throw std::invalid_argument("Invalid sheaf learning inputs");
    }

    const auto edges = collectDirectedEdges(graph_adjacency);
    std::vector<Eigen::MatrixXd> restriction_maps;
    restriction_maps.reserve(edges.size());
    for (const DirectedEdge &edge : edges)
    {
        restriction_maps.push_back(computeOptimalRestrictionMap(
            node_data[static_cast<size_t>(edge.src)], node_data[static_cast<size_t>(edge.dst)]));
    }

    SheafLearningResult result;
    result.restriction_maps = std::move(restriction_maps);
    result.learned_sheaf_laplacian =
        constructSheafLaplacian(graph_adjacency, node_dimensions, result.restriction_maps);
    result.final_total_variation =
        computeTotalVariation(graph_adjacency, node_data, result.restriction_maps);
    result.iterations_used = 1;
    result.converged = true;
    const auto stop = std::chrono::steady_clock::now();
    result.compute_time_ms =
        static_cast<double>(
            std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count()) /
        1000.0;
    return result;
}

Eigen::MatrixXd SheafLearner::computeOptimalRestrictionMap(const Eigen::MatrixXd &data_i,
                                                           const Eigen::MatrixXd &data_j) const
{
    if (data_i.cols() != data_j.cols())
    {
        throw std::invalid_argument("Observation count mismatch for restriction-map estimation");
    }
    if (data_i.cols() == 0 || data_i.rows() == 0 || data_j.rows() == 0)
    {
        throw std::invalid_argument("Restriction-map estimation requires non-empty data");
    }
    if (!data_i.allFinite() || !data_j.allFinite())
    {
        throw std::invalid_argument("Restriction-map data must be finite");
    }

    const Eigen::MatrixXd cross = data_i * data_j.transpose();
    Eigen::MatrixXd gram = data_j * data_j.transpose();
    gram.diagonal().array() += kRidgeRegularization;

    Eigen::LDLT<Eigen::MatrixXd> solver(gram);
    if (solver.info() != Eigen::Success)
    {
        throw std::runtime_error("Failed to factorize Gram matrix for restriction map");
    }

    const Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(data_j.rows(), data_j.rows());
    Eigen::MatrixXd result = cross * solver.solve(identity);
    if (!result.allFinite())
    {
        throw std::invalid_argument("Restriction map must remain finite");
    }
    return result;
}

double SheafLearner::computeTotalVariation(const Eigen::SparseMatrix<double> &graph_adjacency,
                                           std::span<const Eigen::MatrixXd> node_data,
                                           std::span<const Eigen::MatrixXd> restriction_maps) const
{
    if (graph_adjacency.rows() != graph_adjacency.cols() || graph_adjacency.rows() <= 0 ||
        node_data.size() != static_cast<size_t>(graph_adjacency.rows()))
    {
        throw std::invalid_argument("Sheaf total-variation inputs have incompatible dimensions");
    }
    if (!hasCompatibleObservationData(node_data))
    {
        throw std::invalid_argument("Sheaf node observations must be non-empty and aligned");
    }
    if (!hasOnlyFiniteValues(graph_adjacency) || !hasOnlyFiniteMatrices(node_data) ||
        !hasOnlyFiniteMatrices(restriction_maps))
    {
        throw std::invalid_argument("Sheaf total-variation inputs must be finite");
    }
    const auto edges = collectDirectedEdges(graph_adjacency);
    if (edges.size() != restriction_maps.size())
    {
        throw std::invalid_argument("Restriction-map count does not match graph edge count");
    }

    double total_variation = 0.0;
    for (size_t idx = 0; idx < edges.size(); ++idx)
    {
        const DirectedEdge &edge = edges[idx];
        const auto &map = restriction_maps[idx];
        const auto &xi = node_data[static_cast<size_t>(edge.src)];
        const auto &xj = node_data[static_cast<size_t>(edge.dst)];
        if (map.rows() != xi.rows() || map.cols() != xj.rows())
        {
            throw std::invalid_argument("Restriction-map dimensions do not match endpoint stalks");
        }
        const Eigen::MatrixXd residual = xi - map * xj;
        total_variation += edge.weight * residual.squaredNorm();
        if (!residual.allFinite() || !std::isfinite(total_variation))
        {
            throw std::invalid_argument("Sheaf total variation must remain finite");
        }
    }
    return total_variation;
}

Eigen::SparseMatrix<double>
SheafLearner::constructSheafLaplacian(const Eigen::SparseMatrix<double> &graph_adjacency,
                                      std::span<const size_t> node_dimensions,
                                      std::span<const Eigen::MatrixXd> restriction_maps) const
{
    if (graph_adjacency.rows() != graph_adjacency.cols() || graph_adjacency.rows() <= 0 ||
        node_dimensions.size() != static_cast<size_t>(graph_adjacency.rows()))
    {
        throw std::invalid_argument("Sheaf Laplacian inputs have incompatible dimensions");
    }
    if (std::any_of(node_dimensions.begin(), node_dimensions.end(),
                    [](size_t dim) { return dim == 0; }))
    {
        throw std::invalid_argument("Sheaf node dimensions must be positive");
    }
    if (!hasOnlyFiniteValues(graph_adjacency) || !hasOnlyFiniteMatrices(restriction_maps))
    {
        throw std::invalid_argument("Sheaf Laplacian inputs must be finite");
    }
    const auto edges = collectDirectedEdges(graph_adjacency);
    if (edges.size() != restriction_maps.size())
    {
        throw std::invalid_argument("Restriction-map count does not match graph edge count");
    }
    const std::vector<size_t> offsets = buildStalkOffsets(node_dimensions);
    const size_t total_dim = offsets.back();

    Eigen::MatrixXd dense = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(total_dim),
                                                  static_cast<Eigen::Index>(total_dim));
    for (size_t idx = 0; idx < edges.size(); ++idx)
    {
        const DirectedEdge &edge = edges[idx];
        const size_t src = static_cast<size_t>(edge.src);
        const size_t dst = static_cast<size_t>(edge.dst);
        const size_t d_src = node_dimensions[src];
        const size_t d_dst = node_dimensions[dst];
        const size_t off_src = offsets[src];
        const size_t off_dst = offsets[dst];
        const Eigen::MatrixXd &map = restriction_maps[idx];
        if (map.rows() != static_cast<Eigen::Index>(d_src) ||
            map.cols() != static_cast<Eigen::Index>(d_dst))
        {
            throw std::invalid_argument("Restriction-map dimensions do not match node dimensions");
        }

        const Eigen::Index src_i = static_cast<Eigen::Index>(off_src);
        const Eigen::Index dst_i = static_cast<Eigen::Index>(off_dst);
        const Eigen::Index d_src_i = static_cast<Eigen::Index>(d_src);
        const Eigen::Index d_dst_i = static_cast<Eigen::Index>(d_dst);
        const double w = edge.weight;

        dense.block(src_i, src_i, d_src_i, d_src_i).diagonal().array() += w;
        dense.block(src_i, dst_i, d_src_i, d_dst_i) -= w * map;
        dense.block(dst_i, src_i, d_dst_i, d_src_i) -= w * map.transpose();
        dense.block(dst_i, dst_i, d_dst_i, d_dst_i).noalias() += w * (map.transpose() * map);
        if (!dense.allFinite())
        {
            throw std::invalid_argument("Sheaf Laplacian entries must remain finite");
        }
    }

    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(total_dim * 4);
    for (Eigen::Index r = 0; r < dense.rows(); ++r)
    {
        for (Eigen::Index c = 0; c < dense.cols(); ++c)
        {
            const double v = dense(r, c);
            if (std::abs(v) > kSparseTolerance)
            {
                triplets.emplace_back(r, c, v);
            }
        }
    }

    Eigen::SparseMatrix<double> out(static_cast<Eigen::Index>(total_dim),
                                    static_cast<Eigen::Index>(total_dim));
    out.setFromTriplets(triplets.begin(), triplets.end());
    return out;
}

bool SheafLearner::validateInputs(const Eigen::SparseMatrix<double> &graph_adjacency,
                                  std::span<const Eigen::MatrixXd> node_data,
                                  std::span<const size_t> node_dimensions) const
{
    if (graph_adjacency.rows() != graph_adjacency.cols())
    {
        return false;
    }
    if (graph_adjacency.rows() <= 0)
    {
        return false;
    }
    if (!hasOnlyFiniteValues(graph_adjacency) || !hasOnlyFiniteMatrices(node_data))
    {
        return false;
    }

    const size_t node_count = static_cast<size_t>(graph_adjacency.rows());
    if (node_data.size() != node_count || node_dimensions.size() != node_count)
    {
        return false;
    }

    size_t observations = 0;
    for (size_t i = 0; i < node_count; ++i)
    {
        if (node_dimensions[i] == 0)
        {
            return false;
        }
        const auto &mat = node_data[i];
        if (mat.rows() != static_cast<Eigen::Index>(node_dimensions[i]) || mat.cols() <= 0)
        {
            return false;
        }
        if (i == 0)
        {
            observations = static_cast<size_t>(mat.cols());
        }
        else if (static_cast<size_t>(mat.cols()) != observations)
        {
            return false;
        }
    }
    return true;
}

// learnSheafFromGraphSignal and benchmarkSheafLearning split to sheaf_learning_ops.cpp

} // namespace learning
} // namespace sheaf
} // namespace nerve
