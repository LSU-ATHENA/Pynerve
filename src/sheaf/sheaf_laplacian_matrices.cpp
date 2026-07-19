#include "nerve/sheaf/sheaf_laplacian.hpp"
#include "nerve/simd/simd_base.hpp"
#include "sheaf_laplacian_detail.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#if HAS_EIGEN && __has_include(<Eigen/Sparse>) && __has_include(<Eigen/Dense>)
namespace nerve
{
namespace sheaf
{

Eigen::SparseMatrix<double> SheafLaplacianRuntime::buildTopologicalLaplacian()
{
    const Size n = nodes_.size();
    Eigen::SparseMatrix<double> laplacian(static_cast<Eigen::Index>(n),
                                          static_cast<Eigen::Index>(n));
    if (n == 0)
    {
        return laplacian;
    }
    const auto node_ids = detail::sortedNodeIds(nodes_);
    const auto index = detail::buildNodeIndexMap(node_ids);
    std::vector<Eigen::Triplet<double>> triplets;
    for (uint32_t node_id : node_ids)
    {
        const auto &node = nodes_.at(node_id);
        const Size row = index.at(node_id);
        double degree = 0.0;
        for (Size i = 0; i < node.neighbors.size() && i < node.edge_weights.size(); ++i)
        {
            const auto it = index.find(node.neighbors[i]);
            if (it == index.end())
            {
                continue;
            }
            const double weight =
                std::isfinite(node.edge_weights[i]) ? std::max(0.0, node.edge_weights[i]) : 0.0;
            degree += weight;
            triplets.emplace_back(static_cast<Eigen::Index>(row),
                                  static_cast<Eigen::Index>(it->second), -weight);
        }
        triplets.emplace_back(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(row),
                              degree);
    }
    laplacian.setFromTriplets(triplets.begin(), triplets.end());
    return laplacian;
}

Eigen::SparseMatrix<double> SheafLaplacianRuntime::buildRestrictionWeightedLaplacian()
{
    const Size n = nodes_.size();
    Eigen::SparseMatrix<double> laplacian(static_cast<Eigen::Index>(n),
                                          static_cast<Eigen::Index>(n));
    const auto node_ids = detail::sortedNodeIds(nodes_);
    const auto index = detail::buildNodeIndexMap(node_ids);
    std::vector<Eigen::Triplet<double>> triplets;
    for (uint32_t node_id : node_ids)
    {
        const auto &node = nodes_.at(node_id);
        const Size row = index.at(node_id);
        double degree = 0.0;
        for (Size i = 0; i < node.neighbors.size() && i < node.edge_weights.size(); ++i)
        {
            const auto neighbor_it = index.find(node.neighbors[i]);
            if (neighbor_it == index.end() || !std::isfinite(node.edge_weights[i]))
            {
                continue;
            }
            const auto restriction = computeRestrictionMap(node_id, node.neighbors[i]);
            double scale = 1.0;
            if (!restriction.empty())
            {
                scale = 0.0;
                for (double value : restriction)
                {
                    scale += std::abs(value);
                }
                scale /= static_cast<double>(restriction.size());
            }
            const double weight = std::max(0.0, node.edge_weights[i]) * scale;
            degree += weight;
            triplets.emplace_back(static_cast<Eigen::Index>(row),
                                  static_cast<Eigen::Index>(neighbor_it->second), -weight);
        }
        triplets.emplace_back(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(row),
                              degree);
    }
    laplacian.setFromTriplets(triplets.begin(), triplets.end());
    return laplacian;
}

Eigen::SparseMatrix<double> SheafLaplacianRuntime::buildAttributeLaplacian()
{
    const Size n = nodes_.size();
    Eigen::SparseMatrix<double> laplacian(static_cast<Eigen::Index>(n),
                                          static_cast<Eigen::Index>(n));
    if (n == 0)
    {
        return laplacian;
    }
    const auto node_ids = detail::sortedNodeIds(nodes_);
    const auto names = detail::collectAttributeNames(config_.attribute_names, nodes_);
    std::vector<Eigen::Triplet<double>> triplets;
    for (const auto &name : names)
    {
        std::vector<double> values;
        values.reserve(n);
        bool complete = true;
        for (uint32_t node_id : node_ids)
        {
            double value = 0.0;
            if (!detail::getNodeAttributeValue(nodes_.at(node_id), name, &value))
            {
                complete = false;
                break;
            }
            values.push_back(value);
        }
        if (!complete)
        {
            continue;
        }
        if (config_.normalize_attributes)
        {
            detail::normalizeValues(values, config_.numerical_tolerance);
        }
        // Pre-allocate batch buffer for SIMD exp
        std::vector<double> exp_args(n);
        for (Size i = 0; i < n; ++i)
        {
            Size count = 0;
            for (Size j = i + 1; j < n; ++j)
            {
                exp_args[count] = -std::abs(values[i] - values[j]);
                ++count;
            }
            if (count > 0)
            {
                nerve::simd::simd_exp(exp_args.data(), count);
                for (Size k = 0; k < count; ++k)
                {
                    const double similarity = exp_args[k] * config_.attribute_weight;
                    const Size j = i + 1 + k;
                    triplets.emplace_back(static_cast<Eigen::Index>(i),
                                          static_cast<Eigen::Index>(j), -similarity);
                    triplets.emplace_back(static_cast<Eigen::Index>(j),
                                          static_cast<Eigen::Index>(i), -similarity);
                    triplets.emplace_back(static_cast<Eigen::Index>(i),
                                          static_cast<Eigen::Index>(i), similarity);
                    triplets.emplace_back(static_cast<Eigen::Index>(j),
                                          static_cast<Eigen::Index>(j), similarity);
                }
            }
        }
    }
    laplacian.setFromTriplets(triplets.begin(), triplets.end());
    return laplacian;
}

void SheafLaplacianRuntime::populateResultMetrics(SheafLaplacianResult &result)
{
    result.matrix_size = static_cast<size_t>(result.sheaf_laplacian.rows());
    result.sheaf_nodes.clear();
    result.sheaf_nodes.reserve(nodes_.size());
    for (uint32_t node_id : detail::sortedNodeIds(nodes_))
    {
        result.sheaf_nodes.push_back(nodes_.at(node_id));
    }
    result.attribute_names = detail::collectAttributeNames(config_.attribute_names, nodes_);
    result.trace = result.sheaf_laplacian.diagonal().sum();
    result.frobenius_norm = result.sheaf_laplacian.norm();
    result.spectral_radius = result.frobenius_norm;
    result.attribute_influence = sheaf_utils::computeAttributeInfluence(result);
    result.topological_influence = sheaf_utils::computeTopologicalInfluence(result);
    result.attribute_contributions = sheaf_utils::computePerAttributeContributions(result);
    result.numerical_residual = 0.0;
    result.stability_certificate = persistence::StabilityCertificate(0.0, 0.0, true);
    result.isStable = true;
}

Eigen::SparseMatrix<double>
SheafLaplacianRuntime::combineLaplacians(const Eigen::SparseMatrix<double> &topological,
                                         const Eigen::SparseMatrix<double> &attribute)
{
    if (topological.rows() != attribute.rows() || topological.cols() != attribute.cols())
    {
        throw std::invalid_argument("Laplacian matrices must have same dimensions");
    }
    return config_.topological_weight * topological + attribute;
}

} // namespace sheaf
} // namespace nerve
#endif
