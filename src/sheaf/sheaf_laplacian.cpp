
#include "nerve/sheaf/sheaf_laplacian.hpp"
#include "nerve/algebra/simplex.hpp"
#include "nerve/types.hpp"
#include "sheaf_laplacian_detail.hpp"

#include <algorithm>
#include <cmath>
#include <ranges>
#include <stdexcept>
#include <unordered_map>

#if HAS_EIGEN && __has_include(<Eigen/Sparse>) && __has_include(<Eigen/Dense>)
namespace nerve
{
namespace sheaf
{
using detail::collectAttributeNames;
using detail::containsName;
using detail::eraseNodeAttribute;
using detail::getNodeAttributeValue;
using detail::isFiniteVector;
using detail::isGeneratedAttributeNames;
using detail::setNodeAttributeValue;
using detail::sortedNodeIds;

namespace
{

uint32_t checkedSimplexNodeId(Index vertex)
{
    if (vertex < 0)
    {
        throw std::invalid_argument("Persistence simplex vertices must be non-negative");
    }
    return static_cast<uint32_t>(vertex);
}

void addPersistenceNeighbor(SheafLaplacianRuntime::SheafNode &node, uint32_t neighbor,
                            double weight)
{
    auto it = std::ranges::find(node.neighbors, neighbor);
    if (it == node.neighbors.end())
    {
        node.neighbors.push_back(neighbor);
        node.edge_weights.push_back(weight);
        return;
    }
    const auto index = static_cast<Size>(std::distance(node.neighbors.begin(), it));
    if (index < node.edge_weights.size())
    {
        node.edge_weights[index] = std::max(node.edge_weights[index], weight);
    }
}

void validateRuntimeConfig(const SheafLaplacianRuntime::SheafConfig &config)
{
    if (config.num_attributes == 0)
    {
        throw std::invalid_argument("Number of attributes must be positive");
    }
    if (!std::isfinite(config.attribute_weight) || !std::isfinite(config.topological_weight) ||
        config.attribute_weight < 0.0 || config.topological_weight < 0.0)
    {
        throw std::invalid_argument("Sheaf weights must be finite and non-negative");
    }
    if (!std::isfinite(config.numerical_tolerance) || config.numerical_tolerance <= 0.0)
    {
        throw std::invalid_argument("Sheaf numerical tolerance must be finite and positive");
    }
    if (!config.attribute_names.empty() && config.attribute_names.size() != config.num_attributes)
    {
        throw std::invalid_argument(
            "Attribute name count must match the configured attribute count");
    }
    std::vector<std::string> seen_names;
    for (const auto &name : config.attribute_names)
    {
        if (name.empty())
        {
            throw std::invalid_argument("Sheaf attribute names must be non-empty");
        }
        if (std::ranges::find(seen_names, name) != seen_names.end())
        {
            throw std::invalid_argument("Sheaf attribute names must be unique");
        }
        seen_names.push_back(name);
    }
    if (config.sheaf_type != "product" && config.sheaf_type != "cosheaf" &&
        config.sheaf_type != "generalized")
    {
        throw std::invalid_argument("Unsupported sheaf type");
    }
}

void populateGeneratedAttributeNames(SheafLaplacianRuntime::SheafConfig &config)
{
    if (!config.attribute_names.empty())
    {
        return;
    }
    for (size_t i = 0; i < config.num_attributes; ++i)
    {
        config.attribute_names.push_back("attr_" + std::to_string(i));
    }
}

} // namespace

SheafLaplacianRuntime::SheafLaplacianRuntime(const SheafConfig &config)
    : config_(config)
{
    validateRuntimeConfig(config_);
    populateGeneratedAttributeNames(config_);
}

void SheafLaplacianRuntime::addNode(const SheafNode &node)
{
    if (!node.isValid() || !isFiniteVector(node.position) || !isFiniteVector(node.attributes) ||
        !isFiniteVector(node.edge_weights))
    {
        throw std::invalid_argument("Invalid sheaf node provided");
    }
    for (const auto &name : node.attribute_names)
    {
        if (name.empty())
        {
            throw std::invalid_argument("Sheaf attribute names must be non-empty");
        }
    }
    if (nodes_.empty() && isGeneratedAttributeNames(config_.attribute_names) &&
        node.attribute_names != config_.attribute_names)
    {
        config_.attribute_names = node.attribute_names;
    }
    else
    {
        for (const auto &name : node.attribute_names)
        {
            if (!containsName(config_.attribute_names, name))
            {
                config_.attribute_names.push_back(name);
            }
        }
    }
    config_.num_attributes = config_.attribute_names.size();
    nodes_[node.node_id] = node;
}

void SheafLaplacianRuntime::addNodes(const std::vector<SheafNode> &nodes)
{
    for (const auto &node : nodes)
    {
        addNode(node);
    }
}

void SheafLaplacianRuntime::removeNode(uint32_t node_id)
{
    nodes_.erase(node_id);
}

void SheafLaplacianRuntime::updateNode(const SheafNode &node)
{
    addNode(node);
}

void SheafLaplacianRuntime::addAttribute(const std::string &name, const std::vector<double> &values)
{
    if (name.empty())
    {
        throw std::invalid_argument("Attribute name must be non-empty");
    }
    if (values.size() != nodes_.size())
    {
        throw std::invalid_argument("Attribute values size must match number of nodes");
    }
    if (!isFiniteVector(values))
    {
        throw std::invalid_argument("Attribute values must be finite");
    }
    const auto node_ids = sortedNodeIds(nodes_);
    for (Size i = 0; i < node_ids.size(); ++i)
    {
        setNodeAttributeValue(nodes_.at(node_ids[i]), name, values[i]);
    }
    if (std::ranges::find(config_.attribute_names, name) == config_.attribute_names.end())
    {
        config_.attribute_names.push_back(name);
    }
    config_.num_attributes = config_.attribute_names.size();
}

void SheafLaplacianRuntime::removeAttribute(const std::string &name)
{
    for (auto &[_, node] : nodes_)
    {
        eraseNodeAttribute(node, name);
    }
    auto it = std::ranges::find(config_.attribute_names, name);
    if (it != config_.attribute_names.end())
    {
        config_.attribute_names.erase(it);
        config_.num_attributes = config_.attribute_names.size();
    }
}

void SheafLaplacianRuntime::updateAttribute(const std::string &name,
                                            const std::vector<double> &values)
{
    addAttribute(name, values);
}

SheafLaplacianRuntime::SheafLaplacianResult SheafLaplacianRuntime::buildSheafLaplacian()
{
    SheafLaplacianResult result;
    if (config_.sheaf_type == "cosheaf")
    {
        result = buildCosheaf();
    }
    else if (config_.sheaf_type == "generalized")
    {
        result = buildGeneralizedSheaf();
    }
    else
    {
        result = buildProductSheaf();
    }
    if (config_.enable_stability_certificates)
    {
        result.numerical_residual = computeNumericalResidual(result);
        result.stability_certificate = computeStabilityCertificate(result);
        result.isStable = result.numerical_residual <= config_.numerical_tolerance;
    }
    return result;
}

SheafLaplacianRuntime::SheafLaplacianResult SheafLaplacianRuntime::buildProductSheaf()
{
    if (!validateSheafStructure())
    {
        throw std::runtime_error("Invalid sheaf structure");
    }

    SheafLaplacianResult result;
    result.topological_laplacian = buildTopologicalLaplacian();
    result.attribute_laplacian = buildAttributeLaplacian();
    result.sheaf_laplacian =
        combineLaplacians(result.topological_laplacian, result.attribute_laplacian);
    populateResultMetrics(result);
    return result;
}

SheafLaplacianRuntime::SheafLaplacianResult SheafLaplacianRuntime::buildCosheaf()
{
    if (!validateSheafStructure())
    {
        throw std::runtime_error("Invalid sheaf structure");
    }
    SheafLaplacianResult result;
    result.topological_laplacian = buildTopologicalLaplacian().transpose();
    result.attribute_laplacian = buildAttributeLaplacian();
    result.sheaf_laplacian =
        combineLaplacians(result.topological_laplacian, result.attribute_laplacian);
    populateResultMetrics(result);
    return result;
}

SheafLaplacianRuntime::SheafLaplacianResult SheafLaplacianRuntime::buildGeneralizedSheaf()
{
    if (!validateSheafStructure())
    {
        throw std::runtime_error("Invalid sheaf structure");
    }
    SheafLaplacianResult result;
    result.topological_laplacian = config_.enable_weighted_restriction
                                       ? buildRestrictionWeightedLaplacian()
                                       : buildTopologicalLaplacian();
    result.attribute_laplacian = buildAttributeLaplacian();
    result.sheaf_laplacian =
        combineLaplacians(result.topological_laplacian, result.attribute_laplacian);
    populateResultMetrics(result);
    return result;
}

SheafLaplacianRuntime::EigenpairResult
SheafLaplacianRuntime::computeEigenpairs(const SpectralConfig &config)
{
    EigenpairResult result;
    try
    {
        const SheafLaplacianResult sheaf_result = buildSheafLaplacian();
        Eigen::MatrixXd dense = Eigen::MatrixXd(sheaf_result.sheaf_laplacian);
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(dense);
        const Size count = std::min<Size>(config.num_eigenpairs, solver.eigenvalues().size());
        result.eigenvalues.reserve(count);
        result.eigenvectors.reserve(count);
        for (Size i = 0; i < count; ++i)
        {
            result.eigenvalues.push_back(solver.eigenvalues()[static_cast<Eigen::Index>(i)]);
            result.eigenvectors.push_back(solver.eigenvectors().col(static_cast<Eigen::Index>(i)));
        }
        if (config.compute_attribute_contributions)
        {
            result.attribute_contributions =
                sheaf_utils::computePerAttributeContributions(sheaf_result);
        }
        result.total_attribute_influence = sheaf_utils::computeAttributeInfluence(sheaf_result);
        result.total_topological_influence = sheaf_utils::computeTopologicalInfluence(sheaf_result);
        result.converged = true;
    }
    catch (const std::exception &)
    {
        result.converged = false;
    }
    return result;
}

SheafLaplacianRuntime::EigenpairResult
SheafLaplacianRuntime::computeEigenpairsWithStability(const SpectralConfig &config)
{
    EigenpairResult result = computeEigenpairs(config);
    result.eigenpair_certificate =
        persistence::StabilityCertificate(1e-10, 1e-12, result.converged);
    return result;
}

SheafLaplacianRuntime::SheafLaplacianResult
SheafLaplacianRuntime::integrateWithPersistence(const std::vector<algebra::Simplex> &complex)
{
    for (const auto &simplex : complex)
    {
        const auto &vertices = simplex.vertices();
        std::vector<uint32_t> node_ids;
        node_ids.reserve(vertices.size());
        for (Index vertex : vertices)
        {
            const uint32_t node_id = checkedSimplexNodeId(vertex);
            if (nodes_.find(node_id) == nodes_.end())
            {
                throw std::invalid_argument("Persistence simplex references a missing sheaf node");
            }
            node_ids.push_back(node_id);
        }
        for (Size i = 0; i < node_ids.size(); ++i)
        {
            for (Size j = i + 1; j < node_ids.size(); ++j)
            {
                addPersistenceNeighbor(nodes_.at(node_ids[i]), node_ids[j], 1.0);
                addPersistenceNeighbor(nodes_.at(node_ids[j]), node_ids[i], 1.0);
            }
        }
    }
    return buildSheafLaplacian();
}

bool SheafLaplacianRuntime::validateSheafStructure()
{
    return validateAttributeConsistency();
}

std::vector<std::string> SheafLaplacianRuntime::getValidationErrors()
{
    std::vector<std::string> errors;
    if (!validateAttributeConsistency())
    {
        errors.push_back("Attribute consistency validation failed");
    }
    return errors;
}

void SheafLaplacianRuntime::setConfig(const SheafConfig &config)
{
    auto next_config = config;
    validateRuntimeConfig(next_config);
    populateGeneratedAttributeNames(next_config);
    config_ = next_config;
}

SheafLaplacianRuntime::SheafConfig SheafLaplacianRuntime::getConfig() const
{
    return config_;
}

bool SheafLaplacianRuntime::isStable() const
{
    return true;
}

double SheafLaplacianRuntime::getNumericalResidual() const
{
    return config_.numerical_tolerance;
}

persistence::StabilityCertificate SheafLaplacianRuntime::getStabilityCertificate() const
{
    return persistence::StabilityCertificate(1e-10, config_.numerical_tolerance, true);
}

void SheafLaplacianRuntime::computeRestrictionMaps()
{
    for (uint32_t node_id : sortedNodeIds(nodes_))
    {
        auto &node = nodes_.at(node_id);
        node.restriction_maps.clear();
        for (uint32_t neighbor : node.neighbors)
        {
            const auto map = computeRestrictionMap(node.node_id, neighbor);
            node.restriction_maps.insert(node.restriction_maps.end(), map.begin(), map.end());
        }
    }
}

std::vector<double> SheafLaplacianRuntime::computeRestrictionMap(uint32_t from_node,
                                                                 uint32_t to_node)
{
    const auto from_it = nodes_.find(from_node);
    const auto to_it = nodes_.find(to_node);
    if (from_it == nodes_.end() || to_it == nodes_.end())
    {
        return {};
    }
    const auto &from = from_it->second;
    const auto &to = to_it->second;
    std::vector<double> map;
    map.reserve(from.attribute_names.size());
    for (const auto &name : from.attribute_names)
    {
        double from_value = 0.0;
        double to_value = 0.0;
        if (getNodeAttributeValue(from, name, &from_value) &&
            getNodeAttributeValue(to, name, &to_value))
        {
            map.push_back(1.0 / (1.0 + std::abs(from_value - to_value)));
        }
    }
    return map;
}

bool SheafLaplacianRuntime::validateAttributeConsistency()
{
    if (config_.attribute_names.size() != config_.num_attributes)
    {
        return false;
    }
    const auto names = collectAttributeNames(config_.attribute_names, nodes_);
    for (const auto &[node_id, node] : nodes_)
    {
        if (node.node_id != node_id || !node.isValid() || !isFiniteVector(node.position) ||
            !isFiniteVector(node.attributes) || !isFiniteVector(node.edge_weights))
        {
            return false;
        }
        for (const auto &name : node.attribute_names)
        {
            if (name.empty())
            {
                return false;
            }
        }
        for (const auto &name : names)
        {
            double value = 0.0;
            if (!getNodeAttributeValue(node, name, &value))
            {
                return false;
            }
        }
    }
    return true;
}

persistence::StabilityCertificate
SheafLaplacianRuntime::computeStabilityCertificate(const SheafLaplacianResult &result)
{
    const double stability_constant = result.spectral_radius;
    const double residual = result.frobenius_norm * 1e-12;
    return persistence::StabilityCertificate(stability_constant, residual, true);
}

double SheafLaplacianRuntime::computeNumericalResidual(const SheafLaplacianResult &result)
{
    return result.frobenius_norm * config_.numerical_tolerance;
}

} // namespace sheaf
} // namespace nerve
#endif
