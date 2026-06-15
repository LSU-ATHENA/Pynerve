
#include "nerve/sheaf/sheaf_laplacian.hpp"

#include <algorithm>
#include <cmath>
#include <ranges>
#include <sstream>
#include <stdexcept>
#if HAS_EIGEN && __has_include(<Eigen/Sparse>) && __has_include(<Eigen/Dense>)
namespace nerve
{
namespace sheaf
{
namespace
{

double computeAttributeWeight(const std::string &attribute_name,
                              const SheafLaplacianRuntime::SheafLaplacianResult &result)
{
    const auto it = std::ranges::find(result.attribute_names, attribute_name);
    if (it == result.attribute_names.end() || result.attribute_laplacian.rows() == 0)
    {
        return 0.0;
    }
    const auto index = static_cast<Eigen::Index>(std::distance(result.attribute_names.begin(), it));
    const auto bounded = std::min<Eigen::Index>(index, result.attribute_laplacian.rows() - 1);
    return std::abs(result.attribute_laplacian.coeff(bounded, bounded));
}

std::string sheafTypeName(SheafLaplacianFactory::SheafType type)
{
    switch (type)
    {
        case SheafLaplacianFactory::SheafType::COSHEAF:
            return "cosheaf";
        case SheafLaplacianFactory::SheafType::GENERALIZED_SHEAF:
        case SheafLaplacianFactory::SheafType::PERSISTENT_SHEAF:
            return "generalized";
        case SheafLaplacianFactory::SheafType::PRODUCT_SHEAF:
        default:
            return "product";
    }
}

bool isSupportedSheafType(const std::string &type)
{
    return type == "product" || type == "cosheaf" || type == "generalized";
}

bool hasOnlyFiniteValues(const Eigen::SparseMatrix<double> &matrix)
{
    for (int outer = 0; outer < matrix.outerSize(); ++outer)
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

} // namespace

SheafLaplacianFactory::SheafLaplacianFactory(const FactoryConfig &config)
    : config_(config)
{}
std::unique_ptr<SheafLaplacianRuntime>
SheafLaplacianFactory::createSheafLaplacian(SheafType type,
                                            const SheafLaplacianRuntime::SheafConfig &config)
{
    auto typed_config = config;
    typed_config.sheaf_type = sheafTypeName(type);
    typed_config.enable_stability_certificates =
        typed_config.enable_stability_certificates && config_.enable_stability_certificates;
    if (!validateConfig(typed_config))
    {
        throw std::invalid_argument("Invalid sheaf Laplacian configuration");
    }
    return std::make_unique<SheafLaplacianRuntime>(typed_config);
}
std::unique_ptr<SheafLaplacianRuntime>
SheafLaplacianFactory::createProductSheaf(const SheafLaplacianRuntime::SheafConfig &config)
{
    return createSheafLaplacian(SheafType::PRODUCT_SHEAF, config);
}
std::unique_ptr<SheafLaplacianRuntime>
SheafLaplacianFactory::createCosheaf(const SheafLaplacianRuntime::SheafConfig &config)
{
    return createSheafLaplacian(SheafType::COSHEAF, config);
}
std::unique_ptr<SheafLaplacianRuntime>
SheafLaplacianFactory::createGeneralizedSheaf(const SheafLaplacianRuntime::SheafConfig &config)
{
    return createSheafLaplacian(SheafType::GENERALIZED_SHEAF, config);
}
bool SheafLaplacianFactory::validateConfig(const SheafLaplacianRuntime::SheafConfig &config)
{
    return getConfigErrors(config).empty();
}
std::vector<std::string>
SheafLaplacianFactory::getConfigErrors(const SheafLaplacianRuntime::SheafConfig &config)
{
    std::vector<std::string> errors;
    if (config.num_attributes == 0)
    {
        errors.push_back("Number of attributes must be positive");
    }
    if (!std::isfinite(config.attribute_weight) || !std::isfinite(config.topological_weight) ||
        config.attribute_weight < 0.0 || config.topological_weight < 0.0)
    {
        errors.push_back("Weights must be non-negative");
    }
    if (!std::isfinite(config.numerical_tolerance) || config.numerical_tolerance <= 0.0)
    {
        errors.push_back("Numerical tolerance must be positive and finite");
    }
    if (!config.attribute_names.empty() && config.attribute_names.size() != config.num_attributes)
    {
        errors.push_back("Attribute name count must match the configured attribute count");
    }
    std::vector<std::string> seen_names;
    for (const auto &name : config.attribute_names)
    {
        if (name.empty())
        {
            errors.push_back("Attribute names must be non-empty");
            break;
        }
        if (std::ranges::find(seen_names, name) != seen_names.end())
        {
            errors.push_back("Attribute names must be unique");
            break;
        }
        seen_names.push_back(name);
    }
    if (!isSupportedSheafType(config.sheaf_type))
    {
        errors.push_back("Unsupported sheaf type");
    }
    return errors;
}
namespace sheaf_utils
{
double computeAttributeInfluence(const SheafLaplacianRuntime::SheafLaplacianResult &result)
{
    const double attribute_norm = result.attribute_laplacian.norm();
    const double topological_norm = result.topological_laplacian.norm();
    const double safe_attribute =
        std::isfinite(attribute_norm) && attribute_norm >= 0.0 ? attribute_norm : 0.0;
    const double safe_topological =
        std::isfinite(topological_norm) && topological_norm >= 0.0 ? topological_norm : 0.0;
    const double denominator = safe_attribute + safe_topological;
    return denominator > 0.0 ? safe_attribute / denominator : 0.0;
}
double computeTopologicalInfluence(const SheafLaplacianRuntime::SheafLaplacianResult &result)
{
    const double attribute_norm = result.attribute_laplacian.norm();
    const double topological_norm = result.topological_laplacian.norm();
    const double safe_attribute =
        std::isfinite(attribute_norm) && attribute_norm >= 0.0 ? attribute_norm : 0.0;
    const double safe_topological =
        std::isfinite(topological_norm) && topological_norm >= 0.0 ? topological_norm : 0.0;
    const double denominator = safe_attribute + safe_topological;
    return denominator > 0.0 ? safe_topological / denominator : 0.0;
}
std::vector<double>
computePerAttributeContributions(const SheafLaplacianRuntime::SheafLaplacianResult &result)
{
    std::vector<double> contributions;
    double total_weight = 0.0;
    for (const auto &attr_name : result.attribute_names)
    {
        double weight = computeAttributeWeight(attr_name, result);
        weight = std::isfinite(weight) && weight >= 0.0 ? weight : 0.0;
        contributions.push_back(weight);
        total_weight += weight;
    }
    if (std::isfinite(total_weight) && total_weight > 0.0)
    {
        for (auto &contribution : contributions)
        {
            contribution /= total_weight;
        }
    }
    return contributions;
}
bool validateEigenpairCorrectness(const SheafLaplacianRuntime::EigenpairResult &result,
                                  const SheafLaplacianRuntime::SheafLaplacianResult &sheaf_result)
{
    if (result.eigenvalues.size() != result.eigenvectors.size())
    {
        return false;
    }
    const auto expected_size = sheaf_result.sheaf_laplacian.rows();
    if (expected_size == 0)
    {
        return result.eigenvalues.empty() && result.eigenvectors.empty();
    }
    for (double eigenvalue : result.eigenvalues)
    {
        if (std::isnan(eigenvalue) || std::isinf(eigenvalue))
        {
            return false;
        }
    }
    for (const auto &eigenvector : result.eigenvectors)
    {
        if (eigenvector.size() != expected_size || !eigenvector.allFinite())
        {
            return false;
        }
    }
    return true;
}
bool validateSheafProperties(const SheafLaplacianRuntime::SheafLaplacianResult &result)
{
    if (result.sheaf_laplacian.rows() != result.sheaf_laplacian.cols())
    {
        return false;
    }
    if (!hasOnlyFiniteValues(result.sheaf_laplacian) ||
        !hasOnlyFiniteValues(result.topological_laplacian) ||
        !hasOnlyFiniteValues(result.attribute_laplacian))
    {
        return false;
    }
    for (int i = 0; i < result.sheaf_laplacian.rows(); ++i)
    {
        if (result.sheaf_laplacian.coeff(i, i) < -1e-10)
        {
            return false;
        }
    }
    return true;
}
Eigen::SparseMatrix<double>
convertToStandardLaplacian(const SheafLaplacianRuntime::SheafLaplacianResult &result)
{
    return result.topological_laplacian;
}
std::vector<algebra::Simplex>
extractTopologicalStructure(const SheafLaplacianRuntime::SheafLaplacianResult &result)
{
    std::vector<algebra::Simplex> simplices;
    for (const auto &node : result.sheaf_nodes)
    {
        simplices.emplace_back(std::vector<int>{static_cast<int>(node.node_id)});
        for (size_t i = 0; i < node.neighbors.size(); ++i)
        {
            if (node.node_id < node.neighbors[i])
            {
                simplices.emplace_back(std::vector<int>{static_cast<int>(node.node_id),
                                                        static_cast<int>(node.neighbors[i])});
            }
        }
    }
    return simplices;
}
errors::ErrorCode mapSheafErrorToErrorCode(const std::exception &e)
{
    std::string error_msg = e.what();
    if (error_msg.find("dimension") != std::string::npos)
    {
        return errors::ErrorCode::E85_MATRIX_STRUCTURE;
    }
    else if (error_msg.find("memory") != std::string::npos)
    {
        return errors::ErrorCode::E41_RESOURCE_LIMIT;
    }
    else if (error_msg.find("convergence") != std::string::npos)
    {
        return errors::ErrorCode::E50_PH_ABORT;
    }
    else if (error_msg.find("invalid") != std::string::npos)
    {
        return errors::ErrorCode::E54_PH4_INVALID_INPUT;
    }
    else
    {
        return errors::ErrorCode::E51_LAPLACIAN_ABORT;
    }
}
std::string formatSheafErrorMessage(const std::exception &e)
{
    std::ostringstream oss;
    oss << "Sheaf Laplacian error: " << e.what();
    return oss.str();
}
} // namespace sheaf_utils
} // namespace sheaf
} // namespace nerve
#endif
