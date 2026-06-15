
#pragma once
#include "nerve/config.hpp"
#include "nerve/errors/errors.hpp"
#include "nerve/persistence/kernels/ph4_ops.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#if HAS_EIGEN && __has_include(<Eigen/Sparse>) && __has_include(<Eigen/Dense>)
#include <Eigen/Dense>
#include <Eigen/Sparse>
namespace nerve
{
namespace sheaf
{
class SheafLaplacianRuntime
{
public:
    struct SheafNode
    {
        uint32_t node_id;
        std::vector<double> position;
        std::vector<double> attributes;
        std::vector<std::string> attribute_names;
        std::vector<uint32_t> neighbors;
        std::vector<double> edge_weights;
        std::vector<double> stalk_values;
        std::vector<double> restriction_maps;
        double local_laplacian;
        bool isValid() const
        {
            return !position.empty() && !attributes.empty() &&
                   attribute_names.size() == attributes.size() &&
                   neighbors.size() == edge_weights.size() && std::isfinite(local_laplacian) &&
                   std::all_of(position.begin(), position.end(),
                               [](double value) { return std::isfinite(value); }) &&
                   std::all_of(attributes.begin(), attributes.end(),
                               [](double value) { return std::isfinite(value); }) &&
                   std::all_of(edge_weights.begin(), edge_weights.end(),
                               [](double value) { return std::isfinite(value); }) &&
                   std::all_of(stalk_values.begin(), stalk_values.end(),
                               [](double value) { return std::isfinite(value); }) &&
                   std::all_of(restriction_maps.begin(), restriction_maps.end(),
                               [](double value) { return std::isfinite(value); });
        }
    };
    struct SheafConfig
    {
        size_t num_attributes = 4;
        std::vector<std::string> attribute_names;
        bool enable_sheaf_laplacian = true;
        bool enable_weighted_restriction = true;
        double attribute_weight = 1.0;
        double topological_weight = 1.0;
        bool normalize_attributes = true;
        std::string sheaf_type = "product";
        bool enable_persistence_integration = false;
        bool enable_stability_certificates = true;
        double numerical_tolerance = 1e-12;
    };
    struct SheafLaplacianResult
    {
        Eigen::SparseMatrix<double> sheaf_laplacian;
        Eigen::SparseMatrix<double> topological_laplacian;
        Eigen::SparseMatrix<double> attribute_laplacian;
        std::vector<SheafNode> sheaf_nodes;
        std::vector<std::string> attribute_names;
        size_t matrix_size;
        double spectral_radius;
        double trace;
        double frobenius_norm;
        double attribute_influence;
        double topological_influence;
        std::vector<double> attribute_contributions;
        persistence::StabilityCertificate stability_certificate;
        double numerical_residual;
        bool isStable;
    };
    struct EigenpairResult
    {
        std::vector<double> eigenvalues;
        std::vector<Eigen::VectorXd> eigenvectors;
        std::vector<double> attribute_contributions;
        double total_attribute_influence;
        double total_topological_influence;
        std::vector<double> error_estimates;
        persistence::StabilityCertificate eigenpair_certificate;
        bool converged;
        size_t iterations_used;
    };
    struct SpectralConfig
    {
        size_t num_eigenpairs = 50;
        double convergence_tolerance = 1e-8;
        bool compute_attribute_contributions = true;
        bool enable_sensitivity_analysis = true;
        size_t max_iterations = 1000;
    };
    explicit SheafLaplacianRuntime(const SheafConfig &config);
    ~SheafLaplacianRuntime() = default;
    void addNode(const SheafNode &node);
    void addNodes(const std::vector<SheafNode> &nodes);
    void removeNode(uint32_t node_id);
    void updateNode(const SheafNode &node);
    void addAttribute(const std::string &name, const std::vector<double> &values);
    void removeAttribute(const std::string &name);
    void updateAttribute(const std::string &name, const std::vector<double> &values);
    SheafLaplacianResult buildSheafLaplacian();
    SheafLaplacianResult buildProductSheaf();
    SheafLaplacianResult buildCosheaf();
    SheafLaplacianResult buildGeneralizedSheaf();
    EigenpairResult computeEigenpairs(const SpectralConfig &config);
    EigenpairResult computeEigenpairsWithStability(const SpectralConfig &config);
    SheafLaplacianResult integrateWithPersistence(const std::vector<algebra::Simplex> &complex);
    bool validateSheafStructure();
    std::vector<std::string> getValidationErrors();
    void setConfig(const SheafConfig &config);
    SheafConfig getConfig() const;
    bool isStable() const;
    double getNumericalResidual() const;
    persistence::StabilityCertificate getStabilityCertificate() const;

private:
    SheafConfig config_;
    std::unordered_map<uint32_t, SheafNode> nodes_;
    Eigen::SparseMatrix<double> buildTopologicalLaplacian();
    Eigen::SparseMatrix<double> buildRestrictionWeightedLaplacian();
    Eigen::SparseMatrix<double> buildAttributeLaplacian();
    Eigen::SparseMatrix<double> combineLaplacians(const Eigen::SparseMatrix<double> &topological,
                                                  const Eigen::SparseMatrix<double> &attribute);
    void populateResultMetrics(SheafLaplacianResult &result);
    void computeRestrictionMaps();
    std::vector<double> computeRestrictionMap(uint32_t from_node, uint32_t to_node);
    bool validateAttributeConsistency();
    persistence::StabilityCertificate
    computeStabilityCertificate(const SheafLaplacianResult &result);
    double computeNumericalResidual(const SheafLaplacianResult &result);
};
class SheafLaplacianFactory
{
public:
    enum class SheafType
    {
        PRODUCT_SHEAF,
        COSHEAF,
        GENERALIZED_SHEAF,
        PERSISTENT_SHEAF
    };
    struct FactoryConfig
    {
        SheafType default_type = SheafType::PRODUCT_SHEAF;
        bool enable_stability_certificates = true;
        double error_tolerance = 1e-10;
        size_t max_memory_mb = 1024;
    };
    explicit SheafLaplacianFactory(const FactoryConfig &config);
    std::unique_ptr<SheafLaplacianRuntime>
    createSheafLaplacian(SheafType type, const SheafLaplacianRuntime::SheafConfig &config);
    std::unique_ptr<SheafLaplacianRuntime>
    createProductSheaf(const SheafLaplacianRuntime::SheafConfig &config);
    std::unique_ptr<SheafLaplacianRuntime>
    createCosheaf(const SheafLaplacianRuntime::SheafConfig &config);
    std::unique_ptr<SheafLaplacianRuntime>
    createGeneralizedSheaf(const SheafLaplacianRuntime::SheafConfig &config);
    bool validateConfig(const SheafLaplacianRuntime::SheafConfig &config);
    std::vector<std::string> getConfigErrors(const SheafLaplacianRuntime::SheafConfig &config);

private:
    FactoryConfig config_;
};
namespace sheaf_utils
{
double computeAttributeInfluence(const SheafLaplacianRuntime::SheafLaplacianResult &result);
double computeTopologicalInfluence(const SheafLaplacianRuntime::SheafLaplacianResult &result);
std::vector<double>
computePerAttributeContributions(const SheafLaplacianRuntime::SheafLaplacianResult &result);
bool validateEigenpairCorrectness(const SheafLaplacianRuntime::EigenpairResult &result,
                                  const SheafLaplacianRuntime::SheafLaplacianResult &sheaf_result);
bool validateSheafProperties(const SheafLaplacianRuntime::SheafLaplacianResult &result);
Eigen::SparseMatrix<double>
convertToStandardLaplacian(const SheafLaplacianRuntime::SheafLaplacianResult &result);
std::vector<algebra::Simplex>
extractTopologicalStructure(const SheafLaplacianRuntime::SheafLaplacianResult &result);
errors::ErrorCode mapSheafErrorToErrorCode(const std::exception &e);
std::string formatSheafErrorMessage(const std::exception &e);
} // namespace sheaf_utils
} // namespace sheaf
} // namespace nerve
#else

namespace nerve::sheaf
{
class SheafLaplacianRuntime;
class SheafLaplacianFactory;
} // namespace nerve::sheaf

#endif
