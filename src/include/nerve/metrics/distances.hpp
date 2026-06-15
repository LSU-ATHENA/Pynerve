
#pragma once

#include "nerve/algebra/complex.hpp"
#include "nerve/core/policy/ownership_checks.hpp"
#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/core_types.hpp"
#include "nerve/persistence/core/core_types.hpp"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace nerve::metrics
{

using Diagram = nerve::persistence::Diagram;
using Pair = ::nerve::Pair;
using SimplicialComplex = nerve::algebra::SimplicialComplex;
using AlgebraSimplex = nerve::algebra::Simplex;

double bottleneckDistance(const Diagram &diagram1, const Diagram &diagram2);
double wassersteinDistance(const Diagram &diagram1, const Diagram &diagram2, double p = 2.0);
double frechetDistance(const Diagram &diagram1, const Diagram &diagram2);
double gromovHausdorffDistance(const Diagram &diagram1, const Diagram &diagram2);
double gromovHausdorffDistance(const SimplicialComplex &complex1,
                               const SimplicialComplex &complex2);
double hausdorffDistance(const SimplicialComplex &complex1, const SimplicialComplex &complex2);
double editDistance(const SimplicialComplex &complex1, const SimplicialComplex &complex2);
double interleavingDistance(const SimplicialComplex &complex1, const SimplicialComplex &complex2);

double hausdorffDistance(const std::vector<std::vector<double>> &points1,
                         const std::vector<std::vector<double>> &points2);
double chamferDistance(const std::vector<std::vector<double>> &points1,
                       const std::vector<std::vector<double>> &points2);
double earthMoversDistance(const std::vector<std::vector<double>> &points1,
                           const std::vector<std::vector<double>> &points2);

double hausdorffDistance(const core::PointBuffer &points1, const core::PointBuffer &points2);
double chamferDistance(const core::PointBuffer &points1, const core::PointBuffer &points2);
double earthMoversDistance(const core::PointBuffer &points1, const core::PointBuffer &points2);

static_assert(core::ownership_utils::is_owned_type<core::PointBuffer>::value,
              "PointBuffer distance functions should use owning types for zero-copy");
static_assert(core::ownership_utils::is_zero_copy_capable<core::PointBuffer>,
              "PointBuffer should support zero-copy operations");

class BottleneckDistance
{
public:
    BottleneckDistance() = default;
    double compute(const Diagram &diagram1, const Diagram &diagram2);
    void setTolerance(double tolerance);
    void setMaxIterations(Size max_iterations);
    void useApproximation(bool use_approx);
    std::vector<std::pair<nerve::persistence::Pair, nerve::persistence::Pair>>
    getOptimalMatching() const;
    double getComputationTime() const;

private:
    double tolerance_ = 1e-6;
    Size max_iterations_ = 1000;
    bool use_approximation_ = false;
    double computation_time_ = 0.0;
    std::vector<std::pair<nerve::persistence::Pair, nerve::persistence::Pair>> optimal_matching_;

    std::vector<std::vector<double>>
    buildCostMatrix(const std::vector<nerve::persistence::Pair> &pairs1,
                    const std::vector<nerve::persistence::Pair> &pairs2) const;
    double solveAssignmentHungarian(const std::vector<std::vector<double>> &cost_matrix);
    double solveAssignmentGreedy(const std::vector<std::vector<double>> &cost_matrix);
    double pairDistance(const nerve::persistence::Pair &pair1,
                        const nerve::persistence::Pair &pair2) const;
};

class WassersteinDistance
{
public:
    WassersteinDistance() = default;
    explicit WassersteinDistance(double p) { setOrder(p); }
    double compute(const Diagram &diagram1, const Diagram &diagram2);
    double computeWithOrder(const Diagram &diagram1, const Diagram &diagram2, double p);
    void setOrder(double p);
    void setRegularization(double epsilon);
    void useSinkhorn(bool useSinkhorn);
    std::vector<std::vector<double>> getOptimalTransportPlan() const;
    double getComputationTime() const;

private:
    double p_ = 2.0;
    double regularization_ = 1e-3;
    bool use_sinkhorn_ = false;
    double computation_time_ = 0.0;
    std::vector<std::vector<double>> transport_plan_;

    std::vector<std::vector<double>>
    buildCostMatrix(const std::vector<nerve::persistence::Pair> &pairs1,
                    const std::vector<nerve::persistence::Pair> &pairs2, double p) const;
    double solveTransportHungarian(const std::vector<std::vector<double>> &cost_matrix);
    double solveTransportSinkhorn(const std::vector<std::vector<double>> &cost_matrix);
    double pairDistance(const nerve::persistence::Pair &pair1,
                        const nerve::persistence::Pair &pair2) const;
};

class GromovHausdorffDistance
{
public:
    GromovHausdorffDistance() = default;
    double compute(const SimplicialComplex &complex1, const SimplicialComplex &complex2);
    void setEmbeddingDimension(Size dim);
    void setDistanceMetric(const std::string &metric);
    void useApproximateEmbedding(bool use_approx);
    std::vector<std::vector<double>> getOptimalCorrespondence() const;
    double getComputationTime() const;

private:
    Size embedding_dimension_ = 3;
    std::string distance_metric_ = "euclidean";
    bool use_approximate_embedding_ = false;
    double computation_time_ = 0.0;
    std::vector<std::vector<double>> optimal_correspondence_;

    std::vector<std::vector<double>> embedComplex(const SimplicialComplex &complex) const;
    double computeHausdorffDistance(const std::vector<std::vector<double>> &points1,
                                    const std::vector<std::vector<double>> &points2) const;
};

class EditDistance
{
public:
    EditDistance() = default;
    double compute(const SimplicialComplex &complex1, const SimplicialComplex &complex2);
    void setInsertionCost(double cost);
    void setDeletionCost(double cost);
    void setSubstitutionCost(double cost);
    std::vector<std::string> getEditOperations() const;
    Size getNumOperations() const;
    double getComputationTime() const;

private:
    double insertion_cost_ = 1.0;
    double deletion_cost_ = 1.0;
    double substitution_cost_ = 1.0;
    double computation_time_ = 0.0;
    std::vector<std::string> edit_operations_;

    double computeEditDistanceDp(const std::vector<AlgebraSimplex> &simplices1,
                                 const std::vector<AlgebraSimplex> &simplices2);
    double simplexDistance(const AlgebraSimplex &simplex1, const AlgebraSimplex &simplex2) const;
};

class InterleavingDistance
{
public:
    InterleavingDistance() = default;
    double compute(const std::vector<std::pair<AlgebraSimplex, double>> &filtration1,
                   const std::vector<std::pair<AlgebraSimplex, double>> &filtration2);
    void setMaxDimension(Size max_dim);
    void setTolerance(double tolerance);
    void useApproximateAlgorithm(bool use_approx);

private:
    Size max_dimension_ = 3;
    double tolerance_ = 1e-6;
    bool use_approximate_algorithm_ = false;
    double computeInterleavingDistanceExact(
        const std::vector<std::pair<AlgebraSimplex, double>> &filtration1,
        const std::vector<std::pair<AlgebraSimplex, double>> &filtration2);
    double computeInterleavingDistanceApprox(
        const std::vector<std::pair<AlgebraSimplex, double>> &filtration1,
        const std::vector<std::pair<AlgebraSimplex, double>> &filtration2);
};

class FrechetDistance
{
public:
    FrechetDistance() = default;
    double compute(const Diagram &diagram1, const Diagram &diagram2);
    void setParameterization(const std::string &param);
    void setTolerance(double tolerance);

private:
    std::string parameterization_ = "linear";
    double tolerance_ = 1e-6;
    double computeFrechetDistanceDp(const std::vector<nerve::persistence::Pair> &pairs1,
                                    const std::vector<nerve::persistence::Pair> &pairs2);
    double curveDistance(const nerve::persistence::Pair &pair1,
                         const nerve::persistence::Pair &pair2) const;
};

class DistanceMetricFactory
{
public:
    enum class MetricType
    {
        BOTTLENECK,
        WASSERSTEIN,
        GROMOV_HAUSDORFF,
        EDIT,
        INTERLEAVING,
        FRECHET,
        FREDHET = FRECHET
    };

    static std::unique_ptr<BottleneckDistance> createBottleneck();
    static std::unique_ptr<WassersteinDistance> createWasserstein(double p = 2.0);
    static std::unique_ptr<GromovHausdorffDistance> createGromovHausdorff();
    static std::unique_ptr<EditDistance> createEdit();
    static std::unique_ptr<InterleavingDistance> createInterleaving();
    static std::unique_ptr<FrechetDistance> createFrechet();

    static double computeDistance(MetricType type, const Diagram &diagram1, const Diagram &diagram2,
                                  double parameter = 0.0);
    static double computeDistance(MetricType type, const SimplicialComplex &complex1,
                                  const SimplicialComplex &complex2);
};

class DistanceMatrix
{
public:
    static std::vector<std::vector<double>> computeDiagramDistanceMatrix(
        const std::vector<Diagram> &diagrams,
        DistanceMetricFactory::MetricType metric = DistanceMetricFactory::MetricType::BOTTLENECK);
    static std::vector<std::vector<double>>
    computeComplexDistanceMatrix(const std::vector<SimplicialComplex> &complexes,
                                 DistanceMetricFactory::MetricType metric =
                                     DistanceMetricFactory::MetricType::GROMOV_HAUSDORFF);
    static std::vector<std::vector<double>> computePointCloudDistanceMatrix(
        const std::vector<std::vector<std::vector<double>>> &point_clouds,
        const std::string &metric = "hausdorff");
    static std::vector<Size>
    findNearestNeighbors(const std::vector<std::vector<double>> &distance_matrix, Size k = 1);
    static std::vector<std::vector<Size>>
    computeClusters(const std::vector<std::vector<double>> &distance_matrix, double threshold);
};

class DistanceStatistics
{
public:
    static double computeMean(const std::vector<std::vector<double>> &distance_matrix);
    static double computeStdDeviation(const std::vector<std::vector<double>> &distance_matrix);
    static std::vector<double>
    computeRowMeans(const std::vector<std::vector<double>> &distance_matrix);
    static double permutationTest(const std::vector<std::vector<double>> &distance_matrix1,
                                  const std::vector<std::vector<double>> &distance_matrix2,
                                  Size num_permutations = 1000);
    static double silhouetteScore(const std::vector<std::vector<double>> &distance_matrix,
                                  const std::vector<Size> &cluster_labels);
    static std::vector<std::vector<double>>
    multidimensionalScaling(const std::vector<std::vector<double>> &distance_matrix,
                            Size target_dimension = 2);
};

} // namespace nerve::metrics
