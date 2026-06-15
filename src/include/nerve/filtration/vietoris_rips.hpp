
#pragma once
#include "nerve/algebra/simplex.hpp"
#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/core/rng/determinism_contract.hpp"
#include "nerve/core_types.hpp"
#include "nerve/errors/errors.hpp"

#include <functional>
#include <string>
#include <vector>
namespace nerve::filtration
{
class VietorisRips
{
public:
    VietorisRips() = default;
    explicit VietorisRips(double max_radius);
    void setMaxRadius(double radius);
    void setMaxDimension(Size max_dim);
    void setDistanceMetric(const std::string &metric);
    errors::ErrorResult<std::vector<std::pair<algebra::Simplex, double>>>
    buildFiltration(const core::ownership_utils::PointView &points, size_t dimension,
                    const core::DeterminismContract &contract = {});
    errors::ErrorResult<std::vector<std::pair<algebra::Simplex, double>>>
    buildFiltration(const core::ownership_utils::PointView &points, size_t dimension,
                    const core::ownership_utils::PointView &custom_radii,
                    const core::DeterminismContract &contract = {});
    void addPoint(const core::ownership_utils::PointView &point, size_t dimension);
    void addPoints(const core::ownership_utils::PointView &points, size_t dimension);
    std::vector<std::pair<algebra::Simplex, double>> getCurrentFiltration();
    core::ownership_utils::OwnedPointBuffer
    computeDistanceMatrix(const core::ownership_utils::PointView &points, size_t dimension) const;
    double computeDistance(const core::ownership_utils::PointView &p1,
                           const core::ownership_utils::PointView &p2, size_t dimension) const;
    Size getNumSimplices() const;
    Size getNumSimplicesOfDimension(Size dim) const;
    std::vector<double> getFiltrationValues() const;
    std::vector<Size> getSimplexDimensions() const;
    double getComputationTime() const;

private:
    double max_radius_ = 1.0;
    Size max_dimension_ = 3;
    std::string distance_metric_ = "euclidean";
    std::vector<std::vector<double>> points_;
    std::vector<std::pair<algebra::Simplex, double>> filtration_;
    std::vector<std::vector<double>> distance_matrix_;
    double computation_time_ = 0.0;
    void buildDistanceMatrix();
    void buildSimplices();
    void buildKSimplices(Size k);
    void buildKSimplicesWithCustomRadii(Size k, const std::vector<double> &custom_radii,
                                        double scaling_factor);
    std::vector<Index> findSimplexVertices(double radius, Size k) const;
    double simplexRadius(const std::vector<Index> &vertices) const;
    bool isSimplexValid(const std::vector<Index> &vertices, double radius) const;
    double euclideanDistance(const std::vector<double> &p1, const std::vector<double> &p2) const;
    double manhattanDistance(const std::vector<double> &p1, const std::vector<double> &p2) const;
    double chebyshevDistance(const std::vector<double> &p1, const std::vector<double> &p2) const;
    double cosineDistance(const std::vector<double> &p1, const std::vector<double> &p2) const;
    void sortFiltration();
    void validatePoints(const std::vector<std::vector<double>> &points) const;
    Size getPointDimension() const;
};
class WeightedVietorisRips
{
public:
    WeightedVietorisRips() = default;
    explicit WeightedVietorisRips(const std::vector<double> &weights);
    void setWeights(const std::vector<double> &weights);
    void setAdaptiveRadius(bool adaptive);
    void setWeightFunction(const std::string &function);
    std::vector<std::pair<algebra::Simplex, double>>
    buildFiltration(const core::ownership_utils::PointView &points, size_t dimension,
                    const core::DeterminismContract &contract = {});
    std::vector<double> computeLocalWeights(const core::ownership_utils::PointView &points,
                                            size_t dimension) const;
    double computeAdaptiveRadius(Index point_index, Size k) const;

private:
    std::vector<double> weights_;
    bool adaptive_radius_ = true;
    std::string weight_function_ = "inverse_distance";
    double computeWeightedRadius(Index i, Index j) const;
    double applyWeightFunction(double distance, double weight) const;
};
class SparseVietorisRips
{
public:
    SparseVietorisRips() = default;
    explicit SparseVietorisRips(Size k_neighbors);
    void setKNeighbors(Size k);
    void setApproximationFactor(double factor);
    void setBatchSize(Size batch_size);
    std::vector<std::pair<algebra::Simplex, double>>
    buildFiltration(const core::ownership_utils::PointView &points, size_t dimension,
                    const core::DeterminismContract &contract = {});
    std::vector<Index> findApproximateNeighbors(Index point_index) const;
    double approximateDistance(Index i, Index j) const;

private:
    Size k_neighbors_ = 10;
    double approximation_factor_ = 1.1;
    Size batch_size_ = 1000;
    std::vector<std::vector<Index>> neighbor_graph_;
    std::vector<std::vector<double>> neighbor_distances_;
    std::vector<std::pair<algebra::Simplex, double>> filtration_;
    void buildNeighborGraph(const std::vector<std::vector<double>> &points);
    void buildApproximateFiltration();
};
std::vector<std::pair<algebra::Simplex, double>>
computeVietorisRipsFiltration(const core::ownership_utils::PointView &points, size_t dimension,
                              double max_radius = 1.0, Size max_dimension = 3);
std::vector<std::pair<algebra::Simplex, double>>
computeWeightedVietorisRips(const core::ownership_utils::PointView &points, size_t dimension,
                            const std::vector<double> &weights, double max_radius = 1.0);
core::ownership_utils::OwnedPointBuffer
computeAllPairDistances(const core::ownership_utils::PointView &points, size_t dimension);
std::vector<std::pair<Index, Index>>
findKNearestNeighbors(const core::ownership_utils::PointView &points, size_t dimension,
                      Index point_index, Size k);
} // namespace nerve::filtration
