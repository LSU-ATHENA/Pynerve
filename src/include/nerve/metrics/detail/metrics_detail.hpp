#pragma once

#include "nerve/algorithms/distance.hpp"
#include "nerve/metrics/bottleneck/bottleneck.hpp"
#include "nerve/metrics/distances.hpp"
#include "nerve/metrics/lazy_distance.hpp"
#include "nerve/metrics/sinkhorn/sinkhorn.hpp"
#include "nerve/persistence/core/core_types.hpp"

#include <utility>
#include <vector>

namespace nerve::metrics
{
// Bottleneck
double bottleneckDistance(const std::vector<std::pair<double, double>> &d1,
                          const std::vector<std::pair<double, double>> &d2);
bool validateDiagram(const std::vector<std::pair<double, double>> &diagram);
double diagonalDistance(const std::pair<double, double> &p);

// Frechet
double frechetDistance(const std::vector<double> &curve1, const std::vector<double> &curve2);

// Lazy distance matrix
class LazyDistanceMatrix
{
public:
    LazyDistanceMatrix(size_t n, size_t dim, const double *data);
    double get(size_t i, size_t j) const;
    void enableCache(bool enabled);
};

// Sparse distance matrix
struct EdgeEntry
{
    size_t i;
    size_t j;
    double distance;
};
class SparseDistanceMatrix
{
public:
    SparseDistanceMatrix(size_t n, double threshold);
    void addEdge(size_t i, size_t j, double dist);
    std::vector<EdgeEntry> getEdges() const;
    double sparsityRatio() const;
};

// Sinkhorn Wasserstein
struct SinkhornConfig
{
    double epsilon = 0.1;
    size_t max_iterations = 100;
    double convergence_threshold = 1e-6;
};
class SinkhornWasserstein
{
public:
    explicit SinkhornWasserstein(const SinkhornConfig &config = {});
    double compute(const std::vector<double> &a, const std::vector<double> &b,
                   const std::vector<std::vector<double>> &cost);
};
} // namespace nerve::metrics
