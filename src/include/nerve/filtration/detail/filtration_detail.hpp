#pragma once
#include "nerve/filtration/level_set.hpp"
#include "nerve/filtration/vietoris_rips.hpp"

#include <vector>

namespace nerve::filtration
{
// Level set connectivity
struct GridConnectivity
{
    std::vector<std::vector<int>> components;
    int num_components;
};
GridConnectivity computeLevelSetConnectivity(const std::vector<std::vector<double>> &grid,
                                             double threshold);

// VR builder helpers
class VRBuilder
{
public:
    VRBuilder(size_t n_points, size_t point_dim);
    size_t numVertices() const;
    std::vector<std::pair<int, double>> allPairDistances(const double *points);
    std::vector<std::pair<int, double>> kNearestNeighbors(const double *points, int k);
};
} // namespace nerve::filtration
