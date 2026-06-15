#pragma once

#include "nerve/algebra/boundary.hpp"
#include "nerve/errors/detail/error_result.hpp"
#include "nerve/gpu/kernel_launcher.hpp"

#include <utility>
#include <vector>

namespace nerve::gpu
{

errors::ErrorResult<void>
buildVietorisRipsFiltration(const std::vector<std::vector<double>> &points, double max_radius,
                            int max_dimension,
                            std::vector<std::pair<algebra::Simplex, double>> &filtration);

errors::ErrorResult<void>
convertVrFiltration(const std::vector<std::pair<algebra::Simplex, double>> &filtration,
                    std::vector<ComputeManager::VRSimplex> &out_simplices);

errors::ErrorResult<void> buildCechLikeComplex(const std::vector<std::vector<double>> &points,
                                               double max_radius, int max_dimension,
                                               std::vector<ComputeManager::CechSimplex> &out);

} // namespace nerve::gpu
