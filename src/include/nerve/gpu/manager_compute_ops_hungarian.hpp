#pragma once

#include "nerve/errors/detail/error_result.hpp"

#include <utility>
#include <vector>

namespace nerve::gpu
{

errors::ErrorResult<void>
validateSquareCostMatrix(const std::vector<std::vector<double>> &cost_matrix, bool allow_infinity);

errors::ErrorResult<double>
solveAssignmentHungarian(const std::vector<std::vector<double>> &cost_matrix,
                         std::vector<std::pair<int, int>> &out_assignment);

bool augmentThresholdMatching(std::size_t left, const std::vector<std::vector<double>> &costs,
                              double threshold, std::vector<char> &seen,
                              std::vector<int> &match_right);

bool thresholdPerfectMatching(const std::vector<std::vector<double>> &costs, double threshold,
                              std::vector<std::pair<int, int>> *assignment);

} // namespace nerve::gpu
