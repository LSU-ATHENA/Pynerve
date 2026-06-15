#pragma once

#include <vector>

namespace nerve::gpu
{

double euclideanDistance(const std::vector<double> &lhs, const std::vector<double> &rhs);

double triangleCircumradius(double a, double b, double c);

double enclosingRadiusEstimate(const std::vector<std::vector<double>> &points,
                               const std::vector<int> &vertices);

} // namespace nerve::gpu
