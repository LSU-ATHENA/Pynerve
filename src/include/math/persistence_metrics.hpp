
#pragma once

#include "math/persistence_metrics/algorithms.hpp"

#include <exception>
#include <string>
#include <vector>

namespace nerve::math
{

inline error::Result<HungarianSolver>
makeHungarianSolver(const std::vector<std::vector<double>> &cost_matrix)
{
    try
    {
        return error::Result<HungarianSolver>::ok(HungarianSolver(cost_matrix));
    }
    catch (const std::exception &ex)
    {
        return error::Result<HungarianSolver>::err(
            error::TDAErrorCode::InvalidFieldOperation,
            std::string("Failed to create Hungarian solver: ") + ex.what());
    }
}

inline error::Result<OptimalTransport> makeOptimalTransport(double p = 2.0)
{
    try
    {
        return error::Result<OptimalTransport>::ok(OptimalTransport(p));
    }
    catch (const std::exception &ex)
    {
        return error::Result<OptimalTransport>::err(
            error::TDAErrorCode::InvalidFieldOperation,
            std::string("Failed to create optimal transport solver: ") + ex.what());
    }
}

} // namespace nerve::math
