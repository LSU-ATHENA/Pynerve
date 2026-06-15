#pragma once

/// @file landmark_selector.hpp

#include <cstddef>
#include <vector>

namespace nerve::persistence
{

class LandmarkSelector
{
public:
    enum class Strategy
    {
        MAXMIN,
        RANDOM,
        GRID,
        DENSITY
    };

    static std::vector<size_t> selectLandmarks(const std::vector<double> &points, size_t point_dim,
                                               size_t num_points, size_t num_landmarks,
                                               Strategy strategy);

private:
    static std::vector<size_t> selectMaxmin(const std::vector<double> &points, size_t point_dim,
                                            size_t num_points, size_t num_landmarks);

    static std::vector<size_t> selectRandom(size_t num_points, size_t num_landmarks);

    static std::vector<size_t> selectGrid(const std::vector<double> &points, size_t point_dim,
                                          size_t num_points, size_t num_landmarks);

    static std::vector<size_t> selectDensityWeighted(const std::vector<double> &points,
                                                     size_t point_dim, size_t num_points,
                                                     size_t num_landmarks);
};

} // namespace nerve::persistence
