#pragma once
#include "nerve/core_types.hpp"

#include <cstddef>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

namespace nerve::cpu::simd
{

class CPUFeatureDetector
{
public:
    [[nodiscard]] static bool hasAVX512F();
    [[nodiscard]] static bool hasAVX512VL();
    [[nodiscard]] static bool hasAVX512BW();
    [[nodiscard]] static bool hasAVX2();
    [[nodiscard]] static bool hasFMA();
    [[nodiscard]] static int getMaxSIMDWidth();
    [[nodiscard]] static std::string getCPUModel();
    [[nodiscard]] static int getNumCores();
    [[nodiscard]] static int getNumThreads();
};

[[nodiscard]] bool hasAVX512F();
[[nodiscard]] bool hasAVX2();
[[nodiscard]] int getMaxSIMDWidth();
[[nodiscard]] std::string getCPUModel();
[[nodiscard]] int getNumCores();
[[nodiscard]] int getNumThreads();

[[nodiscard]] nerve::persistence::PersistenceDiagram
computeCPUOptimized(std::span<const double> points, size_t n_points, size_t point_dim,
                    double max_distance);

template <typename T>
[[nodiscard]] inline nerve::persistence::PersistenceDiagram
computeCPUOptimized(std::span<const T> points, size_t n_points, size_t point_dim, T max_distance)
{
    if constexpr (std::is_same_v<T, double>)
    {
        return computeCPUOptimized(std::span<const double>(points.data(), points.size()), n_points,
                                   point_dim, static_cast<double>(max_distance));
    }
    else
    {
        std::vector<double> converted(points.size());
        for (size_t i = 0; i < points.size(); ++i)
        {
            converted[i] = static_cast<double>(points[i]);
        }
        return computeCPUOptimized(std::span<const double>(converted.data(), converted.size()),
                                   n_points, point_dim, static_cast<double>(max_distance));
    }
}

[[nodiscard]] std::string getCPUOptimizationReport();

} // namespace nerve::cpu::simd
