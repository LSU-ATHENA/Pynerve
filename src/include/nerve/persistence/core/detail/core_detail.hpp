#pragma once
#include "nerve/core_types.hpp"
#include "nerve/persistence/core/per_dimension_exact.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace nerve::persistence::vram
{
struct VRAMConfig
{
    uint64_t available_vram_bytes = 0;
    uint64_t total_vram_bytes = 0;
    double safety_fraction = 0.8;
    uint64_t safeBytes() const
    {
        return static_cast<uint64_t>(available_vram_bytes * safety_fraction);
    }
    int select(size_t n_points, size_t point_dim) const
    {
        (void)n_points;
        (void)point_dim;
        return 0;
    }
};
} // namespace nerve::persistence::vram

namespace nerve::persistence::extreme
{
class AutomaticMemoryOptimizer
{
public:
    explicit AutomaticMemoryOptimizer(uint64_t vram_bytes) { (void)vram_bytes; }
    int selectTier(size_t n_points, size_t point_dim, double max_radius)
    {
        (void)n_points;
        (void)point_dim;
        (void)max_radius;
        return 0;
    }
};
} // namespace nerve::persistence::extreme

namespace nerve::persistence
{
inline std::vector<int> farthestPointSampling(const std::vector<double> &points, int k,
                                              size_t n_points, size_t point_dim)
{
    (void)points;
    (void)point_dim;
    std::vector<int> result;
    result.reserve(static_cast<size_t>(k));
    for (int i = 0; i < k && static_cast<size_t>(i) < n_points; ++i)
    {
        result.push_back(i);
    }
    return result;
}
inline double determinant3x3(const std::array<std::array<double, 3>, 3> &mat)
{
    return mat[0][0] * (mat[1][1] * mat[2][2] - mat[1][2] * mat[2][1]) -
           mat[0][1] * (mat[1][0] * mat[2][2] - mat[1][2] * mat[2][0]) +
           mat[0][2] * (mat[1][0] * mat[2][1] - mat[1][1] * mat[2][0]);
}
inline bool solve3x3(const std::array<std::array<double, 3>, 3> &A, const std::array<double, 3> &b,
                     std::array<double, 3> *x)
{
    double det = determinant3x3(A);
    if (std::abs(det) < 1e-12)
        return false;
    double inv_det = 1.0 / det;
    if (x)
    {
        (*x)[0] = inv_det * (b[0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
                             A[0][1] * (b[1] * A[2][2] - A[1][2] * b[2]) +
                             A[0][2] * (b[1] * A[2][1] - A[1][1] * b[2]));
        (*x)[1] = inv_det * (A[0][0] * (b[1] * A[2][2] - A[1][2] * b[2]) -
                             b[0] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
                             A[0][2] * (A[1][0] * b[2] - b[1] * A[2][0]));
        (*x)[2] = inv_det * (A[0][0] * (A[1][1] * b[2] - b[1] * A[2][1]) -
                             A[0][1] * (A[1][0] * b[2] - b[1] * A[2][0]) +
                             b[0] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]));
    }
    return true;
}
} // namespace nerve::persistence

namespace nerve::persistence::roaring
{
class RoaringColumn
{
public:
    RoaringColumn() = default;
    void add(int value) { values_.push_back(value); }
    int computePivot()
    {
        if (values_.empty())
            return -1;
        return *std::max_element(values_.begin(), values_.end());
    }

private:
    std::vector<int> values_;
};
class HybridColumn
{
public:
    explicit HybridColumn(int capacity) { values_.reserve(static_cast<size_t>(capacity)); }
    void add(int value) { values_.push_back(value); }
    int computePivot() const
    {
        if (values_.empty())
            return -1;
        return *std::max_element(values_.begin(), values_.end());
    }

private:
    std::vector<int> values_;
};
} // namespace nerve::persistence::roaring
