#pragma once
#include "nerve/core_types.hpp"

#include <array>
#include <vector>

namespace nerve::persistence
{
struct H0Result
{
    std::vector<std::pair<int, double>> pairs;
    int num_pairs = 0;
    double num_components = 0;
};
struct H1Result
{
    int num_pairs = 0;
};
struct H2Result
{
    int num_pairs = 0;
};
} // namespace nerve::persistence

namespace nerve::persistence::perdim
{
H0Result computeH0UnionFind(const std::vector<std::vector<int>> &simplices,
                            const std::vector<double> &filtration_values);
H1Result computeH1ReducedVR(const std::vector<std::vector<int>> &simplices,
                            const std::vector<double> &filt, const std::vector<int> &dims);
H2Result computeH2AlphaComplex(const std::vector<std::vector<int>> &simplices,
                               const std::vector<double> &filt, const std::vector<int> &dims);
} // namespace nerve::persistence::perdim

namespace nerve::persistence::vram
{
struct VRAMConfig
{
    uint64_t available_vram_bytes = 0;
    uint64_t total_vram_bytes = 0;
    double safety_fraction = 0.8;
    uint64_t safeBytes() const;
    int select(size_t n_points, size_t point_dim) const;
};
} // namespace nerve::persistence::vram

namespace nerve::persistence::extreme
{
class AutomaticMemoryOptimizer
{
public:
    explicit AutomaticMemoryOptimizer(uint64_t vram_bytes);
    int selectTier(size_t n_points, size_t point_dim, double max_radius);
};
} // namespace nerve::persistence::extreme

namespace nerve::persistence
{
std::vector<int> farthestPointSampling(const std::vector<double> &points, int k, size_t n_points,
                                       size_t point_dim);
double determinant3x3(const std::array<std::array<double, 3>, 3> &mat);
bool solve3x3(const std::array<std::array<double, 3>, 3> &A, const std::array<double, 3> &b,
              std::array<double, 3> *x);
} // namespace nerve::persistence

namespace nerve::persistence::roaring
{
class RoaringColumn
{
public:
    RoaringColumn();
    void add(int value);
    int computePivot();
};
class HybridColumn
{
public:
    explicit HybridColumn(int capacity);
    void add(int value);
    int computePivot();
};
} // namespace nerve::persistence::roaring
