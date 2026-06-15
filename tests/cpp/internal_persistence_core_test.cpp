
#include "nerve/persistence/core/core_types.hpp"
#include "nerve/persistence/core/detail/core_detail.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

namespace
{

bool check_per_dimension_h0()
{
    std::vector<std::vector<int>> simplices = {{0}, {1}, {2}, {0, 1}, {1, 2}};
    std::vector<double> filt = {0.0, 0.0, 0.0, 1.0, 1.5};
    auto result = nerve::persistence::perdim::computeH0UnionFind(simplices, filt);
    if (result.pairs.empty())
        return false;
    if (result.num_pairs <= 0)
        return false;
    return true;
}

bool check_per_dimension_h1()
{
    std::vector<std::vector<int>> simplices = {{0}, {1}, {2}, {0, 1}, {1, 2}, {0, 2}};
    std::vector<double> filt = {0.0, 0.0, 0.0, 1.0, 1.5, 2.0};
    std::vector<int> dims = {0, 0, 0, 1, 1, 1};
    auto result = nerve::persistence::perdim::computeH1ReducedVR(simplices, filt, dims);
    if (result.num_pairs < 0)
        return false;
    return true;
}

bool check_per_dimension_h2()
{
    std::vector<std::vector<int>> simplices = {{0},    {1},    {2},    {3},    {0, 1},   {0, 2},
                                               {0, 3}, {1, 2}, {1, 3}, {2, 3}, {0, 1, 2}};
    std::vector<double> filt = {0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2};
    std::vector<int> dims = {0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2};
    auto result = nerve::persistence::perdim::computeH2AlphaComplex(simplices, filt, dims);
    if (result.num_pairs < 0)
        return false;
    return true;
}

bool check_vram_selection()
{
    nerve::persistence::vram::VRAMConfig cfg;
    cfg.available_vram_bytes = 1024ULL * 1024ULL * 1024ULL;
    cfg.total_vram_bytes = 2ULL * 1024ULL * 1024ULL * 1024ULL;
    cfg.safety_fraction = 0.8;
    auto safe = cfg.safeBytes();
    if (safe == 0)
        return false;
    auto algo = cfg.select(100, 3);
    (void)algo;
    return true;
}

bool check_memory_reduction_config()
{
    nerve::persistence::extreme::AutomaticMemoryOptimizer opt(4ULL * 1024ULL * 1024ULL * 1024ULL);
    auto tier = opt.selectTier(100, 3, 1.0);
    (void)tier;
    return true;
}

bool check_flood_complex_utils()
{
    std::vector<double> points = {0.0, 0.0, 1.0, 0.0, 0.0, 1.0};
    auto selected = nerve::persistence::farthestPointSampling(points, 2, 3, 2);
    if (selected.empty())
        return false;
    if (selected.size() != 2)
        return false;
    return true;
}

bool check_delaunay_helpers()
{
    if (nerve::persistence::determinant3x3({{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}}) != 1.0)
        return false;
    std::array<double, 3> x;
    bool solved = nerve::persistence::solve3x3({{{2, 0, 0}, {0, 2, 0}, {0, 0, 2}}}, {1, 1, 1}, &x);
    if (!solved)
        return false;
    if (std::abs(x[0] - 0.5) > 1e-12)
        return false;
    return true;
}

bool check_roaring_column_basic()
{
    nerve::persistence::roaring::RoaringColumn col;
    col.add(5);
    col.add(10);
    int pivot = col.computePivot();
    if (pivot < 0)
        return false;
    return true;
}

bool check_roaring_hybrid_basic()
{
    nerve::persistence::roaring::HybridColumn hcol(64);
    hcol.add(3);
    hcol.add(7);
    int pivot = hcol.computePivot();
    if (pivot != 7)
        return false;
    return true;
}

} // namespace

int main()
{
    if (!check_per_dimension_h0())
        return 1;
    if (!check_per_dimension_h1())
        return 1;
    if (!check_per_dimension_h2())
        return 1;
    if (!check_vram_selection())
        return 1;
    if (!check_memory_reduction_config())
        return 1;
    if (!check_flood_complex_utils())
        return 1;
    if (!check_delaunay_helpers())
        return 1;
    if (!check_roaring_column_basic())
        return 1;
    if (!check_roaring_hybrid_basic())
        return 1;
    return 0;
}
