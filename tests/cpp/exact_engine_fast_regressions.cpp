#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/core_types.hpp"
#include "nerve/persistence/core/core_types.hpp"
#include "nerve/persistence/utils/exact_engine_fast.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace
{

using nerve::Size;
using nerve::core::BufferView;

bool check_fast_engine_triangle()
{
    constexpr int kDim = 2;
    constexpr int kN = 3;
    constexpr double kThr = 2.0;

    std::vector<std::vector<int>> neighbors(kN);
    for (int i = 0; i < kN; ++i)
    {
        for (int j = 0; j < kN; ++j)
        {
            if (i != j)
                neighbors[i].push_back(j);
        }
    }

    std::unordered_map<std::uint64_t, double> edge_w;
    auto key = [](int a, int b) -> std::uint64_t {
        return (static_cast<std::uint64_t>(a) << 32) | static_cast<std::uint64_t>(b);
    };
    auto dist = [](int i, int j) -> double {
        (void)i;
        (void)j;
        return 1.0;
    };
    for (int i = 0; i < kN; ++i)
        for (int j = i + 1; j < kN; ++j)
            edge_w[key(i, j)] = dist(i, j);

    auto result =
        nerve::persistence::computeExactCohomologyZ2Fast(kN, kDim, kThr, neighbors, edge_w);

    if (result.pairs.empty())
    {
        std::cerr << "fast engine: triangle should produce pairs\n";
        return false;
    }

    bool found_h0 = false;
    bool found_h0_ess = false;
    for (const auto &p : result.pairs)
    {
        if (p.dimension == 0 && !std::isinf(p.death))
            found_h0 = true;
        if (p.dimension == 0 && std::isinf(p.death))
            found_h0_ess = true;
    }

    if (!found_h0)
    {
        std::cerr << "fast engine: triangle should have finite H0 pairs\n";
        return false;
    }
    if (!found_h0_ess)
    {
        std::cerr << "fast engine: triangle should have essential H0 pair\n";
        return false;
    }

    return true;
}

bool check_fast_engine_square()
{
    constexpr int kDim = 2;
    constexpr int kN = 4;
    constexpr double kThr = 2.0;

    std::vector<std::vector<int>> neighbors(kN);
    for (int i = 0; i < kN; ++i)
    {
        for (int j = 0; j < kN; ++j)
        {
            if (i != j)
                neighbors[i].push_back(j);
        }
    }

    std::unordered_map<std::uint64_t, double> edge_w;
    auto key = [](int a, int b) -> std::uint64_t {
        return (static_cast<std::uint64_t>(a) << 32) | static_cast<std::uint64_t>(b);
    };
    double coords[4][2] = {{0, 0}, {1, 0}, {0, 1}, {1, 1}};
    for (int i = 0; i < kN; ++i)
    {
        for (int j = i + 1; j < kN; ++j)
        {
            double dx = coords[i][0] - coords[j][0];
            double dy = coords[i][1] - coords[j][1];
            edge_w[key(i, j)] = std::sqrt(dx * dx + dy * dy);
        }
    }

    auto result =
        nerve::persistence::computeExactCohomologyZ2Fast(kN, kDim, kThr, neighbors, edge_w);

    if (result.pairs.empty())
    {
        std::cerr << "fast engine: square should produce pairs\n";
        return false;
    }

    bool found_h0_ess = false;
    for (const auto &p : result.pairs)
    {
        if (p.dimension == 0 && std::isinf(p.death))
            found_h0_ess = true;
    }

    if (!found_h0_ess)
    {
        std::cerr << "fast engine: square should have H0 essential\n";
        return false;
    }

    return true;
}

bool check_fast_engine_tetrahedron()
{
    constexpr int kDim = 3;
    constexpr int kN = 4;
    constexpr double kThr = 2.0;

    std::vector<std::vector<int>> neighbors(kN);
    for (int i = 0; i < kN; ++i)
    {
        for (int j = 0; j < kN; ++j)
        {
            if (i != j)
                neighbors[i].push_back(j);
        }
    }

    std::unordered_map<std::uint64_t, double> edge_w;
    auto key = [](int a, int b) -> std::uint64_t {
        return (static_cast<std::uint64_t>(a) << 32) | static_cast<std::uint64_t>(b);
    };
    double coords[4][3] = {{0, 0, 0}, {1, 0, 0}, {0.5, 0.866, 0}, {0.5, 0.289, 0.816}};
    for (int i = 0; i < kN; ++i)
    {
        for (int j = i + 1; j < kN; ++j)
        {
            double dx = coords[i][0] - coords[j][0];
            double dy = coords[i][1] - coords[j][1];
            double dz = coords[i][2] - coords[j][2];
            edge_w[key(i, j)] = std::sqrt(dx * dx + dy * dy + dz * dz);
        }
    }

    auto result =
        nerve::persistence::computeExactCohomologyZ2Fast(kN, kDim, kThr, neighbors, edge_w);

    if (result.pairs.empty())
    {
        std::cerr << "fast engine: tetrahedron should produce pairs\n";
        return false;
    }

    bool found_h0_ess = false;
    for (const auto &p : result.pairs)
    {
        if (p.dimension == 0 && std::isinf(p.death))
            found_h0_ess = true;
    }

    if (!found_h0_ess)
    {
        std::cerr << "fast engine: tetrahedron should have H0 essential\n";
        return false;
    }

    return true;
}

bool check_fast_engine_pair_counts()
{
    constexpr int kDim = 2;
    constexpr int kN = 3;
    constexpr double kThr = 2.0;

    std::vector<std::vector<int>> neighbors(kN);
    for (int i = 0; i < kN; ++i)
    {
        for (int j = 0; j < kN; ++j)
        {
            if (i != j)
                neighbors[i].push_back(j);
        }
    }

    std::unordered_map<std::uint64_t, double> edge_w;
    auto key = [](int a, int b) -> std::uint64_t {
        return (static_cast<std::uint64_t>(a) << 32) | static_cast<std::uint64_t>(b);
    };
    for (int i = 0; i < kN; ++i)
        for (int j = i + 1; j < kN; ++j)
            edge_w[key(i, j)] = 1.0;

    auto result =
        nerve::persistence::computeExactCohomologyZ2Fast(kN, kDim, kThr, neighbors, edge_w);

    Size finite_h0 = 0;
    Size essential_h0 = 0;
    for (const auto &p : result.pairs)
    {
        if (p.dimension == 0)
        {
            if (std::isinf(p.death))
                ++essential_h0;
            else
                ++finite_h0;
        }
    }

    if (finite_h0 != 2)
    {
        std::cerr << "fast engine: expected 2 finite H0, got " << finite_h0 << "\n";
        return false;
    }
    if (essential_h0 != 1)
    {
        std::cerr << "fast engine: expected 1 essential H0, got " << essential_h0 << "\n";
        return false;
    }

    return true;
}

} // namespace

int main()
{
    if (!check_fast_engine_triangle())
    {
        std::cerr << "FAIL: fast engine triangle\n";
        return 1;
    }
    if (!check_fast_engine_square())
    {
        std::cerr << "FAIL: fast engine square\n";
        return 1;
    }
    if (!check_fast_engine_tetrahedron())
    {
        std::cerr << "FAIL: fast engine tetrahedron\n";
        return 1;
    }
    if (!check_fast_engine_pair_counts())
    {
        std::cerr << "FAIL: fast engine pair counts\n";
        return 1;
    }
    return 0;
}
