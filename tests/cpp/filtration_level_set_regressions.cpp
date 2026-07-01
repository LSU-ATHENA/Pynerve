#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/core/rng/determinism_contract.hpp"
#include "nerve/core_types.hpp"
#include "nerve/filtration/level_set.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <vector>

namespace
{

using nerve::Index;
using nerve::Size;
using nerve::algebra::Simplex;
using nerve::core::BufferView;
using nerve::core::DeterminismContract;

constexpr double kTol = 1e-10;

bool check_level_set_construction()
{
    nerve::filtration::LevelSet ls({4, 4});
    ls.setFiltrationType("sublevel");
    ls.setNumLevels(10);
    ls.setAdaptiveLevels(false);
    return true;
}

bool check_level_set_build_2d()
{
    nerve::filtration::LevelSet ls({3, 3});
    ls.setFiltrationType("sublevel");
    ls.setNumLevels(5);
    ls.setAdaptiveLevels(false);
    std::vector<double> field{0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
    BufferView<const double> view(field.data(), field.size());
    DeterminismContract contract;
    auto filtration = ls.build2dFiltration(view, 3, 3, contract);
    if (filtration.empty())
    {
        std::cerr << "2D filtration should not be empty\n";
        return false;
    }
    return true;
}

bool check_filtration_monotonic()
{
    nerve::filtration::LevelSet ls({3, 3});
    ls.setFiltrationType("sublevel");
    ls.setNumLevels(5);
    ls.setAdaptiveLevels(false);
    std::vector<double> field{0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
    BufferView<const double> view(field.data(), field.size());
    DeterminismContract contract;
    auto filtration = ls.build2dFiltration(view, 3, 3, contract);
    for (size_t i = 1; i < filtration.size(); ++i)
    {
        if (filtration[i].second < filtration[i - 1].second - kTol)
        {
            std::cerr << "filtration not monotonic at " << i << "\n";
            return false;
        }
    }
    return true;
}

bool check_simplex_dimensions()
{
    nerve::filtration::LevelSet ls({3, 3});
    ls.setFiltrationType("sublevel");
    ls.setNumLevels(5);
    ls.setAdaptiveLevels(false);
    std::vector<double> field{0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
    BufferView<const double> view(field.data(), field.size());
    DeterminismContract contract;
    auto filtration = ls.build2dFiltration(view, 3, 3, contract);
    for (const auto &entry : filtration)
    {
        auto dim = entry.first.dimension();
        if (dim < 0 || dim > 2)
        {
            std::cerr << "invalid simplex dimension: " << dim << "\n";
            return false;
        }
    }
    return true;
}

bool check_get_num_simplices()
{
    nerve::filtration::LevelSet ls({3, 3});
    ls.setFiltrationType("sublevel");
    ls.setNumLevels(5);
    ls.setAdaptiveLevels(false);
    std::vector<double> field{0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
    BufferView<const double> view(field.data(), field.size());
    DeterminismContract contract;
    auto filtration = ls.build2dFiltration(view, 3, 3, contract);
    Size n = ls.getNumSimplices();
    if (n == 0)
    {
        std::cerr << "num simplices should be non-zero\n";
        return false;
    }
    Size n_dim1 = ls.getNumSimplicesOfDimension(1);
    (void)n_dim1;
    return true;
}

bool check_filtration_values()
{
    nerve::filtration::LevelSet ls({3, 3});
    ls.setFiltrationType("sublevel");
    ls.setNumLevels(5);
    ls.setAdaptiveLevels(false);
    std::vector<double> field{0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
    BufferView<const double> view(field.data(), field.size());
    DeterminismContract contract;
    auto filtration = ls.build2dFiltration(view, 3, 3, contract);
    auto values = ls.getFiltrationValues();
    if (values.size() != filtration.size())
    {
        std::cerr << "filtration values size mismatch\n";
        return false;
    }
    for (double v : values)
    {
        if (!std::isfinite(v))
        {
            std::cerr << "non-finite filtration value\n";
            return false;
        }
    }
    return true;
}

bool check_grid_dimension()
{
    nerve::filtration::LevelSet ls({4, 5});
    Size dim = ls.getGridDimension();
    if (dim != 2)
    {
        std::cerr << "grid dimension should be 2, got " << dim << "\n";
        return false;
    }
    Size total = ls.getTotalGridPoints();
    if (total != 20)
    {
        std::cerr << "total grid points should be 20, got " << total << "\n";
        return false;
    }
    return true;
}

bool check_build_with_connectivity()
{
    nerve::filtration::LevelSet ls({3, 3});
    ls.setFiltrationType("sublevel");
    ls.setNumLevels(5);
    ls.setAdaptiveLevels(false);
    std::vector<double> field{0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
    BufferView<const double> view(field.data(), field.size());
    std::vector<std::vector<Index>> conn{{0, 1}, {1, 2}, {0, 3}, {3, 4}};
    DeterminismContract contract;
    auto result = ls.buildFiltration(view, conn, contract);
    if (!result.isError())
    {
        auto filtration = result.value();
        if (!filtration.empty())
        {
            for (const auto &entry : filtration)
            {
                if (entry.second < 0.0)
                {
                    std::cerr << "negative filtration value with connectivity\n";
                    return false;
                }
            }
        }
    }
    return true;
}

bool check_build_without_connectivity()
{
    nerve::filtration::LevelSet ls({3, 3});
    ls.setFiltrationType("sublevel");
    ls.setNumLevels(5);
    ls.setAdaptiveLevels(false);
    std::vector<double> field{0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
    BufferView<const double> view(field.data(), field.size());
    DeterminismContract contract;
    auto result = ls.buildFiltration(view, contract);
    if (!result.isError())
    {
        auto filtration = result.value();
        if (filtration.empty())
        {
            std::cerr << "filtration should not be empty\n";
            return false;
        }
    }
    return true;
}

bool check_compute_levels()
{
    nerve::filtration::LevelSet ls({3, 3});
    ls.setNumLevels(5);
    std::vector<double> field{0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
    BufferView<const double> view(field.data(), field.size());
    auto levels = ls.computeLevels(view);
    if (levels.empty())
    {
        std::cerr << "levels should not be empty\n";
        return false;
    }
    if (levels.size() != 5)
    {
        std::cerr << "expected 5 levels, got " << levels.size() << "\n";
        return false;
    }
    return true;
}

} // namespace

int main()
{
    if (!check_level_set_construction())
    {
        std::cerr << "FAIL: level set construction\n";
        return 1;
    }
    if (!check_level_set_build_2d())
    {
        std::cerr << "FAIL: 2D build\n";
        return 1;
    }
    if (!check_filtration_monotonic())
    {
        std::cerr << "FAIL: monotonic\n";
        return 1;
    }
    if (!check_simplex_dimensions())
    {
        std::cerr << "FAIL: simplex dims\n";
        return 1;
    }
    if (!check_get_num_simplices())
    {
        std::cerr << "FAIL: num simplices\n";
        return 1;
    }
    if (!check_filtration_values())
    {
        std::cerr << "FAIL: filtration values\n";
        return 1;
    }
    if (!check_grid_dimension())
    {
        std::cerr << "FAIL: grid dimension\n";
        return 1;
    }
    if (!check_build_with_connectivity())
    {
        std::cerr << "FAIL: build with connectivity\n";
        return 1;
    }
    if (!check_build_without_connectivity())
    {
        std::cerr << "FAIL: build without connectivity\n";
        return 1;
    }
    if (!check_compute_levels())
    {
        std::cerr << "FAIL: compute levels\n";
        return 1;
    }
    return 0;
}
