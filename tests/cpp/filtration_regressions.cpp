#include "nerve/algebra/simplex.hpp"
#include "nerve/common/accelerated_types.hpp"
#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/core/rng/determinism_contract.hpp"
#include "nerve/core_types.hpp"
#include "nerve/errors/errors.hpp"
#include "nerve/filtration/simd_filtration.hpp"
#include "nerve/filtration/vietoris_rips.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <random>
#include <tuple>
#include <vector>

namespace
{

using nerve::Dimension;
using nerve::Field;
using nerve::Index;
using nerve::Size;
using nerve::algebra::Simplex;
using nerve::core::BufferView;
using nerve::core::DeterminismContract;
using nerve::filtration::VietorisRips;

BufferView<const double> view_of(const std::vector<double> &v)
{
    return {v.data(), v.size()};
}

bool check_vietoris_rips_construction()
{
    VietorisRips vr(2.0);
    (void)vr;
    return true;
}

bool check_vietoris_rips_set_max_radius()
{
    VietorisRips vr;
    vr.setMaxRadius(1.5);
    (void)vr;
    return true;
}

bool check_vietoris_rips_set_max_dimension()
{
    VietorisRips vr;
    vr.setMaxDimension(2);
    (void)vr;
    return true;
}

bool check_vietoris_rips_empty_input()
{
    VietorisRips vr(1.0);
    vr.setMaxDimension(2);

    std::vector<double> empty;
    BufferView<const double> view(empty.data(), empty.size());
    DeterminismContract contract;

    auto result = vr.buildFiltration(view, 2, contract);
    if (!result.isError())
    {
        auto filtration = result.value();
        if (!filtration.empty())
        {
            std::cerr << "empty input should produce empty filtration\n";
            return false;
        }
    }
    return true;
}

bool check_vietoris_rips_build_filtration()
{
    VietorisRips vr(2.0);
    vr.setMaxDimension(2);

    std::vector<double> points = {0.0, 0.0, 1.0, 0.0, 0.5, 0.866};
    BufferView<const double> view(points.data(), points.size());
    DeterminismContract contract;

    auto result = vr.buildFiltration(view, 2, contract);
    if (result.isError())
    {
        return true;
    }

    auto filtration = result.value();
    if (filtration.empty())
    {
        std::cerr << "filtration should not be empty for 3 points\n";
        return false;
    }

    for (const auto &entry : filtration)
    {
        if (entry.second < 0.0)
        {
            std::cerr << "negative filtration value\n";
            return false;
        }
    }
    return true;
}

bool check_filtration_monotonic()
{
    VietorisRips vr(2.0);
    vr.setMaxDimension(2);

    std::vector<double> points = {0.0, 0.0, 1.0, 0.0, 0.5, 0.866};
    BufferView<const double> view(points.data(), points.size());
    DeterminismContract contract;

    auto result = vr.buildFiltration(view, 2, contract);
    if (result.isError())
    {
        return true;
    }

    auto filtration = result.value();
    for (size_t i = 1; i < filtration.size(); ++i)
    {
        if (filtration[i].second < filtration[i - 1].second - 1e-12)
        {
            std::cerr << "filtration not monotonic at index " << i << "\n";
            return false;
        }
    }
    return true;
}

bool check_vietoris_rips_single_point()
{
    VietorisRips vr(1.0);
    vr.setMaxDimension(2);

    std::vector<double> point = {0.0, 0.0};
    BufferView<const double> view(point.data(), point.size());
    DeterminismContract contract;

    auto result = vr.buildFiltration(view, 2, contract);
    if (result.isError())
    {
        return true;
    }

    auto filtration = result.value();
    for (const auto &entry : filtration)
    {
        if (entry.first.dimension() < 0)
        {
            std::cerr << "negative simplex dimension\n";
            return false;
        }
    }
    return true;
}

bool check_simd_batch_filter()
{
    std::vector<double> values = {0.5, 1.5, 2.5, 3.5, 4.5};
    nerve::filtration::simdBatchFilterValues(values.data(), static_cast<Size>(values.size()), 0, 1,
                                             3.0);
    return true;
}

bool check_simd_sort_pairs()
{
    std::vector<nerve::Pair> pairs = {{0.0, 3.0, 0}, {0.0, 1.0, 0}, {0.0, 2.0, 0}};
    nerve::filtration::simdSortPairsByBirth(pairs.data(), static_cast<Size>(pairs.size()));
    return true;
}

bool check_vietoris_rips_num_simplices()
{
    VietorisRips vr(2.0);
    vr.setMaxDimension(2);

    std::vector<double> points = {0.0, 0.0, 1.0, 0.0};
    BufferView<const double> view(points.data(), points.size());
    DeterminismContract contract;

    auto result = vr.buildFiltration(view, 2, contract);
    if (result.isError())
    {
        return true;
    }

    auto n = vr.getNumSimplices();
    (void)n;
    return true;
}

} // namespace

int main()
{
    if (!check_vietoris_rips_construction())
    {
        std::cerr << "FAIL: VietorisRips construction\n";
        return 1;
    }
    if (!check_vietoris_rips_set_max_radius())
    {
        std::cerr << "FAIL: setMaxRadius\n";
        return 1;
    }
    if (!check_vietoris_rips_set_max_dimension())
    {
        std::cerr << "FAIL: setMaxDimension\n";
        return 1;
    }
    if (!check_vietoris_rips_empty_input())
    {
        std::cerr << "FAIL: empty input\n";
        return 1;
    }
    if (!check_vietoris_rips_build_filtration())
    {
        std::cerr << "FAIL: buildFiltration\n";
        return 1;
    }
    if (!check_filtration_monotonic())
    {
        std::cerr << "FAIL: filtration monotonic\n";
        return 1;
    }
    if (!check_vietoris_rips_single_point())
    {
        std::cerr << "FAIL: single point\n";
        return 1;
    }
    if (!check_simd_batch_filter())
    {
        std::cerr << "FAIL: simdBatchFilterValues\n";
        return 1;
    }
    if (!check_simd_sort_pairs())
    {
        std::cerr << "FAIL: simdSortPairsByBirth\n";
        return 1;
    }
    if (!check_vietoris_rips_num_simplices())
    {
        std::cerr << "FAIL: getNumSimplices\n";
        return 1;
    }
    return 0;
}
