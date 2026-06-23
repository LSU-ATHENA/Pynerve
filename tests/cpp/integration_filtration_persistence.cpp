#include "nerve/algebra/complex.hpp"
#include "nerve/algebra/simplex.hpp"
#include "nerve/common/accelerated_types.hpp"
#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/core_types.hpp"
#include "nerve/filtration/vietoris_rips.hpp"
#include "nerve/persistence/utils/exact_engine.hpp"
#include "nerve/persistence/vr/vr_fast_ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <tuple>
#include <vector>

namespace
{

using nerve::Dimension;
using nerve::Field;
using nerve::Size;
using nerve::algebra::Simplex;
using nerve::common::VRConfig;
using nerve::core::BufferView;
using nerve::persistence::Pair;
using nerve::persistence::VRAlgorithmSelection;

constexpr double kTol = 1e-10;

BufferView<const double> view_of(const std::vector<double> &v)
{
    return {v.data(), v.size()};
}

std::vector<Pair> canonical(std::vector<Pair> pairs)
{
    std::sort(pairs.begin(), pairs.end(), [](const Pair &a, const Pair &b) {
        return std::tuple(a.dimension, a.birth, a.death) <
               std::tuple(b.dimension, b.birth, b.death);
    });
    return pairs;
}

bool pairs_equal(const Pair &a, const Pair &b)
{
    if (a.dimension != b.dimension)
        return false;
    if (std::abs(a.birth - b.birth) > kTol)
        return false;
    if (a.isInfinite() && b.isInfinite())
        return true;
    if (a.isInfinite() || b.isInfinite())
        return false;
    return std::abs(a.death - b.death) < kTol;
}

static bool is_diagonal_pair(const Pair &p)
{
    return std::abs(p.birth - p.death) < 1e-12 || (std::isinf(p.death) && p.birth > 0);
}

bool assert_same_pairs(const std::vector<Pair> &expected, const std::vector<Pair> &actual)
{
    auto c1 = canonical(expected);
    auto c2 = canonical(actual);
    c1.erase(std::remove_if(c1.begin(), c1.end(), is_diagonal_pair), c1.end());
    c2.erase(std::remove_if(c2.begin(), c2.end(), is_diagonal_pair), c2.end());
    if (c1.size() != c2.size())
    {
        std::cerr << "finite pair count mismatch: " << c1.size() << " vs " << c2.size() << "\n";
        return false;
    }
    for (std::size_t i = 0; i < c1.size(); ++i)
    {
        if (!pairs_equal(c1[i], c2[i]))
        {
            std::cerr << "pair " << i << " differs: dim=" << c1[i].dimension
                      << " birth=" << c1[i].birth << " death=" << c1[i].death
                      << " vs dim=" << c2[i].dimension << " birth=" << c2[i].birth
                      << " death=" << c2[i].death << "\n";
            return false;
        }
    }
    return true;
}

bool check_filtration_vs_direct_triangle()
{
    const std::vector<double> pts = {0.0, 0.0, 1.0, 0.0, 0.0, 1.0};

    nerve::filtration::VietorisRips vr;
    vr.setMaxRadius(2.0);
    vr.setMaxDimension(2);
    auto filtration_result = vr.buildFiltration(view_of(pts), 2);
    if (filtration_result.isError())
    {
        std::cerr << "filtration build failed\n";
        return false;
    }
    auto filtration = filtration_result.moveValue();
    if (filtration.empty())
    {
        std::cerr << "filtration is empty\n";
        return false;
    }

    nerve::persistence::IncrementalExactZ2 engine;
    for (const auto &entry : filtration)
    {
        engine.addSimplex(entry.first, entry.second);
    }
    auto filtration_pairs = engine.snapshot();

    VRConfig cfg;
    cfg.max_radius = 2.0;
    cfg.max_dim = 2;
    cfg.algorithm = VRAlgorithmSelection::EXACT_STANDARD;

    const auto direct_pairs = nerve::persistence::computeVrPersistenceFast(view_of(pts), 2, cfg);

    if (!assert_same_pairs(direct_pairs, filtration_pairs.pairs))
    {
        std::cerr << "FAIL: filtration vs direct triangle\n";
        return false;
    }
    return true;
}

bool check_filtration_vs_direct_square()
{
    const std::vector<double> pts = {0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0};

    nerve::filtration::VietorisRips vr;
    vr.setMaxRadius(2.0);
    vr.setMaxDimension(2);
    auto filtration_result = vr.buildFiltration(view_of(pts), 2);
    if (filtration_result.isError())
    {
        std::cerr << "square filtration build failed\n";
        return false;
    }
    auto filtration = filtration_result.moveValue();

    nerve::persistence::IncrementalExactZ2 engine;
    for (const auto &entry : filtration)
    {
        engine.addSimplex(entry.first, entry.second);
    }
    auto filtration_pairs = engine.snapshot();

    VRConfig cfg;
    cfg.max_radius = 2.0;
    cfg.max_dim = 2;
    cfg.algorithm = VRAlgorithmSelection::EXACT_STANDARD;

    const auto direct_pairs = nerve::persistence::computeVrPersistenceFast(view_of(pts), 2, cfg);

    if (!assert_same_pairs(direct_pairs, filtration_pairs.pairs))
    {
        std::cerr << "FAIL: filtration vs direct square\n";
        return false;
    }
    return true;
}

bool check_filtration_vs_direct_tetrahedron()
{
    const std::vector<double> pts = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};

    nerve::filtration::VietorisRips vr;
    vr.setMaxRadius(2.0);
    vr.setMaxDimension(3);
    auto filtration_result = vr.buildFiltration(view_of(pts), 3);
    if (filtration_result.isError())
    {
        std::cerr << "tetrahedron filtration build failed\n";
        return false;
    }
    auto filtration = filtration_result.moveValue();

    nerve::persistence::IncrementalExactZ2 engine;
    for (const auto &entry : filtration)
    {
        engine.addSimplex(entry.first, entry.second);
    }
    auto filtration_pairs = engine.snapshot();

    VRConfig cfg;
    cfg.max_radius = 2.0;
    cfg.max_dim = 3;
    cfg.algorithm = VRAlgorithmSelection::EXACT_STANDARD;

    const auto direct_pairs = nerve::persistence::computeVrPersistenceFast(view_of(pts), 3, cfg);

    if (!assert_same_pairs(direct_pairs, filtration_pairs.pairs))
    {
        std::cerr << "FAIL: filtration vs direct tetrahedron\n";
        return false;
    }
    return true;
}

bool check_filtration_values_monotonic()
{
    const std::vector<double> pts = {0.0, 0.0, 1.0, 0.0, 0.0, 1.0};

    nerve::filtration::VietorisRips vr;
    vr.setMaxRadius(2.0);
    vr.setMaxDimension(2);
    auto filtration_result = vr.buildFiltration(view_of(pts), 2);
    if (filtration_result.isError())
    {
        std::cerr << "monotonic: filtration build failed\n";
        return false;
    }
    auto filtration = filtration_result.moveValue();

    std::vector<double> dims = vr.getFiltrationValues();
    if (dims.size() != filtration.size())
    {
        std::cerr << "monotonic: filtration values size mismatch\n";
        return false;
    }

    for (std::size_t i = 0; i + 1 < filtration.size(); ++i)
    {
        if (filtration[i].second > filtration[i + 1].second + kTol)
        {
            std::cerr << "monotonic: filtration value decreased at " << i << "\n";
            return false;
        }
    }

    return true;
}

bool check_filtration_with_custom_radii()
{
    const std::vector<double> pts = {0.0, 0.0, 1.0, 0.0};

    nerve::filtration::VietorisRips vr;
    vr.setMaxRadius(2.0);
    vr.setMaxDimension(1);
    auto filtration_result = vr.buildFiltration(view_of(pts), 2);
    if (filtration_result.isError())
    {
        std::cerr << "custom radii: build failed\n";
        return false;
    }
    auto filtration = filtration_result.moveValue();

    VRConfig cfg;
    cfg.max_radius = 2.0;
    cfg.max_dim = 1;
    cfg.algorithm = VRAlgorithmSelection::EXACT_STANDARD;

    const auto direct_pairs = nerve::persistence::computeVrPersistenceFast(view_of(pts), 2, cfg);

    nerve::persistence::IncrementalExactZ2 engine;
    for (const auto &entry : filtration)
    {
        engine.addSimplex(entry.first, entry.second);
    }
    auto filtration_pairs = engine.snapshot();

    if (!assert_same_pairs(direct_pairs, filtration_pairs.pairs))
    {
        std::cerr << "FAIL: filtration vs direct custom radii\n";
        return false;
    }
    return true;
}

} // namespace

int main()
{
    if (!check_filtration_vs_direct_triangle())
    {
        std::cerr << "FAIL: filtration vs direct triangle\n";
        return 1;
    }
    if (!check_filtration_vs_direct_square())
    {
        std::cerr << "FAIL: filtration vs direct square\n";
        return 1;
    }
    if (!check_filtration_values_monotonic())
    {
        std::cerr << "FAIL: filtration values monotonic\n";
        return 1;
    }
    if (!check_filtration_with_custom_radii())
    {
        std::cerr << "FAIL: filtration custom radii\n";
        return 1;
    }
    return 0;
}
