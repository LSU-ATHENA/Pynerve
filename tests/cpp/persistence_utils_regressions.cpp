#include "nerve/common/accelerated_types.hpp"
#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/core_types.hpp"
#include "nerve/persistence/core/core_types.hpp"
#include "nerve/persistence/utils/diagram_statistics.hpp"

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
using nerve::core::BufferView;
using nerve::persistence::Diagram;
using nerve::persistence::Pair;

std::vector<Pair> canonical(std::vector<Pair> pairs)
{
    std::sort(pairs.begin(), pairs.end(), [](const Pair &a, const Pair &b) {
        return std::tuple(a.dimension, a.birth, a.death) <
               std::tuple(b.dimension, b.birth, b.death);
    });
    return pairs;
}

bool check_betti_numbers_from_pairs()
{
    std::vector<Pair> pairs;
    pairs.push_back({0.0, std::numeric_limits<double>::infinity(), 0});
    pairs.push_back({0.0, 1.0, 0});
    pairs.push_back({0.0, std::numeric_limits<double>::infinity(), 1});
    pairs.push_back({1.0, 2.0, 1});

    auto betti = nerve::persistence::bettiNumbersFromPairs(pairs);
    if (betti.size() < 2)
    {
        std::cerr << "betti size expected >=2, got " << betti.size() << "\n";
        return false;
    }
    if (betti[0] != 1)
    {
        std::cerr << "betti[0] expected 1, got " << betti[0] << "\n";
        return false;
    }
    if (betti[1] != 1)
    {
        std::cerr << "betti[1] expected 1, got " << betti[1] << "\n";
        return false;
    }
    return true;
}

bool check_shannon_entropy_normalized()
{
    std::vector<double> weights = {1.0, 2.0, 3.0, 4.0};
    double entropy = nerve::persistence::shannonEntropyNormalized(weights);
    if (!std::isfinite(entropy) || entropy <= 0.0)
    {
        std::cerr << "shannon entropy expected positive, got " << entropy << "\n";
        return false;
    }
    return true;
}

bool check_diagram_construct_known_pairs()
{
    Diagram d;
    d.addPair({0.0, std::numeric_limits<double>::infinity(), 0});
    d.addPair({0.0, 1.0, 0});
    d.addPair({1.0, std::numeric_limits<double>::infinity(), 1});

    if (d.count() != 3)
    {
        std::cerr << "diagram count expected 3, got " << d.count() << "\n";
        return false;
    }
    auto pairs = d.getPairsByDimension(0);
    if (pairs.size() != 2)
    {
        std::cerr << "dim-0 pairs expected 2, got " << pairs.size() << "\n";
        return false;
    }
    pairs = d.getPairsByDimension(1);
    if (pairs.size() != 1)
    {
        std::cerr << "dim-1 pairs expected 1, got " << pairs.size() << "\n";
        return false;
    }
    return true;
}

bool check_empty_diagram_zero_betti()
{
    Diagram d;
    auto betti = d.computeBetti();
    if (!betti.empty())
    {
        std::cerr << "empty diagram betti expected empty, got size " << betti.size() << "\n";
        return false;
    }
    return true;
}

bool check_birth_death_dimension_access()
{
    Pair p{2.5, 7.3, 1};
    if (std::abs(p.birth - 2.5) > 1e-12)
    {
        std::cerr << "birth expected 2.5, got " << p.birth << "\n";
        return false;
    }
    if (std::abs(p.death - 7.3) > 1e-12)
    {
        std::cerr << "death expected 7.3, got " << p.death << "\n";
        return false;
    }
    if (p.dimension != 1)
    {
        std::cerr << "dimension expected 1, got " << p.dimension << "\n";
        return false;
    }
    return true;
}

bool check_filtration_simplification_invariant()
{
    Diagram d;
    d.addPair({0.0, 0.5, 0});
    d.addPair({0.0, 1.0, 0});
    d.addPair({0.0, std::numeric_limits<double>::infinity(), 0});
    d.addPair({1.0, 1.5, 1});

    auto betti = d.computeBetti();
    bool has_h0_essential = false;
    for (size_t i = 0; i < betti.size(); ++i)
    {
        if (betti[i] > 0 && i == 0)
            has_h0_essential = true;
    }
    if (!has_h0_essential)
    {
        std::cerr << "filtration simplification: expected H0 essential\n";
        return false;
    }
    return true;
}

bool check_diagram_max_persistence()
{
    Diagram d;
    d.addPair({0.0, 5.0, 0});
    d.addPair({0.0, 3.0, 0});
    d.addPair({0.0, std::numeric_limits<double>::infinity(), 0});

    double maxp = d.getMaxPersistence();
    if (std::abs(maxp - 5.0) > 1e-12)
    {
        std::cerr << "max persistence expected 5.0, got " << maxp << "\n";
        return false;
    }
    return true;
}

bool check_diagram_average_persistence()
{
    Diagram d;
    d.addPair({0.0, 4.0, 0});
    d.addPair({0.0, 6.0, 0});

    double avg = d.getAveragePersistence();
    if (std::abs(avg - 5.0) > 1e-12)
    {
        std::cerr << "average persistence expected 5.0, got " << avg << "\n";
        return false;
    }
    return true;
}

} // namespace

int main()
{
    if (!check_betti_numbers_from_pairs())
    {
        std::cerr << "FAIL: betti numbers from pairs\n";
        return 1;
    }
    if (!check_shannon_entropy_normalized())
    {
        std::cerr << "FAIL: shannon entropy normalized\n";
        return 1;
    }
    if (!check_diagram_construct_known_pairs())
    {
        std::cerr << "FAIL: diagram construct known pairs\n";
        return 1;
    }
    if (!check_empty_diagram_zero_betti())
    {
        std::cerr << "FAIL: empty diagram zero betti\n";
        return 1;
    }
    if (!check_birth_death_dimension_access())
    {
        std::cerr << "FAIL: birth death dimension access\n";
        return 1;
    }
    if (!check_filtration_simplification_invariant())
    {
        std::cerr << "FAIL: filtration simplification invariant\n";
        return 1;
    }
    if (!check_diagram_max_persistence())
    {
        std::cerr << "FAIL: diagram max persistence\n";
        return 1;
    }
    if (!check_diagram_average_persistence())
    {
        std::cerr << "FAIL: diagram average persistence\n";
        return 1;
    }
    return 0;
}
