#include "nerve/common/accelerated_types.hpp"
#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/core_types.hpp"
#include "nerve/persistence/core/core_types.hpp"

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

using nerve::core::BufferView;
using nerve::persistence::Diagram;
using nerve::persistence::Pair;

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

bool check_pair_construction()
{
    Pair p;
    if (p.birth != 0.0 || p.death != 0.0 || p.dimension != 0)
    {
        std::cerr << "default Pair not zero-initialized\n";
        return false;
    }
    return true;
}

bool check_pair_value_construction()
{
    Pair p{1.0, 5.0, 2};
    if (std::abs(p.birth - 1.0) > 1e-12 || std::abs(p.death - 5.0) > 1e-12 || p.dimension != 2)
    {
        std::cerr << "value Pair construction incorrect\n";
        return false;
    }
    return true;
}

bool check_is_infinite()
{
    Pair finite{0.0, 1.0, 0};
    Pair infinite{0.0, std::numeric_limits<double>::infinity(), 0};

    if (finite.isInfinite())
    {
        std::cerr << "finite pair reported as infinite\n";
        return false;
    }
    if (!infinite.isInfinite())
    {
        std::cerr << "infinite pair reported as finite\n";
        return false;
    }
    return true;
}

bool check_lifetime()
{
    Pair p{2.0, 5.0, 0};
    auto life = p.lifetime();
    if (std::abs(life - 3.0) > 1e-12)
    {
        std::cerr << "lifetime expected 3.0, got " << life << "\n";
        return false;
    }
    return true;
}

bool check_lifetime_infinite()
{
    Pair p{0.0, std::numeric_limits<double>::infinity(), 0};
    auto life = p.lifetime();
    if (std::isfinite(life))
    {
        std::cerr << "infinite pair should have infinite lifetime\n";
        return false;
    }
    return true;
}

bool check_dimension_accessor()
{
    Pair p{0.0, 1.0, 3};
    if (p.dimension != 3)
    {
        std::cerr << "dimension should be 3\n";
        return false;
    }
    return true;
}

bool check_diagram_construction()
{
    Diagram d;
    if (!d.isEmpty())
    {
        std::cerr << "default Diagram should be empty\n";
        return false;
    }
    if (d.count() != 0)
    {
        std::cerr << "default Diagram count should be 0\n";
        return false;
    }
    return true;
}

bool check_diagram_add_pair()
{
    Diagram d;
    d.addPair(Pair{1.0, 2.0, 0});
    d.addPair(Pair{0.0, std::numeric_limits<double>::infinity(), 0});

    if (d.count() != 2)
    {
        std::cerr << "expected count 2, got " << d.count() << "\n";
        return false;
    }
    if (d.isEmpty())
    {
        std::cerr << "diagram with pairs should not be empty\n";
        return false;
    }
    return true;
}

bool check_diagram_clear()
{
    Diagram d;
    d.addPair(Pair{0.0, 1.0, 0});
    d.clear();
    if (!d.isEmpty())
    {
        std::cerr << "cleared diagram should be empty\n";
        return false;
    }
    return true;
}

bool check_betti_computation()
{
    Diagram d;
    d.addPair(Pair{0.0, std::numeric_limits<double>::infinity(), 0});
    d.addPair(Pair{1.0, std::numeric_limits<double>::infinity(), 1});

    auto betti = d.computeBetti();
    if (betti.size() < 2)
    {
        std::cerr << "expected at least 2 Betti numbers\n";
        return false;
    }
    if (betti[0] < 1 || betti[1] < 1)
    {
        std::cerr << "expected at least B0=1, B1=1\n";
        return false;
    }
    return true;
}

bool check_pairs_by_dimension()
{
    Diagram d;
    d.addPair(Pair{0.0, 1.0, 0});
    d.addPair(Pair{0.0, std::numeric_limits<double>::infinity(), 0});
    d.addPair(Pair{1.0, 2.0, 1});

    auto dim0 = d.getPairsByDimension(0);
    auto dim1 = d.getPairsByDimension(1);

    if (dim0.size() != 2)
    {
        std::cerr << "expected 2 dim-0 pairs, got " << dim0.size() << "\n";
        return false;
    }
    if (dim1.size() != 1)
    {
        std::cerr << "expected 1 dim-1 pair, got " << dim1.size() << "\n";
        return false;
    }
    return true;
}

} // namespace

int main()
{
    if (!check_pair_construction())
    {
        std::cerr << "FAIL: Pair construction\n";
        return 1;
    }
    if (!check_pair_value_construction())
    {
        std::cerr << "FAIL: Pair value construction\n";
        return 1;
    }
    if (!check_is_infinite())
    {
        std::cerr << "FAIL: isInfinite\n";
        return 1;
    }
    if (!check_lifetime())
    {
        std::cerr << "FAIL: lifetime\n";
        return 1;
    }
    if (!check_lifetime_infinite())
    {
        std::cerr << "FAIL: lifetime infinite\n";
        return 1;
    }
    if (!check_dimension_accessor())
    {
        std::cerr << "FAIL: dimension\n";
        return 1;
    }
    if (!check_diagram_construction())
    {
        std::cerr << "FAIL: Diagram construction\n";
        return 1;
    }
    if (!check_diagram_add_pair())
    {
        std::cerr << "FAIL: Diagram addPair\n";
        return 1;
    }
    if (!check_diagram_clear())
    {
        std::cerr << "FAIL: Diagram clear\n";
        return 1;
    }
    if (!check_betti_computation())
    {
        std::cerr << "FAIL: Betti computation\n";
        return 1;
    }
    if (!check_pairs_by_dimension())
    {
        std::cerr << "FAIL: pairs by dimension\n";
        return 1;
    }
    return 0;
}
