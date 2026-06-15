
#include "nerve/streaming/diagram_sorting.hpp"

#include <algorithm>
#include <cmath>
#include <ranges>

namespace nerve::streaming::detail
{

namespace
{

double pairNorm(const Pair &pair)
{
    if (pair.isInfinite())
    {
        return std::fabs(pair.birth);
    }
    return std::fabs(pair.death - pair.birth);
}

} // namespace

std::vector<Pair> canonicalPairs(const persistence::Diagram &diagram)
{
    const auto &pairs = diagram.getPairs();
    std::vector<Pair> out(pairs.begin(), pairs.end());
    std::ranges::sort(out, {},
                      [](const Pair &p) { return std::tuple(p.dimension, p.birth, p.death); });
    return out;
}

double diagramSupDistance(const persistence::Diagram &lhs, const persistence::Diagram &rhs)
{
    const auto left = canonicalPairs(lhs);
    const auto right = canonicalPairs(rhs);
    const Size limit = std::max(left.size(), right.size());
    double max_diff = 0.0;
    for (Size i = 0; i < limit; ++i)
    {
        if (i >= left.size())
        {
            max_diff = std::max(max_diff, pairNorm(right[i]));
            continue;
        }
        if (i >= right.size())
        {
            max_diff = std::max(max_diff, pairNorm(left[i]));
            continue;
        }

        const auto &a = left[i];
        const auto &b = right[i];
        double diff = std::fabs(a.birth - b.birth);
        if (a.isInfinite() != b.isInfinite())
        {
            diff = std::max(diff, std::max(pairNorm(a), pairNorm(b)));
        }
        else if (!a.isInfinite() && !b.isInfinite())
        {
            diff = std::max(diff, std::fabs(a.death - b.death));
        }
        max_diff = std::max(max_diff, diff);
    }
    return max_diff;
}

} // namespace nerve::streaming::detail
