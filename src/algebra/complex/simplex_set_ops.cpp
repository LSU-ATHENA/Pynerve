
#include "nerve/algebra/simplex.hpp"

#include <algorithm>
#include <ranges>
#include <vector>

namespace nerve::algebra
{

[[nodiscard]] bool SimplexSet::insert(const Simplex &simplex)
{
    return simplices_.insert(simplex).second;
}

[[nodiscard]] bool SimplexSet::erase(const Simplex &simplex)
{
    return simplices_.erase(simplex) > 0;
}

[[nodiscard]] bool SimplexSet::contains(const Simplex &simplex) const
{
    return simplices_.find(simplex) != simplices_.end();
}

[[nodiscard]] Size SimplexSet::size() const noexcept
{
    return simplices_.size();
}

[[nodiscard]] bool SimplexSet::empty() const noexcept
{
    return simplices_.empty();
}

void SimplexSet::clear()
{
    simplices_.clear();
}
[[nodiscard]] SimplexSet SimplexSet::intersection(const SimplexSet &other) const
{
    SimplexSet result;
    std::ranges::copy_if(simplices_, std::inserter(result.simplices_, result.simplices_.end()),
                         [&other](const auto &simplex) { return other.contains(simplex); });
    return result;
}

[[nodiscard]] SimplexSet SimplexSet::unionSet(const SimplexSet &other) const
{
    SimplexSet result = *this;
    std::ranges::copy(other.simplices_, std::inserter(result.simplices_, result.simplices_.end()));
    return result;
}

[[nodiscard]] SimplexSet SimplexSet::difference(const SimplexSet &other) const
{
    SimplexSet result;
    std::ranges::copy_if(simplices_, std::inserter(result.simplices_, result.simplices_.end()),
                         [&other](const auto &simplex) { return !other.contains(simplex); });
    return result;
}
SimplexSet SimplexSet::kSimplices(Size k) const
{
    SimplexSet result;
    for (const auto &simplex : simplices_)
    {
        if (simplex.dimension() == k)
        {
            static_cast<void>(result.insert(simplex));
        }
    }
    return result;
}
Size SimplexSet::numKSimplices(Size k) const
{
    Size count = 0;
    for (const auto &simplex : simplices_)
    {
        if (simplex.dimension() == k)
        {
            ++count;
        }
    }
    return count;
}
Size SimplexSet::maxDimension() const
{
    Size max_dim = 0;
    for (const auto &simplex : simplices_)
    {
        max_dim = std::max(max_dim, simplex.dimension());
    }
    return max_dim;
}
SimplexSet SimplexSet::boundary() const
{
    SimplexSet result;
    for (const auto &simplex : simplices_)
    {
        auto faces = simplex.faces({});
        for (const auto &face : faces)
        {
            static_cast<void>(result.insert(face));
        }
    }
    return result;
}
SimplexSet SimplexSet::kBoundary(Size k) const
{
    SimplexSet result;
    for (const auto &simplex : simplices_)
    {
        auto kFaces = simplex.kFaces(k, {});
        for (const auto &face : kFaces)
        {
            static_cast<void>(result.insert(face));
        }
    }
    return result;
}
SimplexSet SimplexSet::star(const Simplex &simplex) const
{
    SimplexSet result;
    for (const auto &s : simplices_)
    {
        if (simplex.isFaceOf(s))
        {
            static_cast<void>(result.insert(s));
        }
    }
    return result;
}
SimplexSet SimplexSet::link(const Simplex &simplex) const
{
    SimplexSet result;
    auto star_simplex = star(simplex);
    for (const auto &tau : star_simplex.simplices_)
    {
        if (simplex.meet(tau, {}).numVertices() == 0)
        {
            static_cast<void>(result.insert(tau));
        }
    }
    return result;
}
std::vector<Simplex> SimplexSet::toVector() const
{
    return std::vector<Simplex>(simplices_.begin(), simplices_.end());
}
} // namespace nerve::algebra
