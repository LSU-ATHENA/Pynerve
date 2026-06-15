
#include "nerve/algebra/cellular.hpp"

#include <algorithm>
#include <ranges>
#include <stdexcept>
#include <vector>

namespace nerve::algebra
{

void CWComplex::addSimplices(const std::vector<Simplex> &simplices)
{
    std::ranges::for_each(simplices, [this](const Simplex &s) { addSimplex(s); });
}

void CWComplex::addSimplex(const Simplex &simplex)
{
    if (simplex_to_index_.find(simplex) != simplex_to_index_.end())
    {
        return;
    }
    const Index index = static_cast<Index>(simplices_.size());
    simplices_.emplace_back(simplex);
    simplex_to_index_[simplex] = index;
    invalidateCaches();
}

[[nodiscard]] Size CWComplex::numSimplices() const
{
    return simplices_.size();
}

[[nodiscard]] int CWComplex::maxDimension() const
{
    if (simplices_.empty())
    {
        return -1;
    }

    auto it = std::ranges::max_element(
        simplices_, {}, [](const Simplex &s) { return static_cast<int>(s.dimension()); });

    return static_cast<int>(it->dimension());
}

[[nodiscard]] const Simplex &CWComplex::getSimplex(Index index) const
{
    if (index < 0 || static_cast<Size>(index) >= simplices_.size())
    {
        throw std::out_of_range("Simplex index out of range");
    }
    return simplices_[index];
}

[[nodiscard]] std::vector<Index> CWComplex::getSimplicesOfDimension(int dimension) const
{
    std::vector<Index> out;

    for (Size i = 0; i < simplices_.size(); ++i)
    {
        if (static_cast<int>(simplices_[i].dimension()) == dimension)
        {
            out.emplace_back(static_cast<Index>(i));
        }
    }
    return out;
}

std::vector<Index> CWComplex::getStar(const Simplex &simplex) const
{
    if (!caches_valid_)
    {
        rebuildCaches();
    }
    const auto it = simplex_to_index_.find(simplex);
    if (it == simplex_to_index_.end())
    {
        return {};
    }
    return star_cache_[static_cast<Size>(it->second)];
}

std::vector<Index> CWComplex::getLink(const Simplex &simplex) const
{
    if (!caches_valid_)
    {
        rebuildCaches();
    }
    const auto it = simplex_to_index_.find(simplex);
    if (it == simplex_to_index_.end())
    {
        return {};
    }
    return link_cache_[static_cast<Size>(it->second)];
}

std::vector<int> CWComplex::computeHomology() const
{
    return computeBettiNumbers();
}

std::vector<int> CWComplex::computeBettiNumbers() const
{
    CellularComplex complex;
    for (const Simplex &simplex : simplices_)
    {
        std::vector<Index> boundary_indices;
        for (const Simplex &face : simplex.faces(core::DeterminismContract{}))
        {
            const auto it = simplex_to_index_.find(face);
            if (it != simplex_to_index_.end())
            {
                boundary_indices.push_back(it->second);
            }
        }
        [[maybe_unused]] const Index cell_index =
            complex.addCell(Cell(static_cast<int>(simplex.dimension()), boundary_indices));
    }
    const auto betti = complex.computeBettiNumbers();
    if (betti.isError())
    {
        return {};
    }
    return betti.value();
}

void CWComplex::invalidateCaches() const
{
    caches_valid_ = false;
    star_cache_.clear();
    link_cache_.clear();
}

void CWComplex::rebuildCaches() const
{
    star_cache_.assign(simplices_.size(), {});
    link_cache_.assign(simplices_.size(), {});

    for (Size i = 0; i < simplices_.size(); ++i)
    {
        for (Size j = 0; j < simplices_.size(); ++j)
        {
            if (i == j)
            {
                continue;
            }
            if (simplices_[i].isFaceOf(simplices_[j]))
            {
                star_cache_[i].push_back(static_cast<Index>(j));

                bool disjoint = true;
                for (const Index v : simplices_[i].vertices())
                {
                    if (simplices_[j].contains(v))
                    {
                        disjoint = false;
                        break;
                    }
                }
                if (disjoint)
                {
                    link_cache_[i].push_back(static_cast<Index>(j));
                }
            }
        }
    }

    caches_valid_ = true;
}

} // namespace nerve::algebra
