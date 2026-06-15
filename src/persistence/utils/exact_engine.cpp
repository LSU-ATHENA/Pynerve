
#include "exact_engine_internal.hpp"
#include "nerve/persistence/utils/exact_engine.hpp"

#include <algorithm>
#include <cmath>
#include <ranges>
#include <stdexcept>
#include <unordered_map>

namespace nerve::persistence
{

namespace detail
{

Column symmetricDifferenceSorted(const Column &a, const Column &b)
{
    Column out;
    out.reserve(a.size() + b.size());
    Size i = 0;
    Size j = 0;
    while (i < a.size() && j < b.size())
    {
        if (a[i] == b[j])
        {
            ++i;
            ++j;
        }
        else if (a[i] < b[j])
        {
            out.push_back(a[i++]);
        }
        else
        {
            out.push_back(b[j++]);
        }
    }
    while (i < a.size())
    {
        out.push_back(a[i++]);
    }
    while (j < b.size())
    {
        out.push_back(b[j++]);
    }
    return out;
}

bool simplexFiltrationOrder(const std::pair<algebra::Simplex, double> &a,
                            const std::pair<algebra::Simplex, double> &b)
{
    if (a.second != b.second)
    {
        return a.second < b.second;
    }
    if (a.first.dimension() != b.first.dimension())
    {
        return a.first.dimension() < b.first.dimension();
    }
    return a.first < b.first;
}

} // namespace detail

namespace
{

void validateSimplexInput(const algebra::Simplex &simplex, double filtration)
{
    if (simplex.numVertices() == 0)
    {
        throw std::invalid_argument("exact persistence simplex must be non-empty");
    }
    for (Index vertex : simplex.vertices())
    {
        if (vertex < 0)
        {
            throw std::invalid_argument("exact persistence simplex vertices must be non-negative");
        }
    }
    if (!std::isfinite(filtration))
    {
        throw std::invalid_argument("exact persistence filtration must be finite");
    }
}

std::vector<Size> computeBettiNumbers(const std::vector<Pair> &pairs)
{
    Size max_dim = 0;
    for (const auto &pair : pairs)
    {
        if (pair.dimension >= 0)
        {
            max_dim = std::max(max_dim, static_cast<Size>(pair.dimension));
        }
    }
    std::vector<Size> betti(max_dim + 1, 0);
    for (const auto &pair : pairs)
    {
        if (pair.dimension >= 0 && pair.isInfinite())
        {
            const Size dim = static_cast<Size>(pair.dimension);
            if (dim < betti.size())
            {
                betti[dim]++;
            }
        }
    }
    return betti;
}

ExactPersistenceResult
buildResultFromReduction(const std::vector<std::pair<algebra::Simplex, double>> &simplices,
                         const std::vector<detail::Column> &reduced, const std::vector<Index> &low,
                         Size max_dim, Size reduction_operations)
{
    ExactPersistenceResult result;
    result.reduction_operations = reduction_operations;

    const Size n = simplices.size();
    std::vector<bool> birthKilled(n, false);
    for (Size col = 0; col < n; ++col)
    {
        if (low[col] >= 0)
        {
            birthKilled[static_cast<Size>(low[col])] = true;
        }
    }

    for (Size col = 0; col < n; ++col)
    {
        if (low[col] < 0)
        {
            continue;
        }
        const Size birth_idx = static_cast<Size>(low[col]);
        const auto &birth_simplex = simplices[birth_idx].first;
        if (birth_simplex.dimension() > max_dim)
        {
            continue;
        }
        const double birth = simplices[birth_idx].second;
        const double death = simplices[col].second;
        if (death < birth)
        {
            continue;
        }
        result.pairs.push_back(Pair{birth, death, static_cast<Dimension>(birth_simplex.dimension()),
                                    static_cast<Index>(birth_idx), static_cast<Index>(col)});
    }

    const double inf = std::numeric_limits<Field>::infinity();
    for (Size col = 0; col < n; ++col)
    {
        if (!reduced[col].empty() || birthKilled[col])
        {
            continue;
        }
        const auto &simplex = simplices[col].first;
        if (simplex.dimension() > max_dim)
        {
            continue;
        }
        result.pairs.push_back(Pair{simplices[col].second, inf,
                                    static_cast<Dimension>(simplex.dimension()),
                                    static_cast<Index>(col), -1});
    }

    std::ranges::sort(result.pairs, {}, &Pair::dimension);

    result.betti_numbers = computeBettiNumbers(result.pairs);
    return result;
}

} // namespace

IncrementalExactZ2::IncrementalExactZ2(Size max_dim)
    : max_dim_(max_dim)
    , simplices_()
    , index_of_simplex_()
    , reduced_columns_()
    , low_()
    , low_row_to_col_()
    , reduction_operations_(0)
    , last_filtration_(-std::numeric_limits<double>::infinity())
{}

void IncrementalExactZ2::setMaxDim(Size max_dim)
{
    max_dim_ = max_dim;
}

void IncrementalExactZ2::clear()
{
    simplices_.clear();
    index_of_simplex_.clear();
    reduced_columns_.clear();
    low_.clear();
    low_row_to_col_.clear();
    reduction_operations_ = 0;
    last_filtration_ = -std::numeric_limits<double>::infinity();
}

bool IncrementalExactZ2::addSimplex(const algebra::Simplex &simplex, double filtration)
{
    validateSimplexInput(simplex, filtration);
    if (filtration + 1e-12 < last_filtration_)
    {
        return false;
    }
    auto existing = index_of_simplex_.find(simplex);
    if (existing != index_of_simplex_.end())
    {
        return true;
    }

    const Size col = simplices_.size();
    simplices_.emplace_back(simplex, filtration);
    index_of_simplex_.emplace(simplex, col);
    reduced_columns_.emplace_back();
    low_.push_back(-1);
    low_row_to_col_.push_back(-1);

    Column boundary;
    if (simplex.dimension() > 0)
    {
        auto faces = simplex.faces({});
        boundary.reserve(faces.size());
        for (const auto &face : faces)
        {
            auto it = index_of_simplex_.find(face);
            if (it == index_of_simplex_.end())
            {
                continue;
            }
            if (it->second < col)
            {
                boundary.push_back(it->second);
            }
        }
        std::ranges::sort(boundary);
        const auto [first, last] = std::ranges::unique(boundary);
        boundary.erase(first, last);
    }

    while (!boundary.empty())
    {
        const Size pivot = boundary.back();
        if (pivot >= low_row_to_col_.size())
        {
            low_row_to_col_.resize(pivot + 1, -1);
        }
        const Index killer = low_row_to_col_[pivot];
        if (killer < 0)
        {
            low_row_to_col_[pivot] = static_cast<Index>(col);
            low_[col] = static_cast<Index>(pivot);
            break;
        }
        boundary = detail::symmetricDifferenceSorted(boundary,
                                                     reduced_columns_[static_cast<Size>(killer)]);
        ++reduction_operations_;
    }

    reduced_columns_[col] = std::move(boundary);
    last_filtration_ = std::max(last_filtration_, filtration);
    return true;
}

void IncrementalExactZ2::rebuildFromComplex(const algebra::SimplicialComplex &complex)
{
    clear();
    const auto filtered = complex.getFilteredSimplices();
    if (filtered.empty())
    {
        return;
    }

    std::unordered_map<algebra::Simplex, double, algebra::Simplex::Hash> min_filtration;
    min_filtration.reserve(filtered.size() * 2);
    for (const auto &entry : filtered)
    {
        auto it = min_filtration.find(entry.first);
        if (it == min_filtration.end() || entry.second < it->second)
        {
            min_filtration[entry.first] = entry.second;
        }
    }

    std::vector<std::pair<algebra::Simplex, double>> simplices;
    simplices.reserve(min_filtration.size());
    for (const auto &item : min_filtration)
    {
        simplices.emplace_back(item.first, item.second);
    }
    std::sort(simplices.begin(), simplices.end(), detail::simplexFiltrationOrder);

    for (const auto &simplex : simplices)
    {
        addSimplex(simplex.first, simplex.second);
    }
}

ExactPersistenceResult IncrementalExactZ2::snapshot() const
{
    return buildResultFromReduction(simplices_, reduced_columns_, low_, max_dim_,
                                    reduction_operations_);
}

ExactPersistenceResult computeExactPersistenceZ2(const algebra::SimplicialComplex &complex,
                                                 Size max_dim)
{
    IncrementalExactZ2 engine(max_dim);
    engine.rebuildFromComplex(complex);
    return engine.snapshot();
}

} // namespace nerve::persistence
