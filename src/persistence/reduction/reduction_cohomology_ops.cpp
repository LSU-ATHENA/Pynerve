
#include "nerve/algebra/complex.hpp"
#include "nerve/algebra/simplex.hpp"
#include "nerve/persistence/reduction/reduction_ops.hpp"

#include <algorithm>
#include <cmath>
#include <ranges>

namespace nerve::persistence
{

void Reducer::computeCoboundaryMatrix()
{
    coboundary_matrix_.assign(reduced_matrix_.size(), {});
    for (Size col = 0; col < reduced_matrix_.size(); ++col)
    {
        for (const auto &entry : reduced_matrix_[col])
        {
            const Size row = entry.first;
            if (row < coboundary_matrix_.size())
            {
                coboundary_matrix_[row].push_back(static_cast<int>(col));
            }
        }
    }
    for (auto &c : coboundary_matrix_)
    {
        std::ranges::sort(c);
        const auto [first, last] = std::ranges::unique(c);
        c.erase(first, last);
    }
}

void Reducer::cohomologyReductionStep(int column)
{
    if (column < 0 || static_cast<Size>(column) >= cocycle_pivots_.size())
    {
        return;
    }
    const int pivot = findCocyclePivot(column);
    cocycle_pivots_[static_cast<Size>(column)] = pivot;
    if (static_cast<Size>(column) < column_processed_.size())
    {
        column_processed_[static_cast<Size>(column)] = true;
    }
}

int Reducer::findCocyclePivot(int column) const
{
    if (column < 0 || static_cast<Size>(column) >= coboundary_matrix_.size())
    {
        return -1;
    }
    const auto &c = coboundary_matrix_[static_cast<Size>(column)];
    if (c.empty())
    {
        return -1;
    }
    return c.back();
}

void Reducer::addCocycle(int target, int source)
{
    if (target < 0 || source < 0)
    {
        return;
    }
    if (static_cast<Size>(target) >= coboundary_matrix_.size())
    {
        return;
    }
    if (static_cast<Size>(source) >= coboundary_matrix_.size())
    {
        return;
    }

    auto &t = coboundary_matrix_[static_cast<Size>(target)];
    auto &s = coboundary_matrix_[static_cast<Size>(source)];
    std::ranges::sort(t);
    std::ranges::sort(s);

    Vector<int> out;
    out.reserve(t.size() + s.size());
    Size i = 0;
    Size j = 0;
    while (i < t.size() && j < s.size())
    {
        if (t[i] == s[j])
        {
            ++i;
            ++j;
        }
        else if (t[i] < s[j])
        {
            out.push_back(t[i++]);
        }
        else
        {
            out.push_back(s[j++]);
        }
    }
    while (i < t.size())
    {
        out.push_back(t[i++]);
    }
    while (j < s.size())
    {
        out.push_back(s[j++]);
    }
    t = std::move(out);
}

void Reducer::compressCocycles()
{
    for (auto &c : coboundary_matrix_)
    {
        c.shrink_to_fit();
    }
}

Index Reducer::findLowestPivotCacheAware(Size column) const
{
    return findLowestPivot(column);
}

// Tolerance for pivot detection in reduced matrix
constexpr double REDUCED_MATRIX_TOLERANCE = 1e-12;

bool Reducer::hasPivotInColumnAtomic(Size column, Index pivot) const
{
    if (column >= reduced_matrix_.size() || pivot < 0)
    {
        return false;
    }
    for (const auto &entry : reduced_matrix_[column])
    {
        if (entry.first == static_cast<Size>(pivot) &&
            std::fabs(entry.second) > REDUCED_MATRIX_TOLERANCE)
        {
            return true;
        }
    }
    return false;
}

void Reducer::addColumnSIMD(Size dest_col, Size src_col, double coefficient)
{
    if (std::fabs(coefficient) < REDUCED_MATRIX_TOLERANCE)
    {
        return;
    }
    eliminatePivot(dest_col, src_col);
}

void Reducer::addCocycle(int target, int source, const std::vector<int> &vertices2, int vertex)
{
    if (target < 0 || source < 0)
    {
        return;
    }
    if (static_cast<Size>(target) >= coboundary_matrix_.size())
    {
        return;
    }
    if (static_cast<Size>(source) >= coboundary_matrix_.size())
    {
        return;
    }

    auto &t = coboundary_matrix_[static_cast<Size>(target)];
    auto &s = coboundary_matrix_[static_cast<Size>(source)];
    std::ranges::sort(t);
    std::ranges::sort(s);

    Vector<int> out;
    out.reserve(t.size() + s.size());
    Size i = 0;
    Size j = 0;
    while (i < t.size() && j < s.size())
    {
        if (t[i] == s[j])
        {
            ++i;
            ++j;
        }
        else if (t[i] < s[j])
        {
            out.push_back(t[i++]);
        }
        else
        {
            out.push_back(s[j++]);
        }
    }
    while (i < t.size())
    {
        out.push_back(t[i++]);
    }
    while (j < s.size())
    {
        out.push_back(s[j++]);
    }
    t = std::move(out);

    if (std::ranges::find(vertices2, vertex) != vertices2.end())
    {
        return;
    }
}

} // namespace nerve::persistence
