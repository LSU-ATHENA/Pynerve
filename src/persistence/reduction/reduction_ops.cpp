
#include "nerve/algebra/boundary.hpp"
#include "nerve/gpu/gpu_compute.hpp"
#include "nerve/persistence/core/core_types.hpp"
#include "nerve/persistence/reduction/reduction_ops.hpp"
#include "nerve/persistence/streaming/streaming_reducer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <ranges>

namespace nerve::persistence
{

namespace
{

using Z2Column = Vector<Size>;

// Tolerance for Z2 column conversion (non-zero entries)
constexpr double Z2_CONVERSION_TOLERANCE = 1e-12;

Z2Column symmetricDifferenceSorted(const Z2Column &a, const Z2Column &b)
{
    Z2Column out;
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

Z2Column columnToZ2Rows(const Vector<std::pair<Size, double>> &column)
{
    Z2Column rows;
    rows.reserve(column.size());
    for (const auto &entry : column)
    {
        if (std::fabs(entry.second) > Z2_CONVERSION_TOLERANCE)
        {
            rows.push_back(entry.first);
        }
    }
    std::ranges::sort(rows);
    const auto [first, last] = std::ranges::unique(rows);
    rows.erase(first, last);
    return rows;
}

Vector<std::pair<Size, double>> fromZ2Rows(const Z2Column &rows)
{
    Vector<std::pair<Size, double>> out;
    out.reserve(rows.size());
    for (Size r : rows)
    {
        out.emplace_back(r, 1.0);
    }
    return out;
}

} // namespace

Reducer::Reducer(const algebra::BoundaryMatrix &boundary_matrix)
    : matrix_(&boundary_matrix)
    , num_operations_(0)
    , computation_time_(0.0)
{
    initializeReduction();
}

void Reducer::compute()
{
    auto start = std::chrono::high_resolution_clock::now();
    initializeReduction();
    reduceMatrix();
    computePersistencePairsFromPivots();
    computeBettiNumbersFromPivots();
    classifyEssentials();
    auto end = std::chrono::high_resolution_clock::now();
    computation_time_ = std::chrono::duration<double>(end - start).count();
}

void Reducer::computeWithCoefficients()
{
    compute();
}

const Vector<Pair> &Reducer::getPairs() const
{
    return pairs_;
}

const Vector<Index> &Reducer::getEssentials() const
{
    return essentials_;
}

const Vector<Size> &Reducer::getBetti() const
{
    return betti_;
}

void Reducer::initializeReduction()
{
    reduced_matrix_.clear();
    pairs_.clear();
    essentials_.clear();
    betti_.clear();
    compact_pivots_.clear();
    simplex_dimensions_.clear();
    pivot_columns_.clear();
    columns_loaded_ = false;
    num_operations_ = 0;

    if (matrix_ == nullptr)
    {
        return;
    }

    const Size cols = matrix_->cols();

    reduced_matrix_.resize(cols);
    pivot_columns_.assign(cols, -1);
    simplex_dimensions_.resize(cols, -1);
}

// Tolerance for boundary reduction (non-zero matrix entries)
constexpr double BOUNDARY_REDUCTION_TOLERANCE = 1e-12;

void Reducer::reduceMatrixColumnByColumn()
{
    if (matrix_ == nullptr)
    {
        columns_loaded_ = true;
        return;
    }

    const Size cols = matrix_->cols();
    const Size rows = matrix_->rows();

    for (Size col = 0; col < cols; ++col)
    {
        Z2Column col_rows;
        for (Size row = 0; row < rows; ++row)
        {
            const double value = matrix_->getMatrixEntry(row, col);
            if (std::fabs(value) > BOUNDARY_REDUCTION_TOLERANCE)
            {
                col_rows.push_back(row);
            }
        }
        std::ranges::sort(col_rows);
        const auto [first2, last2] = std::ranges::unique(col_rows);
        col_rows.erase(first2, last2);
        reduced_matrix_[col] = fromZ2Rows(col_rows);
        simplex_dimensions_[col] = matrix_->getColSimplexDimension(col);
    }
    columns_loaded_ = true;
}

void Reducer::computePersistencePairsFromPivots()
{
    pairs_.clear();
    essentials_.clear();

    if (matrix_ == nullptr)
    {
        return;
    }

    const Field inf = std::numeric_limits<Field>::infinity();
    const Size cols = reduced_matrix_.size();
    std::vector<bool> killedBirthColumns(cols, false);

    for (Size col = 0; col < cols; ++col)
    {
        const Index pivot = getPivot(col);
        if (pivot < 0)
        {
            continue;
        }
        const Index birth_col = matrix_->getColumnIndexForRowSimplex(static_cast<Size>(pivot));
        if (birth_col >= 0 && static_cast<Size>(birth_col) < killedBirthColumns.size())
        {
            killedBirthColumns[static_cast<Size>(birth_col)] = true;
        }
    }

    for (Size col = 0; col < cols; ++col)
    {
        const Index pivot = getPivot(col);
        if (pivot >= 0)
        {
            Pair pair;
            pair.birth =
                static_cast<Field>(matrix_->getRowFiltrationValue(static_cast<Size>(pivot)));
            pair.death = static_cast<Field>(matrix_->getFiltrationValue(col));
            pair.dimension =
                static_cast<Dimension>(matrix_->getRowSimplexDimension(static_cast<Size>(pivot)));
            if (pair.death >= pair.birth)
            {
                pairs_.push_back(pair);
            }
            continue;
        }

        const int dim = matrix_->getColSimplexDimension(col);
        if (dim < 0)
        {
            continue;
        }
        if (killedBirthColumns[col])
        {
            continue;
        }

        Pair essential;
        essential.birth = static_cast<Field>(matrix_->getFiltrationValue(col));
        essential.death = inf;
        essential.dimension = static_cast<Dimension>(dim);
        pairs_.push_back(essential);
        essentials_.push_back(static_cast<Index>(col));
    }

    std::ranges::sort(pairs_, {}, &Pair::birth);
}

void Reducer::computeBettiNumbersFromPivots()
{
    Size max_dim = 0;
    for (const auto &pair : pairs_)
    {
        if (pair.dimension >= 0)
        {
            max_dim = std::max(max_dim, static_cast<Size>(pair.dimension));
        }
    }
    betti_.assign(max_dim + 1, 0);
    for (const auto &pair : pairs_)
    {
        if (pair.isInfinite() && pair.dimension >= 0)
        {
            const Size d = static_cast<Size>(pair.dimension);
            if (d < betti_.size())
            {
                betti_[d]++;
            }
        }
    }
}

void Reducer::classifyEssentials()
{
    if (!essentials_.empty())
    {
        return;
    }
    std::vector<bool> killedBirthColumns(reduced_matrix_.size(), false);
    for (Size col = 0; col < reduced_matrix_.size(); ++col)
    {
        const Index pivot = getPivot(col);
        if (pivot < 0)
        {
            continue;
        }
        const Index birth_col = matrix_->getColumnIndexForRowSimplex(static_cast<Size>(pivot));
        if (birth_col >= 0 && static_cast<Size>(birth_col) < killedBirthColumns.size())
        {
            killedBirthColumns[static_cast<Size>(birth_col)] = true;
        }
    }
    for (Size col = 0; col < reduced_matrix_.size(); ++col)
    {
        if (getPivot(col) == -1 && !killedBirthColumns[col])
        {
            essentials_.push_back(static_cast<Index>(col));
        }
    }
}

void Reducer::clearLowestPivot(Size j)
{
    if (j >= reduced_matrix_.size() || reduced_matrix_[j].empty())
    {
        return;
    }
    reduced_matrix_[j].pop_back();
}

Index Reducer::findLowestPivot(Size j) const
{
    if (j >= reduced_matrix_.size() || reduced_matrix_[j].empty())
    {
        return -1;
    }
    return static_cast<Index>(reduced_matrix_[j].back().first);
}

void Reducer::setPivot(Size j, Index pivot)
{
    if (j >= pivot_columns_.size())
    {
        pivot_columns_.resize(j + 1, -1);
    }
    pivot_columns_[j] = pivot;
}

void Reducer::eliminatePivot(Size j, Size k)
{
    if (j >= reduced_matrix_.size() || k >= reduced_matrix_.size())
    {
        return;
    }
    const Z2Column a = columnToZ2Rows(reduced_matrix_[j]);
    const Z2Column b = columnToZ2Rows(reduced_matrix_[k]);
    reduced_matrix_[j] = fromZ2Rows(symmetricDifferenceSorted(a, b));
    ++num_operations_;
}

void Reducer::clearBirthColumn(Size pivotRow, std::vector<bool> &cleared)
{
    Index birth_col = matrix_->getColumnIndexForRowSimplex(pivotRow);
    if (birth_col >= 0 && static_cast<Size>(birth_col) < reduced_matrix_.size())
    {
        const Size bIdx = static_cast<Size>(birth_col);
        if (!cleared[bIdx])
        {
            cleared[bIdx] = true;
            reduced_matrix_[bIdx].clear();
            if (bIdx < pivot_columns_.size())
            {
                pivot_columns_[bIdx] = -1;
            }
        }
    }
}

std::vector<Reducer::ApparentPairInfo> Reducer::findApparentPairs(std::vector<bool> &cleared)
{
    std::vector<ApparentPairInfo> result;
    const Size cols = reduced_matrix_.size();

    for (Size j = 0; j < cols; ++j)
    {
        if (cleared[j])
            continue;
        const Index pivotJ = findLowestPivot(j);
        if (pivotJ < 0)
            continue;

        const Size i = static_cast<Size>(pivotJ);
        if (i >= cols || cleared[i])
            continue;

        const Index pivotI = findLowestPivot(i);
        if (pivotI == pivotJ)
        {
            cleared[i] = true;
            cleared[j] = true;
            setPivot(j, pivotJ);
            reduced_matrix_[i].clear();
            reduced_matrix_[j].clear();
            result.push_back({i, j});
        }
    }

    return result;
}

void Reducer::reduceTwist(std::vector<Index> &lowRowToCol, std::vector<bool> &cleared)
{
    const Size rows = matrix_->rows();
    const Size cols = reduced_matrix_.size();

    std::vector<Index> pivotToCol(rows, -1);
    for (Size col = 0; col < cols; ++col)
    {
        if (cleared[col])
            continue;
        const Index pivot = findLowestPivot(col);
        if (pivot < 0)
        {
            cleared[col] = true;
            continue;
        }
        const Size p = static_cast<Size>(pivot);
        if (p < rows)
        {
            pivotToCol[p] = static_cast<Index>(col);
        }
    }

    for (Size p = rows; p-- > 0;)
    {
        const Index j = pivotToCol[p];
        if (j < 0)
            continue;

        const Size col = static_cast<Size>(j);
        if (cleared[col])
            continue;

        while (true)
        {
            const Index pivot = findLowestPivot(col);
            if (pivot < 0)
                break;

            const Size pIdx = static_cast<Size>(pivot);
            const Index killer = lowRowToCol[pIdx];

            if (killer < 0)
            {
                lowRowToCol[pIdx] = static_cast<Index>(col);
                setPivot(col, pivot);
                clearBirthColumn(pIdx, cleared);
                break;
            }

            eliminatePivot(col, static_cast<Size>(killer));
        }
    }
}

Z2Column Reducer::toZ2Rows(const Vector<std::pair<Size, double>> &column)
{
    return columnToZ2Rows(column);
}

void Reducer::reduceMatrix()
{
    if (matrix_ == nullptr)
    {
        return;
    }
    if (!columns_loaded_)
    {
        reduceMatrixColumnByColumn();
    }

    const Size rows = matrix_->rows();
    const Size cols = reduced_matrix_.size();

    // Try GPU acceleration for large matrices (CUDA only)
    bool used_gpu = false;
#ifdef __CUDACC__
    if (use_gpu_ && cols > gpu_threshold_ && nerve::gpu::isAvailable())
    {
        auto gpu_result = nerve::gpu::computeReduction(*this, *matrix_);
        if (gpu_result.isSuccess())
        {
            used_gpu = true;
        }
    }
#endif

    if (!used_gpu)
    {
        std::vector<Index> lowRowToCol(rows, -1);
        pivot_columns_.assign(cols, -1);
        std::vector<bool> cleared(cols, false);

        for (Size col = 0; col < cols; ++col)
        {
            if (reduced_matrix_[col].empty())
            {
                cleared[col] = true;
            }
        }

        findApparentPairs(cleared);

        reduceTwist(lowRowToCol, cleared);

        std::fill(lowRowToCol.begin(), lowRowToCol.end(), Index(-1));

        for (Size col = 0; col < cols; ++col)
        {
            if (cleared[col])
                continue;
            while (true)
            {
                const Index pivot = findLowestPivot(col);
                if (pivot < 0)
                    break;
                const Index killer = lowRowToCol[static_cast<Size>(pivot)];
                if (killer < 0)
                {
                    lowRowToCol[static_cast<Size>(pivot)] = static_cast<Index>(col);
                    setPivot(col, pivot);
                    clearBirthColumn(static_cast<Size>(pivot), cleared);
                    break;
                }
                eliminatePivot(col, static_cast<Size>(killer));
            }
        }

        compact_pivots_ = pivot_columns_;
    }
}

} // namespace nerve::persistence
