#include "nerve/persistence/reduction/reduction_mpi_ops.hpp"

#if NERVE_HAS_MPI && __has_include(<mpi.h>)

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <ranges>
#include <stdexcept>
#include <unordered_map>

namespace nerve::persistence
{

namespace
{

constexpr double kZ2Tolerance = 1e-12;

using Z2Column = std::vector<Size>;

Z2Column symmetricDifferenceZ2(const Z2Column &a, const Z2Column &b)
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

Z2Column extractColumn(const algebra::BoundaryMatrix &matrix, Size col)
{
    Z2Column rows;
    const Size num_rows = matrix.rows();
    for (Size r = 0; r < num_rows; ++r)
    {
        if (std::fabs(matrix.getMatrixEntry(r, col)) > kZ2Tolerance)
        {
            rows.push_back(r);
        }
    }
    std::ranges::sort(rows);
    const auto [first, last] = std::ranges::unique(rows);
    rows.erase(first, last);
    return rows;
}

void checkMpiSuccess(int status, const char *context)
{
    if (status != MPI_SUCCESS)
    {
        throw std::runtime_error(context);
    }
}

int checkedInt(std::size_t value, const char *context)
{
    if (value > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::length_error(context);
    }
    return static_cast<int>(value);
}

} // namespace

void MpiDistributedReducer::compute(const algebra::BoundaryMatrix &matrix)
{
    matrix_ = &matrix;

    const int rank = comm_.rank();
    const int size = comm_.size();
    const Size cols = matrix_->cols();

    if (cols == 0)
    {
        return;
    }

    local_columns_.clear();
    for (Size c = 0; c < cols; ++c)
    {
        if (static_cast<int>(c % static_cast<std::size_t>(size)) == rank)
        {
            local_columns_.push_back(c);
        }
    }

    reduceLocalColumns();
    exchangePivots();
    resolveRemoteDependencies();
    computePersistencePairsFromPivots();
    computeBettiNumbersFromPivots();
}

void MpiDistributedReducer::reduceLocalColumns()
{
    if (matrix_ == nullptr || local_columns_.empty())
    {
        return;
    }

    const Size cols = matrix_->cols();
    local_pivots_.assign(cols, -1);
    std::unordered_map<Size, Z2Column> local_col_data;

    for (Size col : local_columns_)
    {
        local_col_data[col] = extractColumn(*matrix_, col);
    }

    std::vector<Size> sorted_locals = local_columns_;
    std::ranges::sort(sorted_locals);

    std::unordered_map<Index, Size> pivot_to_column;

    for (Size col : sorted_locals)
    {
        Z2Column &column = local_col_data[col];

        while (!column.empty())
        {
            const Index pivot = static_cast<Index>(column.back());
            const auto it = pivot_to_column.find(pivot);
            if (it == pivot_to_column.end())
            {
                break;
            }
            const Z2Column &reducer_col = local_col_data[it->second];
            column = symmetricDifferenceZ2(column, reducer_col);
        }

        if (!column.empty())
        {
            const Index pivot = static_cast<Index>(column.back());
            pivot_to_column[pivot] = col;
            local_pivots_[col] = pivot;
        }
    }
}

void MpiDistributedReducer::exchangePivots()
{
    const int size = comm_.size();

    if (matrix_ == nullptr)
    {
        return;
    }

    const Size cols = matrix_->cols();
    const int send_count = checkedInt(cols, "pivot count exceeds MPI int range");

    std::vector<int> pivots_send(cols, -1);
    for (Size c = 0; c < cols; ++c)
    {
        pivots_send[c] = static_cast<int>(local_pivots_[c]);
    }

    std::vector<int> pivots_recv(static_cast<std::size_t>(send_count * size), -1);

    checkMpiSuccess(MPI_Allgather(pivots_send.data(), send_count, MPI_INT, pivots_recv.data(),
                                  send_count, MPI_INT, MPI_COMM_WORLD),
                    "MPI_Allgather failed in MpiDistributedReducer::exchangePivots");

    global_pivots_.resize(cols, -1);
    for (Size col = 0; col < cols; ++col)
    {
        for (int src = 0; src < size; ++src)
        {
            const int pivot =
                pivots_recv[static_cast<std::size_t>(src * send_count + static_cast<int>(col))];
            if (pivot >= 0 && global_pivots_[col] < 0)
            {
                global_pivots_[col] = static_cast<Index>(pivot);
            }
        }
    }
}

void MpiDistributedReducer::resolveRemoteDependencies()
{
    if (matrix_ == nullptr)
    {
        return;
    }

    const Size cols = matrix_->cols();
    std::unordered_map<Index, Size> row_to_column;

    for (Size col = 0; col < cols; ++col)
    {
        const Index pivot = global_pivots_[col];
        if (pivot < 0)
        {
            continue;
        }
        const auto it = row_to_column.find(pivot);
        if (it == row_to_column.end() || col < it->second)
        {
            row_to_column[pivot] = col;
        }
    }

    global_pivots_.assign(cols, -1);
    for (const auto &[row, col] : row_to_column)
    {
        if (col < cols)
        {
            global_pivots_[col] = row;
        }
    }
}

void MpiDistributedReducer::gatherResults()
{
    const int rank = comm_.rank();
    const int size = comm_.size();

    if (matrix_ == nullptr)
    {
        return;
    }

    int local_pair_count = static_cast<int>(pairs_.size());
    int local_essential_count = static_cast<int>(essentials_.size());

    checkMpiSuccess(MPI_Bcast(&local_pair_count, 1, MPI_INT, 0, MPI_COMM_WORLD),
                    "MPI_Bcast pair count failed");
    checkMpiSuccess(MPI_Bcast(&local_essential_count, 1, MPI_INT, 0, MPI_COMM_WORLD),
                    "MPI_Bcast essential count failed");

    std::vector<int> recv_counts;
    if (rank == 0)
    {
        recv_counts.resize(static_cast<std::size_t>(size));
    }

    checkMpiSuccess(MPI_Gather(&local_pair_count, 1, MPI_INT,
                               rank == 0 ? recv_counts.data() : nullptr, 1, MPI_INT, 0,
                               MPI_COMM_WORLD),
                    "MPI_Gather pair counts failed");

    (void)size;
}

void MpiDistributedReducer::computePersistencePairsFromPivots()
{
    pairs_.clear();
    essentials_.clear();

    if (matrix_ == nullptr)
    {
        return;
    }

    const Field inf = std::numeric_limits<Field>::infinity();
    const Size cols = matrix_->cols();
    std::vector<bool> killed_birth(cols, false);

    for (Size col = 0; col < cols; ++col)
    {
        const Index pivot = global_pivots_[col];
        if (pivot < 0)
        {
            continue;
        }
        const Index birth_col = matrix_->getColumnIndexForRowSimplex(static_cast<Size>(pivot));
        if (birth_col >= 0 && static_cast<Size>(birth_col) < killed_birth.size())
        {
            killed_birth[static_cast<Size>(birth_col)] = true;
        }
    }

    for (Size col = 0; col < cols; ++col)
    {
        const Index pivot = global_pivots_[col];
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
        if (killed_birth[col])
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

void MpiDistributedReducer::computeBettiNumbersFromPivots()
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

} // namespace nerve::persistence

#endif // NERVE_HAS_MPI
