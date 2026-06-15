#include "nerve/algebra/bit_packed.hpp"
#include "nerve/persistence/streaming/streaming_reducer.hpp"

#include <algorithm>
#include <cstddef>

namespace nerve::persistence::streaming
{

StreamingReducer::StreamingReducer(bool use_bit_packed)
    : use_bit_packed_(use_bit_packed)
{}

PersistenceDiagram StreamingReducer::reduce(const StreamingColumnGenerator &generator)
{
    PersistenceDiagram result;

    const size_t n_simplices = generator.getNumSimplices();

    if (use_bit_packed_)
    {
        bit_matrix_ = std::make_unique<algebra::compressed::BitPackedZ2Matrix>(
            n_simplices, std::min(MAX_PIVOT_CACHE_SIZE, n_simplices));
    }

    for (size_t col_idx = 0; col_idx < n_simplices; ++col_idx)
    {
        std::vector<int> column = generator.generateColumn(col_idx);

        loadColumn(column);

        std::ptrdiff_t pivot = findPivot();

        while (pivot >= 0 && hasCachedColumn(static_cast<int>(pivot)))
        {
            xorWithCachedColumn(static_cast<int>(pivot));
            pivot = findPivot();
        }

        if (pivot >= 0)
        {
            PersistencePair pair;
            pair.birth = generator.getFiltrationValue(static_cast<size_t>(pivot));
            pair.death = generator.getFiltrationValue(col_idx);
            pair.dimension = generator.getSimplexDimension(col_idx);
            result.pairs.push_back(pair);

            cacheColumn(static_cast<int>(pivot), col_idx);
        }
        else
        {
            PersistencePair pair;
            pair.birth = generator.getFiltrationValue(col_idx);
            pair.death = std::numeric_limits<double>::infinity();
            pair.dimension = generator.getSimplexDimension(col_idx);
            result.pairs.push_back(pair);
        }

        clearCurrentColumn();
    }

    return result;
}

size_t StreamingReducer::getMemoryUsage() const
{
    size_t usage = 0;

    usage += current_column_.capacity() * sizeof(int);

    usage += pivot_to_column_.size() * sizeof(std::pair<int, int>);

    if (bit_matrix_)
    {
        usage += bit_matrix_->memoryBytes();
    }
    else
    {
        for (const auto &[_, col] : cached_columns_)
        {
            usage += col.size() * sizeof(int);
        }
    }

    return usage;
}

void StreamingReducer::loadColumn(const std::vector<int> &column)
{
    current_column_ = column;
}

std::ptrdiff_t StreamingReducer::findPivot() const
{
    if (current_column_.empty())
    {
        return -1;
    }
    return *std::max_element(current_column_.begin(), current_column_.end());
}

bool StreamingReducer::hasCachedColumn(int pivot) const
{
    if (pivot < 0)
    {
        return false;
    }
    if (cached_columns_.find(pivot) != cached_columns_.end())
    {
        return true;
    }
    return bit_matrix_ && bit_matrix_->hasColumn(pivot);
}

void StreamingReducer::xorWithCachedColumn(int pivot)
{
    std::vector<int> cached;
    if (bit_matrix_)
    {
        cached = bit_matrix_->getColumn(pivot);
    }
    else
    {
        auto it = cached_columns_.find(pivot);
        if (it != cached_columns_.end())
        {
            cached = it->second;
        }
    }

    std::vector<int> result;
    std::set_symmetric_difference(current_column_.begin(), current_column_.end(), cached.begin(),
                                  cached.end(), std::back_inserter(result));

    current_column_ = std::move(result);
}

void StreamingReducer::cacheColumn(int pivot, size_t col_idx)
{
    if (pivot < 0)
    {
        return;
    }

    const auto pivot_col = static_cast<size_t>(pivot);
    if (bit_matrix_ && pivot_col < bit_matrix_->getNumCols())
    {
        bit_matrix_->clearColumn(pivot_col);
        bit_matrix_->addColumn(pivot_col, current_column_);
    }
    else
    {
        cached_columns_[pivot] = current_column_;
    }

    pivot_to_column_[pivot] = static_cast<int>(col_idx);
}

void StreamingReducer::clearCurrentColumn()
{
    current_column_.clear();
}

} // namespace nerve::persistence::streaming
