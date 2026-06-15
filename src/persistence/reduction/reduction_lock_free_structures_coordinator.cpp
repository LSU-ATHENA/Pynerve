
#include "nerve/persistence/reduction/reduction_lock_free_structures.hpp"

#include <limits>
#include <stdexcept>

namespace nerve::persistence::lockfree
{

namespace
{

size_t checkedColumnCount(size_t num_columns)
{
    if (num_columns > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        throw std::overflow_error("lockfree coordinator column count exceeds int range");
    }
    return num_columns;
}

} // namespace

LockFreeReductionCoordinator::LockFreeReductionCoordinator(size_t num_columns)
    : total_columns_(checkedColumnCount(num_columns))
    , reduced_flags_(total_columns_)
{}

int LockFreeReductionCoordinator::claimColumn()
{
    size_t col = next_column_.fetch_add(1, std::memory_order_relaxed);
    if (col >= total_columns_)
    {
        return -1;
    }
    return static_cast<int>(col);
}

void LockFreeReductionCoordinator::markReduced(int column_idx)
{
    if (column_idx < 0 || static_cast<size_t>(column_idx) >= total_columns_)
    {
        throw std::out_of_range("lockfree reduction column index out of range");
    }
    if (!reduced_flags_[static_cast<size_t>(column_idx)].testAndSet())
    {
        num_reduced_.fetch_add(1, std::memory_order_relaxed);
    }
}

bool LockFreeReductionCoordinator::allReduced() const
{
    return num_reduced_.load(std::memory_order_relaxed) >= total_columns_;
}

size_t LockFreeReductionCoordinator::numReduced() const
{
    return num_reduced_.load(std::memory_order_relaxed);
}

size_t LockFreeReductionCoordinator::totalColumns() const
{
    return total_columns_;
}

} // namespace nerve::persistence::lockfree
