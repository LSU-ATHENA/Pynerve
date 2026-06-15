
#include "nerve/persistence/reduction/reduction_lock_free_structures.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace nerve::persistence::lockfree
{

namespace
{

constexpr int EMPTY_PIVOT = -1;

size_t checkedPowerOfTwoCapacity(size_t requested_capacity)
{
    size_t capacity = 1;
    const size_t minimum_capacity = std::max<size_t>(requested_capacity, 2);
    while (capacity < minimum_capacity)
    {
        if (capacity > std::numeric_limits<size_t>::max() / 2)
        {
            throw std::length_error("lockfree capacity exceeds addressable range");
        }
        capacity *= 2;
    }
    return capacity;
}

} // namespace

LockFreePivotTable::LockFreePivotTable(size_t initial_capacity)
{
    table_.resize(checkedPowerOfTwoCapacity(initial_capacity));
}

size_t LockFreePivotTable::hash(int pivot) const
{
    std::hash<int> hasher;
    return hasher(pivot) & (table_.size() - 1);
}

bool LockFreePivotTable::tryInsert(int pivot, int column)
{
    if (pivot < 0 || column < 0)
    {
        throw std::invalid_argument("lockfree pivot table requires non-negative keys");
    }

    size_t idx = hash(pivot);

    for (size_t probe = 0; probe < table_.size(); ++probe)
    {
        size_t pos = (idx + probe) & (table_.size() - 1);

        Entry &entry = table_[pos];

        const int current_pivot = entry.pivot.load(std::memory_order_acquire);
        if (current_pivot == pivot)
        {
            return false;
        }
        if (current_pivot != EMPTY_PIVOT)
        {
            continue;
        }

        int expected = EMPTY_PIVOT;
        if (entry.pivot.compare_exchange_strong(expected, pivot, std::memory_order_acq_rel,
                                                std::memory_order_acquire))
        {
            entry.column.store(column, std::memory_order_release);
            entry.occupied.store(true, std::memory_order_release);
            size_.fetch_add(1, std::memory_order_relaxed);
            return true;
        }

        if (expected == pivot)
        {
            return false;
        }
    }

    return false;
}

int LockFreePivotTable::find(int pivot) const
{
    size_t idx = hash(pivot);

    for (size_t probe = 0; probe < table_.size(); ++probe)
    {
        size_t pos = (idx + probe) & (table_.size() - 1);

        const Entry &entry = table_[pos];
        const int current_pivot = entry.pivot.load(std::memory_order_acquire);

        if (current_pivot == EMPTY_PIVOT)
        {
            return -1;
        }

        if (current_pivot == pivot && entry.occupied.load(std::memory_order_acquire))
        {
            return entry.column.load(std::memory_order_acquire);
        }
    }

    return -1;
}

size_t LockFreePivotTable::size() const
{
    return size_.load(std::memory_order_relaxed);
}

size_t LockFreePivotTable::memoryUsage() const
{
    return table_.size() * sizeof(Entry);
}

} // namespace nerve::persistence::lockfree
