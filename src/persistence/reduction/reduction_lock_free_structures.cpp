
// For many-core scaling (32+ threads)

#include "nerve/persistence/reduction/reduction_lock_free_structures.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>

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

size_t checkedThreadIndex(int thread_id, size_t size)
{
    if (thread_id < 0 || static_cast<size_t>(thread_id) >= size)
    {
        throw std::out_of_range("lockfree thread id out of range");
    }
    return static_cast<size_t>(thread_id);
}

size_t checkedColumnCount(size_t num_columns)
{
    if (num_columns > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        throw std::overflow_error("lockfree coordinator column count exceeds int range");
    }
    return num_columns;
}

void validateBenchmarkInputs(int num_threads, int operations_per_thread, size_t table_size)
{
    if (num_threads <= 0)
    {
        throw std::invalid_argument("lockfree benchmark requires at least one thread");
    }
    if (operations_per_thread <= 0)
    {
        throw std::invalid_argument("lockfree benchmark requires at least one operation");
    }
    if (table_size == 0)
    {
        throw std::invalid_argument("lockfree benchmark requires a non-empty table");
    }
    if (table_size > static_cast<size_t>(std::numeric_limits<int>::max()) / 2)
    {
        throw std::overflow_error("lockfree benchmark table size exceeds int key range");
    }
}

} // namespace

} // namespace nerve::persistence::lockfree
