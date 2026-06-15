
#include "nerve/persistence/reduction/reduction_lock_free_structures.hpp"

#include <stdexcept>

namespace nerve::persistence::lockfree
{

namespace
{

size_t checkedThreadIndex(int thread_id, size_t size)
{
    if (thread_id < 0 || static_cast<size_t>(thread_id) >= size)
    {
        throw std::out_of_range("lockfree thread id out of range");
    }
    return static_cast<size_t>(thread_id);
}

} // namespace

LockFreePivotAnnounce::LockFreePivotAnnounce(size_t num_threads)
    : announcements_(num_threads)
{
    for (auto &a : announcements_)
    {
        a.store(-1, std::memory_order_relaxed);
    }
}

void LockFreePivotAnnounce::announce(int thread_id, int pivot)
{
    announcements_[checkedThreadIndex(thread_id, announcements_.size())].store(
        pivot, std::memory_order_release);
}

bool LockFreePivotAnnounce::isBeingWorkedOn(int pivot) const
{
    for (const auto &a : announcements_)
    {
        if (a.load(std::memory_order_acquire) == pivot)
        {
            return true;
        }
    }
    return false;
}

void LockFreePivotAnnounce::clear(int thread_id)
{
    announcements_[checkedThreadIndex(thread_id, announcements_.size())].store(
        -1, std::memory_order_release);
}

} // namespace nerve::persistence::lockfree
