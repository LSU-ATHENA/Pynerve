#include "memory/safe_memory_pool.hpp"

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <new>

namespace nerve::memory
{

namespace
{

template <typename T, Size SlabCapacity>
struct PoolDiagnostics
{
    std::atomic<Size> alloc_count{0};
    std::atomic<Size> free_count{0};
    std::atomic<Size> slab_count{0};

    static PoolDiagnostics &instance()
    {
        static PoolDiagnostics diag;
        return diag;
    }
};

} // namespace

Size getSlabAllocatorDiagnosticCount() noexcept
{
    return PoolDiagnostics<void, 256>::instance().alloc_count.load(std::memory_order_relaxed);
}

Size estimateMemoryOverhead(Size object_count, Size object_size, Size slab_capacity) noexcept
{
    if (slab_capacity == 0)
        return 0;
    Size slabs = (object_count + slab_capacity - 1) / slab_capacity;
    Size data_overhead = slabs * slab_capacity * object_size;
    Size ptr_overhead = slabs * slab_capacity * sizeof(void *);
    return data_overhead + ptr_overhead;
}

} // namespace nerve::memory
