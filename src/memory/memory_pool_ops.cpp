#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <new>

#ifdef __linux__
#include <sys/mman.h>
#if defined(NERVE_HAS_NUMA)
#include <numa.h>
#endif
#endif

#include "memory/safe_memory_pool.hpp"

namespace nerve::memory
{

namespace
{

struct GlobalMemoryTracker
{
    std::atomic<Size> total_allocations{0};
    std::atomic<Size> total_deallocations{0};
    std::atomic<Size> current_bytes{0};
    std::atomic<Size> peak_bytes{0};

    static GlobalMemoryTracker &instance()
    {
        static GlobalMemoryTracker tracker;
        return tracker;
    }
};

} // namespace

Size getGlobalAllocationCount() noexcept
{
    return GlobalMemoryTracker::instance().total_allocations.load(std::memory_order_relaxed);
}

Size getGlobalDeallocationCount() noexcept
{
    return GlobalMemoryTracker::instance().total_deallocations.load(std::memory_order_relaxed);
}

Size getGlobalCurrentBytes() noexcept
{
    return GlobalMemoryTracker::instance().current_bytes.load(std::memory_order_relaxed);
}

Size getGlobalPeakBytes() noexcept
{
    return GlobalMemoryTracker::instance().peak_bytes.load(std::memory_order_relaxed);
}

void trackAllocEvent(Size bytes) noexcept
{
    auto &t = GlobalMemoryTracker::instance();
    t.total_allocations.fetch_add(1, std::memory_order_relaxed);
    t.current_bytes.fetch_add(bytes, std::memory_order_relaxed);
    Size curr = t.current_bytes.load(std::memory_order_relaxed);
    Size peak = t.peak_bytes.load(std::memory_order_relaxed);
    while (curr > peak)
        t.peak_bytes.compare_exchange_weak(peak, curr, std::memory_order_relaxed,
                                           std::memory_order_relaxed);
}

void trackFreeEvent(Size bytes) noexcept
{
    auto &t = GlobalMemoryTracker::instance();
    t.total_deallocations.fetch_add(1, std::memory_order_relaxed);
    t.current_bytes.fetch_sub(bytes, std::memory_order_relaxed);
}

void resetGlobalMemoryStats() noexcept
{
    auto &t = GlobalMemoryTracker::instance();
    t.total_allocations.store(0, std::memory_order_relaxed);
    t.total_deallocations.store(0, std::memory_order_relaxed);
    t.current_bytes.store(0, std::memory_order_relaxed);
    t.peak_bytes.store(0, std::memory_order_relaxed);
}

Size getSlabAllocatorDiagnosticCount() noexcept
{
    return GlobalMemoryTracker::instance().total_allocations.load(std::memory_order_relaxed);
}

Size estimateMemoryOverhead(Size object_count, Size object_size, Size slab_capacity) noexcept
{
    if (slab_capacity == 0)
        return 0;
    Size slabs = (object_count + slab_capacity - 1) / slab_capacity;
    return slabs * slab_capacity * (object_size + sizeof(void *));
}

GlobalPagePool &GlobalPagePool::instance()
{
    static GlobalPagePool pool;
    return pool;
}

static void *tryHugepageAlloc()
{
#ifdef __linux__
    void *p =
        mmap(nullptr, kHugePageSize, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p != MAP_FAILED)
    {
        madvise(p, kHugePageSize, MADV_HUGEPAGE);
        return p;
    }
    p = mmap(nullptr, kHugePageSize, PROT_READ | PROT_WRITE,
             MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
    if (p != MAP_FAILED)
        return p;
#endif
    return nullptr;
}

GlobalPagePool::GlobalPagePool()
    : hugepages_available_(false)
{
    void *probe = tryHugepageAlloc();
    if (probe)
    {
        hugepages_available_ = true;
        auto *node = static_cast<PageNode *>(probe);
        node->next = free_list_.load(std::memory_order_relaxed);
        free_list_.store(node, std::memory_order_release);
    }
}

GlobalPagePool::~GlobalPagePool()
{
    PageNode *node = free_list_.load(std::memory_order_relaxed);
    while (node)
    {
        PageNode *next = node->next;
        munmap(node, kHugePageSize);
        node = next;
    }
}

void *GlobalPagePool::allocFromOS() noexcept
{
    void *p = hugepages_available_ ? tryHugepageAlloc() : nullptr;
    if (p)
    {
        hugetlb_pages_.fetch_add(1, std::memory_order_relaxed);
        return p;
    }
    p = std::aligned_alloc(kHugePageSize, kHugePageSize);
    if (!p)
    {
        p = std::malloc(kHugePageSize);
    }
    return p;
}

void *GlobalPagePool::allocatePage() noexcept
{
    PageNode *node = free_list_.load(std::memory_order_acquire);
    while (node)
    {
        if (free_list_.compare_exchange_weak(node, node->next, std::memory_order_release,
                                             std::memory_order_acquire))
        {
            allocated_pages_.fetch_add(1, std::memory_order_relaxed);
            return node;
        }
    }
    void *p = allocFromOS();
    if (p)
        allocated_pages_.fetch_add(1, std::memory_order_relaxed);
    return p;
}

void GlobalPagePool::deallocatePage(void *page) noexcept
{
    if (!page)
        return;
    auto *node = static_cast<PageNode *>(page);
    node->next = free_list_.load(std::memory_order_acquire);
    while (!free_list_.compare_exchange_weak(node->next, node, std::memory_order_release,
                                             std::memory_order_acquire))
    {
    }
    allocated_pages_.fetch_sub(1, std::memory_order_relaxed);
}

Size GlobalPagePool::pagesAllocated() const noexcept
{
    return allocated_pages_.load(std::memory_order_relaxed);
}

Size GlobalPagePool::hugetlbPagesAllocated() const noexcept
{
    return hugetlb_pages_.load(std::memory_order_relaxed);
}

RawArrayPool::RawArrayPool(Size initial_bytes, bool use_hugepages)
    : state_(std::make_unique<PoolState>())
    , initial_byte_size_(initial_bytes)
    , use_hugepages_(use_hugepages)
{
    Size alloc_size = aligned(initial_bytes);
    void *base = nullptr;
    if (use_hugepages && alloc_size >= kHugePageSize)
    {
        base = GlobalPagePool::instance().allocatePage();
        alloc_size = kHugePageSize;
    }
    if (!base)
        base = std::malloc(alloc_size);
    if (!base)
        throw std::bad_alloc();
    state_->blocks.emplace_back();
    auto &b = state_->blocks.back();
    b.base = base;
    b.total = alloc_size;
    b.used.store(0, std::memory_order_relaxed);
    trackAllocEvent(alloc_size);
}

RawArrayPool::~RawArrayPool()
{
    for (auto &b : state_->blocks)
    {
        if (use_hugepages_ && b.total >= kHugePageSize &&
            reinterpret_cast<uintptr_t>(b.base) % kHugePageSize == 0)
        {
            GlobalPagePool::instance().deallocatePage(b.base);
        }
        else
        {
            std::free(b.base);
        }
        trackFreeEvent(b.total);
    }
}

void *RawArrayPool::allocate(Size bytes)
{
    bytes = aligned(bytes);
    Size idx = state_->current_block_idx.load(std::memory_order_acquire);
    Size n = state_->blocks.size();

    for (Size attempt = 0; attempt < n; ++attempt)
    {
        Block &b = state_->blocks[idx % n];
        Size used = b.used.load(std::memory_order_acquire);
        Size new_used = used + bytes;
        if (new_used <= b.total &&
            b.used.compare_exchange_weak(used, new_used, std::memory_order_release,
                                         std::memory_order_acquire))
        {
            total_allocated_.fetch_add(bytes, std::memory_order_relaxed);
            Size curr = total_allocated_.load(std::memory_order_relaxed);
            Size peak = peak_used_.load(std::memory_order_relaxed);
            while (curr > peak)
                peak_used_.compare_exchange_weak(peak, curr, std::memory_order_relaxed,
                                                 std::memory_order_relaxed);
            return static_cast<char *>(b.base) + used;
        }
        idx = (idx + 1) % n;
    }

    Size new_size = std::max(bytes, state_->blocks.back().total * 2);
    if (use_hugepages_)
        new_size = (new_size + kHugePageSize - 1) & ~(kHugePageSize - 1);
    void *base = use_hugepages_ ? GlobalPagePool::instance().allocatePage() : std::malloc(new_size);
    if (!base)
        return nullptr;
    state_->blocks.emplace_back();
    auto &b = state_->blocks.back();
    b.base = base;
    b.total = new_size;
    b.used.store(bytes, std::memory_order_relaxed);
    state_->current_block_idx.store(state_->blocks.size() - 1, std::memory_order_release);
    trackAllocEvent(new_size);
    total_allocated_.fetch_add(bytes, std::memory_order_relaxed);
    return base;
}

void RawArrayPool::deallocate(void * /*ptr*/, Size bytes)
{
    total_allocated_.fetch_sub(bytes, std::memory_order_relaxed);
}

void RawArrayPool::reset() noexcept
{
    for (auto &b : state_->blocks)
        b.used.store(0, std::memory_order_relaxed);
    total_allocated_.store(0, std::memory_order_relaxed);
    state_->current_block_idx.store(0, std::memory_order_release);
}

struct NumaAwareAllocator::NumaPolicy
{
    bool numa_available;
    int preferred_node;
    int actual_nodes;
    explicit NumaPolicy(int node)
        : preferred_node(node)
        , actual_nodes(0)
    {
#if defined(__linux__) && defined(NERVE_HAS_NUMA)
        if (numa_available() != -1)
        {
            numa_available = true;
            actual_nodes = numa_max_node() + 1;
        }
        else
#endif
        {
            numa_available = false;
        }
    }
};

NumaAwareAllocator::NumaAwareAllocator(int preferred_node)
    : preferred_node_(preferred_node)
    , policy_(std::make_unique<NumaPolicy>(preferred_node))
{}

NumaAwareAllocator::~NumaAwareAllocator() = default;

void *NumaAwareAllocator::allocate(Size bytes, Size alignment)
{
#if defined(__linux__) && defined(NERVE_HAS_NUMA)
    if (policy_->numa_available && policy_->preferred_node >= 0)
    {
        void *p = numa_alloc_onnode(bytes, policy_->preferred_node);
        if (p)
        {
            trackAllocEvent(bytes);
            return p;
        }
    }
#endif
    if (alignment <= alignof(std::max_align_t))
    {
        void *p = std::aligned_alloc(alignment, bytes);
        if (p)
            trackAllocEvent(bytes);
        return p ? p : std::malloc(bytes);
    }
    Size padded = bytes + alignment + sizeof(void *);
    void *raw = std::malloc(padded);
    if (!raw)
        return nullptr;
    void *aligned = raw;
    std::align(alignment, bytes, aligned, padded);
    *(static_cast<void **>(aligned) - 1) = raw;
    trackAllocEvent(bytes + alignment + sizeof(void *));
    return aligned;
}

void NumaAwareAllocator::deallocate(void *ptr, Size bytes)
{
    if (!ptr)
        return;
#if defined(__linux__) && defined(NERVE_HAS_NUMA)
    if (policy_->numa_available)
    {
        numa_free(ptr, bytes);
        trackFreeEvent(bytes);
        return;
    }
#endif
    Size alignment = kCacheLineSize;
    if (reinterpret_cast<uintptr_t>(ptr) % alignment >
        static_cast<uintptr_t>(alignof(std::max_align_t)))
    {
        void *raw = *(static_cast<void **>(ptr) - 1);
        std::free(raw);
    }
    else
    {
        std::free(ptr);
    }
    trackFreeEvent(bytes);
}

int NumaAwareAllocator::getPreferredNode() const noexcept
{
    return preferred_node_;
}
void NumaAwareAllocator::setPreferredNode(int node)
{
    preferred_node_ = node;
}

SizeClassAllocator::SizeClassAllocator()
{
    for (Size i = 0; i < kNumSizeClasses; ++i)
    {
        Size bytes = size_classes_[i];
        pools_[i] = std::make_unique<RawArrayPool>(bytes * 256, false);
    }
}

void *SizeClassAllocator::allocate(Size bytes)
{
    Size idx = findPoolIndex(bytes);
    if (idx < kNumSizeClasses)
    {
        void *p = pools_[idx]->allocate(bytes);
        if (p)
            return p;
    }
    return std::malloc(bytes);
}

void SizeClassAllocator::deallocate(void *ptr, Size bytes)
{
    Size idx = findPoolIndex(bytes);
    if (idx < kNumSizeClasses)
    {
        pools_[idx]->deallocate(ptr, bytes);
        return;
    }
    std::free(ptr);
}

Size SizeClassAllocator::totalAllocated() const noexcept
{
    Size total = 0;
    for (Size i = 0; i < kNumSizeClasses; ++i)
        total += pools_[i]->totalAllocated();
    return total;
}

Size SizeClassAllocator::getSizeClass(Size bytes) const noexcept
{
    return size_classes_[findPoolIndex(bytes)];
}

Size SizeClassAllocator::findPoolIndex(Size bytes) const noexcept
{
    for (Size i = 0; i < kNumSizeClasses; ++i)
    {
        if (bytes <= size_classes_[i])
            return i;
    }
    return kNumSizeClasses;
}

} // namespace nerve::memory
