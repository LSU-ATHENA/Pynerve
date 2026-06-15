#pragma once
#include "nerve/core_types.hpp"

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <new>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

namespace nerve::memory
{

static constexpr Size kCacheLineSize = 64;
static constexpr Size kHugePageSize = 2 * 1024 * 1024;

template <typename T>
struct alignas(kCacheLineSize) CacheLineAligned
{
    T value;
    CacheLineAligned()
        : value{}
    {}
    explicit CacheLineAligned(T v)
        : value(v)
    {}
    operator T &() noexcept { return value; }
    operator const T &() const noexcept { return value; }
    T *operator&() noexcept { return &value; }
};

class GlobalPagePool
{
public:
    static GlobalPagePool &instance();

    void *allocatePage() noexcept;
    void deallocatePage(void *page) noexcept;
    Size pagesAllocated() const noexcept;
    Size hugetlbPagesAllocated() const noexcept;
    Size pageSize() const noexcept { return kHugePageSize; }

private:
    GlobalPagePool();
    ~GlobalPagePool();

    struct PageNode
    {
        PageNode *next;
    };

    std::atomic<PageNode *> free_list_{nullptr};
    std::atomic<Size> allocated_pages_{0};
    std::atomic<Size> hugetlb_pages_{0};
    bool hugepages_available_{false};

    void *allocFromOS() noexcept;
};

template <typename T, Size SlabCapacity = 256>
class SlabAllocator
{
    static_assert(SlabCapacity > 0, "SlabCapacity must be positive");

public:
    SlabAllocator() { addSlab(); }
    SlabAllocator(const SlabAllocator &) = delete;
    SlabAllocator &operator=(const SlabAllocator &) = delete;

    ~SlabAllocator() noexcept
    {
#ifndef NDEBUG
        std::lock_guard<std::mutex> lock(mutex_);
        if (!live_.empty())
        {
            fprintf(stderr, "[Nerve MemPool] WARNING: %zu objects live at destruction\n",
                    live_.size());
        }
#endif
    }

    T *allocate()
    {
        if (free_head_.load(std::memory_order_acquire) == nullptr)
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (free_head_.load(std::memory_order_relaxed) == nullptr)
                addSlab();
        }
        PageNode *node = free_head_.load(std::memory_order_acquire);
        while (node && !free_head_.compare_exchange_weak(
                           node, node->next, std::memory_order_release, std::memory_order_acquire))
        {
        }
        T *p = reinterpret_cast<T *>(node);
#ifndef NDEBUG
        std::lock_guard<std::mutex> dbg_lock(mutex_);
        live_.insert(p);
#endif
        return p;
    }

    void deallocate(T *p) noexcept
    {
        if (!p)
            return;
#ifndef NDEBUG
        {
            std::lock_guard<std::mutex> dbg_lock(mutex_);
            auto it = live_.find(p);
            assert(it != live_.end() && "double-free or invalid pointer");
            if (it == live_.end())
                return;
            live_.erase(it);
        }
#endif
        auto *node = reinterpret_cast<PageNode *>(p);
        node->next = free_head_.load(std::memory_order_acquire);
        while (!free_head_.compare_exchange_weak(node->next, node, std::memory_order_release,
                                                 std::memory_order_acquire))
        {
        }
    }

    void prefetchSlab() noexcept
    {
        if (free_head_.load(std::memory_order_relaxed) == nullptr)
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (free_head_.load(std::memory_order_relaxed) == nullptr)
                addSlab();
        }
    }

    void reset() noexcept
    {
        std::lock_guard<std::mutex> lock(mutex_);
        free_head_.store(nullptr, std::memory_order_release);
#ifndef NDEBUG
        live_.clear();
#endif
        for (auto &slab : slabs_)
        {
            for (Size i = 0; i < SlabCapacity; ++i)
            {
                auto *node = reinterpret_cast<PageNode *>(&(*slab)[i]);
                node->next = free_head_.load(std::memory_order_relaxed);
                free_head_.store(node, std::memory_order_release);
            }
        }
    }

    Size capacity() const noexcept
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return slabs_.size() * SlabCapacity;
    }

    Size available() const noexcept { return capacity(); }

private:
    struct alignas(kCacheLineSize) Slab
    {
        alignas(T) std::byte data_storage[SlabCapacity * sizeof(T)];
        T *data() { return reinterpret_cast<T *>(data_storage); }
        T &operator[](Size i) { return data()[i]; }
    };

    struct alignas(kCacheLineSize) PageNode
    {
        PageNode *next;
    };

    std::vector<std::unique_ptr<Slab>> slabs_;
    std::atomic<PageNode *> free_head_{nullptr};
    mutable std::mutex mutex_;
#ifndef NDEBUG
    std::unordered_set<T *> live_;
#endif

    static_assert(sizeof(Slab) <= SlabCapacity * sizeof(T) + kCacheLineSize,
                  "Slab fits in expected alignment boundary");

    void addSlab()
    {
        auto &slab = slabs_.emplace_back(std::make_unique<Slab>());
        PageNode *head = free_head_.load(std::memory_order_relaxed);
        for (Size i = 0; i < SlabCapacity; ++i)
        {
            auto *node = reinterpret_cast<PageNode *>(&(*slab)[i]);
            node->next = head;
            head = node;
        }
        free_head_.store(head, std::memory_order_release);
    }
};

template <typename T, Size SlabCapacity = 256>
class ThreadLocalPool
{
public:
    using allocator_type = SlabAllocator<T, SlabCapacity>;

    static ThreadLocalPool &instance()
    {
        thread_local ThreadLocalPool pool;
        return pool;
    }

    T *allocate() { return getLocalAllocator().allocate(); }
    void deallocate(T *p) noexcept { getLocalAllocator().deallocate(p); }
    void prefetch() noexcept { getLocalAllocator().prefetchSlab(); }
    allocator_type &getLocalAllocator() { return allocator_; }
    const allocator_type &getLocalAllocator() const { return allocator_; }

private:
    ThreadLocalPool() = default;
    allocator_type allocator_;
};

template <typename T>
class PooledPtr
{
public:
    using pool_type = ThreadLocalPool<T>;

    explicit PooledPtr(T *ptr = nullptr) noexcept
        : ptr_(ptr)
    {}
    ~PooledPtr()
    {
        if (ptr_)
        {
            ptr_->~T();
            pool_type::instance().deallocate(ptr_);
        }
    }

    PooledPtr(const PooledPtr &) = delete;
    PooledPtr &operator=(const PooledPtr &) = delete;

    PooledPtr(PooledPtr &&other) noexcept
        : ptr_(other.ptr_)
    {
        other.ptr_ = nullptr;
    }
    PooledPtr &operator=(PooledPtr &&other) noexcept
    {
        if (this != &other)
        {
            if (ptr_)
            {
                ptr_->~T();
                pool_type::instance().deallocate(ptr_);
            }
            ptr_ = other.ptr_;
            other.ptr_ = nullptr;
        }
        return *this;
    }

    T &operator*() const noexcept { return *ptr_; }
    T *operator->() const noexcept { return ptr_; }
    T *get() const noexcept { return ptr_; }
    explicit operator bool() const noexcept { return ptr_ != nullptr; }
    T *release() noexcept
    {
        T *tmp = ptr_;
        ptr_ = nullptr;
        return tmp;
    }

private:
    T *ptr_;
};

template <typename T, typename... Args>
PooledPtr<T> makePooled(Args &&...args)
{
    auto *ptr = ThreadLocalPool<T>::instance().allocate();
    if (!ptr)
        throw std::bad_alloc();
    try
    {
        new (ptr) T(std::forward<Args>(args)...);
        return PooledPtr<T>(ptr);
    }
    catch (...)
    {
        ThreadLocalPool<T>::instance().deallocate(ptr);
        throw;
    }
}

class RawArrayPool
{
public:
    explicit RawArrayPool(Size initial_bytes = 64 * 1024 * 1024, bool use_hugepages = false);
    ~RawArrayPool();

    void *allocate(Size bytes);
    void deallocate(void *ptr, Size bytes);
    Size totalAllocated() const noexcept
    {
        return total_allocated_.load(std::memory_order_relaxed);
    }
    Size peakUtilization() const noexcept { return peak_used_.load(std::memory_order_relaxed); }
    void reset() noexcept;

private:
    struct alignas(kCacheLineSize) Block
    {
        void *base = nullptr;
        Size total = 0;
        std::atomic<Size> used{0};
        Block() = default;
        Block(Block &&other) noexcept
            : base(other.base)
            , total(other.total)
            , used(other.used.load(std::memory_order_relaxed))
        {
            other.base = nullptr;
            other.total = 0;
        }
        Block &operator=(Block &&other) noexcept
        {
            if (this != &other)
            {
                base = other.base;
                total = other.total;
                used.store(other.used.load(std::memory_order_relaxed), std::memory_order_relaxed);
                other.base = nullptr;
                other.total = 0;
            }
            return *this;
        }
        Block(const Block &) = delete;
        Block &operator=(const Block &) = delete;
    };
    struct alignas(kCacheLineSize) PoolState
    {
        std::deque<Block> blocks;
        std::atomic<Size> current_block_idx{0};
    };
    std::unique_ptr<PoolState> state_;
    Size initial_byte_size_;
    bool use_hugepages_;
    std::atomic<Size> total_allocated_{0};
    std::atomic<Size> peak_used_{0};

    [[nodiscard]] Size aligned(Size bytes) const noexcept
    {
        return (bytes + kCacheLineSize - 1) & ~(kCacheLineSize - 1);
    }
};

class NumaAwareAllocator
{
public:
    explicit NumaAwareAllocator(int preferred_node = -1);
    ~NumaAwareAllocator();

    void *allocate(Size bytes, Size alignment = kCacheLineSize);
    void deallocate(void *ptr, Size bytes);
    int getPreferredNode() const noexcept;
    void setPreferredNode(int node);

private:
    int preferred_node_;
    struct NumaPolicy;
    std::unique_ptr<NumaPolicy> policy_;
};

template <typename T>
class PoolBackedVector
{
public:
    using value_type = T;
    using size_type = Size;
    using pointer = T *;
    using const_pointer = const T *;
    using reference = T &;
    using const_reference = const T &;

    PoolBackedVector(RawArrayPool &pool)
        : pool_(&pool)
        , data_(nullptr)
        , size_(0)
        , capacity_(0)
    {}
    ~PoolBackedVector()
    {
        if (data_)
            pool_->deallocate(data_, capacity_ * sizeof(T));
    }

    PoolBackedVector(const PoolBackedVector &) = delete;
    PoolBackedVector &operator=(const PoolBackedVector &) = delete;
    PoolBackedVector(PoolBackedVector &&) = delete;
    PoolBackedVector &operator=(PoolBackedVector &&) = delete;

    void reserve(Size n)
    {
        if (n <= capacity_)
            return;
        Size new_cap = std::max(n, capacity_ * 2);
        T *new_data = static_cast<T *>(pool_->allocate(new_cap * sizeof(T)));
        if (data_)
        {
            for (Size i = 0; i < size_; ++i)
                new (new_data + i) T(std::move(data_[i]));
            for (Size i = 0; i < size_; ++i)
                data_[i].~T();
            pool_->deallocate(data_, capacity_ * sizeof(T));
        }
        data_ = new_data;
        capacity_ = new_cap;
    }

    void push_back(const T &value)
    {
        if (size_ >= capacity_)
            reserve(capacity_ ? capacity_ * 2 : 8);
        new (data_ + size_) T(value);
        ++size_;
    }

    void push_back(T &&value)
    {
        if (size_ >= capacity_)
            reserve(capacity_ ? capacity_ * 2 : 8);
        new (data_ + size_) T(std::move(value));
        ++size_;
    }

    template <typename... Args>
    T &emplace_back(Args &&...args)
    {
        if (size_ >= capacity_)
            reserve(capacity_ ? capacity_ * 2 : 8);
        new (data_ + size_) T(std::forward<Args>(args)...);
        return data_[size_++];
    }

    void clear()
    {
        for (Size i = 0; i < size_; ++i)
            data_[i].~T();
        size_ = 0;
    }

    T &operator[](Size i) { return data_[i]; }
    const T &operator[](Size i) const { return data_[i]; }
    T *data() noexcept { return data_; }
    const T *data() const noexcept { return data_; }
    Size size() const noexcept { return size_; }
    Size capacity() const noexcept { return capacity_; }
    bool empty() const noexcept { return size_ == 0; }
    T *begin() noexcept { return data_; }
    T *end() noexcept { return data_ + size_; }
    const T *begin() const noexcept { return data_; }
    const T *end() const noexcept { return data_ + size_; }

private:
    RawArrayPool *pool_;
    T *data_;
    Size size_;
    Size capacity_;
};

enum class SizeClass : Size
{
    Tiny16 = 16,
    Tiny32 = 32,
    Tiny64 = 64,
    Tiny128 = 128,
    Tiny256 = 256,
    Small512 = 512,
    Small1024 = 1024,
    Small2048 = 2048,
    Small4096 = 4096,
    Medium8192 = 8192,
    Medium16384 = 16384,
    Medium32768 = 32768,
    Large65536 = 65536,
    Large131072 = 131072,
    Huge262144 = 262144
};

class SizeClassAllocator
{
public:
    SizeClassAllocator();

    void *allocate(Size bytes);
    void deallocate(void *ptr, Size bytes);
    Size totalAllocated() const noexcept;
    Size getSizeClass(Size bytes) const noexcept;

private:
    static constexpr Size kNumSizeClasses = 16;
    std::unique_ptr<RawArrayPool> pools_[kNumSizeClasses];
    static constexpr Size size_classes_[kNumSizeClasses] = {
        16,   32,   64,    128,   256,   512,    1024,   2048,
        4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288};
    Size findPoolIndex(Size bytes) const noexcept;
};

Size getGlobalAllocationCount() noexcept;
Size getGlobalDeallocationCount() noexcept;
Size getGlobalCurrentBytes() noexcept;
Size getGlobalPeakBytes() noexcept;
void trackAllocEvent(Size bytes) noexcept;
void trackFreeEvent(Size bytes) noexcept;
void resetGlobalMemoryStats() noexcept;
Size getSlabAllocatorDiagnosticCount() noexcept;
Size estimateMemoryOverhead(Size object_count, Size object_size, Size slab_capacity = 256) noexcept;

} // namespace nerve::memory
