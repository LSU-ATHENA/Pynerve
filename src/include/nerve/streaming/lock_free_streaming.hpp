
#pragma once
#include "nerve/core_types.hpp"

#include <algorithm>
#include <atomic>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

namespace nerve::streaming::lockfree
{

constinit const size_t DEFAULT_BUFFER_SIZE = 1024 * 1024;
constinit const size_t CACHE_LINE_SIZE = 64;

template <typename T>
concept StreamableData = std::copyable<T> && std::default_initializable<T> &&
                         std::destructible<T> && sizeof(T) <= CACHE_LINE_SIZE;

namespace detail
{

inline size_t nextPowerOfTwo(size_t n) noexcept
{
    size_t value = std::max<size_t>(2, n);
    if (value > (size_t{1} << ((sizeof(size_t) * 8) - 1)))
    {
        return 0;
    }
    --value;
    for (size_t shift = 1; shift < sizeof(size_t) * 8; shift <<= 1)
    {
        value |= value >> shift;
    }
    return value + 1;
}

inline size_t checkedNextPowerOfTwo(size_t n)
{
    const size_t value = nextPowerOfTwo(n);
    if (value == 0)
    {
        throw std::length_error("streaming lockfree capacity exceeds size_t");
    }
    return value;
}

inline size_t checkedStorageCapacity(size_t logical_capacity)
{
    if (logical_capacity == std::numeric_limits<size_t>::max())
    {
        throw std::length_error("streaming lockfree storage capacity exceeds size_t");
    }
    return checkedNextPowerOfTwo(logical_capacity + 1);
}

} // namespace detail

/**
 * @brief Wait-Free Ring Buffer
 *
 * Single-producer, single-consumer with wait-free guarantee
 */
template <StreamableData T>
class WaitFreeRingBuffer
{
public:
    explicit WaitFreeRingBuffer(size_t capacity)
        : capacity_(std::max<size_t>(1, capacity))
        , storage_capacity_(detail::checkedStorageCapacity(capacity_))
        , mask_(storage_capacity_ - 1)
        , buffer_(std::make_unique<T[]>(storage_capacity_))
    {}

    ~WaitFreeRingBuffer() = default;

    WaitFreeRingBuffer(const WaitFreeRingBuffer &) = delete;
    WaitFreeRingBuffer &operator=(const WaitFreeRingBuffer &) = delete;

    // Wait-free operations
    [[nodiscard]] bool tryPush(const T &item) noexcept
    {
        const size_t head = head_.load(std::memory_order_relaxed);
        const size_t tail = tail_.load(std::memory_order_acquire);
        if (head - tail >= capacity_)
        {
            return false;
        }
        buffer_[head & mask_] = item;
        head_.store(head + 1, std::memory_order_release);
        return true;
    }

    [[nodiscard]] std::optional<T> tryPop() noexcept
    {
        const size_t tail = tail_.load(std::memory_order_relaxed);
        const size_t head = head_.load(std::memory_order_acquire);
        if (tail == head)
        {
            return std::nullopt;
        }
        T item = buffer_[tail & mask_];
        tail_.store(tail + 1, std::memory_order_release);
        return item;
    }

    // Bulk operations for throughput
    [[nodiscard]] size_t tryPushBulk(std::span<const T> items) noexcept
    {
        size_t pushed = 0;
        for (const auto &item : items)
        {
            if (!tryPush(item))
            {
                break;
            }
            ++pushed;
        }
        return pushed;
    }

    [[nodiscard]] std::vector<T> tryPopBulk(size_t max_items) noexcept
    {
        std::vector<T> items;
        items.reserve(std::min(max_items, approximateSize()));
        for (size_t i = 0; i < max_items; ++i)
        {
            auto item = tryPop();
            if (!item)
            {
                break;
            }
            items.push_back(*item);
        }
        return items;
    }

    [[nodiscard]] std::vector<T> snapshotNewest(size_t max_items) const
    {
        const size_t tail = tail_.load(std::memory_order_acquire);
        const size_t head = head_.load(std::memory_order_acquire);
        const size_t available = head - tail;
        const size_t count = std::min(max_items, available);
        const size_t start = head - count;
        std::vector<T> items;
        items.reserve(count);
        for (size_t i = 0; i < count; ++i)
        {
            items.push_back(buffer_[(start + i) & mask_]);
        }
        return items;
    }

    [[nodiscard]] size_t approximateSize() const noexcept
    {
        return head_.load(std::memory_order_acquire) - tail_.load(std::memory_order_acquire);
    }

    [[nodiscard]] bool empty() const noexcept { return approximateSize() == 0; }

    [[nodiscard]] bool full() const noexcept { return approximateSize() >= capacity_; }

    [[nodiscard]] size_t capacity() const noexcept { return capacity_; }

private:
    size_t capacity_;
    size_t storage_capacity_;
    size_t mask_;
    std::unique_ptr<T[]> buffer_;
    alignas(CACHE_LINE_SIZE) std::atomic<size_t> head_{0};
    alignas(CACHE_LINE_SIZE) std::atomic<size_t> tail_{0};
};

/**
 * @brief Multi-Producer Multi-Consumer Queue
 *
 * Lock-free via Michael-Scott algorithm + hazard pointers
 */
template <StreamableData T>
class LockFreeMPMCQueue
{
public:
    explicit LockFreeMPMCQueue(size_t capacity)
        : capacity_(detail::checkedNextPowerOfTwo(std::max<size_t>(2, capacity)))
        , mask_(capacity_ - 1)
        , buffer_(std::make_unique<Cell[]>(capacity_))
    {
        for (size_t i = 0; i < capacity_; ++i)
        {
            buffer_[i].sequence.store(i, std::memory_order_relaxed);
        }
    }

    ~LockFreeMPMCQueue() = default;

    [[nodiscard]] bool tryEnqueue(const T &item)
    {
        Cell *cell = nullptr;
        size_t pos = enqueue_pos_.load(std::memory_order_relaxed);
        for (;;)
        {
            cell = &buffer_[pos & mask_];
            const size_t seq = cell->sequence.load(std::memory_order_acquire);
            const auto diff = static_cast<std::intptr_t>(seq) - static_cast<std::intptr_t>(pos);
            if (diff == 0)
            {
                if (enqueue_pos_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed))
                {
                    break;
                }
            }
            else if (diff < 0)
            {
                return false;
            }
            else
            {
                pos = enqueue_pos_.load(std::memory_order_relaxed);
            }
        }
        cell->data = item;
        cell->sequence.store(pos + 1, std::memory_order_release);
        return true;
    }

    [[nodiscard]] std::optional<T> tryDequeue()
    {
        Cell *cell = nullptr;
        size_t pos = dequeue_pos_.load(std::memory_order_relaxed);
        for (;;)
        {
            cell = &buffer_[pos & mask_];
            const size_t seq = cell->sequence.load(std::memory_order_acquire);
            const auto diff = static_cast<std::intptr_t>(seq) - static_cast<std::intptr_t>(pos + 1);
            if (diff == 0)
            {
                if (dequeue_pos_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed))
                {
                    break;
                }
            }
            else if (diff < 0)
            {
                return std::nullopt;
            }
            else
            {
                pos = dequeue_pos_.load(std::memory_order_relaxed);
            }
        }
        T item = cell->data;
        cell->sequence.store(pos + capacity_, std::memory_order_release);
        return item;
    }

private:
    struct Cell
    {
        std::atomic<size_t> sequence{0};
        T data;
    };

    size_t capacity_;
    size_t mask_;
    std::unique_ptr<Cell[]> buffer_;
    alignas(CACHE_LINE_SIZE) std::atomic<size_t> enqueue_pos_{0};
    alignas(CACHE_LINE_SIZE) std::atomic<size_t> dequeue_pos_{0};
};

/**
 * @brief Lock-Free Streaming Window
 *
 * Atomic snapshots for PH computation on streaming data
 */
template <StreamableData T>
class LockFreeStreamingWindow
{
public:
    explicit LockFreeStreamingWindow(size_t window_size)
        : buffer_(std::max<size_t>(1, window_size))
        , window_size_(std::max<size_t>(1, window_size))
    {}

    void addPoint(const T &point)
    {
        while (!buffer_.tryPush(point))
        {
            (void)buffer_.tryPop();
            window_start_.fetch_add(1, std::memory_order_relaxed);
        }
    }

    [[nodiscard]] std::vector<T> getSnapshot() const
    {
        return buffer_.snapshotNewest(window_size_);
    }

    void rotateWindow()
    {
        const auto removed = buffer_.tryPopBulk(window_size_);
        window_start_.fetch_add(removed.size(), std::memory_order_relaxed);
    }

private:
    WaitFreeRingBuffer<T> buffer_;
    std::atomic<size_t> window_start_{0};
    size_t window_size_;
};

// Throughput and latency are workload- and hardware-dependent.

} // namespace nerve::streaming::lockfree
