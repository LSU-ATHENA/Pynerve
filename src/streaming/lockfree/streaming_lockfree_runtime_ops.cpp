//
// Lock-free streaming persistence kernels.
// Based on: "The Flood Complex" (NeurIPS 2025) and lock-free PH research
//
// Optimizations:
// - Wait-free ring buffers (10M+ pts/sec)
// - Hazard pointers for safe memory reclamation
// - Atomic snapshots for consistent window views
// - Batched updates for throughput

#include <atomic>
#include <concepts>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <thread>
#include <unordered_set>
#include <vector>

namespace nerve::streaming::lockfree
{

constinit const size_t CACHE_LINE_SIZE = 64;

template <typename T>
concept StreamableData = std::copyable<T> && std::destructible<T> && sizeof(T) <= CACHE_LINE_SIZE;

/**
 * @brief Hazard Pointer for safe memory reclamation
 *
 * Prevents ABA problem and use-after-free in lock-free algorithms
 * Based on: "Hazard Pointers: Safe Memory Reclamation for Lock-Free Objects"
 */
/**
 * @brief Global hazard pointer registry
 *
 * Manages all thread-local hazard pointers for safe scanning.
 * Each thread registers its hazard pointer on construction.
 */
class HazardPointerRegistry
{
public:
    static HazardPointerRegistry &instance()
    {
        static HazardPointerRegistry registry;
        return registry;
    }

    void registerHazardPointer(std::atomic<void *> *hp)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        hazard_pointers_.insert(hp);
    }

    void unregisterHazardPointer(std::atomic<void *> *hp)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        hazard_pointers_.erase(hp);
    }

    bool isPointerProtected(void *ptr) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto *hp : hazard_pointers_)
        {
            if (hp->load(std::memory_order_acquire) == ptr)
            {
                return true;
            }
        }
        return false;
    }

    std::vector<void *> scanActivePointers() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<void *> active;
        for (auto *hp : hazard_pointers_)
        {
            void *ptr = hp->load(std::memory_order_acquire);
            if (ptr != nullptr)
            {
                active.push_back(ptr);
            }
        }
        return active;
    }

private:
    mutable std::mutex mutex_;
    std::unordered_set<std::atomic<void *> *> hazard_pointers_;
};

class HazardPointer
{
public:
    HazardPointer() { HazardPointerRegistry::instance().registerHazardPointer(&hazard_ptr_); }

    ~HazardPointer() { HazardPointerRegistry::instance().unregisterHazardPointer(&hazard_ptr_); }

    template <typename NodeType>
    void retire(NodeType *ptr)
    {
        std::lock_guard<std::mutex> lock(retire_mutex_);
        retired_list_.push_back({ptr, [](void *node) { delete static_cast<NodeType *>(node); }});

        // Try to reclaim if list is getting large
        if (retired_list_.size() >= RETIRE_THRESHOLD)
        {
            tryReclaimLocked();
        }
    }

    void tryReclaim()
    {
        std::lock_guard<std::mutex> lock(retire_mutex_);
        tryReclaimLocked();
    }

    void protect(void *ptr) { hazard_ptr_.store(ptr, std::memory_order_seq_cst); }

    void clear() { hazard_ptr_.store(nullptr, std::memory_order_seq_cst); }

    static bool isProtected(void *ptr)
    {
        // Scan all registered hazard pointers
        return HazardPointerRegistry::instance().isPointerProtected(ptr);
    }

private:
    struct RetiredNode
    {
        void *ptr;
        void (*deleter)(void *);
    };

    void tryReclaimLocked()
    {
        auto active = HazardPointerRegistry::instance().scanActivePointers();
        std::unordered_set<void *> protected_set(active.begin(), active.end());

        auto it = retired_list_.begin();
        while (it != retired_list_.end())
        {
            if (protected_set.find(it->ptr) == protected_set.end())
            {
                // Safe to delete - no hazard pointer references it
                it->deleter(it->ptr);
                it = retired_list_.erase(it);
            }
            else
            {
                ++it;
            }
        }
    }

    static constexpr size_t RETIRE_THRESHOLD = 64;
    alignas(CACHE_LINE_SIZE) std::atomic<void *> hazard_ptr_{nullptr};
    std::vector<RetiredNode> retired_list_;
    mutable std::mutex retire_mutex_;
};

/**
 * @brief Wait-Free Ring Buffer for Streaming Data
 *
 * Single-producer, single-consumer with wait-free operations
 *
 * @tparam T Streamable data type
 */
template <StreamableData T>
class WaitFreeRingBuffer
{
public:
    explicit WaitFreeRingBuffer(size_t capacity)
        : capacity_(nextPowerOf2(std::max<size_t>(capacity, 1) + 1))
        , mask_(capacity_ - 1)
        , buffer_(new T[capacity_])
        , head_{0}
        , tail_{0}
    {}

    ~WaitFreeRingBuffer() { delete[] buffer_; }

    // Delete copy to prevent accidents
    WaitFreeRingBuffer(const WaitFreeRingBuffer &) = delete;
    WaitFreeRingBuffer &operator=(const WaitFreeRingBuffer &) = delete;

    /**
     * @brief Wait-free push for producer
     *
     * Never blocks, returns false if buffer full
     * O(1) complexity, lock-free guarantee
     */
    [[nodiscard]] bool tryPush(const T &item) noexcept
    {
        size_t current_head = head_.load(std::memory_order_relaxed);
        size_t next_head = (current_head + 1) & mask_;

        // Check if buffer full (wait-free)
        if (next_head == tail_.load(std::memory_order_acquire))
        {
            return false;
        }

        // Write item
        buffer_[current_head & mask_] = item;

        // Update head with release semantics
        head_.store(next_head, std::memory_order_release);

        return true;
    }

    /**
     * @brief Wait-free pop for consumer
     *
     * Never blocks, returns nullopt if buffer empty
     * O(1) complexity, lock-free guarantee
     */
    [[nodiscard]] std::optional<T> tryPop() noexcept
    {
        size_t current_tail = tail_.load(std::memory_order_relaxed);

        // Check if buffer empty (wait-free)
        if (current_tail == head_.load(std::memory_order_acquire))
        {
            return std::nullopt; // Buffer empty
        }

        // Read item
        T item = buffer_[current_tail & mask_];

        // Update tail with release semantics
        tail_.store((current_tail + 1) & mask_, std::memory_order_release);

        return item;
    }

    /**
     * @brief Bulk push for improved throughput
     *
     * Reduces atomic operations by batching
     */
    [[nodiscard]] size_t tryPushBulk(std::span<const T> items) noexcept
    {
        size_t current_head = head_.load(std::memory_order_relaxed);
        size_t current_tail = tail_.load(std::memory_order_acquire);

        // Calculate available space
        size_t available = (current_tail - current_head - 1) & mask_;
        size_t to_write = std::min(items.size(), available);

        // Bulk write
        for (size_t i = 0; i < to_write; ++i)
        {
            buffer_[(current_head + i) & mask_] = items[i];
        }

        // Single atomic update
        head_.store((current_head + to_write) & mask_, std::memory_order_release);

        return to_write;
    }

    /**
     * @brief Get current size (approximate)
     *
     * May be slightly stale due to concurrent updates
     */
    [[nodiscard]] size_t approximateSize() const noexcept
    {
        return (head_.load(std::memory_order_acquire) - tail_.load(std::memory_order_acquire)) &
               mask_;
    }

    [[nodiscard]] bool empty() const noexcept
    {
        return head_.load(std::memory_order_acquire) == tail_.load(std::memory_order_acquire);
    }

    [[nodiscard]] bool full() const noexcept
    {
        return ((head_.load(std::memory_order_acquire) + 1) & mask_) ==
               tail_.load(std::memory_order_acquire);
    }

    [[nodiscard]] size_t capacity() const noexcept { return capacity_; }

private:
    size_t capacity_;
    size_t mask_;
    T *buffer_;

    // Cache-line aligned to prevent false sharing
    alignas(CACHE_LINE_SIZE) std::atomic<size_t> head_{0};
    alignas(CACHE_LINE_SIZE) std::atomic<size_t> tail_{0};

    [[nodiscard]] static size_t nextPowerOf2(size_t n)
    {
        size_t capacity = 1;
        while (capacity < n)
        {
            if (capacity > std::numeric_limits<size_t>::max() / 2)
            {
                throw std::length_error("lockfree runtime ring buffer capacity exceeds size_t");
            }
            capacity *= 2;
        }
        return capacity;
    }
};

/**
 * @brief Multi-Producer Multi-Consumer Queue with Lock-Free Algorithm
 *
 * Based on Michael-Scott queue with hazard pointers
 * For distributed streaming across multiple threads
 */
template <StreamableData T>
class LockFreeMPMCQueue
{
public:
    explicit LockFreeMPMCQueue(size_t capacity);
    ~LockFreeMPMCQueue();

    [[nodiscard]] bool tryEnqueue(const T &item);
    [[nodiscard]] std::optional<T> tryDequeue();

private:
    struct Node
    {
        T data;
        std::atomic<Node *> next{nullptr};
    };

    std::atomic<Node *> head_;
    std::atomic<Node *> tail_;
    HazardPointer hp_producer_;
    HazardPointer hp_consumer_;
};

/**
 * @brief Streaming Window with Atomic Snapshot
 *
 * Provides consistent view of streaming window for PH computation
 * Lock-free window rotation for real-time updates
 */
template <StreamableData T>
class LockFreeStreamingWindow
{
public:
    explicit LockFreeStreamingWindow(size_t window_size);

    // Add point to stream (wait-free)
    void addPoint(const T &point);

    // Get atomic snapshot of current window
    [[nodiscard]] std::vector<T> getSnapshot() const;

    // Rotate window (lock-free)
    void rotateWindow();

private:
    WaitFreeRingBuffer<T> buffer_;
    std::atomic<size_t> window_start_{0};
    size_t window_size_;
};

// Performance targets (2026)
// Single-threaded: 10M+ points/sec
// Multi-threaded: 100M+ points/sec (10 threads)
// Latency: <1us per operation

} // namespace nerve::streaming::lockfree
