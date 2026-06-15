
// Lock-Free Streaming Data Structures
//
// Optimizations:
// - Lock-free single-producer single-consumer queue
// - Circular buffer for streaming windows
// - Memory-mapped file I/O
// - Batched updates for throughput

#include "nerve/core.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

namespace nerve::streaming::lockfree
{

/**
 * @brief Lock-free single-producer single-consumer queue
 *
 * Lock-free queue using atomic operations with proper memory ordering.
 * Implements the classic array-based SPSC queue with cache-line padding
 * for head/tail pointers to prevent false sharing.
 */
template <typename T>
class LockFreeSPSCQueue
{
public:
    explicit LockFreeSPSCQueue(size_t capacity)
        : capacity_(nextPowerOf2(capacity))
        , mask_(capacity_ - 1)
        , buffer_(new T[capacity_])
        , head_(0)
        , tail_(0)
    {}

    LockFreeSPSCQueue(const LockFreeSPSCQueue &) = delete;
    LockFreeSPSCQueue &operator=(const LockFreeSPSCQueue &) = delete;
    LockFreeSPSCQueue(LockFreeSPSCQueue &&) = delete;
    LockFreeSPSCQueue &operator=(LockFreeSPSCQueue &&) = delete;

    ~LockFreeSPSCQueue() { delete[] buffer_; }

    // Producer only
    bool push(const T &item)
    {
        size_t head = head_.load(std::memory_order_relaxed);
        size_t next_head = (head + 1) & mask_;

        if (next_head == tail_.load(std::memory_order_acquire))
        {
            return false;
        }

        buffer_[head] = item;
        head_.store(next_head, std::memory_order_release);
        return true;
    }

    // Consumer only
    std::optional<T> pop()
    {
        size_t tail = tail_.load(std::memory_order_relaxed);

        if (tail == head_.load(std::memory_order_acquire))
        {
            return std::nullopt; // Empty
        }

        T item = buffer_[tail];
        tail_.store((tail + 1) & mask_, std::memory_order_release);
        return item;
    }

    size_t size() const
    {
        return (head_.load(std::memory_order_acquire) - tail_.load(std::memory_order_acquire)) &
               mask_;
    }

    bool empty() const
    {
        return head_.load(std::memory_order_acquire) == tail_.load(std::memory_order_acquire);
    }

private:
    size_t capacity_;
    size_t mask_;
    T *buffer_;

    alignas(64) std::atomic<size_t> head_;
    alignas(64) std::atomic<size_t> tail_;

    static size_t nextPowerOf2(size_t n)
    {
        const size_t target = std::max<size_t>(n, 2);
        size_t capacity = 1;
        while (capacity < target)
        {
            if (capacity > std::numeric_limits<size_t>::max() / 2)
            {
                throw std::length_error("lockfree streaming queue capacity exceeds size_t");
            }
            capacity *= 2;
        }
        return capacity;
    }
};

/**
 * @brief Hazard pointer for safe memory reclamation
 *
 * Each thread registers nodes it's currently accessing.
 * Nodes can only be deleted when no hazard pointers point to them.
 */
class HazardPointer
{
public:
    template <typename NodeType>
    void retireNode(NodeType *node)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        retired_nodes_.push_back({node, [](void *ptr) { delete static_cast<NodeType *>(ptr); }});

        // Try to reclaim retired nodes periodically
        if (retired_nodes_.size() >= RETIRE_THRESHOLD)
        {
            reclaimNodesLocked();
        }
    }

    void setHazardPointer(void *ptr) { threadLocalHazardSlot().ptr = ptr; }

    void clearHazardPointer() { threadLocalHazardSlot().ptr = nullptr; }

    void reclaimNodes()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        reclaimNodesLocked();
    }

    static HazardPointer &instance()
    {
        static HazardPointer hp;
        return hp;
    }

private:
    struct RetiredNode
    {
        void *ptr;
        void (*deleter)(void *);
    };

    struct HazardSlot
    {
        void *ptr = nullptr;

        HazardSlot() { HazardPointer::instance().registerHazardPointer(&ptr); }

        ~HazardSlot() { HazardPointer::instance().unregisterHazardPointer(&ptr); }
    };

    static HazardSlot &threadLocalHazardSlot()
    {
        thread_local HazardSlot slot;
        return slot;
    }

    void registerHazardPointer(void **ptr)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        active_hps_.insert(ptr);
    }

    void unregisterHazardPointer(void **ptr)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        active_hps_.erase(ptr);
    }

    void reclaimNodesLocked()
    {
        // Collect all active hazard pointers
        std::unordered_set<void *> protected_ptrs;
        for (void **hp : active_hps_)
        {
            if (*hp != nullptr)
            {
                protected_ptrs.insert(*hp);
            }
        }

        // Delete nodes that are not protected
        auto it = retired_nodes_.begin();
        while (it != retired_nodes_.end())
        {
            if (protected_ptrs.find(it->ptr) == protected_ptrs.end())
            {
                // Safe to delete - no hazard pointers point to it
                it->deleter(it->ptr);
                it = retired_nodes_.erase(it);
            }
            else
            {
                ++it;
            }
        }
    }

    static constexpr size_t RETIRE_THRESHOLD = 100;
    std::mutex mutex_;
    std::vector<RetiredNode> retired_nodes_;
    std::unordered_set<void **> active_hps_;
};

/**
 * @brief Lock-free multi-producer multi-consumer queue with hazard pointer
 * protection
 *
 * Uses hazard pointers for safe memory reclamation in concurrent environments.
 * Prevents use-after-free when multiple threads access the queue
 * simultaneously.
 */
template <typename T>
class LockFreeMPMCQueue
{
public:
    LockFreeMPMCQueue()
        : head_(new Node())
        , tail_(head_.load())
    {}

    ~LockFreeMPMCQueue()
    {
        for (;;)
        {
            auto drained = pop();
            if (!drained.has_value())
            {
                break;
            }
        }

        // Reclaim any remaining retired nodes
        HazardPointer::instance().reclaimNodes();

        // Delete remaining sentinel node
        delete head_.load();
    }

    void push(const T &item)
    {
        Node *node = new Node(item);
        Node *prev_tail = tail_.exchange(node, std::memory_order_acq_rel);
        prev_tail->next_.store(node, std::memory_order_release);
    }

    std::optional<T> pop()
    {
        for (;;)
        {
            Node *head = head_.load(std::memory_order_acquire);

            // Set hazard pointer before accessing node
            HazardPointer::instance().setHazardPointer(head);

            // Verify head hasn't changed (hazard pointer check)
            Node *current_head = head_.load(std::memory_order_acquire);
            if (current_head != head)
            {
                HazardPointer::instance().clearHazardPointer();
                continue;
            }

            Node *next = head->next_.load(std::memory_order_acquire);

            if (next == nullptr)
            {
                HazardPointer::instance().clearHazardPointer();
                return std::nullopt;
            }

            if (head_.compare_exchange_strong(head, next, std::memory_order_acq_rel,
                                              std::memory_order_acquire))
            {
                T item = next->data_;

                // Clear hazard pointer before retiring
                HazardPointer::instance().clearHazardPointer();

                // Safely retire the old head node via hazard pointer mechanism
                HazardPointer::instance().retireNode(head);

                return item;
            }

            HazardPointer::instance().clearHazardPointer();
        }
    }

private:
    struct Node
    {
        T data_;
        std::atomic<Node *> next_;

        Node()
            : next_(nullptr)
        {}
        explicit Node(const T &data)
            : data_(data)
            , next_(nullptr)
        {}
    };

    std::atomic<Node *> head_;
    std::atomic<Node *> tail_;
};

/**
 * @brief Circular buffer for streaming windows
 *
 * Fixed-size circular buffer with overwrite capability
 */
template <typename T>
class CircularBuffer
{
public:
    explicit CircularBuffer(size_t capacity)
        : capacity_(std::max<size_t>(capacity, 1))
        , buffer_(capacity_)
        , read_idx_(0)
        , write_idx_(0)
        , size_(0)
    {}

    // Write to buffer (overwrites if full)
    void write(const T &item)
    {
        size_t idx = write_idx_.fetch_add(1, std::memory_order_relaxed) % capacity_;
        buffer_[idx] = item;

        size_t current_size = size_.fetch_add(1, std::memory_order_relaxed);
        if (current_size >= capacity_)
        {
            // Buffer full, advance read pointer
            read_idx_.fetch_add(1, std::memory_order_relaxed);
            size_.store(capacity_, std::memory_order_relaxed);
        }
    }

    // Read from buffer (non-blocking)
    std::optional<T> read()
    {
        size_t idx = read_idx_.load(std::memory_order_relaxed) % capacity_;

        if (size_.load(std::memory_order_acquire) == 0)
        {
            return std::nullopt;
        }

        T item = buffer_[idx];
        read_idx_.fetch_add(1, std::memory_order_relaxed);
        size_.fetch_sub(1, std::memory_order_relaxed);
        return item;
    }

    // Get current window (for persistence computation)
    std::vector<T> getWindow(size_t window_size) const
    {
        std::vector<T> window;
        window_size = std::min(window_size, size_.load(std::memory_order_acquire));
        window.reserve(window_size);

        size_t start = read_idx_.load(std::memory_order_relaxed);
        for (size_t i = 0; i < window_size; ++i)
        {
            window.push_back(buffer_[(start + i) % capacity_]);
        }

        return window;
    }

    size_t size() const { return size_.load(std::memory_order_acquire); }

private:
    size_t capacity_;
    std::vector<T> buffer_;

    alignas(64) std::atomic<size_t> read_idx_;
    alignas(64) std::atomic<size_t> write_idx_;
    alignas(64) std::atomic<size_t> size_;
};

#include "detail/streaming_lockfree_windows.inl"
} // namespace nerve::streaming::lockfree
