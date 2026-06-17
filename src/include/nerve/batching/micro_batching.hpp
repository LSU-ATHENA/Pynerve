
#pragma once
#include "nerve/core_types.hpp"

#include <atomic>
#include <chrono>
#include <concepts>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace nerve::batching
{

class MicroBatchProcessor
{
public:
    MicroBatchProcessor(Size batch_size, Size max_pending);
    ~MicroBatchProcessor();
    void submit(const std::vector<double> &input);
    void flushBatch();
    Size processed() const;
    Size batchesCompleted() const;
    Size pending() const;
    void setBatchSize(Size batch_size);

private:
    std::vector<double> processSingle(const std::vector<double> &input) const;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

template <typename T>
concept Batchable = requires(T t) {
    { t.empty() } -> std::convertible_to<bool>;
};

struct BatchConfig
{
    size_t max_batch_size = 32;
    size_t max_wait_time_ms = 10;
    size_t min_batch_size = 1;
    size_t num_batch_threads = 4;
    size_t max_queue_size = 1000;
    bool enable_zero_copy = true;
};
template <Batchable T>
struct BatchItem
{
    T data;
    int64_t timestamp_ns;
    int64_t symbol_id;
    uint32_t priority;
    uint64_t sequence_id;
    std::function<void(const std::vector<T> &)> callback;
    std::chrono::steady_clock::time_point enqueue_time;
    bool isValid() const { return !data.empty() && timestamp_ns > 0 && symbol_id >= 0; }
};
template <typename T>
class MicroBatchingEngine
{
public:
    using BatchItemPtr = std::shared_ptr<BatchItem<T>>;
    using Batch = std::vector<BatchItemPtr>;
    using BatchProcessor = std::function<void(const Batch &)>;

    struct BatchStats
    {
        uint64_t total_items_processed = 0;
        uint64_t total_batches_processed = 0;
        double average_batch_size = 0.0;
        double average_wait_time_ms = 0.0;
        double average_processing_time_ms = 0.0;
        uint64_t queue_overflows = 0;
        uint64_t timeouts = 0;
    };

    explicit MicroBatchingEngine(const BatchConfig &config)
        : config_(config)
    {}

    ~MicroBatchingEngine() { stop(); }

    uint64_t enqueueItem(T data, int64_t timestamp_ns, int64_t symbol_id, uint32_t priority = 0,
                         std::function<void(const std::vector<T> &)> callback = nullptr)
    {
        auto item = std::make_shared<BatchItem<T>>();
        item->data = std::move(data);
        item->timestamp_ns = timestamp_ns;
        item->symbol_id = symbol_id;
        item->priority = priority;
        item->callback = std::move(callback);
        item->enqueue_time = std::chrono::steady_clock::now();
        item->sequence_id = stats_.total_items_processed;

        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            if (item_queue_.size() >= config_.max_queue_size)
            {
                stats_.queue_overflows++;
                return 0;
            }
            item_queue_.push(std::move(item));
        }
        queue_cv_.notify_one();
        stats_.total_items_processed++;
        return stats_.total_items_processed;
    }

    void setBatchProcessor(BatchProcessor processor) { batch_processor_ = std::move(processor); }

    void start()
    {
        running_ = true;
        batch_threads_.reserve(config_.num_batch_threads);
        for (size_t i = 0; i < config_.num_batch_threads; ++i)
        {
            batch_threads_.emplace_back(&MicroBatchingEngine::batchWorkerThread, this);
        }
    }

    void stop()
    {
        running_ = false;
        queue_cv_.notify_all();
        for (auto &t : batch_threads_)
        {
            if (t.joinable())
            {
                t.join();
            }
        }
        batch_threads_.clear();
    }

    void flush()
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        if (item_queue_.empty())
            return;
        auto batch = collectBatchLocked();
        lock.unlock();
        processBatch(batch);
        lock.lock();
    }

    BatchStats getStats() const
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        BatchStats copy = stats_;
        if (stats_.total_batches_processed > 0)
        {
            copy.average_batch_size = static_cast<double>(stats_.total_items_processed) /
                                      static_cast<double>(stats_.total_batches_processed);
        }
        return copy;
    }

private:
    BatchConfig config_;
    BatchProcessor batch_processor_;
    std::vector<std::thread> batch_threads_;
    std::atomic<bool> running_{false};
    std::queue<BatchItemPtr> item_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    mutable std::mutex stats_mutex_;
    BatchStats stats_;

    Batch collectBatchLocked()
    {
        Batch batch;
        batch.reserve(config_.max_batch_size);
        while (!item_queue_.empty() && batch.size() < config_.max_batch_size)
        {
            batch.push_back(std::move(item_queue_.front()));
            item_queue_.pop();
        }
        return batch;
    }

    Batch collectBatch()
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        if (config_.max_wait_time_ms > 0)
        {
            queue_cv_.wait_for(lock, std::chrono::milliseconds(config_.max_wait_time_ms), [this] {
                return !running_ || item_queue_.size() >= config_.min_batch_size;
            });
        }
        if (item_queue_.empty())
            return {};
        return collectBatchLocked();
    }

    void batchWorkerThread()
    {
        while (running_)
        {
            auto batch = collectBatch();
            if (batch.empty())
            {
                if (!running_)
                    break;
                stats_.timeouts++;
                continue;
            }
            processBatch(batch);
        }
    }

    void processBatch(const Batch &batch)
    {
        auto start = std::chrono::steady_clock::now();
        if (batch_processor_)
        {
            batch_processor_(batch);
        }
        auto end = std::chrono::steady_clock::now();

        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.total_batches_processed++;
        stats_.average_processing_time_ms =
            (stats_.average_processing_time_ms * (stats_.total_batches_processed - 1) +
             std::chrono::duration<double, std::milli>(end - start).count()) /
            static_cast<double>(stats_.total_batches_processed);
    }
};
template <typename T>
class SymbolBatchingEngine
{
public:
    struct SymbolBatchConfig
    {
        std::vector<int64_t> symbol_ids;
        size_t max_batch_size_per_symbol = 8;
        size_t max_total_batch_size = 32;
        int64_t max_time_window_ns = 100'000'000;
    };

    explicit SymbolBatchingEngine(const SymbolBatchConfig &config)
        : config_(config)
    {}

    uint64_t enqueueSymbolItem(T data, int64_t symbol_id, int64_t timestamp_ns)
    {
        std::lock_guard<std::mutex> lock(queues_mutex_);
        auto &q = symbol_queues_[symbol_id];
        if (q.size() >= config_.max_batch_size_per_symbol)
        {
            return 0;
        }
        q.push(std::move(data));
        return q.size();
    }

    std::vector<T> getBatchForSymbols(const std::vector<int64_t> &symbol_ids)
    {
        std::lock_guard<std::mutex> lock(queues_mutex_);
        std::vector<T> result;
        size_t total_collected = 0;
        for (auto sid : symbol_ids)
        {
            auto it = symbol_queues_.find(sid);
            if (it == symbol_queues_.end())
                continue;
            auto &q = it->second;
            while (!q.empty() && total_collected < config_.max_total_batch_size)
            {
                result.push_back(std::move(q.front()));
                q.pop();
                total_collected++;
            }
            if (q.empty())
            {
                symbol_queues_.erase(it);
            }
        }
        return result;
    }

private:
    SymbolBatchConfig config_;
    std::unordered_map<int64_t, std::queue<T>> symbol_queues_;
    std::mutex queues_mutex_;
};
} // namespace nerve::batching
