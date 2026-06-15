
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
    explicit MicroBatchingEngine(const BatchConfig &config);
    ~MicroBatchingEngine();
    uint64_t enqueueItem(T data, int64_t timestamp_ns, int64_t symbol_id, uint32_t priority = 0,
                         std::function<void(const std::vector<T> &)> callback = nullptr);
    void setBatchProcessor(BatchProcessor processor);
    void start();
    void stop();
    void flush();
    struct BatchStats
    {
        uint64_t total_items_processed;
        uint64_t total_batches_processed;
        double average_batch_size;
        double average_wait_time_ms;
        double average_processing_time_ms;
        uint64_t queue_overflows;
        uint64_t timeouts;
    };
    BatchStats getStats() const;

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
    void batchWorkerThread();
    Batch collectBatch();
    void processBatch(const Batch &batch);
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
    explicit SymbolBatchingEngine(const SymbolBatchConfig &config);
    uint64_t enqueueSymbolItem(T data, int64_t symbol_id, int64_t timestamp_ns);
    std::vector<T> getBatchForSymbols(const std::vector<int64_t> &symbol_ids);

private:
    SymbolBatchConfig config_;
    std::unordered_map<int64_t, std::queue<T>> symbol_queues_;
    std::mutex queues_mutex_;
};
} // namespace nerve::batching
