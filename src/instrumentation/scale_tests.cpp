
#include "nerve/instrumentation/error_events.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace nerve::instrumentation
{

// Memory Unit Constants
constexpr size_t BYTES_PER_KB = 1024ULL;
constexpr size_t BYTES_PER_MB = 1024ULL * 1024ULL;
constexpr size_t BYTES_PER_GB = 1024ULL * 1024ULL * 1024ULL;

class MemoryTracker
{
public:
    struct AllocationInfo
    {
        void *pointer = nullptr;
        std::size_t size = 0;
        std::chrono::high_resolution_clock::time_point timestamp{};
        std::string function_name;
        std::string file;
        std::uint32_t line_number = 0;
    };

    explicit MemoryTracker(bool enable_tracking = false)
        : tracking_enabled_(enable_tracking)
    {}

    void trackAllocation(void *ptr, std::size_t size, const std::string &function_name,
                         const std::string &file = "", std::uint32_t line_number = 0)
    {
        if (!tracking_enabled_.load() || ptr == nullptr || size == 0)
        {
            return;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        AllocationInfo info{ptr,           size, std::chrono::high_resolution_clock::now(),
                            function_name, file, line_number};
        active_allocations_[ptr] = info;
        allocation_history_.push_back(info);
        ++total_allocations_;
        const std::size_t current = getCurrentMemoryUsageLocked();
        peak_memory_usage_.store(std::max(peak_memory_usage_.load(), current));
    }

    void trackDeallocation(void *ptr, const std::string & /*function_name*/,
                           const std::string & /*file*/ = "", std::uint32_t /*line_number*/ = 0)
    {
        if (!tracking_enabled_.load() || ptr == nullptr)
        {
            return;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        active_allocations_.erase(ptr);
    }

    std::size_t getTotalAllocations() const { return total_allocations_.load(); }
    std::size_t getPeakMemoryUsage() const { return peak_memory_usage_.load(); }

    std::size_t getCurrentMemoryUsage() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return getCurrentMemoryUsageLocked();
    }

    std::vector<AllocationInfo> getAllocationHistory() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return allocation_history_;
    }

    void clearHistory()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        allocation_history_.clear();
        active_allocations_.clear();
        total_allocations_.store(0);
        peak_memory_usage_.store(0);
    }

private:
    std::size_t getCurrentMemoryUsageLocked() const
    {
        std::size_t total = 0;
        for (const auto &entry : active_allocations_)
        {
            total += entry.second.size;
        }
        return total;
    }

    mutable std::mutex mutex_;
    std::atomic<bool> tracking_enabled_{false};
    std::vector<AllocationInfo> allocation_history_;
    std::unordered_map<void *, AllocationInfo> active_allocations_;
    std::atomic<std::size_t> total_allocations_{0};
    std::atomic<std::size_t> peak_memory_usage_{0};
};

class NUMABufferManager
{
public:
    struct PinnedBuffer
    {
        void *data = nullptr;
        std::size_t size = 0;
        int numa_node = -1;
    };

    explicit NUMABufferManager(std::size_t initial_pool_size = BYTES_PER_MB)
        : total_allocated_(0)
        , peak_allocated_(initial_pool_size)
    {}

    PinnedBuffer allocatePinnedBuffer(std::size_t size, int numa_node = -1)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto raw = std::make_unique<std::uint8_t[]>(size);
        void *ptr = raw.get();
        allocations_[ptr] = std::move(raw);
        total_allocated_ += size;
        peak_allocated_ = std::max(peak_allocated_, total_allocated_);
        return PinnedBuffer{ptr, size, numa_node};
    }

    void free_pinned_buffer(PinnedBuffer &buffer)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        allocations_.erase(buffer.data);
        if (total_allocated_ >= buffer.size)
        {
            total_allocated_ -= buffer.size;
        }
        buffer = {};
    }

    std::size_t getTotalAllocated() const { return total_allocated_; }
    std::size_t getPeakAllocated() const { return peak_allocated_; }

private:
    mutable std::mutex mutex_;
    std::unordered_map<void *, std::unique_ptr<std::uint8_t[]>> allocations_;
    std::size_t total_allocated_;
    std::size_t peak_allocated_;
};

class ScaleTestSuite
{
public:
    struct TestConfig
    {
        std::size_t min_points = 1000;
        std::size_t max_points = 10000;
        std::size_t max_dimension = 8;
        std::size_t num_iterations = 5;
    };

    struct TestResults
    {
        bool passed = true;
        std::size_t peak_memory_mb = 0;
        std::size_t total_allocations = 0;
        std::vector<std::string> failures;
    };

    ScaleTestSuite()
        : memory_tracker_(true)
        , buffer_manager_()
    {}

    void runAllTests(const TestConfig &config)
    {
        std::mt19937_64 rng(1337);
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        for (std::size_t it = 0; it < config.num_iterations; ++it)
        {
            const std::size_t n = config.min_points + ((config.max_points - config.min_points) *
                                                       (it + 1) / config.num_iterations);
            const std::size_t d = std::max<std::size_t>(1, config.max_dimension);
            std::vector<double> points(n * d);
            for (double &v : points)
            {
                v = dist(rng);
            }

            auto buffer = buffer_manager_.allocatePinnedBuffer(points.size() * sizeof(double));
            memory_tracker_.trackAllocation(buffer.data, buffer.size,
                                            "ScaleTestSuite::runAllTests");
            memory_tracker_.trackDeallocation(buffer.data, "ScaleTestSuite::runAllTests");
            buffer_manager_.free_pinned_buffer(buffer);
        }

        results_.total_allocations = memory_tracker_.getTotalAllocations();
        results_.peak_memory_mb = memory_tracker_.getPeakMemoryUsage() / BYTES_PER_MB;
    }

    const TestResults &getResults() const { return results_; }

    void printResults() const
    {
        std::cout << "Scale tests passed=" << (results_.passed ? "true" : "false")
                  << ", total_allocations=" << results_.total_allocations
                  << ", peak_memory_mb=" << results_.peak_memory_mb << '\n';
    }

private:
    MemoryTracker memory_tracker_;
    NUMABufferManager buffer_manager_;
    TestResults results_;
};

} // namespace nerve::instrumentation
