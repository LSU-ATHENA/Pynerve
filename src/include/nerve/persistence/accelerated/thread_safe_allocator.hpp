
#pragma once

#include "nerve/core.hpp"
#include "nerve/errors/errors.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <exception>
#include <iterator>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace nerve::persistence::accelerated
{

constexpr size_t BYTES_PER_KB = 1024ULL;
constexpr size_t BYTES_PER_MB = 1024ULL * 1024ULL;
constexpr size_t BYTES_PER_GB = 1024ULL * 1024ULL * 1024ULL;

class CudaMemoryPool
{
public:
    struct Config
    {
        size_t initial_size = BYTES_PER_GB;
        size_t max_size = 8ULL * BYTES_PER_GB;
        size_t alignment = 256;
        bool auto_grow = true;
    };

    static errors::ErrorResult<std::unique_ptr<CudaMemoryPool>> create()
    {
        return create(Config{});
    }

    static errors::ErrorResult<std::unique_ptr<CudaMemoryPool>> create(const Config &config)
    {
        const auto validation = validateConfig(config);
        if (validation.isError())
        {
            return errors::ErrorResult<std::unique_ptr<CudaMemoryPool>>::error(
                validation.errorCode(), validation.error().message);
        }

        try
        {
            auto pool = std::unique_ptr<CudaMemoryPool>(new CudaMemoryPool(config));
            auto init_result = pool->initialize();
            if (init_result.isError())
            {
                return errors::ErrorResult<std::unique_ptr<CudaMemoryPool>>::error(
                    init_result.errorCode(), init_result.error().message);
            }
            return errors::ErrorResult<std::unique_ptr<CudaMemoryPool>>::success(std::move(pool));
        }
        catch (const std::exception &e)
        {
            return errors::ErrorResult<std::unique_ptr<CudaMemoryPool>>::error(
                errors::ErrorCode::E50_PH_ABORT, e.what());
        }
    }

    ~CudaMemoryPool() { cleanup(); }

    errors::ErrorResult<void *> allocate(size_t size, size_t alignment = 256)
    {
        if (size == 0)
        {
            return errors::ErrorResult<void *>::error(errors::ErrorCode::E51_PH_INPUT,
                                                      "allocation size must be positive");
        }
        if (!isPowerOfTwo(alignment))
        {
            return errors::ErrorResult<void *>::error(errors::ErrorCode::E52_PH_CONFIG,
                                                      "alignment must be a nonzero power of two");
        }

        std::lock_guard<std::mutex> lock(mutex_);

        const size_t remainder = size % alignment;
        if (remainder != 0)
        {
            const size_t padding = alignment - remainder;
            if (size > static_cast<size_t>(-1) - padding)
            {
                return errors::ErrorResult<void *>::error(errors::ErrorCode::E41_RESOURCE_LIMIT,
                                                          "aligned allocation size overflow");
            }
            size += padding;
        }

        auto allocation = allocate_from_free_blocks(size);
        if (allocation.isSuccess())
        {
            return allocation;
        }

        if (config_.auto_grow && total_allocated_ < config_.max_size &&
            size <= config_.max_size - total_allocated_)
        {
            auto grown = growPool(size);
            if (grown.isError())
            {
                return errors::ErrorResult<void *>::error(grown.errorCode(), grown.error().message);
            }
            return allocate_from_free_blocks(size);
        }

        return errors::ErrorResult<void *>::error(errors::ErrorCode::E41_RESOURCE_LIMIT,
                                                  "CUDA memory pool exhausted");
    }

    errors::ErrorResult<void> deallocate(void *ptr)
    {
        if (!ptr)
        {
            return errors::ErrorResult<void>::success();
        }

        std::lock_guard<std::mutex> lock(mutex_);

        auto it = used_blocks_.find(ptr);
        if (it == used_blocks_.end())
        {
            return errors::ErrorResult<void>::error(errors::ErrorCode::E50_PH_ABORT);
        }

        Block block = it->second;
        used_blocks_.erase(it);
        free_blocks_.push_back(block);

        mergeAdjacentBlocks();

        return errors::ErrorResult<void>::success();
    }

    size_t getAllocatedSize() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return total_allocated_;
    }

    size_t getUsedSize() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        size_t used = 0;
        for (const auto &block : used_blocks_)
        {
            used += block.second.size;
        }
        return used;
    }

private:
    struct Block
    {
        void *ptr = nullptr;
        size_t size = 0;
    };

    explicit CudaMemoryPool(const Config &config)
        : config_(config)
    {}

    static bool isPowerOfTwo(size_t value) { return value != 0 && (value & (value - 1)) == 0; }

    static errors::ErrorResult<void> validateConfig(const Config &config)
    {
        if (config.initial_size == 0)
        {
            return errors::ErrorResult<void>::error(errors::ErrorCode::E52_PH_CONFIG,
                                                    "initial pool size must be positive");
        }
        if (config.max_size == 0 || config.initial_size > config.max_size)
        {
            return errors::ErrorResult<void>::error(
                errors::ErrorCode::E52_PH_CONFIG,
                "maximum pool size must be positive and at least the initial size");
        }
        if (!isPowerOfTwo(config.alignment))
        {
            return errors::ErrorResult<void>::error(errors::ErrorCode::E52_PH_CONFIG,
                                                    "alignment must be a nonzero power of two");
        }
        return errors::ErrorResult<void>::success();
    }

    errors::ErrorResult<void> initialize()
    {
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess || device_count == 0)
        {
            return errors::ErrorResult<void>::error(errors::ErrorCode::E50_PH_ABORT,
                                                    "no CUDA device available");
        }

        void *initial_memory = nullptr;
        err = cudaMalloc(&initial_memory, config_.initial_size);
        if (err != cudaSuccess)
        {
            return errors::ErrorResult<void>::error(errors::ErrorCode::E10_GPU_OOM,
                                                    "failed to allocate initial CUDA pool");
        }

        total_allocated_ = config_.initial_size;
        free_blocks_.push_back(Block{.ptr = initial_memory, .size = config_.initial_size});

        return errors::ErrorResult<void>::success();
    }

    errors::ErrorResult<void> growPool(size_t size)
    {
        if (total_allocated_ >= config_.max_size)
        {
            return errors::ErrorResult<void>::error(errors::ErrorCode::E41_RESOURCE_LIMIT,
                                                    "CUDA memory pool is at maximum size");
        }
        const size_t remaining = config_.max_size - total_allocated_;
        size_t grow_size = std::min(std::max(size, config_.initial_size), remaining);
        if (grow_size < size)
        {
            return errors::ErrorResult<void>::error(errors::ErrorCode::E41_RESOURCE_LIMIT,
                                                    "requested allocation exceeds pool limit");
        }

        void *new_memory = nullptr;
        cudaError_t err = cudaMalloc(&new_memory, grow_size);
        if (err != cudaSuccess)
        {
            return errors::ErrorResult<void>::error(errors::ErrorCode::E10_GPU_OOM,
                                                    "failed to grow CUDA memory pool");
        }

        total_allocated_ += grow_size;
        free_blocks_.push_back(Block{.ptr = new_memory, .size = grow_size});
        return errors::ErrorResult<void>::success();
    }

    errors::ErrorResult<void *> allocate_from_free_blocks(size_t size)
    {
        for (size_t i = 0; i < free_blocks_.size(); ++i)
        {
            Block &block = free_blocks_[i];
            if (block.size < size)
            {
                continue;
            }
            void *ptr = block.ptr;
            if (block.size == size)
            {
                free_blocks_.erase(free_blocks_.begin() + static_cast<std::ptrdiff_t>(i));
            }
            else
            {
                block.ptr = static_cast<void *>(static_cast<char *>(block.ptr) + size);
                block.size -= size;
            }
            used_blocks_[ptr] = Block{.ptr = ptr, .size = size};
            return errors::ErrorResult<void *>::success(static_cast<void *>(ptr));
        }
        return errors::ErrorResult<void *>::error(errors::ErrorCode::E50_PH_ABORT);
    }

    void mergeAdjacentBlocks()
    {
        if (free_blocks_.size() < 2)
        {
            return;
        }
        std::sort(free_blocks_.begin(), free_blocks_.end(), [](const Block &a, const Block &b) {
            return reinterpret_cast<std::uintptr_t>(a.ptr) <
                   reinterpret_cast<std::uintptr_t>(b.ptr);
        });

        for (size_t i = 0; i < free_blocks_.size() - 1; ++i)
        {
            char *end_of_current = static_cast<char *>(free_blocks_[i].ptr) + free_blocks_[i].size;
            if (end_of_current == free_blocks_[i + 1].ptr)
            {
                free_blocks_[i].size += free_blocks_[i + 1].size;
                using BlockIteratorDiff = std::vector<Block>::difference_type;
                free_blocks_.erase(
                    std::next(free_blocks_.begin(), static_cast<BlockIteratorDiff>(i + 1)));
                --i;
            }
        }
    }

    void cleanup()
    {
        std::lock_guard<std::mutex> lock(mutex_);

        for (const auto &block : free_blocks_)
        {
            if (block.ptr != nullptr)
            {
                cudaFree(block.ptr);
            }
        }

        for (const auto &block : used_blocks_)
        {
            if (block.second.ptr != nullptr)
            {
                cudaFree(block.second.ptr);
            }
        }

        free_blocks_.clear();
        used_blocks_.clear();
        total_allocated_ = 0;
    }

    Config config_;
    std::vector<Block> free_blocks_;
    std::unordered_map<void *, Block> used_blocks_;
    size_t total_allocated_ = 0;
    mutable std::mutex mutex_;
};

class ThreadSafeMemoryManager
{
public:
    static errors::ErrorResult<std::unique_ptr<ThreadSafeMemoryManager>> create()
    {
        try
        {
            auto manager = std::unique_ptr<ThreadSafeMemoryManager>(new ThreadSafeMemoryManager());
            auto init_result = manager->initialize();
            if (init_result.isError())
            {
                return errors::ErrorResult<std::unique_ptr<ThreadSafeMemoryManager>>::error(
                    init_result.errorCode(), init_result.error().message);
            }
            return errors::ErrorResult<std::unique_ptr<ThreadSafeMemoryManager>>::success(
                std::move(manager));
        }
        catch (const std::exception &e)
        {
            return errors::ErrorResult<std::unique_ptr<ThreadSafeMemoryManager>>::error(
                errors::ErrorCode::E50_PH_ABORT, e.what());
        }
    }

    ~ThreadSafeMemoryManager() { cleanup(); }

    errors::ErrorResult<void *> allocateGpuMemory(size_t size, size_t alignment = 256)
    {
        if (!cuda_pool_)
        {
            return errors::ErrorResult<void *>::error(errors::ErrorCode::E50_PH_ABORT);
        }
        return cuda_pool_->allocate(size, alignment);
    }

    errors::ErrorResult<void> deallocateGpuMemory(void *ptr)
    {
        if (!cuda_pool_)
        {
            return errors::ErrorResult<void>::error(errors::ErrorCode::E50_PH_ABORT);
        }
        return cuda_pool_->deallocate(ptr);
    }

    template <typename T>
    errors::ErrorResult<core::BufferView<const T>> createBufferView(const std::vector<T> &data)
    {
        return errors::ErrorResult<core::BufferView<const T>>::success(
            core::BufferView<const T>(data.data(), data.size()));
    }

    template <typename T>
    errors::ErrorResult<core::BufferView<const T>> createBufferView(const T *data, size_t size)
    {
        return errors::ErrorResult<core::BufferView<const T>>::success(
            core::BufferView<const T>(data, size));
    }

    size_t getGpuMemoryUsage() const
    {
        if (cuda_pool_)
        {
            return cuda_pool_->getUsedSize();
        }
        return 0;
    }

    size_t getGpuMemoryAllocated() const
    {
        if (cuda_pool_)
        {
            return cuda_pool_->getAllocatedSize();
        }
        return 0;
    }

private:
    ThreadSafeMemoryManager() = default;

    errors::ErrorResult<void> initialize()
    {
        auto pool_result = CudaMemoryPool::create();
        if (pool_result.isError())
        {
            return errors::ErrorResult<void>::error(pool_result.errorCode(),
                                                    pool_result.error().message);
        }

        cuda_pool_ = std::move(pool_result.value());
        return errors::ErrorResult<void>::success();
    }

    void cleanup() { cuda_pool_.reset(); }

    std::unique_ptr<CudaMemoryPool> cuda_pool_;
};

} // namespace nerve::persistence::accelerated
