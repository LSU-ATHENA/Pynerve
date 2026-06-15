
#pragma once
#include "nerve/error/error_registry.hpp"

#include <cuda_runtime.h>

#include <concepts>
#include <memory>
#include <stdexcept>

namespace nerve::gpu
{

template <typename T>
concept GPUCompatible = std::trivially_copyable<T> || std::is_arithmetic_v<T>;

template <GPUCompatible T>
class GPUMemory
{
public:
    // Constructor allocates GPU memory
    explicit GPUMemory(size_t count)
        : size_(count * sizeof(T))
    {
        void *ptr;
        cudaError_t result = cudaMalloc(&ptr, size_);
        if (result != cudaSuccess)
        {
            throw std::runtime_error(std::string("cudaMalloc failed: ") +
                                     cudaGetErrorString(result));
        }
        ptr_ = static_cast<T *>(ptr);
    }

    // Destructor automatically frees GPU memory
    ~GPUMemory()
    {
        if (ptr_)
        {
            cudaFree(ptr_);
        }
    }

    // Move-only semantics for efficient transfers
    GPUMemory(const GPUMemory &) = delete;
    GPUMemory &operator=(const GPUMemory &) = delete;

    GPUMemory(GPUMemory &&other) noexcept
        : ptr_(other.ptr_)
        , size_(other.size_)
    {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    GPUMemory &operator=(GPUMemory &&other) noexcept
    {
        if (this != &other)
        {
            // Clean up current resource
            if (ptr_)
            {
                cudaFree(ptr_);
            }

            // Transfer ownership
            ptr_ = other.ptr_;
            size_ = other.size_;

            // Leave other in valid state
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    // Accessors
    T *get() const noexcept { return ptr_; }
    T *operator->() const noexcept { return ptr_; }
    T &operator*() const noexcept { return *ptr_; }
    size_t size() const noexcept { return size_; }
    size_t count() const noexcept { return size_ / sizeof(T); }

    // Check if memory is allocated
    explicit operator bool() const noexcept { return ptr_ != nullptr; }

    // Reset to different size
    error::Result<void> reset(size_t new_count)
    {
        size_t new_size = new_count * sizeof(T);

        // Free old memory
        if (ptr_)
        {
            cudaError_t result = cudaFree(ptr_);
            if (result != cudaSuccess)
            {
                return error::Result<void>::err(error::TDAErrorCode::AllocationFailed,
                                                "Failed to free GPU memory during reset");
            }
            ptr_ = nullptr;
        }

        // Allocate new memory
        if (new_count > 0)
        {
            void *new_ptr;
            cudaError_t result = cudaMalloc(&new_ptr, new_size);
            if (result != cudaSuccess)
            {
                return error::Result<void>::err(error::TDAErrorCode::AllocationFailed,
                                                "Failed to allocate GPU memory during reset");
            }
            ptr_ = static_cast<T *>(new_ptr);
            size_ = new_size;
        }
        else
        {
            size_ = 0;
        }

        return error::Result<void>::ok();
    }

    // Copy data from host to device
    error::Result<void> copyFromHost(const T *host_data, size_t count)
    {
        if (!ptr_ || count > this->count())
        {
            return error::Result<void>::err(error::TDAErrorCode::InvalidDimension,
                                            "Invalid copy parameters: insufficient GPU memory");
        }

        cudaError_t result = cudaMemcpy(ptr_, host_data, count * sizeof(T), cudaMemcpyHostToDevice);
        if (result != cudaSuccess)
        {
            return error::Result<void>::err(error::TDAErrorCode::AllocationFailed,
                                            "Failed to copy data from host to device");
        }

        return error::Result<void>::ok();
    }

    // Copy data from device to host
    error::Result<void> copyToHost(T *host_data, size_t count) const
    {
        if (!ptr_ || count > this->count())
        {
            return error::Result<void>::err(error::TDAErrorCode::InvalidDimension,
                                            "Invalid copy parameters: insufficient GPU memory");
        }

        cudaError_t result = cudaMemcpy(host_data, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost);
        if (result != cudaSuccess)
        {
            return error::Result<void>::err(error::TDAErrorCode::AllocationFailed,
                                            "Failed to copy data from device to host");
        }

        return error::Result<void>::ok();
    }

    // Zero out GPU memory
    error::Result<void> zero()
    {
        if (!ptr_)
        {
            return error::Result<void>::err(error::TDAErrorCode::AllocationFailed,
                                            "Cannot zero unallocated GPU memory");
        }

        cudaError_t result = cudaMemset(ptr_, 0, size_);
        if (result != cudaSuccess)
        {
            return error::Result<void>::err(error::TDAErrorCode::AllocationFailed,
                                            "Failed to zero GPU memory");
        }

        return error::Result<void>::ok();
    }

private:
    T *ptr_ = nullptr;
    size_t size_ = 0;
};

// Convenience factory functions
template <typename T>
error::Result<GPUMemory<T>> makeGpuMemory(size_t count)
{
    try
    {
        return error::Result<GPUMemory<T>>::ok(GPUMemory<T>(count));
    }
    catch (const std::bad_alloc &)
    {
        return error::Result<GPUMemory<T>>::err(error::TDAErrorCode::AllocationFailed,
                                                "Failed to allocate GPU memory");
    }
}

template <typename T>
error::Result<GPUMemory<T>> makeGpuMemoryWithData(const T *host_data, size_t count)
{
    auto gpu_mem_result = makeGpuMemory<T>(count);
    if (gpu_mem_result.isErr())
    {
        return gpu_mem_result;
    }

    auto &gpu_mem = gpu_mem_result.value();
    auto copy_result = gpu_mem.copyFromHost(host_data, count);
    if (copy_result.isErr())
    {
        return error::Result<GPUMemory<T>>::err(copy_result.error(), copy_result.detail());
    }

    return gpu_mem_result;
}

} // namespace nerve::gpu
