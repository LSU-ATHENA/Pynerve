#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <utility>

namespace nerve::gpu
{

template <typename T>
class DeviceArray
{
public:
    DeviceArray() = default;

    explicit DeviceArray(size_t count)
        : count_(count)
    {
        if (count_ > 0)
        {
            cudaError_t err = cudaMalloc(&ptr_, count_ * sizeof(T));
            if (err != cudaSuccess)
            {
                throw std::runtime_error("cudaMalloc failed: " +
                                         std::string(cudaGetErrorString(err)));
            }
        }
    }

    ~DeviceArray() { free(); }

    DeviceArray(const DeviceArray &) = delete;
    DeviceArray &operator=(const DeviceArray &) = delete;

    DeviceArray(DeviceArray &&other) noexcept
        : ptr_(other.ptr_)
        , count_(other.count_)
    {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }

    DeviceArray &operator=(DeviceArray &&other) noexcept
    {
        if (this != &other)
        {
            free();
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    T *get() noexcept { return ptr_; }
    const T *get() const noexcept { return ptr_; }
    size_t size() const noexcept { return count_; }
    size_t bytes() const noexcept { return count_ * sizeof(T); }

    void reset()
    {
        free();
        ptr_ = nullptr;
        count_ = 0;
    }

    void copyFromHost(const T *host_ptr, size_t count, cudaStream_t stream = 0)
    {
        cudaError_t err =
            cudaMemcpyAsync(ptr_, host_ptr, count * sizeof(T), cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess)
        {
            throw std::runtime_error("cudaMemcpy H2D failed: " +
                                     std::string(cudaGetErrorString(err)));
        }
    }

    void copyToHost(T *host_ptr, size_t count, cudaStream_t stream = 0)
    {
        cudaError_t err =
            cudaMemcpyAsync(host_ptr, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost, stream);
        if (err != cudaSuccess)
        {
            throw std::runtime_error("cudaMemcpy D2H failed: " +
                                     std::string(cudaGetErrorString(err)));
        }
    }

private:
    void free()
    {
        if (ptr_ != nullptr)
        {
            cudaFree(ptr_);
        }
    }

    T *ptr_ = nullptr;
    size_t count_ = 0;
};

} // namespace nerve::gpu
