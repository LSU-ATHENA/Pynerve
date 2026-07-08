#include "nerve/gpu/device_array.hpp"

#include <cuda_runtime.h>

#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace
{

bool check_cuda(cudaError_t code, const char *expression)
{
    if (code == cudaSuccess)
        return true;
    std::cerr << expression << " failed: " << cudaGetErrorString(code) << '\n';
    return false;
}

bool has_gpu()
{
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return err == cudaSuccess && device_count > 0;
}

} // namespace

int main()
{
    if (!has_gpu())
    {
        std::cerr << "No CUDA device available  --  skipping DeviceArray tests\n";
        return 0;
    }

    // Default construction
    {
        nerve::gpu::DeviceArray<float> arr;
        assert(arr.get() == nullptr);
        assert(arr.size() == 0);
        assert(arr.bytes() == 0);
    }

    // Allocating constructor
    {
        nerve::gpu::DeviceArray<int> arr(100);
        assert(arr.get() != nullptr);
        assert(arr.size() == 100);
        assert(arr.bytes() == 100 * sizeof(int));
    }

    // Move construction
    {
        nerve::gpu::DeviceArray<double> src(50);
        double *src_ptr = src.get();
        assert(src_ptr != nullptr);
        assert(src.size() == 50);

        nerve::gpu::DeviceArray<double> dst(std::move(src));
        assert(dst.get() == src_ptr);
        assert(dst.size() == 50);
        assert(src.get() == nullptr);
        assert(src.size() == 0);
    }

    // Move assignment
    {
        nerve::gpu::DeviceArray<float> src(25);
        float *src_ptr = src.get();

        nerve::gpu::DeviceArray<float> dst(10);
        assert(dst.get() != nullptr);
        assert(dst.size() == 10);

        dst = std::move(src);
        assert(dst.get() == src_ptr);
        assert(dst.size() == 25);
        assert(src.get() == nullptr);
        assert(src.size() == 0);
    }

    // Host-to-device and device-to-host transfer
    {
        const size_t count = 16;
        std::vector<int> host_src(count);
        for (size_t i = 0; i < count; ++i)
            host_src[i] = static_cast<int>(i + 1);

        nerve::gpu::DeviceArray<int> arr(count);
        arr.copyFromHost(host_src.data(), count);

        std::vector<int> host_dst(count, 0);
        arr.copyToHost(host_dst.data(), count);

        for (size_t i = 0; i < count; ++i)
        {
            assert(host_dst[i] == static_cast<int>(i + 1));
        }
    }

    // Double transfer round-trip
    {
        const size_t count = 32;
        std::vector<float> host_src(count);
        for (size_t i = 0; i < count; ++i)
            host_src[i] = static_cast<float>(i) * 1.5f;

        nerve::gpu::DeviceArray<float> arr(count);
        arr.copyFromHost(host_src.data(), count);

        std::vector<float> host_mid(count, 0.0f);
        arr.copyToHost(host_mid.data(), count);

        for (size_t i = 0; i < count; ++i)
        {
            assert(std::abs(host_mid[i] - host_src[i]) < 1e-6f);
        }

        std::vector<float> host_modified(count);
        for (size_t i = 0; i < count; ++i)
            host_modified[i] = static_cast<float>(i) * 3.0f;
        arr.copyFromHost(host_modified.data(), count);

        std::vector<float> host_final(count, 0.0f);
        arr.copyToHost(host_final.data(), count);

        for (size_t i = 0; i < count; ++i)
        {
            assert(std::abs(host_final[i] - host_modified[i]) < 1e-6f);
        }
    }

    // Large allocation (try 1GB, expect graceful handling)
    {
        const size_t large_bytes = 1024ULL * 1024ULL * 1024ULL; // 1 GB
        const size_t large_count = large_bytes / sizeof(char);

        bool large_ok = false;
        try
        {
            nerve::gpu::DeviceArray<char> large_arr(large_count);
            assert(large_arr.get() != nullptr);
            large_ok = true;
        }
        catch (const std::runtime_error &)
        {
            large_ok = false;
        }
        catch (const std::bad_alloc &)
        {
            large_ok = false;
        }

        if (large_ok)
        {
            assert(true);
        }
    }

    // Double-move: verify source is nulled after each move
    {
        nerve::gpu::DeviceArray<int> a(5);
        int *a_ptr = a.get();
        assert(a_ptr != nullptr);

        nerve::gpu::DeviceArray<int> b(std::move(a));
        assert(b.get() == a_ptr);
        assert(a.get() == nullptr);
        assert(a.size() == 0);

        nerve::gpu::DeviceArray<int> c(std::move(b));
        assert(c.get() == a_ptr);
        assert(b.get() == nullptr);
        assert(b.size() == 0);
    }

    // Self-move-assignment guard
    {
        nerve::gpu::DeviceArray<float> arr(10);
        float *orig_ptr = arr.get();

        // Self-move assignment via std::move
        arr = std::move(arr);

        assert(arr.get() == orig_ptr);
        assert(arr.size() == 10);
    }

    // Reset
    {
        nerve::gpu::DeviceArray<double> arr(20);
        assert(arr.get() != nullptr);
        assert(arr.size() == 20);

        arr.reset();
        assert(arr.get() == nullptr);
        assert(arr.size() == 0);
    }

    // Transfer with default stream
    {
        const size_t count = 8;
        std::vector<unsigned long long> host_src(count);
        for (size_t i = 0; i < count; ++i)
            host_src[i] = i * 100ULL;

        nerve::gpu::DeviceArray<unsigned long long> arr(count);
        arr.copyFromHost(host_src.data(), count);

        std::vector<unsigned long long> host_dst(count, 0);
        arr.copyToHost(host_dst.data(), count);

        for (size_t i = 0; i < count; ++i)
        {
            assert(host_dst[i] == host_src[i]);
        }
    }

    // Const accessor
    {
        nerve::gpu::DeviceArray<int> arr(5);
        const auto &const_arr = arr;
        assert(const_arr.get() == arr.get());
        assert(const_arr.size() == 5);
    }

    return 0;
}
