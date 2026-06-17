#if __has_include("nerve/gpu/gpu_memory.hpp")
#include "nerve/gpu/gpu_memory.hpp"
#else
int main()
{
    return 0;
}
#endif

#if __has_include("nerve/gpu/gpu_memory.hpp")

#include "nerve/errors/errors.hpp"

#include <algorithm>#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

namespace
{

using nerve::error::Result;
using nerve::error::TDAErrorCode;

bool check_gpu_memory_concept()
{
    bool check_float = nerve::gpu::GPUCompatible<float>;
    bool check_int = nerve::gpu::GPUCompatible<int>;
    bool check_double = nerve::gpu::GPUCompatible<double>;
    if (!check_float)
        return false;
    if (!check_int)
        return false;
    if (!check_double)
        return false;
    return true;
}

bool check_make_gpu_memory_zero_count()
{
    auto result = nerve::gpu::makeGpuMemory<float>(0);
    if (!result.isOk())
        return true;
    auto &mem = result.value();
    if (mem.count() != 0)
        return false;
    if (mem.size() != 0)
        return false;
    return true;
}

bool check_make_gpu_memory_with_data()
{
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    auto result = nerve::gpu::makeGpuMemoryWithData(data.data(), data.size());
    if (!result.isOk())
        return true;
    auto &mem = result.value();
    if (mem.count() != data.size())
        return false;
    return true;
}

bool check_gpu_memory_move()
{
    auto r1 = nerve::gpu::makeGpuMemory<double>(64);
    if (!r1.isOk())
        return true;
    auto m1 = std::move(r1.value());
    nerve::gpu::GPUMemory<double> m2(std::move(m1));
    if (m2.count() != 64)
        return false;
    nerve::gpu::GPUMemory<double> m3(32);
    m3 = std::move(m2);
    if (m3.count() != 64)
        return false;
    return true;
}

bool check_gpu_memory_reset()
{
    auto r1 = nerve::gpu::makeGpuMemory<int>(16);
    if (!r1.isOk())
        return true;
    auto &mem = r1.value();
    auto reset_result = mem.reset(32);
    if (!reset_result.isOk())
        return false;
    if (mem.count() != 32)
        return false;
    auto reset_zero = mem.reset(0);
    if (!reset_zero.isOk())
        return false;
    if (mem.count() != 0)
        return false;
    return true;
}

bool check_gpu_memory_bool_cast()
{
    auto r1 = nerve::gpu::makeGpuMemory<char>(8);
    if (!r1.isOk())
        return true;
    auto &mem = r1.value();
    if (!static_cast<bool>(mem))
        return false;
    nerve::gpu::GPUMemory<char> empty(0);
    if (static_cast<bool>(empty))
        return false;
    return true;
}

bool check_gpu_memory_copy_errors()
{
    auto r1 = nerve::gpu::makeGpuMemory<double>(4);
    if (!r1.isOk())
        return true;
    auto &mem = r1.value();
    std::vector<double> src(8, 1.0);
    auto copy_result = mem.copyFromHost(src.data(), 8);
    if (copy_result.isOk())
        return false;
    std::vector<double> dst(2, 0.0);
    auto copy_to = mem.copyToHost(dst.data(), 8);
    if (copy_to.isOk())
        return false;
    return true;
}

bool check_gpu_memory_zero_error()
{
    nerve::gpu::GPUMemory<double> empty(0);
    auto result = empty.zero();
    if (result.isOk())
        return false;
    return true;
}

} // namespace

int main()
{
    if (!check_gpu_memory_concept())
    {
        std::cerr << "FAIL: gpu memory concept\n";
        return 1;
    }
    if (!check_make_gpu_memory_zero_count())
    {
        std::cerr << "FAIL: make gpu memory zero count\n";
        return 1;
    }
    if (!check_make_gpu_memory_with_data())
    {
        std::cerr << "FAIL: make gpu memory with data\n";
        return 1;
    }
    if (!check_gpu_memory_move())
    {
        std::cerr << "FAIL: gpu memory move\n";
        return 1;
    }
    if (!check_gpu_memory_reset())
    {
        std::cerr << "FAIL: gpu memory reset\n";
        return 1;
    }
    if (!check_gpu_memory_bool_cast())
    {
        std::cerr << "FAIL: gpu memory bool cast\n";
        return 1;
    }
    if (!check_gpu_memory_copy_errors())
    {
        std::cerr << "FAIL: gpu memory copy errors\n";
        return 1;
    }
    if (!check_gpu_memory_zero_error())
    {
        std::cerr << "FAIL: gpu memory zero error\n";
        return 1;
    }
    return 0;
}
#endif
