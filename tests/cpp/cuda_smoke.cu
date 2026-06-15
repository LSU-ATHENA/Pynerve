#include <cuda_runtime.h>

#include <array>
#include <iostream>

namespace
{

__global__ void add_one_kernel(int *values)
{
    const int index = static_cast<int>(threadIdx.x);
    values[index] += 1;
}

bool check_cuda(cudaError_t code, const char *expression)
{
    if (code == cudaSuccess)
    {
        return true;
    }
    std::cerr << expression << " failed: " << cudaGetErrorString(code) << '\n';
    return false;
}

} // namespace

int main()
{
    int device_count = 0;
    if (!check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount"))
    {
        return 1;
    }
    if (device_count <= 0)
    {
        std::cerr << "CUDA smoke test requires at least one CUDA device\n";
        return 1;
    }

    std::array<int, 4> host_values{1, 2, 3, 4};
    int *device_values = nullptr;
    const std::size_t bytes = host_values.size() * sizeof(int);
    if (!check_cuda(cudaMalloc(reinterpret_cast<void **>(&device_values), bytes), "cudaMalloc"))
    {
        return 1;
    }
    if (!check_cuda(cudaMemcpy(device_values, host_values.data(), bytes, cudaMemcpyHostToDevice),
                    "cudaMemcpy host-to-device"))
    {
        cudaFree(device_values);
        return 1;
    }

    add_one_kernel<<<1, static_cast<int>(host_values.size())>>>(device_values);
    if (!check_cuda(cudaPeekAtLastError(), "add_one_kernel launch"))
    {
        cudaFree(device_values);
        return 1;
    }
    if (!check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize"))
    {
        cudaFree(device_values);
        return 1;
    }
    if (!check_cuda(cudaMemcpy(host_values.data(), device_values, bytes, cudaMemcpyDeviceToHost),
                    "cudaMemcpy device-to-host"))
    {
        cudaFree(device_values);
        return 1;
    }
    if (!check_cuda(cudaFree(device_values), "cudaFree"))
    {
        return 1;
    }

    const std::array<int, 4> expected{2, 3, 4, 5};
    if (host_values != expected)
    {
        std::cerr << "CUDA smoke result mismatch\n";
        return 1;
    }
    return 0;
}
