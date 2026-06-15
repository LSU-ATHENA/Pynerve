#pragma once
#include "nerve/core_types.hpp"

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <random>

namespace nerve::determinism
{

extern thread_local uint64_t tls_seed;

inline void seed(uint64_t s) noexcept
{
    tls_seed = s;
}
inline uint64_t get_seed() noexcept
{
    return tls_seed;
}

inline uint64_t next_seed() noexcept
{
    constexpr uint64_t kUnseeded = 0;
    thread_local std::mt19937_64 rng;
    if (tls_seed != kUnseeded)
    {
        rng.seed(tls_seed);
        tls_seed = kUnseeded;
    }
    return rng();
}

#if defined(__CUDACC__)

template <int kBlockSize>
__device__ inline double warpReduceSum(double val)
{
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset, 32);
    return val;
}

template <int kBlockSize>
__device__ double blockReduceSum(double val)
{
    extern __shared__ double shared_[];
    int tid = threadIdx.x;
    shared_[tid] = val;
    __syncthreads();
    if constexpr (kBlockSize > 512)
    {
        if (tid < 512)
        {
            shared_[tid] += shared_[tid + 512];
        }
        __syncthreads();
    }
    if constexpr (kBlockSize > 256)
    {
        if (tid < 256)
        {
            shared_[tid] += shared_[tid + 256];
        }
        __syncthreads();
    }
    if constexpr (kBlockSize > 128)
    {
        if (tid < 128)
        {
            shared_[tid] += shared_[tid + 128];
        }
        __syncthreads();
    }
    if (tid < 64)
        shared_[tid] += shared_[tid + 32];
    if (tid < 32)
        shared_[tid] = warpReduceSum<kBlockSize>(shared_[tid]);
    return shared_[0];
}
#endif

#if defined(NERVE_HAS_MPI)
#include <mpi.h>

void deterministic_reduce(const double *send, double *recv, int n, int root, MPI_Comm comm);
void deterministic_allreduce(const double *send, double *recv, int n, MPI_Comm comm);
#endif

} // namespace nerve::determinism
