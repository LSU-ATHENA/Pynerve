#pragma once

#include <cstddef>
// CUDA >= 12 removed the 'Size' typedef; provide it for compatibility
typedef std::size_t Size;

#if defined(__has_include_next)
#if __has_include_next(<cuda_runtime.h>)
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif
#include_next <cuda_runtime.h>
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
#define NERVE_CUDA_RUNTIME_AVAILABLE 1
#else
#define NERVE_CUDA_RUNTIME_AVAILABLE 0

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>

#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#ifndef __global__
#define __global__
#endif
#ifndef __shared__
#define __shared__
#endif
#ifndef __constant__
#define __constant__
#endif

typedef int cudaError_t;
struct CUstream_st;
typedef CUstream_st *cudaStream_t;
struct CUevent_st;
typedef CUevent_st *cudaEvent_t;
struct CUgraph_st;
typedef CUgraph_st *cudaGraph_t;
struct CUgraphExec_st;
typedef CUgraphExec_st *cudaGraphExec_t;

// CUDA error constants
constexpr cudaError_t cudaSuccess = 0;
constexpr cudaError_t cudaErrorInvalidValue = 1;
constexpr cudaError_t cudaErrorMemoryAllocation = 2;
constexpr cudaError_t cudaErrorInitializationError = 3;
constexpr cudaError_t cudaErrorLaunchFailure = 4;
constexpr cudaError_t cudaErrorLaunchTimeout = 5;
constexpr cudaError_t cudaErrorLaunchOutOfResources = 6;
constexpr cudaError_t cudaErrorInvalidDevice = 7;
constexpr cudaError_t cudaErrorPeerAccessAlreadyEnabled = 8;
constexpr cudaError_t cudaErrorNotReady = 34;
constexpr cudaError_t cudaErrorNotSupported = 9;

enum cudaMemcpyKind
{
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4
};

enum cudaDeviceAttr
{
    cudaDevAttrComputeCapabilityMajor = 75,
    cudaDevAttrComputeCapabilityMinor = 76,
    cudaDevAttrMultiProcessorCount = 16
};

enum cudaFuncCache
{
    cudaFuncCachePreferNone = 0
};

enum cudaSharedMemConfig
{
    cudaSharedMemBankSizeDefault = 0
};

struct cudaStreamAttrValue
{
    int dummy;
};

enum cudaStreamAttrID
{
    cudaStreamAttributeAccessPolicyWindow = 1
};

struct dim3
{
    unsigned int x;
    unsigned int y;
    unsigned int z;
    constexpr dim3(unsigned int x_val = 1, unsigned int y_val = 1, unsigned int z_val = 1)
        : x(x_val)
        , y(y_val)
        , z(z_val)
    {}
};

struct cudaDeviceProp
{
    char name[256];
    std::size_t totalGlobalMem;
    int major;
    int minor;
    int multiProcessorCount;
    int clockRate;
    int warpSize;
    int maxThreadsPerBlock;
    int maxThreadsPerMultiProcessor;
    std::size_t sharedMemPerBlock;
    int canMapHostMemory;
    int concurrentKernels;
};

inline const char *cudaGetErrorString(cudaError_t error)
{
    switch (error)
    {
        case cudaSuccess:
            return "cudaSuccess";
        case cudaErrorInvalidValue:
            return "cudaErrorInvalidValue";
        case cudaErrorMemoryAllocation:
            return "cudaErrorMemoryAllocation";
        case cudaErrorInitializationError:
            return "cudaErrorInitializationError";
        case cudaErrorLaunchFailure:
            return "cudaErrorLaunchFailure";
        case cudaErrorLaunchTimeout:
            return "cudaErrorLaunchTimeout";
        case cudaErrorLaunchOutOfResources:
            return "cudaErrorLaunchOutOfResources";
        case cudaErrorInvalidDevice:
            return "cudaErrorInvalidDevice";
        case cudaErrorPeerAccessAlreadyEnabled:
            return "cudaErrorPeerAccessAlreadyEnabled";
        default:
            return "cudaErrorUnknown";
    }
}

inline cudaError_t cudaGetLastError()
{
    return cudaSuccess;
}

inline cudaError_t cudaGetDeviceCount(int *count)
{
    if (count != nullptr)
    {
        *count = 0;
    }
    return cudaSuccess;
}

inline cudaError_t cudaGetDevice(int *device)
{
    if (device != nullptr)
    {
        *device = 0;
    }
    return cudaSuccess;
}

inline cudaError_t cudaSetDevice(int /*device*/)
{
    return cudaSuccess;
}

inline cudaError_t cudaDeviceCanAccessPeer(int *can_access_peer, int /*device*/,
                                           int /*peer_device*/)
{
    if (can_access_peer == nullptr)
    {
        return cudaErrorInvalidValue;
    }
    *can_access_peer = 0;
    return cudaSuccess;
}

inline cudaError_t cudaDeviceEnablePeerAccess(int /*peer_device*/, unsigned int /*flags*/)
{
    return cudaErrorInvalidDevice;
}

inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp *prop, int /*device*/)
{
    if (prop != nullptr)
    {
        std::memset(prop, 0, sizeof(cudaDeviceProp));
        std::strncpy(prop->name, "CPU Runtime", sizeof(prop->name) - 1);
        prop->major = 0;
        prop->minor = 0;
    }
    return cudaSuccess;
}

inline cudaError_t cudaDeviceGetAttribute(int *value, cudaDeviceAttr attr, int /*device*/)
{
    if (value == nullptr)
    {
        return cudaErrorInvalidValue;
    }
    switch (attr)
    {
        case cudaDevAttrComputeCapabilityMajor:
        case cudaDevAttrComputeCapabilityMinor:
            *value = 0;
            break;
        case cudaDevAttrMultiProcessorCount:
            *value = 1;
            break;
        default:
            *value = 0;
            break;
    }
    return cudaSuccess;
}

inline cudaError_t cudaMalloc(void **ptr, std::size_t size)
{
    if (ptr == nullptr)
    {
        return cudaErrorInvalidValue;
    }
    *ptr = ::operator new(size, std::nothrow);
    return *ptr == nullptr ? cudaErrorMemoryAllocation : cudaSuccess;
}

template <typename T>
inline cudaError_t cudaMalloc(T **ptr, std::size_t size)
{
    return cudaMalloc(reinterpret_cast<void **>(ptr), size);
}

inline cudaError_t cudaFree(void *ptr)
{
    ::operator delete(ptr);
    return cudaSuccess;
}

inline cudaError_t cudaMemcpy(void *dst, const void *src, std::size_t count,
                              cudaMemcpyKind /*kind*/)
{
    if (dst == nullptr || src == nullptr)
    {
        return cudaErrorInvalidValue;
    }
    std::memcpy(dst, src, count);
    return cudaSuccess;
}

inline cudaError_t cudaMemcpyPeer(void *dst, int /*dst_device*/, const void *src,
                                  int /*src_device*/, std::size_t count)
{
    return cudaMemcpy(dst, src, count, cudaMemcpyDeviceToDevice);
}

inline cudaError_t cudaMemcpyAsync(void *dst, const void *src, std::size_t count,
                                   cudaMemcpyKind kind, cudaStream_t /*stream*/ = nullptr)
{
    return cudaMemcpy(dst, src, count, kind);
}

inline cudaError_t cudaMemset(void *dev_ptr, int value, std::size_t count)
{
    if (dev_ptr == nullptr)
    {
        return cudaErrorInvalidValue;
    }
    std::memset(dev_ptr, value, count);
    return cudaSuccess;
}

inline cudaError_t cudaMemGetInfo(std::size_t *free_mem, std::size_t *total_mem)
{
    if (free_mem != nullptr)
    {
        *free_mem = 0;
    }
    if (total_mem != nullptr)
    {
        *total_mem = 0;
    }
    return cudaSuccess;
}

inline cudaError_t cudaDeviceSynchronize()
{
    return cudaSuccess;
}

inline cudaError_t cudaDeviceReset()
{
    return cudaSuccess;
}

inline cudaError_t cudaDeviceSetCacheConfig(cudaFuncCache /*config*/)
{
    return cudaSuccess;
}

inline cudaError_t cudaDeviceSetSharedMemConfig(cudaSharedMemConfig /*config*/)
{
    return cudaSuccess;
}

inline cudaError_t cudaDriverGetVersion(int *version)
{
    if (version != nullptr)
    {
        *version = 0;
    }
    return cudaSuccess;
}

inline cudaError_t cudaRuntimeGetVersion(int *version)
{
    if (version != nullptr)
    {
        *version = 0;
    }
    return cudaSuccess;
}

inline cudaError_t cudaStreamCreate(cudaStream_t *stream)
{
    if (stream != nullptr)
    {
        *stream = nullptr;
    }
    return cudaSuccess;
}

inline cudaError_t cudaStreamCreateWithFlags(cudaStream_t *stream, unsigned int /*flags*/)
{
    if (stream != nullptr)
    {
        *stream = nullptr;
    }
    return cudaSuccess;
}

inline cudaError_t cudaStreamDestroy(cudaStream_t /*stream*/)
{
    return cudaSuccess;
}

inline cudaError_t cudaStreamSynchronize(cudaStream_t /*stream*/)
{
    return cudaSuccess;
}

inline cudaError_t cudaStreamQuery(cudaStream_t /*stream*/)
{
    return cudaSuccess;
}

inline cudaError_t cudaEventCreate(cudaEvent_t *event)
{
    if (event != nullptr)
    {
        *event = nullptr;
    }
    return cudaSuccess;
}

inline cudaError_t cudaEventDestroy(cudaEvent_t /*event*/)
{
    return cudaSuccess;
}

inline cudaError_t cudaEventRecord(cudaEvent_t /*event*/, cudaStream_t /*stream*/ = nullptr)
{
    return cudaSuccess;
}

inline cudaError_t cudaStreamWaitEvent(cudaStream_t /*stream*/, cudaEvent_t /*event*/,
                                       unsigned int /*flags*/
)
{
    return cudaSuccess;
}

inline cudaError_t cudaStreamSetAttribute(cudaStream_t /*stream*/, cudaStreamAttrID /*attr*/,
                                          const cudaStreamAttrValue * /*value*/
)
{
    return cudaSuccess;
}

inline cudaError_t cudaOccupancyMaxPotentialBlockSize(int *minGridSize, int *blockSize,
                                                      const void * /*func*/,
                                                      std::size_t /*dynamicSMemSize*/,
                                                      int /*blockSizeLimit*/
)
{
    if (minGridSize != nullptr)
    {
        *minGridSize = 1;
    }
    if (blockSize != nullptr)
    {
        *blockSize = 128;
    }
    return cudaSuccess;
}

inline cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks,
                                                                 const void * /*func*/,
                                                                 int /*blockSize*/,
                                                                 std::size_t /*dynamicSMemSize*/
)
{
    if (numBlocks != nullptr)
    {
        *numBlocks = 1;
    }
    return cudaSuccess;
}

inline cudaError_t cudaEventSynchronize(cudaEvent_t /*event*/)
{
    return cudaSuccess;
}

inline cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t /*start*/, cudaEvent_t /*end*/)
{
    if (ms != nullptr)
    {
        *ms = 0.0f;
    }
    return cudaSuccess;
}

#endif
#else
#include <cuda_runtime.h>
#define NERVE_CUDA_RUNTIME_AVAILABLE 1
#endif
