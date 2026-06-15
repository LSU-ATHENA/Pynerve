#pragma once

#include "nerve/error/error_registry.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <functional>
#include <optional>
#include <source_location>
#include <string>
#include <type_traits>

namespace nerve::gpu
{

enum class CudaErrorKind
{
    Success,
    OutOfMemory,        // cudaErrorMemoryAllocation -> try smaller batch
    InvalidDevice,      // cudaErrorInvalidDevice    -> invalid device selection
    KernelLaunchFailed, // cudaErrorLaunchFailure    -> check grid/block dims
    SyncFailed,         // cudaErrorLaunchTimeout    -> kernel hung
    InvalidValue,       // cudaErrorInvalidValue     -> programming error
    NotReady,           // cudaErrorNotReady         -> async op pending
    Unknown,
};

struct CudaError
{
    cudaError_t raw_code;
    CudaErrorKind kind;
    std::string message;
    std::string call_expr;
    std::source_location where;

    static CudaError from(cudaError_t code, std::string call,
                          std::source_location loc = std::source_location::current())
    {
        return {.raw_code = code,
                .kind = classify(code),
                .message = cudaGetErrorString(code),
                .call_expr = std::move(call),
                .where = loc};
    }

    bool isOom() const noexcept { return kind == CudaErrorKind::OutOfMemory; }
    bool isHwError() const noexcept
    {
        return kind == CudaErrorKind::KernelLaunchFailed || kind == CudaErrorKind::SyncFailed;
    }

    std::string format() const
    {
        return "[CUDA] " + call_expr + " -> " + message + " at " + where.file_name() + ":" +
               std::to_string(where.line());
    }

private:
    static CudaErrorKind classify(cudaError_t c) noexcept
    {
        switch (c)
        {
            case cudaSuccess:
                return CudaErrorKind::Success;
            case cudaErrorMemoryAllocation:
                return CudaErrorKind::OutOfMemory;
            case cudaErrorInvalidDevice:
                return CudaErrorKind::InvalidDevice;
            case cudaErrorLaunchFailure:
                return CudaErrorKind::KernelLaunchFailed;
            case cudaErrorLaunchTimeout:
                return CudaErrorKind::SyncFailed;
            case cudaErrorInvalidValue:
                return CudaErrorKind::InvalidValue;
            case cudaErrorNotReady:
                return CudaErrorKind::NotReady;
            default:
                return CudaErrorKind::Unknown;
        }
    }
};

template <typename T>
using CudaResult = nerve::error::Result<T>;

inline CudaResult<void> cuda_check(cudaError_t code, const char *expr,
                                   std::source_location loc = std::source_location::current())
{
    if (code == cudaSuccess)
        return CudaResult<void>::ok();

    auto cuda_err = CudaError::from(code, expr, loc);
    auto tda_code = [&]() -> nerve::error::TDAErrorCode {
        switch (cuda_err.kind)
        {
            case CudaErrorKind::OutOfMemory:
                return nerve::error::TDAErrorCode::AllocationFailed;
            case CudaErrorKind::InvalidDevice:
                return nerve::error::TDAErrorCode::GpuInvalidDevice;
            case CudaErrorKind::KernelLaunchFailed:
                return nerve::error::TDAErrorCode::GpuKernelLaunchFailed;
            case CudaErrorKind::SyncFailed:
                return nerve::error::TDAErrorCode::GpuSyncFailed;
            case CudaErrorKind::InvalidValue:
                return nerve::error::TDAErrorCode::InvalidInput;
            case CudaErrorKind::NotReady:
                return nerve::error::TDAErrorCode::ResourceLimit;
            default:
                return nerve::error::TDAErrorCode::GpuKernelLaunchFailed;
        }
    }();
    return CudaResult<void>::err(tda_code, cuda_err.format());
}

inline CudaResult<void>
cuda_check_kernel(const char *kernel_name,
                  std::source_location loc = std::source_location::current())
{
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess)
    {
        auto cuda_err =
            CudaError::from(launch_err, std::string("kernel launch: ") + kernel_name, loc);
        return CudaResult<void>::err(nerve::error::TDAErrorCode::GpuKernelLaunchFailed,
                                     cuda_err.format());
    }

    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess)
    {
        auto cuda_err =
            CudaError::from(sync_err, std::string("kernel execute: ") + kernel_name, loc);
        return CudaResult<void>::err(nerve::error::TDAErrorCode::GpuSyncFailed, cuda_err.format());
    }

    return CudaResult<void>::ok();
}

inline CudaResult<void>
cuda_check_kernel_async(const char *kernel_name, cudaStream_t stream = nullptr,
                        std::source_location loc = std::source_location::current())
{
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess)
    {
        auto cuda_err =
            CudaError::from(launch_err, std::string("kernel launch: ") + kernel_name, loc);
        return CudaResult<void>::err(nerve::error::TDAErrorCode::GpuKernelLaunchFailed,
                                     cuda_err.format());
    }

    if (stream)
    {
        cudaError_t sync_err = cudaStreamSynchronize(stream);
        if (sync_err != cudaSuccess)
        {
            auto cuda_err =
                CudaError::from(sync_err, std::string("kernel async: ") + kernel_name, loc);
            return CudaResult<void>::err(nerve::error::TDAErrorCode::GpuSyncFailed,
                                         cuda_err.format());
        }
    }

    return CudaResult<void>::ok();
}

#define CUDA_CALL(expr)                                                                            \
    do                                                                                             \
    {                                                                                              \
        auto _r = ::nerve::gpu::cuda_check((expr), #expr);                                         \
        if (!_r.isOk())                                                                            \
            return _r;                                                                             \
    } while (0)

#define CUDA_KERNEL_CHECK(name)                                                                    \
    do                                                                                             \
    {                                                                                              \
        auto _r = ::nerve::gpu::cuda_check_kernel(name);                                           \
        if (!_r.isOk())                                                                            \
            return _r;                                                                             \
    } while (0)

// Variant that logs and continues
#define CUDA_CALL_LOG(expr)                                                                        \
    do                                                                                             \
    {                                                                                              \
        cudaError_t _e = (expr);                                                                   \
        if (_e != cudaSuccess)                                                                     \
        {                                                                                          \
            ::nerve::gpu::CudaErrorLogger::log(::nerve::gpu::CudaError::from(_e, #expr));          \
        }                                                                                          \
    } while (0)

struct CudaErrorLogger
{
    using Sink = std::function<void(const CudaError &)>;
    static Sink &sink()
    {
        static Sink s = [](const CudaError &e) { fprintf(stderr, "%s\n", e.format().c_str()); };
        return s;
    }
    static void log(const CudaError &e) { sink()(e); }
};

template <typename T>
class RecoveryPolicy
{
public:
    static CudaResult<T> fail(const CudaError &e)
    {
        auto tda_code = [&]() -> nerve::error::TDAErrorCode {
            switch (e.kind)
            {
                case CudaErrorKind::OutOfMemory:
                    return nerve::error::TDAErrorCode::AllocationFailed;
                case CudaErrorKind::InvalidDevice:
                    return nerve::error::TDAErrorCode::GpuInvalidDevice;
                case CudaErrorKind::KernelLaunchFailed:
                    return nerve::error::TDAErrorCode::GpuKernelLaunchFailed;
                case CudaErrorKind::SyncFailed:
                    return nerve::error::TDAErrorCode::GpuSyncFailed;
                case CudaErrorKind::InvalidValue:
                    return nerve::error::TDAErrorCode::InvalidInput;
                case CudaErrorKind::NotReady:
                    return nerve::error::TDAErrorCode::ResourceLimit;
                default:
                    return nerve::error::TDAErrorCode::GpuKernelLaunchFailed;
            }
        }();
        return CudaResult<T>::err(tda_code, e.format());
    }

    static CudaResult<T> retrySmaller(std::size_t initial_batch, int max_retries,
                                      std::function<CudaResult<T>(std::size_t batch)> fn)
    {
        std::size_t batch = initial_batch;
        for (int attempt = 0; attempt <= max_retries; ++attempt)
        {
            auto result = fn(batch);
            if (result.isOk())
                return result;
            if (!result.error().isOom())
                return result;
            batch /= 2;
            if (batch == 0)
                break;
            CudaErrorLogger::log(result.error());
            fprintf(stderr, "[CUDA] OOM retry %d/%d with batch=%zu\n", attempt + 1, max_retries,
                    batch);
        }
        return CudaResult<T>::err(
            CudaError{.raw_code = cudaErrorMemoryAllocation,
                      .kind = CudaErrorKind::OutOfMemory,
                      .message = "OOM after " + std::to_string(max_retries) + " retries",
                      .call_expr = std::string(),
                      .where = std::source_location::current()});
    }
};

} // namespace nerve::gpu
