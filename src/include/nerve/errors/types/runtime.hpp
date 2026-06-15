
#pragma once

#include "nerve/errors/detail/base.hpp"

#include <cstdint>
#include <optional>
#include <vector>

#if defined(__has_include)
#if __has_include(<cuda_runtime.h>)
#include <cuda_runtime.h>
#else
typedef int cudaError_t;
#endif
#else
#ifndef CUDA_TYPES_H
typedef int cudaError_t;
#endif
#endif

namespace nerve::errors
{

class GPUError : public NerveError
{
public:
    GPUError()
        : NerveError("GPU operation failed")
    {}

    GPUError(const char *file, int line, const char *function)
        : NerveError("GPU operation failed", file, line, function)
    {}

    GPUError &requestedBytes(uint64_t bytes)
    {
        requested_bytes_ = bytes;
        addContext("requested", formatBytes(bytes));
        return *this;
    }

    GPUError &availableBytes(uint64_t bytes)
    {
        available_bytes_ = bytes;
        addContext("available", formatBytes(bytes));
        return *this;
    }

    GPUError &setTensorShape(const std::vector<size_t> &shape)
    {
        addContext("tensor_shape", formatShape(shape));
        return *this;
    }

    GPUError &setDtype(const std::string &dtype)
    {
        addContext("dtype", dtype);
        return *this;
    }

    GPUError &setCudaError(cudaError_t err)
    {
        cudaError_ = err;
        addContext("cudaError_code", std::to_string(static_cast<int>(err)));
        return *this;
    }

    GPUError &setMessage(const std::string &msg)
    {
        setBaseMessage(msg);
        return *this;
    }

    std::string errorTypeName() const override { return "GPUError"; }

    uint32_t errorCode() const override { return 0x00000200; }

protected:
    std::optional<uint64_t> requested_bytes_;
    std::optional<uint64_t> available_bytes_;
    std::optional<int> cudaError_;
};

class GPUMemoryError : public GPUError
{
public:
    GPUMemoryError()
        : GPUError()
    {
        setBaseMessage("GPU out of memory");
    }

    GPUMemoryError(const char *file, int line, const char *function)
        : GPUError(file, line, function)
    {
        setBaseMessage("GPU out of memory");
    }

    std::string errorTypeName() const override { return "GPUMemoryError"; }

    uint32_t errorCode() const override { return 0x00000200; }
};

class GPULaunchError : public GPUError
{
public:
    GPULaunchError()
        : GPUError()
    {
        setBaseMessage("GPU kernel launch failed");
    }

    GPULaunchError(const char *file, int line, const char *function)
        : GPUError(file, line, function)
    {
        setBaseMessage("GPU kernel launch failed");
    }

    GPULaunchError &setKernelName(const std::string &name)
    {
        addContext("kernel", name);
        return *this;
    }

    GPULaunchError &setGridDim(int x, int y = 1, int z = 1)
    {
        addContext("grid_dim", "[" + std::to_string(x) + ", " + std::to_string(y) + ", " +
                                   std::to_string(z) + "]");
        return *this;
    }

    GPULaunchError &setBlockDim(int x, int y = 1, int z = 1)
    {
        addContext("block_dim", "[" + std::to_string(x) + ", " + std::to_string(y) + ", " +
                                    std::to_string(z) + "]");
        return *this;
    }

    std::string errorTypeName() const override { return "GPULaunchError"; }

    uint32_t errorCode() const override { return 0x00000201; }
};

class MemoryError : public NerveError
{
public:
    MemoryError()
        : NerveError("Memory operation failed")
    {}

    MemoryError(const char *file, int line, const char *function)
        : NerveError("Memory operation failed", file, line, function)
    {}

    MemoryError &requestedBytes(uint64_t bytes)
    {
        requested_bytes_ = bytes;
        addContext("requested", formatBytes(bytes));
        return *this;
    }

    MemoryError &availableBytes(uint64_t bytes)
    {
        available_bytes_ = bytes;
        addContext("available", formatBytes(bytes));
        return *this;
    }

    MemoryError &setAllocationType(const std::string &type)
    {
        addContext("allocation_type", type);
        return *this;
    }

    std::string errorTypeName() const override { return "MemoryError"; }

protected:
    std::optional<uint64_t> requested_bytes_;
    std::optional<uint64_t> available_bytes_;
};

class OutOfMemoryError : public MemoryError
{
public:
    OutOfMemoryError()
        : MemoryError()
    {
        setBaseMessage("Out of memory");
    }

    OutOfMemoryError(const char *file, int line, const char *function)
        : MemoryError(file, line, function)
    {
        setBaseMessage("Out of memory");
    }

    std::string errorTypeName() const override { return "OutOfMemoryError"; }

    uint32_t errorCode() const override { return 0x00000501; }
};

class AllocationError : public MemoryError
{
public:
    AllocationError()
        : MemoryError()
    {
        setBaseMessage("Allocation failed");
    }

    AllocationError(const char *file, int line, const char *function)
        : MemoryError(file, line, function)
    {
        setBaseMessage("Allocation failed");
    }

    std::string errorTypeName() const override { return "AllocationError"; }
};

class NumericalError : public NerveError
{
public:
    NumericalError()
        : NerveError("Numerical error")
    {}

    NumericalError(const char *file, int line, const char *function)
        : NerveError("Numerical error", file, line, function)
    {}

    NumericalError &setValue(double val)
    {
        addContext("value", std::to_string(val));
        return *this;
    }

    NumericalError &setExpectedRange(double min, double max)
    {
        addContext("expected_range", "[" + std::to_string(min) + ", " + std::to_string(max) + "]");
        return *this;
    }

    std::string errorTypeName() const override { return "NumericalError"; }
};

class ConvergenceError : public NumericalError
{
public:
    ConvergenceError()
        : NumericalError()
    {
        setBaseMessage("Failed to converge");
    }

    ConvergenceError(const char *file, int line, const char *function)
        : NumericalError(file, line, function)
    {
        setBaseMessage("Failed to converge");
    }

    ConvergenceError &setIterations(size_t iters)
    {
        addContext("iterations", std::to_string(iters));
        return *this;
    }

    ConvergenceError &setMaxIterations(size_t max_iters)
    {
        addContext("max_iterations", std::to_string(max_iters));
        return *this;
    }

    ConvergenceError &setResidual(double residual)
    {
        addContext("final_residual", std::to_string(residual));
        return *this;
    }

    ConvergenceError &setTolerance(double tol)
    {
        addContext("tolerance", std::to_string(tol));
        return *this;
    }

    std::string errorTypeName() const override { return "ConvergenceError"; }

    uint32_t errorCode() const override { return 0x00000301; }
};

class PrecisionError : public NumericalError
{
public:
    PrecisionError()
        : NumericalError()
    {
        setBaseMessage("Precision loss detected");
    }

    PrecisionError(const char *file, int line, const char *function)
        : NumericalError(file, line, function)
    {
        setBaseMessage("Precision loss detected");
    }

    PrecisionError &setPrecisionLoss(double loss)
    {
        addContext("precision_loss", std::to_string(loss));
        return *this;
    }

    PrecisionError &setUnderflow(bool underflow)
    {
        addContext("underflow", underflow ? "true" : "false");
        return *this;
    }

    PrecisionError &setOverflow(bool overflow)
    {
        addContext("overflow", overflow ? "true" : "false");
        return *this;
    }

    std::string errorTypeName() const override { return "PrecisionError"; }

    uint32_t errorCode() const override { return 0x00000801; }
};

class NumericalInstabilityError : public NumericalError
{
public:
    NumericalInstabilityError()
        : NumericalError()
    {
        setBaseMessage("Numerical instability detected");
    }

    NumericalInstabilityError(const char *file, int line, const char *function)
        : NumericalError(file, line, function)
    {
        setBaseMessage("Numerical instability detected");
    }

    NumericalInstabilityError &setConditionNumber(double cond)
    {
        addContext("condition_number", std::to_string(cond));
        return *this;
    }

    NumericalInstabilityError &setMatrixSize(size_t n)
    {
        addContext("matrix_size", std::to_string(n) + "x" + std::to_string(n));
        return *this;
    }

    std::string errorTypeName() const override { return "NumericalInstabilityError"; }

    uint32_t errorCode() const override { return 0x00000803; }
};

} // namespace nerve::errors
