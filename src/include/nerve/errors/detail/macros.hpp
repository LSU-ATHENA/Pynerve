
#pragma once

#include "nerve/errors/detail/base.hpp"
#include "nerve/errors/types/runtime.hpp"
#include "nerve/errors/types/type.hpp"
#include "nerve/errors/types/value.hpp"

#ifndef NERVE_ERROR_THROW
#define NERVE_ERROR_THROW(exceptionType, ...)                                                      \
    throw exceptionType(__FILE__, __LINE__, __func__) __VA_ARGS__
#endif

#ifndef NERVE_CHECK_SHAPE
#define NERVE_CHECK_SHAPE(tensor, expected_shape)                                                  \
    do                                                                                             \
    {                                                                                              \
        const auto &_actual = (tensor).shape();                                                    \
        const auto &_expected = (expected_shape);                                                  \
        if (_actual != _expected)                                                                  \
        {                                                                                          \
            throw nerve::errors::ShapeMismatchError(__FILE__, __LINE__, __func__)                  \
                .setExpectedShape(_expected)                                                       \
                .setActualShape(_actual)                                                           \
                .setOperation(#tensor " shape check")                                              \
                .addSuggestion("Verify tensor dimensions match expected shape");                   \
        }                                                                                          \
    } while (0)
#endif

#ifndef NERVE_CHECK_DIM
#define NERVE_CHECK_DIM(value, expected)                                                           \
    do                                                                                             \
    {                                                                                              \
        auto _actual = (value);                                                                    \
        auto _expected = (expected);                                                               \
        if (_actual != _expected)                                                                  \
        {                                                                                          \
            throw nerve::errors::DimensionError(__FILE__, __LINE__, __func__)                      \
                .expected(_expected)                                                               \
                .actual(_actual)                                                                   \
                .setOperation(#value " dimension check")                                           \
                .addSuggestion("Check dimension calculation or input parameters");                 \
        }                                                                                          \
    } while (0)
#endif

#ifndef NERVE_CHECK_RANGE
#define NERVE_CHECK_RANGE(value, min, max)                                                         \
    do                                                                                             \
    {                                                                                              \
        auto _val = (value);                                                                       \
        auto _min = (min);                                                                         \
        auto _max = (max);                                                                         \
        if (_val < _min || _val > _max)                                                            \
        {                                                                                          \
            throw nerve::errors::InvalidArgumentError(__FILE__, __LINE__, __func__)                \
                .setArgumentName(#value)                                                           \
                .setArgumentValue(_val)                                                            \
                .setExpectedRange(_min, _max)                                                      \
                .setReason("Value out of valid range")                                             \
                .addSuggestion("Ensure value is within [" #min ", " #max "]");                     \
        }                                                                                          \
    } while (0)
#endif

#ifndef NERVE_CHECK_NOT_NULL
#define NERVE_CHECK_NOT_NULL(ptr)                                                                  \
    do                                                                                             \
    {                                                                                              \
        if ((ptr) == nullptr)                                                                      \
        {                                                                                          \
            throw nerve::errors::InvalidArgumentError(__FILE__, __LINE__, __func__)                \
                .setArgumentName(#ptr)                                                             \
                .setReason("Null pointer not allowed")                                             \
                .addSuggestion("Check that pointer is initialized before use");                    \
        }                                                                                          \
    } while (0)
#endif

#ifndef NERVE_CHECK_NOT_EMPTY
#define NERVE_CHECK_NOT_EMPTY(container)                                                           \
    do                                                                                             \
    {                                                                                              \
        if ((container).empty())                                                                   \
        {                                                                                          \
            throw nerve::errors::InvalidArgumentError(__FILE__, __LINE__, __func__)                \
                .setArgumentName(#container)                                                       \
                .setReason("Container cannot be empty")                                            \
                .addSuggestion("Check that container has elements before processing");             \
        }                                                                                          \
    } while (0)
#endif

#ifndef NERVE_CHECK_INDEX
#define NERVE_CHECK_INDEX(index, size)                                                             \
    do                                                                                             \
    {                                                                                              \
        auto _idx = (index);                                                                       \
        auto _size = (size);                                                                       \
        if (_idx >= _size)                                                                         \
        {                                                                                          \
            throw nerve::errors::ShapeMismatchError(__FILE__, __LINE__, __func__)                  \
                .setExpectedDims(static_cast<size_t>(_size), 1)                                    \
                .setActualAccess(static_cast<size_t>(_idx), 0)                                     \
                .setOperation(#index " bounds check")                                              \
                .addSuggestion("Check index calculation or container size");                       \
        }                                                                                          \
    } while (0)
#endif

#ifndef NERVE_CHECK_CUDA
#define NERVE_CHECK_CUDA(call)                                                                     \
    do                                                                                             \
    {                                                                                              \
        cudaError_t _err = (call);                                                                 \
        if (_err != cudaSuccess)                                                                   \
        {                                                                                          \
            throw nerve::errors::GPUError(__FILE__, __LINE__, __func__)                            \
                .setCudaError(_err)                                                                \
                .setOperation(#call)                                                               \
                .addContext("cuda_error_string", cudaGetErrorString(_err))                         \
                .addSuggestion("Check CUDA driver version and GPU availability");                  \
        }                                                                                          \
    } while (0)
#endif

#ifndef NERVE_CHECK_ALLOC
#define NERVE_CHECK_ALLOC(ptr, size)                                                               \
    do                                                                                             \
    {                                                                                              \
        if ((ptr) == nullptr)                                                                      \
        {                                                                                          \
            throw nerve::errors::AllocationError(__FILE__, __LINE__, __func__)                     \
                .requestedBytes(static_cast<uint64_t>(size))                                       \
                .setAllocationType(#ptr)                                                           \
                .addSuggestion("Reduce allocation size or free memory");                           \
        }                                                                                          \
    } while (0)
#endif

#ifndef NERVE_CHECK
#define NERVE_CHECK(cond, ...)                                                                     \
    do                                                                                             \
    {                                                                                              \
        if (!(cond))                                                                               \
        {                                                                                          \
            throw nerve::errors::NerveError(#cond " failed", __FILE__, __LINE__, __func__)         \
                __VA_ARGS__;                                                                       \
        }                                                                                          \
    } while (0)
#endif

#ifndef NERVE_THROW
#define NERVE_THROW(msg) throw nerve::errors::NerveError(msg, __FILE__, __LINE__, __func__)
#endif

#ifndef NERVE_THROW_SHAPE_MISMATCH
#define NERVE_THROW_SHAPE_MISMATCH(expected, actual)                                               \
    throw nerve::errors::ShapeMismatchError(__FILE__, __LINE__, __func__)                          \
        .setExpectedShape(expected)                                                                \
        .setActualShape(actual)
#endif

#ifndef NERVE_THROW_DIMENSION_ERROR
#define NERVE_THROW_DIMENSION_ERROR(expected, actual)                                              \
    throw nerve::errors::DimensionError(__FILE__, __LINE__, __func__)                              \
        .expected(expected)                                                                        \
        .actual(actual)
#endif

#ifndef NERVE_THROW_GPU_OOM
#define NERVE_THROW_GPU_OOM(requested, available)                                                  \
    throw nerve::errors::GPUMemoryError(__FILE__, __LINE__, __func__)                              \
        .requestedBytes(requested)                                                                 \
        .availableBytes(available)                                                                 \
        .addSuggestion("Reduce batch size or use CPU implementation")
#endif

#ifndef NERVE_THROW_OOM
#define NERVE_THROW_OOM(requested, available)                                                      \
    throw nerve::errors::OutOfMemoryError(__FILE__, __LINE__, __func__)                            \
        .requestedBytes(requested)                                                                 \
        .availableBytes(available)                                                                 \
        .addSuggestion("Reduce memory usage or increase available memory")
#endif

#ifndef NERVE_THROW_INVALID_ARG
#define NERVE_THROW_INVALID_ARG(name, reason)                                                      \
    throw nerve::errors::InvalidArgumentError(__FILE__, __LINE__, __func__)                        \
        .setArgumentName(name)                                                                     \
        .setReason(reason)
#endif

#ifndef NERVE_THROW_BUDGET_EXCEEDED
#define NERVE_THROW_BUDGET_EXCEEDED(type, limit, usage)                                            \
    throw nerve::errors::BudgetExceededError(__FILE__, __LINE__, __func__)                         \
        .setBudgetType(type)                                                                       \
        .setBudgetLimit(limit)                                                                     \
        .setActualUsage(usage)
#endif

#ifndef NERVE_TRY
#define NERVE_TRY try
#endif

#ifndef NERVE_CATCH
#define NERVE_CATCH                                                                                \
    catch (const nerve::errors::NerveError &e)                                                     \
    {                                                                                              \
        throw;                                                                                     \
    }                                                                                              \
    catch (const std::exception &e)                                                                \
    {                                                                                              \
        throw nerve::errors::NerveError(e.what(), __FILE__, __LINE__, __func__);                   \
    }
#endif

#ifndef NERVE_RETHROW_WITH_CONTEXT
#define NERVE_RETHROW_WITH_CONTEXT(...)                                                            \
    catch (nerve::errors::NerveError & e)                                                          \
    {                                                                                              \
        e __VA_ARGS__;                                                                             \
        throw;                                                                                     \
    }
#endif

template <size_t N, size_t M>
struct CompileTimeShape
{
    static constexpr size_t rows = N;
    static constexpr size_t cols = M;
};

#ifndef NERVE_CHECK_SHAPE_CT
#define NERVE_CHECK_SHAPE_CT(tensor, N, M)                                                         \
    do                                                                                             \
    {                                                                                              \
        const auto &_shape = (tensor).shape();                                                     \
        if (_shape.size() < 2 || _shape[0] != N || _shape[1] != M)                                 \
        {                                                                                          \
            throw nerve::errors::ShapeMismatchError(__FILE__, __LINE__, __func__)                  \
                .setExpectedDims(N, M)                                                             \
                .setActualShape(_shape)                                                            \
                .setOperation(#tensor " compile-time shape check")                                 \
                .addSuggestion("Ensure tensor dimensions match compile-time expectations");        \
        }                                                                                          \
    } while (0)
#endif
