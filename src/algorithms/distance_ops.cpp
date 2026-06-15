#include "nerve/algorithms/distance.hpp"
#include "nerve/algorithms/distance_c.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace
{
nerve_status_t nerve_status_from_exception(const std::exception &error) noexcept
{
    if (dynamic_cast<const std::invalid_argument *>(&error) != nullptr ||
        dynamic_cast<const std::length_error *>(&error) != nullptr ||
        dynamic_cast<const std::out_of_range *>(&error) != nullptr)
    {
        return NERVE_STATUS_INVALID_ARGUMENT;
    }
    if (dynamic_cast<const std::bad_alloc *>(&error) != nullptr)
    {
        return NERVE_STATUS_ALLOCATION_FAILED;
    }
    return NERVE_STATUS_RUNTIME_ERROR;
}
} // namespace

extern "C" nerve_status_t nerve_pairwise_distances_f32_status(const float *points, size_t n,
                                                              size_t dim, float *output) noexcept
{
    try
    {
        nerve::algorithms::nerve_pairwise_distances_f32(points, n, dim, output);
        return NERVE_STATUS_SUCCESS;
    }
    catch (const std::exception &error)
    {
        return nerve_status_from_exception(error);
    }
    catch (...)
    {
        return NERVE_STATUS_UNKNOWN_ERROR;
    }
}

extern "C" nerve_status_t nerve_pairwise_distances_f64_status(const double *points, size_t n,
                                                              size_t dim, double *output) noexcept
{
    try
    {
        nerve::algorithms::nerve_pairwise_distances_f64(points, n, dim, output);
        return NERVE_STATUS_SUCCESS;
    }
    catch (const std::exception &error)
    {
        return nerve_status_from_exception(error);
    }
    catch (...)
    {
        return NERVE_STATUS_UNKNOWN_ERROR;
    }
}

extern "C" nerve_status_t nerve_knn_f32_status(const float *points, size_t n, size_t dim, size_t k,
                                               float *out_distances, size_t *out_indices) noexcept
{
    try
    {
        nerve::algorithms::nerve_knn_f32(points, n, dim, k, out_distances, out_indices);
        return NERVE_STATUS_SUCCESS;
    }
    catch (const std::exception &error)
    {
        return nerve_status_from_exception(error);
    }
    catch (...)
    {
        return NERVE_STATUS_UNKNOWN_ERROR;
    }
}

extern "C" nerve_status_t nerve_knn_f64_status(const double *points, size_t n, size_t dim, size_t k,
                                               double *out_distances, size_t *out_indices) noexcept
{
    try
    {
        nerve::algorithms::nerve_knn_f64(points, n, dim, k, out_distances, out_indices);
        return NERVE_STATUS_SUCCESS;
    }
    catch (const std::exception &error)
    {
        return nerve_status_from_exception(error);
    }
    catch (...)
    {
        return NERVE_STATUS_UNKNOWN_ERROR;
    }
}
