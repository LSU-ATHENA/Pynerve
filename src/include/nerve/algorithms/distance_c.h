#pragma once

#include <stddef.h>

#ifdef __cplusplus
#define NERVE_DISTANCE_C_NOEXCEPT noexcept
extern "C"
{
#else
#define NERVE_DISTANCE_C_NOEXCEPT
#endif

    typedef enum nerve_status_t
    {
        NERVE_STATUS_SUCCESS = 0,
        NERVE_STATUS_INVALID_ARGUMENT = 1,
        NERVE_STATUS_ALLOCATION_FAILED = 2,
        NERVE_STATUS_RUNTIME_ERROR = 3,
        NERVE_STATUS_UNKNOWN_ERROR = 4
    } nerve_status_t;

    nerve_status_t nerve_pairwise_distances_f32_status(const float *points, size_t n, size_t dim,
                                                       float *output) NERVE_DISTANCE_C_NOEXCEPT;
    nerve_status_t nerve_pairwise_distances_f64_status(const double *points, size_t n, size_t dim,
                                                       double *output) NERVE_DISTANCE_C_NOEXCEPT;

    nerve_status_t nerve_knn_f32_status(const float *points, size_t n, size_t dim, size_t k,
                                        float *out_distances,
                                        size_t *out_indices) NERVE_DISTANCE_C_NOEXCEPT;
    nerve_status_t nerve_knn_f64_status(const double *points, size_t n, size_t dim, size_t k,
                                        double *out_distances,
                                        size_t *out_indices) NERVE_DISTANCE_C_NOEXCEPT;

#ifdef __cplusplus
}
#endif

#undef NERVE_DISTANCE_C_NOEXCEPT
