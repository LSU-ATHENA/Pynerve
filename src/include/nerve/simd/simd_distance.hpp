#pragma once
#include "nerve/simd/simd_base.hpp"

#include <cstring>

namespace nerve::simd
{

// Euclidean distance: sqrt(sum((a[i] - b[i])^2))
inline double simd_euclidean(const double *a, const double *b, std::size_t dim)
{
    simd_init();
    return std::sqrt(SIMD.sqdiff_sum(a, b, dim));
}

// Manhattan distance: sum(|a[i] - b[i]|), composed from dispatch primitives
// Uses sub + abs + reduce_sum in block-sized chunks
inline double simd_manhattan(const double *a, const double *b, std::size_t dim)
{
    simd_init();
    double sum = 0.0;
    std::size_t i = 0;
    // Process in blocks sized to the largest SIMD register (AVX-512 = 8 doubles);
    // the dispatch backends internally handle their natural width.
    while (i < dim)
    {
        const std::size_t remaining = dim - i;
        const std::size_t block = remaining < 8 ? remaining : 8;
        double buf[8];
        std::memcpy(buf, a + i, block * sizeof(double));
        SIMD.sub(buf, b + i, block); // buf = a - b
        SIMD.abs(buf, block);        // buf = |a - b|
        sum += SIMD.reduce_sum(buf, block);
        i += block;
    }
    return sum;
}

// Cosine distance: 1 - dot(a,b) / (|a| * |b|)
inline double simd_cosine(const double *a, const double *b, std::size_t dim)
{
    simd_init();
    double dot_val = SIMD.dot(a, b, dim);
    double na = SIMD.norm2(a, dim);
    double nb = SIMD.norm2(b, dim);
    if (na == 0.0 || nb == 0.0)
        return 1.0;
    double cos_sim = dot_val / (na * nb);
    if (cos_sim < -1.0)
        cos_sim = -1.0;
    if (cos_sim > 1.0)
        cos_sim = 1.0;
    return 1.0 - cos_sim;
}

// Float32 composite: sum(|a[i] - b[i]|) composed from dispatch primitives
// Uses sub_f32 + abs_f32 + reduce_sum_f32 in block-sized chunks
inline float simd_manhattan_f32(const float *a, const float *b, std::size_t n)
{
    simd_init();
    float sum = 0.0f;
    std::size_t i = 0;
    // Process in blocks sized to the largest SIMD register (AVX-512 = 16 floats);
    // the dispatch backends internally handle their natural width.
    while (i < n)
    {
        const std::size_t remaining = n - i;
        const std::size_t block = remaining < 16 ? remaining : 16;
        float buf[16];
        std::memcpy(buf, a + i, block * sizeof(float));
        SIMD.sub_f32(buf, b + i, block); // buf = a - b
        SIMD.abs_f32(buf, block);        // buf = |a - b|
        sum += SIMD.reduce_sum_f32(buf, block);
        i += block;
    }
    return sum;
}

// Float16 composite wrappers (compose dispatch primitives, no dedicated dispatch entries)
inline float simd_manhattan_f16(const half *a, const half *b, std::size_t n)
{
    float sum = 0.0f;
    for (std::size_t i = 0; i < n; ++i)
    {
        float d = static_cast<float>(a[i]) - static_cast<float>(b[i]);
        sum += (d < 0.0f) ? -d : d;
    }
    return sum;
}

inline float simd_euclidean_f16(const half *a, const half *b, std::size_t dim)
{
    simd_init();
    return std::sqrt(SIMD.sqdiff_sum_f16(a, b, dim));
}

inline float simd_cosine_f16(const half *a, const half *b, std::size_t dim)
{
    simd_init();
    float dot_val = SIMD.dot_f16(a, b, dim);
    float na = SIMD.norm2_f16(a, dim);
    float nb = SIMD.norm2_f16(b, dim);
    if (na == 0.0f || nb == 0.0f)
        return 1.0f;
    float cos_sim = dot_val / (na * nb);
    if (cos_sim < -1.0f)
        cos_sim = -1.0f;
    if (cos_sim > 1.0f)
        cos_sim = 1.0f;
    return 1.0f - cos_sim;
}

} // namespace nerve::simd
