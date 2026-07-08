#pragma once
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <limits>
#include "nerve/simd/simd_base.hpp"

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define NERVE_HAS_NEON 1
#else
#define NERVE_HAS_NEON 0
#endif

#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#define NERVE_HAS_SVE 1
#else
#define NERVE_HAS_SVE 0
#endif

#if defined(__ARM_FEATURE_SVE2)
#define NERVE_HAS_SVE2 1
#else
#define NERVE_HAS_SVE2 0
#endif

// NEON primitives -- float64 (2-wide) and float32 (4-wide)
// Each function matches the signature used in SimdDispatchTable.

namespace nerve::simd::neon
{

// Memory

inline void memcpy(void *dst, const void *src, std::size_t bytes)
{
#if NERVE_HAS_NEON
    std::size_t i = 0;
    auto *d = static_cast<std::uint8_t *>(dst);
    const auto *s = static_cast<const std::uint8_t *>(src);
    // 16 bytes per NEON store/load (128-bit)
    for (; i + 16 <= bytes; i += 16)
    {
        uint8x16_t v = vld1q_u8(s + i);
        vst1q_u8(d + i, v);
    }
    for (; i < bytes; ++i)
        d[i] = s[i];
#else
    std::memcpy(dst, src, bytes);
#endif
}

inline void memset(void *dst, int value, std::size_t bytes)
{
#if NERVE_HAS_NEON
    std::size_t i = 0;
    auto *d = static_cast<std::uint8_t *>(dst);
    uint8x16_t v = vdupq_n_u8(static_cast<std::uint8_t>(value));
    for (; i + 16 <= bytes; i += 16)
        vst1q_u8(d + i, v);
    for (; i < bytes; ++i)
        d[i] = static_cast<std::uint8_t>(value);
#else
    std::memset(dst, value, bytes);
#endif
}

// Arithmetic (float64)

inline void add(double *a, const double *b, std::size_t n)
{
#if NERVE_HAS_NEON
    std::size_t i = 0;
    for (; i + 2 <= n; i += 2)
    {
        float64x2_t va = vld1q_f64(a + i);
        float64x2_t vb = vld1q_f64(b + i);
        vst1q_f64(a + i, vaddq_f64(va, vb));
    }
    for (; i < n; ++i)
        a[i] += b[i];
#else
    for (std::size_t i = 0; i < n; ++i)
        a[i] += b[i];
#endif
}

inline void sub(double *a, const double *b, std::size_t n)
{
#if NERVE_HAS_NEON
    std::size_t i = 0;
    for (; i + 2 <= n; i += 2)
    {
        float64x2_t va = vld1q_f64(a + i);
        float64x2_t vb = vld1q_f64(b + i);
        vst1q_f64(a + i, vsubq_f64(va, vb));
    }
    for (; i < n; ++i)
        a[i] -= b[i];
#else
    for (std::size_t i = 0; i < n; ++i)
        a[i] -= b[i];
#endif
}

inline void mul(double *a, const double *b, std::size_t n)
{
#if NERVE_HAS_NEON
    std::size_t i = 0;
    for (; i + 2 <= n; i += 2)
    {
        float64x2_t va = vld1q_f64(a + i);
        float64x2_t vb = vld1q_f64(b + i);
        vst1q_f64(a + i, vmulq_f64(va, vb));
    }
    for (; i < n; ++i)
        a[i] *= b[i];
#else
    for (std::size_t i = 0; i < n; ++i)
        a[i] *= b[i];
#endif
}

inline void scale(double *a, double alpha, std::size_t n)
{
#if NERVE_HAS_NEON
    std::size_t i = 0;
    float64x2_t v_alpha = vdupq_n_f64(alpha);
    for (; i + 2 <= n; i += 2)
    {
        float64x2_t va = vld1q_f64(a + i);
        vst1q_f64(a + i, vmulq_f64(va, v_alpha));
    }
    for (; i < n; ++i)
        a[i] *= alpha;
#else
    for (std::size_t i = 0; i < n; ++i)
        a[i] *= alpha;
#endif
}

inline void axpy(double alpha, const double *x, double *y, std::size_t n)
{
#if NERVE_HAS_NEON
    std::size_t i = 0;
    float64x2_t v_alpha = vdupq_n_f64(alpha);
    for (; i + 2 <= n; i += 2)
    {
        float64x2_t vx = vld1q_f64(x + i);
        float64x2_t vy = vld1q_f64(y + i);
        vst1q_f64(y + i, vfmaq_f64(vy, v_alpha, vx));
    }
    for (; i < n; ++i)
        y[i] += alpha * x[i];
#else
    for (std::size_t i = 0; i < n; ++i)
        y[i] += alpha * x[i];
#endif
}

inline void fmad(const double *a, const double *b, double *c, std::size_t n)
{
#if NERVE_HAS_NEON
    std::size_t i = 0;
    for (; i + 2 <= n; i += 2)
    {
        float64x2_t va = vld1q_f64(a + i);
        float64x2_t vb = vld1q_f64(b + i);
        float64x2_t vc = vld1q_f64(c + i);
        vst1q_f64(c + i, vfmaq_f64(vc, va, vb));
    }
    for (; i < n; ++i)
        c[i] += a[i] * b[i];
#else
    for (std::size_t i = 0; i < n; ++i)
        c[i] += a[i] * b[i];
#endif
}

// Reductions

inline double reduce_sum(const double *data, std::size_t n)
{
    double sum = 0.0;
#if NERVE_HAS_NEON
    std::size_t i = 0;
    float64x2_t vacc = vdupq_n_f64(0.0);
    for (; i + 2 <= n; i += 2)
    {
        float64x2_t v = vld1q_f64(data + i);
        vacc = vaddq_f64(vacc, v);
    }
    double tmp[2];
    vst1q_f64(tmp, vacc);
    sum = tmp[0] + tmp[1];
    for (; i < n; ++i)
        sum += data[i];
#else
    for (std::size_t i = 0; i < n; ++i)
        sum += data[i];
#endif
    return sum;
}

inline double reduce_max(const double *data, std::size_t n)
{
    if (n == 0) return 0.0;
    double m = data[0];
#if NERVE_HAS_NEON
    std::size_t i = 0;
    float64x2_t vmax = vdupq_n_f64(data[0]);
    for (; i + 2 <= n; i += 2)
    {
        float64x2_t v = vld1q_f64(data + i);
        vmax = vmaxq_f64(vmax, v);
    }
    double tmp[2];
    vst1q_f64(tmp, vmax);
    m = tmp[0] > tmp[1] ? tmp[0] : tmp[1];
    for (; i < n; ++i)
        if (data[i] > m) m = data[i];
#else
    for (std::size_t i = 1; i < n; ++i)
        if (data[i] > m) m = data[i];
#endif
    return m;
}

inline double dot(const double *a, const double *b, std::size_t n)
{
    double sum = 0.0;
#if NERVE_HAS_NEON
    std::size_t i = 0;
    float64x2_t vacc = vdupq_n_f64(0.0);
    for (; i + 2 <= n; i += 2)
    {
        float64x2_t va = vld1q_f64(a + i);
        float64x2_t vb = vld1q_f64(b + i);
        vacc = vfmaq_f64(vacc, va, vb);
    }
    double tmp[2];
    vst1q_f64(tmp, vacc);
    sum = tmp[0] + tmp[1];
    for (; i < n; ++i)
        sum += a[i] * b[i];
#else
    for (std::size_t i = 0; i < n; ++i)
        sum += a[i] * b[i];
#endif
    return sum;
}

inline double norm2(const double *vec, std::size_t n)
{
    double sum = 0.0;
#if NERVE_HAS_NEON
    std::size_t i = 0;
    float64x2_t vacc = vdupq_n_f64(0.0);
    for (; i + 2 <= n; i += 2)
    {
        float64x2_t v = vld1q_f64(vec + i);
        vacc = vfmaq_f64(vacc, v, v);
    }
    double tmp[2];
    vst1q_f64(tmp, vacc);
    sum = tmp[0] + tmp[1];
    for (; i < n; ++i)
        sum += vec[i] * vec[i];
#else
    for (std::size_t i = 0; i < n; ++i)
        sum += vec[i] * vec[i];
#endif
    return std::sqrt(sum);
}

inline double sqdiff_sum(const double *a, const double *b, std::size_t n)
{
    double sum = 0.0;
#if NERVE_HAS_NEON
    std::size_t i = 0;
    float64x2_t vacc = vdupq_n_f64(0.0);
    for (; i + 2 <= n; i += 2)
    {
        float64x2_t va = vld1q_f64(a + i);
        float64x2_t vb = vld1q_f64(b + i);
        float64x2_t diff = vsubq_f64(va, vb);
        vacc = vfmaq_f64(vacc, diff, diff);
    }
    double tmp[2];
    vst1q_f64(tmp, vacc);
    sum = tmp[0] + tmp[1];
    for (; i < n; ++i)
    {
        double d = a[i] - b[i];
        sum += d * d;
    }
#else
    for (std::size_t i = 0; i < n; ++i)
    {
        double d = a[i] - b[i];
        sum += d * d;
    }
#endif
    return sum;
}

// Element-wise

inline void relu(double *data, std::size_t n)
{
#if NERVE_HAS_NEON
    std::size_t i = 0;
    float64x2_t vzero = vdupq_n_f64(0.0);
    for (; i + 2 <= n; i += 2)
    {
        float64x2_t v = vld1q_f64(data + i);
        vst1q_f64(data + i, vmaxq_f64(v, vzero));
    }
    for (; i < n; ++i)
        if (data[i] < 0.0) data[i] = 0.0;
#else
    for (std::size_t i = 0; i < n; ++i)
        if (data[i] < 0.0) data[i] = 0.0;
#endif
}

inline void clamp(double *data, double lo, double hi, std::size_t n)
{
#if NERVE_HAS_NEON
    std::size_t i = 0;
    float64x2_t vlo = vdupq_n_f64(lo);
    float64x2_t vhi = vdupq_n_f64(hi);
    for (; i + 2 <= n; i += 2)
    {
        float64x2_t v = vld1q_f64(data + i);
        v = vmaxq_f64(v, vlo);
        v = vminq_f64(v, vhi);
        vst1q_f64(data + i, v);
    }
    for (; i < n; ++i)
    {
        if (data[i] < lo) data[i] = lo;
        if (data[i] > hi) data[i] = hi;
    }
#else
    for (std::size_t i = 0; i < n; ++i)
    {
        if (data[i] < lo) data[i] = lo;
        if (data[i] > hi) data[i] = hi;
    }
#endif
}

// Float32 arithmetic

inline void fmad_f32(const float *a, const float *b, float *c, std::size_t n)
{
#if NERVE_HAS_NEON
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4)
    {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t vc = vld1q_f32(c + i);
        vst1q_f32(c + i, vfmaq_f32(vc, va, vb));
    }
    for (; i < n; ++i)
        c[i] += a[i] * b[i];
#else
    for (std::size_t i = 0; i < n; ++i)
        c[i] += a[i] * b[i];
#endif
}

// Float32 variants (4-wide)

inline void add_f32(float *a, const float *b, std::size_t n)
{
#if NERVE_HAS_NEON
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4)
    {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(a + i, vaddq_f32(va, vb));
    }
    for (; i < n; ++i)
        a[i] += b[i];
#else
    for (std::size_t i = 0; i < n; ++i)
        a[i] += b[i];
#endif
}

inline void sub_f32(float *a, const float *b, std::size_t n)
{
#if NERVE_HAS_NEON
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4)
    {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(a + i, vsubq_f32(va, vb));
    }
    for (; i < n; ++i)
        a[i] -= b[i];
#else
    for (std::size_t i = 0; i < n; ++i)
        a[i] -= b[i];
#endif
}

inline void mul_f32(float *a, const float *b, std::size_t n)
{
#if NERVE_HAS_NEON
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4)
    {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(a + i, vmulq_f32(va, vb));
    }
    for (; i < n; ++i)
        a[i] *= b[i];
#else
    for (std::size_t i = 0; i < n; ++i)
        a[i] *= b[i];
#endif
}

inline void scale_f32(float *a, float alpha, std::size_t n)
{
#if NERVE_HAS_NEON
    std::size_t i = 0;
    float32x4_t v_alpha = vdupq_n_f32(alpha);
    for (; i + 4 <= n; i += 4)
    {
        float32x4_t va = vld1q_f32(a + i);
        vst1q_f32(a + i, vmulq_f32(va, v_alpha));
    }
    for (; i < n; ++i)
        a[i] *= alpha;
#else
    for (std::size_t i = 0; i < n; ++i)
        a[i] *= alpha;
#endif
}

inline void axpy_f32(float alpha, const float *x, float *y, std::size_t n)
{
#if NERVE_HAS_NEON
    std::size_t i = 0;
    float32x4_t v_alpha = vdupq_n_f32(alpha);
    for (; i + 4 <= n; i += 4)
    {
        float32x4_t vx = vld1q_f32(x + i);
        float32x4_t vy = vld1q_f32(y + i);
        vst1q_f32(y + i, vfmaq_f32(vy, v_alpha, vx));
    }
    for (; i < n; ++i)
        y[i] += alpha * x[i];
#else
    for (std::size_t i = 0; i < n; ++i)
        y[i] += alpha * x[i];
#endif
}

inline float reduce_sum_f32(const float *data, std::size_t n)
{
    float sum = 0.0f;
#if NERVE_HAS_NEON
    std::size_t i = 0;
    float32x4_t vacc = vdupq_n_f32(0.0f);
    for (; i + 4 <= n; i += 4)
    {
        float32x4_t v = vld1q_f32(data + i);
        vacc = vaddq_f32(vacc, v);
    }
    float tmp[4];
    vst1q_f32(tmp, vacc);
    sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    for (; i < n; ++i)
        sum += data[i];
#else
    for (std::size_t i = 0; i < n; ++i)
        sum += data[i];
#endif
    return sum;
}

inline float reduce_max_f32(const float *data, std::size_t n)
{
    if (n == 0) return 0.0f;
    float m = data[0];
#if NERVE_HAS_NEON
    std::size_t i = 0;
    float32x4_t vmax = vdupq_n_f32(data[0]);
    for (; i + 4 <= n; i += 4)
    {
        float32x4_t v = vld1q_f32(data + i);
        vmax = vmaxq_f32(vmax, v);
    }
    float tmp[4];
    vst1q_f32(tmp, vmax);
    m = tmp[0];
    for (int k = 1; k < 4; ++k) if (tmp[k] > m) m = tmp[k];
    for (; i < n; ++i)
        if (data[i] > m) m = data[i];
#else
    for (std::size_t i = 1; i < n; ++i)
        if (data[i] > m) m = data[i];
#endif
    return m;
}

inline float reduce_min_f32(const float *data, std::size_t n)
{
    if (n == 0) return 0.0f;
    float m = data[0];
#if NERVE_HAS_NEON
    std::size_t i = 0;
    float32x4_t vmin = vdupq_n_f32(data[0]);
    for (; i + 4 <= n; i += 4)
    {
        float32x4_t v = vld1q_f32(data + i);
        vmin = vminq_f32(vmin, v);
    }
    float tmp[4];
    vst1q_f32(tmp, vmin);
    m = tmp[0];
    for (int k = 1; k < 4; ++k) if (tmp[k] < m) m = tmp[k];
    for (; i < n; ++i)
        if (data[i] < m) m = data[i];
#else
    for (std::size_t i = 1; i < n; ++i)
        if (data[i] < m) m = data[i];
#endif
    return m;
}

inline float dot_f32(const float *a, const float *b, std::size_t n)
{
    float sum = 0.0f;
#if NERVE_HAS_NEON
    std::size_t i = 0;
    float32x4_t vacc = vdupq_n_f32(0.0f);
    for (; i + 4 <= n; i += 4)
    {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vacc = vfmaq_f32(vacc, va, vb);
    }
    float tmp[4];
    vst1q_f32(tmp, vacc);
    sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    for (; i < n; ++i)
        sum += a[i] * b[i];
#else
    for (std::size_t i = 0; i < n; ++i)
        sum += a[i] * b[i];
#endif
    return sum;
}

inline float norm2_f32(const float *vec, std::size_t n)
{
    float sum = 0.0f;
#if NERVE_HAS_NEON
    std::size_t i = 0;
    float32x4_t vacc = vdupq_n_f32(0.0f);
    for (; i + 4 <= n; i += 4)
    {
        float32x4_t v = vld1q_f32(vec + i);
        vacc = vfmaq_f32(vacc, v, v);
    }
    float tmp[4];
    vst1q_f32(tmp, vacc);
    sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    for (; i < n; ++i)
        sum += vec[i] * vec[i];
#else
    for (std::size_t i = 0; i < n; ++i)
        sum += vec[i] * vec[i];
#endif
    return std::sqrt(sum);
}

// Float32 distance primitives

inline float sqdiff_sum_f32(const float *a, const float *b, std::size_t n)
{
    float sum = 0.0f;
#if NERVE_HAS_NEON
    std::size_t i = 0;
    float32x4_t vacc = vdupq_n_f32(0.0f);
    for (; i + 4 <= n; i += 4)
    {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t diff = vsubq_f32(va, vb);
        vacc = vfmaq_f32(vacc, diff, diff);
    }
    float tmp[4];
    vst1q_f32(tmp, vacc);
    sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    for (; i < n; ++i)
    {
        float d = a[i] - b[i];
        sum += d * d;
    }
#else
    for (std::size_t i = 0; i < n; ++i)
    {
        float d = a[i] - b[i];
        sum += d * d;
    }
#endif
    return sum;
}

inline float euclidean_f32(const float *a, const float *b, std::size_t n)
{
    return std::sqrt(sqdiff_sum_f32(a, b, n));
}

inline float cosine_f32(const float *a, const float *b, std::size_t n)
{
    float dot_val = dot_f32(a, b, n);
    float na = norm2_f32(a, n);
    float nb = norm2_f32(b, n);
    if (na == 0.0f || nb == 0.0f) return 1.0f;
    float cos_sim = dot_val / (na * nb);
    if (cos_sim < -1.0f) cos_sim = -1.0f;
    if (cos_sim > 1.0f)  cos_sim = 1.0f;
    return 1.0f - cos_sim;
}

// Float32 element-wise unary

inline void neg_f32(float *data, std::size_t n)
{
#if NERVE_HAS_NEON
    std::size_t i = 0;
    float32x4_t vzero = vdupq_n_f32(0.0f);
    for (; i + 4 <= n; i += 4)
    {
        float32x4_t v = vld1q_f32(data + i);
        vst1q_f32(data + i, vsubq_f32(vzero, v));
    }
    for (; i < n; ++i)
        data[i] = -data[i];
#else
    for (std::size_t i = 0; i < n; ++i)
        data[i] = -data[i];
#endif
}

inline void sqrt_f32(float *data, std::size_t n)
{
#if NERVE_HAS_NEON
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4)
    {
        float32x4_t v = vld1q_f32(data + i);
        vst1q_f32(data + i, vsqrtq_f32(v));
    }
    for (; i < n; ++i)
        data[i] = std::sqrt(data[i]);
#else
    for (std::size_t i = 0; i < n; ++i)
        data[i] = std::sqrt(data[i]);
#endif
}

inline void exp_f32(float *data, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        data[i] = std::exp(data[i]);
}

inline void log_f32(float *data, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        data[i] = std::log(data[i]);
}

inline void sigmoid_f32(float *data, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        data[i] = 1.0f / (1.0f + std::exp(-data[i]));
}

inline void tanh_f32(float *data, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        data[i] = std::tanh(data[i]);
}

// Float32 element-wise

inline void abs_f32(float *data, std::size_t n)
{
#if NERVE_HAS_NEON
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4)
    {
        float32x4_t v = vld1q_f32(data + i);
        vst1q_f32(data + i, vabsq_f32(v));
    }
    for (; i < n; ++i)
        data[i] = std::abs(data[i]);
#else
    for (std::size_t i = 0; i < n; ++i)
        data[i] = std::abs(data[i]);
#endif
}

inline void relu_f32(float *data, std::size_t n)
{
#if NERVE_HAS_NEON
    std::size_t i = 0;
    float32x4_t vzero = vdupq_n_f32(0.0f);
    for (; i + 4 <= n; i += 4)
    {
        float32x4_t v = vld1q_f32(data + i);
        vst1q_f32(data + i, vmaxq_f32(v, vzero));
    }
    for (; i < n; ++i)
        if (data[i] < 0.0f) data[i] = 0.0f;
#else
    for (std::size_t i = 0; i < n; ++i)
        if (data[i] < 0.0f) data[i] = 0.0f;
#endif
}

inline void min_f32(float *a, const float *b, std::size_t n)
{
#if NERVE_HAS_NEON
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4)
    {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(a + i, vminq_f32(va, vb));
    }
    for (; i < n; ++i)
        a[i] = a[i] < b[i] ? a[i] : b[i];
#else
    for (std::size_t i = 0; i < n; ++i)
        a[i] = a[i] < b[i] ? a[i] : b[i];
#endif
}

inline void max_f32(float *a, const float *b, std::size_t n)
{
#if NERVE_HAS_NEON
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4)
    {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(a + i, vmaxq_f32(va, vb));
    }
    for (; i < n; ++i)
        a[i] = a[i] > b[i] ? a[i] : b[i];
#else
    for (std::size_t i = 0; i < n; ++i)
        a[i] = a[i] > b[i] ? a[i] : b[i];
#endif
}

inline void clamp_f32(float *data, float lo, float hi, std::size_t n)
{
#if NERVE_HAS_NEON
    std::size_t i = 0;
    float32x4_t vlo = vdupq_n_f32(lo);
    float32x4_t vhi = vdupq_n_f32(hi);
    for (; i + 4 <= n; i += 4)
    {
        float32x4_t v = vld1q_f32(data + i);
        v = vmaxq_f32(v, vlo);
        v = vminq_f32(v, vhi);
        vst1q_f32(data + i, v);
    }
    for (; i < n; ++i)
    {
        if (data[i] < lo) data[i] = lo;
        if (data[i] > hi) data[i] = hi;
    }
#else
    for (std::size_t i = 0; i < n; ++i)
    {
        if (data[i] < lo) data[i] = lo;
        if (data[i] > hi) data[i] = hi;
    }
#endif
}

} // namespace nerve::simd::neon

// SVE primitives -- vector-length agnostic (128-2048 bit)

namespace nerve::simd::sve
{

#if NERVE_HAS_SVE

inline void add(double *a, const double *b, std::size_t n)
{
    std::size_t i = 0;
    svbool_t pg = svwhilelt_b64(i, n);
    while (svptest_any(svptrue_b64(), pg))
    {
        svfloat64_t va = svld1_f64(pg, a + i);
        svfloat64_t vb = svld1_f64(pg, b + i);
        svst1_f64(pg, a + i, svadd_f64_m(pg, va, vb));
        i += svcntd();
        pg = svwhilelt_b64(i, n);
    }
}

inline double reduce_sum(const double *data, std::size_t n)
{
    std::size_t i = 0;
    svfloat64_t vacc = svdup_n_f64(0.0);
    svbool_t pg = svwhilelt_b64(i, n);
    while (svptest_any(svptrue_b64(), pg))
    {
        svfloat64_t v = svld1_f64(pg, data + i);
        vacc = svadd_f64_m(pg, vacc, v);
        i += svcntd();
        pg = svwhilelt_b64(i, n);
    }
    return svaddv_f64(svptrue_b64(), vacc);
}

inline double dot(const double *a, const double *b, std::size_t n)
{
    std::size_t i = 0;
    svfloat64_t vacc = svdup_n_f64(0.0);
    svbool_t pg = svwhilelt_b64(i, n);
    while (svptest_any(svptrue_b64(), pg))
    {
        svfloat64_t va = svld1_f64(pg, a + i);
        svfloat64_t vb = svld1_f64(pg, b + i);
        vacc = svmla_f64_m(pg, vacc, va, vb);
        i += svcntd();
        pg = svwhilelt_b64(i, n);
    }
    return svaddv_f64(svptrue_b64(), vacc);
}

inline double norm2(const double *vec, std::size_t n)
{
    std::size_t i = 0;
    svfloat64_t vacc = svdup_n_f64(0.0);
    svbool_t pg = svwhilelt_b64(i, n);
    while (svptest_any(svptrue_b64(), pg))
    {
        svfloat64_t v = svld1_f64(pg, vec + i);
        vacc = svmla_f64_m(pg, vacc, v, v);
        i += svcntd();
        pg = svwhilelt_b64(i, n);
    }
    double sum = svaddv_f64(svptrue_b64(), vacc);
    return std::sqrt(sum);
}

// Float32 SVE primitives (vector-length agnostic)

inline void add_f32(float *a, const float *b, std::size_t n)
{
#if NERVE_HAS_SVE
    std::size_t i = 0;
    svbool_t pg = svwhilelt_b32(i, n);
    while (svptest_any(svptrue_b32(), pg))
    {
        svfloat32_t va = svld1_f32(pg, a + i);
        svfloat32_t vb = svld1_f32(pg, b + i);
        svst1_f32(pg, a + i, svadd_f32_m(pg, va, vb));
        i += svcntw();
        pg = svwhilelt_b32(i, n);
    }
#else
    for (std::size_t i = 0; i < n; ++i)
        a[i] += b[i];
#endif
}

inline void sub_f32(float *a, const float *b, std::size_t n)
{
#if NERVE_HAS_SVE
    std::size_t i = 0;
    svbool_t pg = svwhilelt_b32(i, n);
    while (svptest_any(svptrue_b32(), pg))
    {
        svfloat32_t va = svld1_f32(pg, a + i);
        svfloat32_t vb = svld1_f32(pg, b + i);
        svst1_f32(pg, a + i, svsub_f32_m(pg, va, vb));
        i += svcntw();
        pg = svwhilelt_b32(i, n);
    }
#else
    for (std::size_t i = 0; i < n; ++i)
        a[i] -= b[i];
#endif
}

inline void mul_f32(float *a, const float *b, std::size_t n)
{
#if NERVE_HAS_SVE
    std::size_t i = 0;
    svbool_t pg = svwhilelt_b32(i, n);
    while (svptest_any(svptrue_b32(), pg))
    {
        svfloat32_t va = svld1_f32(pg, a + i);
        svfloat32_t vb = svld1_f32(pg, b + i);
        svst1_f32(pg, a + i, svmul_f32_m(pg, va, vb));
        i += svcntw();
        pg = svwhilelt_b32(i, n);
    }
#else
    for (std::size_t i = 0; i < n; ++i)
        a[i] *= b[i];
#endif
}

inline void scale_f32(float *a, float alpha, std::size_t n)
{
#if NERVE_HAS_SVE
    std::size_t i = 0;
    svfloat32_t va = svdup_n_f32(alpha);
    svbool_t pg = svwhilelt_b32(i, n);
    while (svptest_any(svptrue_b32(), pg))
    {
        svfloat32_t v = svld1_f32(pg, a + i);
        svst1_f32(pg, a + i, svmul_f32_m(pg, v, va));
        i += svcntw();
        pg = svwhilelt_b32(i, n);
    }
#else
    for (std::size_t i = 0; i < n; ++i)
        a[i] *= alpha;
#endif
}

inline void axpy_f32(float alpha, const float *x, float *y, std::size_t n)
{
#if NERVE_HAS_SVE
    std::size_t i = 0;
    svfloat32_t va = svdup_n_f32(alpha);
    svbool_t pg = svwhilelt_b32(i, n);
    while (svptest_any(svptrue_b32(), pg))
    {
        svfloat32_t vx = svld1_f32(pg, x + i);
        svfloat32_t vy = svld1_f32(pg, y + i);
        svst1_f32(pg, y + i, svmla_f32_m(pg, vy, va, vx));
        i += svcntw();
        pg = svwhilelt_b32(i, n);
    }
#else
    for (std::size_t i = 0; i < n; ++i)
        y[i] += alpha * x[i];
#endif
}

inline float reduce_sum_f32(const float *data, std::size_t n)
{
    float sum = 0.0f;
#if NERVE_HAS_SVE
    std::size_t i = 0;
    svfloat32_t vacc = svdup_n_f32(0.0f);
    svbool_t pg = svwhilelt_b32(i, n);
    while (svptest_any(svptrue_b32(), pg))
    {
        svfloat32_t v = svld1_f32(pg, data + i);
        vacc = svadd_f32_m(pg, vacc, v);
        i += svcntw();
        pg = svwhilelt_b32(i, n);
    }
    sum = svaddv_f32(svptrue_b32(), vacc);
#else
    for (std::size_t i = 0; i < n; ++i)
        sum += data[i];
#endif
    return sum;
}

inline float reduce_max_f32(const float *data, std::size_t n)
{
    if (n == 0) return 0.0f;
    float m = data[0];
#if NERVE_HAS_SVE
    std::size_t i = 0;
    svfloat32_t vmax = svdup_n_f32(-std::numeric_limits<float>::infinity());
    svbool_t pg = svwhilelt_b32(i, n);
    while (svptest_any(svptrue_b32(), pg))
    {
        svfloat32_t v = svld1_f32(pg, data + i);
        vmax = svmax_f32_m(pg, vmax, v);
        i += svcntw();
        pg = svwhilelt_b32(i, n);
    }
    m = svmaxv_f32(svptrue_b32(), vmax);
#else
    for (std::size_t i = 1; i < n; ++i)
        if (data[i] > m) m = data[i];
#endif
    return m;
}

inline float dot_f32(const float *a, const float *b, std::size_t n)
{
    float sum = 0.0f;
#if NERVE_HAS_SVE
    std::size_t i = 0;
    svfloat32_t vacc = svdup_n_f32(0.0f);
    svbool_t pg = svwhilelt_b32(i, n);
    while (svptest_any(svptrue_b32(), pg))
    {
        svfloat32_t va = svld1_f32(pg, a + i);
        svfloat32_t vb = svld1_f32(pg, b + i);
        vacc = svmla_f32_m(pg, vacc, va, vb);
        i += svcntw();
        pg = svwhilelt_b32(i, n);
    }
    sum = svaddv_f32(svptrue_b32(), vacc);
#else
    for (std::size_t i = 0; i < n; ++i)
        sum += a[i] * b[i];
#endif
    return sum;
}

inline float norm2_f32(const float *vec, std::size_t n)
{
    float sum = 0.0f;
#if NERVE_HAS_SVE
    std::size_t i = 0;
    svfloat32_t vacc = svdup_n_f32(0.0f);
    svbool_t pg = svwhilelt_b32(i, n);
    while (svptest_any(svptrue_b32(), pg))
    {
        svfloat32_t v = svld1_f32(pg, vec + i);
        vacc = svmla_f32_m(pg, vacc, v, v);
        i += svcntw();
        pg = svwhilelt_b32(i, n);
    }
    sum = svaddv_f32(svptrue_b32(), vacc);
#else
    for (std::size_t i = 0; i < n; ++i)
        sum += vec[i] * vec[i];
#endif
    return std::sqrt(sum);
}

// SVE float32 arithmetic

inline void fmad_f32(const float *a, const float *b, float *c, std::size_t n)
{
#if NERVE_HAS_SVE
    std::size_t i = 0;
    svbool_t pg = svwhilelt_b32(i, n);
    while (svptest_any(svptrue_b32(), pg))
    {
        svfloat32_t va = svld1_f32(pg, a + i);
        svfloat32_t vb = svld1_f32(pg, b + i);
        svfloat32_t vc = svld1_f32(pg, c + i);
        svst1_f32(pg, c + i, svmla_f32_m(pg, vc, va, vb));
        i += svcntw();
        pg = svwhilelt_b32(i, n);
    }
#else
    for (std::size_t i = 0; i < n; ++i)
        c[i] += a[i] * b[i];
#endif
}

// SVE float32 reductions

inline float reduce_min_f32(const float *data, std::size_t n)
{
    if (n == 0) return 0.0f;
    float m = data[0];
#if NERVE_HAS_SVE
    std::size_t i = 0;
    svfloat32_t vmin = svdup_n_f32(std::numeric_limits<float>::infinity());
    svbool_t pg = svwhilelt_b32(i, n);
    while (svptest_any(svptrue_b32(), pg))
    {
        svfloat32_t v = svld1_f32(pg, data + i);
        vmin = svmin_f32_m(pg, vmin, v);
        i += svcntw();
        pg = svwhilelt_b32(i, n);
    }
    m = svminv_f32(svptrue_b32(), vmin);
#else
    for (std::size_t i = 1; i < n; ++i)
        if (data[i] < m) m = data[i];
#endif
    return m;
}

// SVE float32 element-wise unary

inline void neg_f32(float *data, std::size_t n)
{
#if NERVE_HAS_SVE
    std::size_t i = 0;
    svbool_t pg = svwhilelt_b32(i, n);
    while (svptest_any(svptrue_b32(), pg))
    {
        svfloat32_t v = svld1_f32(pg, data + i);
        svst1_f32(pg, data + i, svneg_f32_m(pg, v, v));
        i += svcntw();
        pg = svwhilelt_b32(i, n);
    }
#else
    for (std::size_t i = 0; i < n; ++i)
        data[i] = -data[i];
#endif
}

inline void sqrt_f32(float *data, std::size_t n)
{
#if NERVE_HAS_SVE
    std::size_t i = 0;
    svbool_t pg = svwhilelt_b32(i, n);
    while (svptest_any(svptrue_b32(), pg))
    {
        svfloat32_t v = svld1_f32(pg, data + i);
        svst1_f32(pg, data + i, svsqrt_f32_m(pg, v, v));
        i += svcntw();
        pg = svwhilelt_b32(i, n);
    }
#else
    for (std::size_t i = 0; i < n; ++i)
        data[i] = std::sqrt(data[i]);
#endif
}

inline void exp_f32(float *data, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        data[i] = std::exp(data[i]);
}

inline void log_f32(float *data, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        data[i] = std::log(data[i]);
}

inline void sigmoid_f32(float *data, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        data[i] = 1.0f / (1.0f + std::exp(-data[i]));
}

inline void tanh_f32(float *data, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        data[i] = std::tanh(data[i]);
}

// SVE float32 distance primitives

inline float sqdiff_sum_f32(const float *a, const float *b, std::size_t n)
{
    float sum = 0.0f;
#if NERVE_HAS_SVE
    std::size_t i = 0;
    svfloat32_t vacc = svdup_n_f32(0.0f);
    svbool_t pg = svwhilelt_b32(i, n);
    while (svptest_any(svptrue_b32(), pg))
    {
        svfloat32_t va = svld1_f32(pg, a + i);
        svfloat32_t vb = svld1_f32(pg, b + i);
        svfloat32_t diff = svsub_f32_m(pg, va, vb);
        vacc = svmla_f32_m(pg, vacc, diff, diff);
        i += svcntw();
        pg = svwhilelt_b32(i, n);
    }
    sum = svaddv_f32(svptrue_b32(), vacc);
#else
    for (std::size_t i = 0; i < n; ++i)
    {
        float d = a[i] - b[i];
        sum += d * d;
    }
#endif
    return sum;
}

inline float euclidean_f32(const float *a, const float *b, std::size_t n)
{
    return std::sqrt(sqdiff_sum_f32(a, b, n));
}

inline float cosine_f32(const float *a, const float *b, std::size_t n)
{
    float dot_val = dot_f32(a, b, n);
    float na = norm2_f32(a, n);
    float nb = norm2_f32(b, n);
    if (na == 0.0f || nb == 0.0f) return 1.0f;
    float cos_sim = dot_val / (na * nb);
    if (cos_sim < -1.0f) cos_sim = -1.0f;
    if (cos_sim > 1.0f)  cos_sim = 1.0f;
    return 1.0f - cos_sim;
}

// SVE float32 element-wise

inline void abs_f32(float *data, std::size_t n)
{
#if NERVE_HAS_SVE
    std::size_t i = 0;
    svbool_t pg = svwhilelt_b32(i, n);
    while (svptest_any(svptrue_b32(), pg))
    {
        svfloat32_t v = svld1_f32(pg, data + i);
        svst1_f32(pg, data + i, svabs_f32_m(pg, v, v));
        i += svcntw();
        pg = svwhilelt_b32(i, n);
    }
#else
    for (std::size_t i = 0; i < n; ++i)
        data[i] = std::abs(data[i]);
#endif
}

inline void relu_f32(float *data, std::size_t n)
{
#if NERVE_HAS_SVE
    std::size_t i = 0;
    svfloat32_t vzero = svdup_n_f32(0.0f);
    svbool_t pg = svwhilelt_b32(i, n);
    while (svptest_any(svptrue_b32(), pg))
    {
        svfloat32_t v = svld1_f32(pg, data + i);
        svst1_f32(pg, data + i, svmax_f32_m(pg, v, vzero));
        i += svcntw();
        pg = svwhilelt_b32(i, n);
    }
#else
    for (std::size_t i = 0; i < n; ++i)
        if (data[i] < 0.0f) data[i] = 0.0f;
#endif
}

inline void min_f32(float *a, const float *b, std::size_t n)
{
#if NERVE_HAS_SVE
    std::size_t i = 0;
    svbool_t pg = svwhilelt_b32(i, n);
    while (svptest_any(svptrue_b32(), pg))
    {
        svfloat32_t va = svld1_f32(pg, a + i);
        svfloat32_t vb = svld1_f32(pg, b + i);
        svst1_f32(pg, a + i, svmin_f32_m(pg, va, vb));
        i += svcntw();
        pg = svwhilelt_b32(i, n);
    }
#else
    for (std::size_t i = 0; i < n; ++i)
        a[i] = a[i] < b[i] ? a[i] : b[i];
#endif
}

inline void max_f32(float *a, const float *b, std::size_t n)
{
#if NERVE_HAS_SVE
    std::size_t i = 0;
    svbool_t pg = svwhilelt_b32(i, n);
    while (svptest_any(svptrue_b32(), pg))
    {
        svfloat32_t va = svld1_f32(pg, a + i);
        svfloat32_t vb = svld1_f32(pg, b + i);
        svst1_f32(pg, a + i, svmax_f32_m(pg, va, vb));
        i += svcntw();
        pg = svwhilelt_b32(i, n);
    }
#else
    for (std::size_t i = 0; i < n; ++i)
        a[i] = a[i] > b[i] ? a[i] : b[i];
#endif
}

inline void clamp_f32(float *data, float lo, float hi, std::size_t n)
{
#if NERVE_HAS_SVE
    std::size_t i = 0;
    svfloat32_t vlo = svdup_n_f32(lo);
    svfloat32_t vhi = svdup_n_f32(hi);
    svbool_t pg = svwhilelt_b32(i, n);
    while (svptest_any(svptrue_b32(), pg))
    {
        svfloat32_t v = svld1_f32(pg, data + i);
        v = svmax_f32_m(pg, v, vlo);
        v = svmin_f32_m(pg, v, vhi);
        svst1_f32(pg, data + i, v);
        i += svcntw();
        pg = svwhilelt_b32(i, n);
    }
#else
    for (std::size_t i = 0; i < n; ++i)
    {
        if (data[i] < lo) data[i] = lo;
        if (data[i] > hi) data[i] = hi;
    }
#endif
}

#endif // NERVE_HAS_SVE

} // namespace nerve::simd::sve

// Float16 NEON primitives (inside nerve::simd::neon namespace)

namespace nerve::simd::neon
{

inline void add_f16(half *a, const half *b, std::size_t n)
{
#if NERVE_HAS_NEON
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4)
    {
        float16x4_t vah = vld1_f16(reinterpret_cast<const float16_t *>(a + i));
        float16x4_t vbh = vld1_f16(reinterpret_cast<const float16_t *>(b + i));
        float32x4_t va = vcvt_f32_f16(vah);
        float32x4_t vb = vcvt_f32_f16(vbh);
        float32x4_t vr = vaddq_f32(va, vb);
        float16x4_t vrh = vcvt_f16_f32(vr);
        vst1_f16(reinterpret_cast<float16_t *>(a + i), vrh);
    }
    for (; i < n; ++i) a[i] = float_to_half(half_to_float(a[i]) + half_to_float(b[i]));
#else
    for (std::size_t i = 0; i < n; ++i) a[i] = float_to_half(half_to_float(a[i]) + half_to_float(b[i]));
#endif
}

inline void sub_f16(half *a, const half *b, std::size_t n)
{
#if NERVE_HAS_NEON
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4)
    {
        float16x4_t vah = vld1_f16(reinterpret_cast<const float16_t *>(a + i));
        float16x4_t vbh = vld1_f16(reinterpret_cast<const float16_t *>(b + i));
        float32x4_t va = vcvt_f32_f16(vah);
        float32x4_t vb = vcvt_f32_f16(vbh);
        float32x4_t vr = vsubq_f32(va, vb);
        float16x4_t vrh = vcvt_f16_f32(vr);
        vst1_f16(reinterpret_cast<float16_t *>(a + i), vrh);
    }
    for (; i < n; ++i) a[i] = float_to_half(half_to_float(a[i]) - half_to_float(b[i]));
#else
    for (std::size_t i = 0; i < n; ++i) a[i] = float_to_half(half_to_float(a[i]) - half_to_float(b[i]));
#endif
}

inline void mul_f16(half *a, const half *b, std::size_t n)
{
#if NERVE_HAS_NEON
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4)
    {
        float16x4_t vah = vld1_f16(reinterpret_cast<const float16_t *>(a + i));
        float16x4_t vbh = vld1_f16(reinterpret_cast<const float16_t *>(b + i));
        float32x4_t va = vcvt_f32_f16(vah);
        float32x4_t vb = vcvt_f32_f16(vbh);
        float32x4_t vr = vmulq_f32(va, vb);
        float16x4_t vrh = vcvt_f16_f32(vr);
        vst1_f16(reinterpret_cast<float16_t *>(a + i), vrh);
    }
    for (; i < n; ++i) a[i] = float_to_half(half_to_float(a[i]) * half_to_float(b[i]));
#else
    for (std::size_t i = 0; i < n; ++i) a[i] = float_to_half(half_to_float(a[i]) * half_to_float(b[i]));
#endif
}

inline void scale_f16(half *a, half alpha, std::size_t n)
{
    float fa = half_to_float(alpha);
#if NERVE_HAS_NEON
    float32x4_t vfa = vdupq_n_f32(fa);
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4)
    {
        float16x4_t vah = vld1_f16(reinterpret_cast<const float16_t *>(a + i));
        float32x4_t va = vcvt_f32_f16(vah);
        float32x4_t vr = vmulq_f32(va, vfa);
        float16x4_t vrh = vcvt_f16_f32(vr);
        vst1_f16(reinterpret_cast<float16_t *>(a + i), vrh);
    }
    for (; i < n; ++i) a[i] = float_to_half(half_to_float(a[i]) * fa);
#else
    for (std::size_t i = 0; i < n; ++i) a[i] = float_to_half(half_to_float(a[i]) * fa);
#endif
}

inline void axpy_f16(half alpha, const half *x, half *y, std::size_t n)
{
    float fa = half_to_float(alpha);
#if NERVE_HAS_NEON
    float32x4_t vfa = vdupq_n_f32(fa);
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4)
    {
        float16x4_t vxh = vld1_f16(reinterpret_cast<const float16_t *>(x + i));
        float16x4_t vyh = vld1_f16(reinterpret_cast<const float16_t *>(y + i));
        float32x4_t vx = vcvt_f32_f16(vxh);
        float32x4_t vy = vcvt_f32_f16(vyh);
        float32x4_t vr = vfmaq_f32(vy, vfa, vx);
        float16x4_t vrh = vcvt_f16_f32(vr);
        vst1_f16(reinterpret_cast<float16_t *>(y + i), vrh);
    }
    for (; i < n; ++i) y[i] = float_to_half(half_to_float(y[i]) + fa * half_to_float(x[i]));
#else
    for (std::size_t i = 0; i < n; ++i) y[i] = float_to_half(half_to_float(y[i]) + fa * half_to_float(x[i]));
#endif
}

inline void fmad_f16(const half *a, const half *b, half *c, std::size_t n)
{
#if NERVE_HAS_NEON
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4)
    {
        float16x4_t vah = vld1_f16(reinterpret_cast<const float16_t *>(a + i));
        float16x4_t vbh = vld1_f16(reinterpret_cast<const float16_t *>(b + i));
        float16x4_t vch = vld1_f16(reinterpret_cast<const float16_t *>(c + i));
        float32x4_t va = vcvt_f32_f16(vah);
        float32x4_t vb = vcvt_f32_f16(vbh);
        float32x4_t vc = vcvt_f32_f16(vch);
        float32x4_t vr = vfmaq_f32(vc, va, vb);
        float16x4_t vrh = vcvt_f16_f32(vr);
        vst1_f16(reinterpret_cast<float16_t *>(c + i), vrh);
    }
    for (; i < n; ++i) c[i] = float_to_half(half_to_float(c[i]) + half_to_float(a[i]) * half_to_float(b[i]));
#else
    for (std::size_t i = 0; i < n; ++i) c[i] = float_to_half(half_to_float(c[i]) + half_to_float(a[i]) * half_to_float(b[i]));
#endif
}

inline float reduce_sum_f16(const half *data, std::size_t n)
{
    float sum = 0.0f;
#if NERVE_HAS_NEON
    std::size_t i = 0;
    float32x4_t vacc = vdupq_n_f32(0.0f);
    for (; i + 4 <= n; i += 4)
    {
        float16x4_t vh = vld1_f16(reinterpret_cast<const float16_t *>(data + i));
        float32x4_t v = vcvt_f32_f16(vh);
        vacc = vaddq_f32(vacc, v);
    }
    float tmp[4];
    vst1q_f32(tmp, vacc);
    sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    for (; i < n; ++i) sum += half_to_float(data[i]);
#else
    for (std::size_t i = 0; i < n; ++i) sum += half_to_float(data[i]);
#endif
    return sum;
}

inline float reduce_max_f16(const half *data, std::size_t n)
{
    if (n == 0) return 0.0f;
    float m = half_to_float(data[0]);
#if NERVE_HAS_NEON
    std::size_t i = 0;
    float16x4_t v0h = vld1_f16(reinterpret_cast<const float16_t *>(data));
    float32x4_t vmax = vcvt_f32_f16(v0h);
    for (; i + 4 <= n; i += 4)
    {
        float16x4_t vh = vld1_f16(reinterpret_cast<const float16_t *>(data + i));
        float32x4_t v = vcvt_f32_f16(vh);
        vmax = vmaxq_f32(vmax, v);
    }
    float tmp[4];
    vst1q_f32(tmp, vmax);
    m = tmp[0];
    for (int k = 1; k < 4; ++k) if (tmp[k] > m) m = tmp[k];
    for (; i < n; ++i) { float v = half_to_float(data[i]); if (v > m) m = v; }
#else
    for (std::size_t i = 1; i < n; ++i) { float v = half_to_float(data[i]); if (v > m) m = v; }
#endif
    return m;
}

inline float reduce_min_f16(const half *data, std::size_t n)
{
    if (n == 0) return 0.0f;
    float m = half_to_float(data[0]);
#if NERVE_HAS_NEON
    std::size_t i = 0;
    float16x4_t v0h = vld1_f16(reinterpret_cast<const float16_t *>(data));
    float32x4_t vmin = vcvt_f32_f16(v0h);
    for (; i + 4 <= n; i += 4)
    {
        float16x4_t vh = vld1_f16(reinterpret_cast<const float16_t *>(data + i));
        float32x4_t v = vcvt_f32_f16(vh);
        vmin = vminq_f32(vmin, v);
    }
    float tmp[4];
    vst1q_f32(tmp, vmin);
    m = tmp[0];
    for (int k = 1; k < 4; ++k) if (tmp[k] < m) m = tmp[k];
    for (; i < n; ++i) { float v = half_to_float(data[i]); if (v < m) m = v; }
#else
    for (std::size_t i = 1; i < n; ++i) { float v = half_to_float(data[i]); if (v < m) m = v; }
#endif
    return m;
}

inline float dot_f16(const half *a, const half *b, std::size_t n)
{
    float sum = 0.0f;
#if NERVE_HAS_NEON
    std::size_t i = 0;
    float32x4_t vacc = vdupq_n_f32(0.0f);
    for (; i + 4 <= n; i += 4)
    {
        float16x4_t vah = vld1_f16(reinterpret_cast<const float16_t *>(a + i));
        float16x4_t vbh = vld1_f16(reinterpret_cast<const float16_t *>(b + i));
        float32x4_t va = vcvt_f32_f16(vah);
        float32x4_t vb = vcvt_f32_f16(vbh);
        vacc = vfmaq_f32(vacc, va, vb);
    }
    float tmp[4];
    vst1q_f32(tmp, vacc);
    sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    for (; i < n; ++i) sum += half_to_float(a[i]) * half_to_float(b[i]);
#else
    for (std::size_t i = 0; i < n; ++i) sum += half_to_float(a[i]) * half_to_float(b[i]);
#endif
    return sum;
}

inline float norm2_f16(const half *vec, std::size_t n)
{
    float sum = 0.0f;
#if NERVE_HAS_NEON
    std::size_t i = 0;
    float32x4_t vacc = vdupq_n_f32(0.0f);
    for (; i + 4 <= n; i += 4)
    {
        float16x4_t vh = vld1_f16(reinterpret_cast<const float16_t *>(vec + i));
        float32x4_t v = vcvt_f32_f16(vh);
        vacc = vfmaq_f32(vacc, v, v);
    }
    float tmp[4];
    vst1q_f32(tmp, vacc);
    sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    for (; i < n; ++i) { float v = half_to_float(vec[i]); sum += v * v; }
#else
    for (std::size_t i = 0; i < n; ++i) { float v = half_to_float(vec[i]); sum += v * v; }
#endif
    return std::sqrt(sum);
}

inline void neg_f16(half *data, std::size_t n)
{
#if NERVE_HAS_NEON
    std::size_t i = 0;
    float32x4_t vzero = vdupq_n_f32(0.0f);
    for (; i + 4 <= n; i += 4)
    {
        float16x4_t vh = vld1_f16(reinterpret_cast<const float16_t *>(data + i));
        float32x4_t v = vcvt_f32_f16(vh);
        float32x4_t vr = vsubq_f32(vzero, v);
        float16x4_t vrh = vcvt_f16_f32(vr);
        vst1_f16(reinterpret_cast<float16_t *>(data + i), vrh);
    }
    for (; i < n; ++i) data[i] = float_to_half(-half_to_float(data[i]));
#else
    for (std::size_t i = 0; i < n; ++i) data[i] = float_to_half(-half_to_float(data[i]));
#endif
}

inline void sqrt_f16(half *data, std::size_t n)
{
#if NERVE_HAS_NEON
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4)
    {
        float16x4_t vh = vld1_f16(reinterpret_cast<const float16_t *>(data + i));
        float32x4_t v = vcvt_f32_f16(vh);
        float32x4_t vr = vsqrtq_f32(v);
        float16x4_t vrh = vcvt_f16_f32(vr);
        vst1_f16(reinterpret_cast<float16_t *>(data + i), vrh);
    }
    for (; i < n; ++i) data[i] = float_to_half(std::sqrt(half_to_float(data[i])));
#else
    for (std::size_t i = 0; i < n; ++i) data[i] = float_to_half(std::sqrt(half_to_float(data[i])));
#endif
}

inline void exp_f16(half *data, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        data[i] = float_to_half(std::exp(half_to_float(data[i])));
}

inline void log_f16(half *data, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        data[i] = float_to_half(std::log(half_to_float(data[i])));
}

inline void relu_f16(half *data, std::size_t n)
{
#if NERVE_HAS_NEON
    std::size_t i = 0;
    float32x4_t vzero = vdupq_n_f32(0.0f);
    for (; i + 4 <= n; i += 4)
    {
        float16x4_t vh = vld1_f16(reinterpret_cast<const float16_t *>(data + i));
        float32x4_t v = vcvt_f32_f16(vh);
        float32x4_t vr = vmaxq_f32(v, vzero);
        float16x4_t vrh = vcvt_f16_f32(vr);
        vst1_f16(reinterpret_cast<float16_t *>(data + i), vrh);
    }
    for (; i < n; ++i)
    {
        float v = half_to_float(data[i]);
        data[i] = float_to_half(v < 0.0f ? 0.0f : v);
    }
#else
    for (std::size_t i = 0; i < n; ++i)
    {
        float v = half_to_float(data[i]);
        data[i] = float_to_half(v < 0.0f ? 0.0f : v);
    }
#endif
}

inline void sigmoid_f16(half *data, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
    {
        float v = half_to_float(data[i]);
        data[i] = float_to_half(1.0f / (1.0f + std::exp(-v)));
    }
}

inline void tanh_f16(half *data, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        data[i] = float_to_half(std::tanh(half_to_float(data[i])));
}

inline void abs_f16(half *data, std::size_t n)
{
#if NERVE_HAS_NEON
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4)
    {
        float16x4_t vh = vld1_f16(reinterpret_cast<const float16_t *>(data + i));
        float32x4_t v = vcvt_f32_f16(vh);
        float32x4_t vr = vabsq_f32(v);
        float16x4_t vrh = vcvt_f16_f32(vr);
        vst1_f16(reinterpret_cast<float16_t *>(data + i), vrh);
    }
    for (; i < n; ++i) data[i] = float_to_half(std::abs(half_to_float(data[i])));
#else
    for (std::size_t i = 0; i < n; ++i) data[i] = float_to_half(std::abs(half_to_float(data[i])));
#endif
}

inline void min_f16(half *a, const half *b, std::size_t n)
{
#if NERVE_HAS_NEON
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4)
    {
        float16x4_t vah = vld1_f16(reinterpret_cast<const float16_t *>(a + i));
        float16x4_t vbh = vld1_f16(reinterpret_cast<const float16_t *>(b + i));
        float32x4_t va = vcvt_f32_f16(vah);
        float32x4_t vb = vcvt_f32_f16(vbh);
        float32x4_t vr = vminq_f32(va, vb);
        float16x4_t vrh = vcvt_f16_f32(vr);
        vst1_f16(reinterpret_cast<float16_t *>(a + i), vrh);
    }
    for (; i < n; ++i)
    {
        float fa = half_to_float(a[i]);
        float fb = half_to_float(b[i]);
        a[i] = float_to_half(fa < fb ? fa : fb);
    }
#else
    for (std::size_t i = 0; i < n; ++i)
    {
        float fa = half_to_float(a[i]);
        float fb = half_to_float(b[i]);
        a[i] = float_to_half(fa < fb ? fa : fb);
    }
#endif
}

inline void max_f16(half *a, const half *b, std::size_t n)
{
#if NERVE_HAS_NEON
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4)
    {
        float16x4_t vah = vld1_f16(reinterpret_cast<const float16_t *>(a + i));
        float16x4_t vbh = vld1_f16(reinterpret_cast<const float16_t *>(b + i));
        float32x4_t va = vcvt_f32_f16(vah);
        float32x4_t vb = vcvt_f32_f16(vbh);
        float32x4_t vr = vmaxq_f32(va, vb);
        float16x4_t vrh = vcvt_f16_f32(vr);
        vst1_f16(reinterpret_cast<float16_t *>(a + i), vrh);
    }
    for (; i < n; ++i)
    {
        float fa = half_to_float(a[i]);
        float fb = half_to_float(b[i]);
        a[i] = float_to_half(fa > fb ? fa : fb);
    }
#else
    for (std::size_t i = 0; i < n; ++i)
    {
        float fa = half_to_float(a[i]);
        float fb = half_to_float(b[i]);
        a[i] = float_to_half(fa > fb ? fa : fb);
    }
#endif
}

inline void clamp_f16(half *data, half lo, half hi, std::size_t n)
{
    float flo = half_to_float(lo);
    float fhi = half_to_float(hi);
#if NERVE_HAS_NEON
    float32x4_t vlo = vdupq_n_f32(flo);
    float32x4_t vhi = vdupq_n_f32(fhi);
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4)
    {
        float16x4_t vh = vld1_f16(reinterpret_cast<const float16_t *>(data + i));
        float32x4_t v = vcvt_f32_f16(vh);
        v = vmaxq_f32(v, vlo);
        v = vminq_f32(v, vhi);
        float16x4_t vrh = vcvt_f16_f32(v);
        vst1_f16(reinterpret_cast<float16_t *>(data + i), vrh);
    }
    for (; i < n; ++i)
    {
        float v = half_to_float(data[i]);
        if (v < flo) v = flo;
        if (v > fhi) v = fhi;
        data[i] = float_to_half(v);
    }
#else
    for (std::size_t i = 0; i < n; ++i)
    {
        float v = half_to_float(data[i]);
        if (v < flo) v = flo;
        if (v > fhi) v = fhi;
        data[i] = float_to_half(v);
    }
#endif
}

inline float sqdiff_sum_f16(const half *a, const half *b, std::size_t n)
{
    float sum = 0.0f;
#if NERVE_HAS_NEON
    std::size_t i = 0;
    float32x4_t vacc = vdupq_n_f32(0.0f);
    for (; i + 4 <= n; i += 4)
    {
        float16x4_t vah = vld1_f16(reinterpret_cast<const float16_t *>(a + i));
        float16x4_t vbh = vld1_f16(reinterpret_cast<const float16_t *>(b + i));
        float32x4_t va = vcvt_f32_f16(vah);
        float32x4_t vb = vcvt_f32_f16(vbh);
        float32x4_t diff = vsubq_f32(va, vb);
        vacc = vfmaq_f32(vacc, diff, diff);
    }
    float tmp[4];
    vst1q_f32(tmp, vacc);
    sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    for (; i < n; ++i)
    {
        float d = half_to_float(a[i]) - half_to_float(b[i]);
        sum += d * d;
    }
#else
    for (std::size_t i = 0; i < n; ++i)
    {
        float d = half_to_float(a[i]) - half_to_float(b[i]);
        sum += d * d;
    }
#endif
    return sum;
}

} // namespace nerve::simd::neon

// Float16 SVE primitives (inside nerve::simd::sve namespace)

namespace nerve::simd::sve
{

#if NERVE_HAS_SVE

inline void add_f16(half *a, const half *b, std::size_t n)
{
    std::size_t i = 0;
    svbool_t pg = svwhilelt_b32(i, n);
    while (svptest_any(svptrue_b32(), pg))
    {
        svfloat16_t vah = svld1_f16(pg, reinterpret_cast<const float16_t *>(a + i));
        svfloat16_t vbh = svld1_f16(pg, reinterpret_cast<const float16_t *>(b + i));
        svfloat32_t va = svcvt_f32_f16(vah);
        svfloat32_t vb = svcvt_f32_f16(vbh);
        svfloat32_t vr = svadd_f32_m(pg, va, vb);
        svfloat16_t vrh = svcvt_f16_f32(vr);
        svst1_f16(pg, reinterpret_cast<float16_t *>(a + i), vrh);
        i += svcntw();
        pg = svwhilelt_b32(i, n);
    }
}

inline void sub_f16(half *a, const half *b, std::size_t n)
{
    std::size_t i = 0;
    svbool_t pg = svwhilelt_b32(i, n);
    while (svptest_any(svptrue_b32(), pg))
    {
        svfloat16_t vah = svld1_f16(pg, reinterpret_cast<const float16_t *>(a + i));
        svfloat16_t vbh = svld1_f16(pg, reinterpret_cast<const float16_t *>(b + i));
        svfloat32_t va = svcvt_f32_f16(vah);
        svfloat32_t vb = svcvt_f32_f16(vbh);
        svfloat32_t vr = svsub_f32_m(pg, va, vb);
        svfloat16_t vrh = svcvt_f16_f32(vr);
        svst1_f16(pg, reinterpret_cast<float16_t *>(a + i), vrh);
        i += svcntw();
        pg = svwhilelt_b32(i, n);
    }
}

inline void mul_f16(half *a, const half *b, std::size_t n)
{
    std::size_t i = 0;
    svbool_t pg = svwhilelt_b32(i, n);
    while (svptest_any(svptrue_b32(), pg))
    {
        svfloat16_t vah = svld1_f16(pg, reinterpret_cast<const float16_t *>(a + i));
        svfloat16_t vbh = svld1_f16(pg, reinterpret_cast<const float16_t *>(b + i));
        svfloat32_t va = svcvt_f32_f16(vah);
        svfloat32_t vb = svcvt_f32_f16(vbh);
        svfloat32_t vr = svmul_f32_m(pg, va, vb);
        svfloat16_t vrh = svcvt_f16_f32(vr);
        svst1_f16(pg, reinterpret_cast<float16_t *>(a + i), vrh);
        i += svcntw();
        pg = svwhilelt_b32(i, n);
    }
}

inline float reduce_sum_f16(const half *data, std::size_t n)
{
    float sum = 0.0f;
    std::size_t i = 0;
    svfloat32_t vacc = svdup_n_f32(0.0f);
    svbool_t pg = svwhilelt_b32(i, n);
    while (svptest_any(svptrue_b32(), pg))
    {
        svfloat16_t vh = svld1_f16(pg, reinterpret_cast<const float16_t *>(data + i));
        svfloat32_t v = svcvt_f32_f16(vh);
        vacc = svadd_f32_m(pg, vacc, v);
        i += svcntw();
        pg = svwhilelt_b32(i, n);
    }
    sum = svaddv_f32(svptrue_b32(), vacc);
    return sum;
}

inline void neg_f16(half *data, std::size_t n)
{
    std::size_t i = 0;
    svbool_t pg = svwhilelt_b32(i, n);
    while (svptest_any(svptrue_b32(), pg))
    {
        svfloat16_t vh = svld1_f16(pg, reinterpret_cast<const float16_t *>(data + i));
        svfloat32_t v = svcvt_f32_f16(vh);
        svfloat32_t vr = svneg_f32_m(pg, v, v);
        svfloat16_t vrh = svcvt_f16_f32(vr);
        svst1_f16(pg, reinterpret_cast<float16_t *>(data + i), vrh);
        i += svcntw();
        pg = svwhilelt_b32(i, n);
    }
}

inline void abs_f16(half *data, std::size_t n)
{
    std::size_t i = 0;
    svbool_t pg = svwhilelt_b32(i, n);
    while (svptest_any(svptrue_b32(), pg))
    {
        svfloat16_t vh = svld1_f16(pg, reinterpret_cast<const float16_t *>(data + i));
        svfloat32_t v = svcvt_f32_f16(vh);
        svfloat32_t vr = svabs_f32_m(pg, v, v);
        svfloat16_t vrh = svcvt_f16_f32(vr);
        svst1_f16(pg, reinterpret_cast<float16_t *>(data + i), vrh);
        i += svcntw();
        pg = svwhilelt_b32(i, n);
    }
}

inline void scale_f16(half *a, half alpha, std::size_t n)
{
    float fa = half_to_float(alpha);
    svfloat32_t vfa = svdup_n_f32(fa);
    std::size_t i = 0;
    svbool_t pg = svwhilelt_b32(i, n);
    while (svptest_any(svptrue_b32(), pg))
    {
        svfloat16_t vah = svld1_f16(pg, reinterpret_cast<const float16_t *>(a + i));
        svfloat32_t va = svcvt_f32_f16(vah);
        svfloat32_t vr = svmul_f32_m(pg, va, vfa);
        svfloat16_t vrh = svcvt_f16_f32(vr);
        svst1_f16(pg, reinterpret_cast<float16_t *>(a + i), vrh);
        i += svcntw();
        pg = svwhilelt_b32(i, n);
    }
}

inline void axpy_f16(half alpha, const half *x, half *y, std::size_t n)
{
    float fa = half_to_float(alpha);
    svfloat32_t vfa = svdup_n_f32(fa);
    std::size_t i = 0;
    svbool_t pg = svwhilelt_b32(i, n);
    while (svptest_any(svptrue_b32(), pg))
    {
        svfloat16_t vxh = svld1_f16(pg, reinterpret_cast<const float16_t *>(x + i));
        svfloat16_t vyh = svld1_f16(pg, reinterpret_cast<const float16_t *>(y + i));
        svfloat32_t vx = svcvt_f32_f16(vxh);
        svfloat32_t vy = svcvt_f32_f16(vyh);
        svfloat32_t vr = svmla_f32_m(pg, vy, vfa, vx);
        svfloat16_t vrh = svcvt_f16_f32(vr);
        svst1_f16(pg, reinterpret_cast<float16_t *>(y + i), vrh);
        i += svcntw();
        pg = svwhilelt_b32(i, n);
    }
}

inline void fmad_f16(const half *a, const half *b, half *c, std::size_t n)
{
    std::size_t i = 0;
    svbool_t pg = svwhilelt_b32(i, n);
    while (svptest_any(svptrue_b32(), pg))
    {
        svfloat16_t vah = svld1_f16(pg, reinterpret_cast<const float16_t *>(a + i));
        svfloat16_t vbh = svld1_f16(pg, reinterpret_cast<const float16_t *>(b + i));
        svfloat16_t vch = svld1_f16(pg, reinterpret_cast<const float16_t *>(c + i));
        svfloat32_t va = svcvt_f32_f16(vah);
        svfloat32_t vb = svcvt_f32_f16(vbh);
        svfloat32_t vc = svcvt_f32_f16(vch);
        svfloat32_t vr = svmla_f32_m(pg, vc, va, vb);
        svfloat16_t vrh = svcvt_f16_f32(vr);
        svst1_f16(pg, reinterpret_cast<float16_t *>(c + i), vrh);
        i += svcntw();
        pg = svwhilelt_b32(i, n);
    }
}

inline float reduce_max_f16(const half *data, std::size_t n)
{
    if (n == 0) return 0.0f;
    std::size_t i = 0;
    svfloat32_t vmax = svdup_n_f32(-std::numeric_limits<float>::infinity());
    svbool_t pg = svwhilelt_b32(i, n);
    while (svptest_any(svptrue_b32(), pg))
    {
        svfloat16_t vh = svld1_f16(pg, reinterpret_cast<const float16_t *>(data + i));
        svfloat32_t v = svcvt_f32_f16(vh);
        vmax = svmax_f32_m(pg, vmax, v);
        i += svcntw();
        pg = svwhilelt_b32(i, n);
    }
    return svmaxv_f32(svptrue_b32(), vmax);
}

inline float reduce_min_f16(const half *data, std::size_t n)
{
    if (n == 0) return 0.0f;
    std::size_t i = 0;
    svfloat32_t vmin = svdup_n_f32(std::numeric_limits<float>::infinity());
    svbool_t pg = svwhilelt_b32(i, n);
    while (svptest_any(svptrue_b32(), pg))
    {
        svfloat16_t vh = svld1_f16(pg, reinterpret_cast<const float16_t *>(data + i));
        svfloat32_t v = svcvt_f32_f16(vh);
        vmin = svmin_f32_m(pg, vmin, v);
        i += svcntw();
        pg = svwhilelt_b32(i, n);
    }
    return svminv_f32(svptrue_b32(), vmin);
}

inline float dot_f16(const half *a, const half *b, std::size_t n)
{
    std::size_t i = 0;
    svfloat32_t vacc = svdup_n_f32(0.0f);
    svbool_t pg = svwhilelt_b32(i, n);
    while (svptest_any(svptrue_b32(), pg))
    {
        svfloat16_t vah = svld1_f16(pg, reinterpret_cast<const float16_t *>(a + i));
        svfloat16_t vbh = svld1_f16(pg, reinterpret_cast<const float16_t *>(b + i));
        svfloat32_t va = svcvt_f32_f16(vah);
        svfloat32_t vb = svcvt_f32_f16(vbh);
        vacc = svmla_f32_m(pg, vacc, va, vb);
        i += svcntw();
        pg = svwhilelt_b32(i, n);
    }
    return svaddv_f32(svptrue_b32(), vacc);
}

inline float norm2_f16(const half *vec, std::size_t n)
{
    std::size_t i = 0;
    svfloat32_t vacc = svdup_n_f32(0.0f);
    svbool_t pg = svwhilelt_b32(i, n);
    while (svptest_any(svptrue_b32(), pg))
    {
        svfloat16_t vh = svld1_f16(pg, reinterpret_cast<const float16_t *>(vec + i));
        svfloat32_t v = svcvt_f32_f16(vh);
        vacc = svmla_f32_m(pg, vacc, v, v);
        i += svcntw();
        pg = svwhilelt_b32(i, n);
    }
    float sum = svaddv_f32(svptrue_b32(), vacc);
    return std::sqrt(sum);
}

inline void sqrt_f16(half *data, std::size_t n)
{
    std::size_t i = 0;
    svbool_t pg = svwhilelt_b32(i, n);
    while (svptest_any(svptrue_b32(), pg))
    {
        svfloat16_t vh = svld1_f16(pg, reinterpret_cast<const float16_t *>(data + i));
        svfloat32_t v = svcvt_f32_f16(vh);
        svfloat32_t vr = svsqrt_f32_m(pg, v, v);
        svfloat16_t vrh = svcvt_f16_f32(vr);
        svst1_f16(pg, reinterpret_cast<float16_t *>(data + i), vrh);
        i += svcntw();
        pg = svwhilelt_b32(i, n);
    }
}

inline void exp_f16(half *data, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        data[i] = float_to_half(std::exp(half_to_float(data[i])));
}

inline void log_f16(half *data, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        data[i] = float_to_half(std::log(half_to_float(data[i])));
}

inline void relu_f16(half *data, std::size_t n)
{
    std::size_t i = 0;
    svfloat32_t vzero = svdup_n_f32(0.0f);
    svbool_t pg = svwhilelt_b32(i, n);
    while (svptest_any(svptrue_b32(), pg))
    {
        svfloat16_t vh = svld1_f16(pg, reinterpret_cast<const float16_t *>(data + i));
        svfloat32_t v = svcvt_f32_f16(vh);
        svfloat32_t vr = svmax_f32_m(pg, v, vzero);
        svfloat16_t vrh = svcvt_f16_f32(vr);
        svst1_f16(pg, reinterpret_cast<float16_t *>(data + i), vrh);
        i += svcntw();
        pg = svwhilelt_b32(i, n);
    }
}

inline void sigmoid_f16(half *data, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
    {
        float v = half_to_float(data[i]);
        data[i] = float_to_half(1.0f / (1.0f + std::exp(-v)));
    }
}

inline void tanh_f16(half *data, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        data[i] = float_to_half(std::tanh(half_to_float(data[i])));
}

inline void min_f16(half *a, const half *b, std::size_t n)
{
    std::size_t i = 0;
    svbool_t pg = svwhilelt_b32(i, n);
    while (svptest_any(svptrue_b32(), pg))
    {
        svfloat16_t vah = svld1_f16(pg, reinterpret_cast<const float16_t *>(a + i));
        svfloat16_t vbh = svld1_f16(pg, reinterpret_cast<const float16_t *>(b + i));
        svfloat32_t va = svcvt_f32_f16(vah);
        svfloat32_t vb = svcvt_f32_f16(vbh);
        svfloat32_t vr = svmin_f32_m(pg, va, vb);
        svfloat16_t vrh = svcvt_f16_f32(vr);
        svst1_f16(pg, reinterpret_cast<float16_t *>(a + i), vrh);
        i += svcntw();
        pg = svwhilelt_b32(i, n);
    }
}

inline void max_f16(half *a, const half *b, std::size_t n)
{
    std::size_t i = 0;
    svbool_t pg = svwhilelt_b32(i, n);
    while (svptest_any(svptrue_b32(), pg))
    {
        svfloat16_t vah = svld1_f16(pg, reinterpret_cast<const float16_t *>(a + i));
        svfloat16_t vbh = svld1_f16(pg, reinterpret_cast<const float16_t *>(b + i));
        svfloat32_t va = svcvt_f32_f16(vah);
        svfloat32_t vb = svcvt_f32_f16(vbh);
        svfloat32_t vr = svmax_f32_m(pg, va, vb);
        svfloat16_t vrh = svcvt_f16_f32(vr);
        svst1_f16(pg, reinterpret_cast<float16_t *>(a + i), vrh);
        i += svcntw();
        pg = svwhilelt_b32(i, n);
    }
}

inline void clamp_f16(half *data, half lo, half hi, std::size_t n)
{
    float flo = half_to_float(lo);
    float fhi = half_to_float(hi);
    svfloat32_t vlo = svdup_n_f32(flo);
    svfloat32_t vhi = svdup_n_f32(fhi);
    std::size_t i = 0;
    svbool_t pg = svwhilelt_b32(i, n);
    while (svptest_any(svptrue_b32(), pg))
    {
        svfloat16_t vh = svld1_f16(pg, reinterpret_cast<const float16_t *>(data + i));
        svfloat32_t v = svcvt_f32_f16(vh);
        v = svmax_f32_m(pg, v, vlo);
        v = svmin_f32_m(pg, v, vhi);
        svfloat16_t vrh = svcvt_f16_f32(v);
        svst1_f16(pg, reinterpret_cast<float16_t *>(data + i), vrh);
        i += svcntw();
        pg = svwhilelt_b32(i, n);
    }
}

inline float sqdiff_sum_f16(const half *a, const half *b, std::size_t n)
{
    std::size_t i = 0;
    svfloat32_t vacc = svdup_n_f32(0.0f);
    svbool_t pg = svwhilelt_b32(i, n);
    while (svptest_any(svptrue_b32(), pg))
    {
        svfloat16_t vah = svld1_f16(pg, reinterpret_cast<const float16_t *>(a + i));
        svfloat16_t vbh = svld1_f16(pg, reinterpret_cast<const float16_t *>(b + i));
        svfloat32_t va = svcvt_f32_f16(vah);
        svfloat32_t vb = svcvt_f32_f16(vbh);
        svfloat32_t diff = svsub_f32_m(pg, va, vb);
        vacc = svmla_f32_m(pg, vacc, diff, diff);
        i += svcntw();
        pg = svwhilelt_b32(i, n);
    }
    return svaddv_f32(svptrue_b32(), vacc);
}

#endif // NERVE_HAS_SVE

} // namespace nerve::simd::sve
