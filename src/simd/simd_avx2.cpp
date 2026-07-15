#include "nerve/simd/simd_base.hpp"

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstring>

// Enable AVX2 + FMA + F16C intrinsics for this file without requiring global flags
#pragma GCC push_options
#pragma GCC target("avx2,fma,f16c")

namespace nerve::simd::avx2
{

void memcpy(void *dst, const void *src, std::size_t bytes)
{
    std::size_t i = 0;
    auto *d = static_cast<std::uint8_t *>(dst);
    const auto *s = static_cast<const std::uint8_t *>(src);
    for (; i + 32 <= bytes; i += 32)
    {
        __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(s + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(d + i), v);
    }
    for (; i < bytes; ++i)
        d[i] = s[i];
}

void memset(void *dst, int value, std::size_t bytes)
{
    std::size_t i = 0;
    auto *d = static_cast<std::uint8_t *>(dst);
    __m256i v = _mm256_set1_epi8(static_cast<char>(value));
    for (; i + 32 <= bytes; i += 32)
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(d + i), v);
    for (; i < bytes; ++i)
        d[i] = static_cast<std::uint8_t>(value);
}

void add(double *a, const double *b, std::size_t n)
{
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4)
    {
        __m256d va = _mm256_loadu_pd(a + i);
        __m256d vb = _mm256_loadu_pd(b + i);
        _mm256_storeu_pd(a + i, _mm256_add_pd(va, vb));
    }
    for (; i < n; ++i)
        a[i] += b[i];
}

void sub(double *a, const double *b, std::size_t n)
{
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4)
    {
        __m256d va = _mm256_loadu_pd(a + i);
        __m256d vb = _mm256_loadu_pd(b + i);
        _mm256_storeu_pd(a + i, _mm256_sub_pd(va, vb));
    }
    for (; i < n; ++i)
        a[i] -= b[i];
}

void mul(double *a, const double *b, std::size_t n)
{
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4)
    {
        __m256d va = _mm256_loadu_pd(a + i);
        __m256d vb = _mm256_loadu_pd(b + i);
        _mm256_storeu_pd(a + i, _mm256_mul_pd(va, vb));
    }
    for (; i < n; ++i)
        a[i] *= b[i];
}

void scale(double *a, double alpha, std::size_t n)
{
    std::size_t i = 0;
    __m256d va = _mm256_set1_pd(alpha);
    for (; i + 4 <= n; i += 4)
    {
        __m256d v = _mm256_loadu_pd(a + i);
        _mm256_storeu_pd(a + i, _mm256_mul_pd(v, va));
    }
    for (; i < n; ++i)
        a[i] *= alpha;
}

void axpy(double alpha, const double *x, double *y, std::size_t n)
{
    std::size_t i = 0;
    __m256d va = _mm256_set1_pd(alpha);
    for (; i + 4 <= n; i += 4)
    {
        __m256d vx = _mm256_loadu_pd(x + i);
        __m256d vy = _mm256_loadu_pd(y + i);
        _mm256_storeu_pd(y + i, _mm256_fmadd_pd(va, vx, vy));
    }
    for (; i < n; ++i)
        y[i] += alpha * x[i];
}

void fmad(const double *a, const double *b, double *c, std::size_t n)
{
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4)
    {
        __m256d va = _mm256_loadu_pd(a + i);
        __m256d vb = _mm256_loadu_pd(b + i);
        __m256d vc = _mm256_loadu_pd(c + i);
        _mm256_storeu_pd(c + i, _mm256_fmadd_pd(va, vb, vc));
    }
    for (; i < n; ++i)
        c[i] += a[i] * b[i];
}

// Reductions

double reduce_sum(const double *data, std::size_t n)
{
    double sum = 0.0;
    std::size_t i = 0;
    __m256d vacc = _mm256_setzero_pd();
    for (; i + 4 <= n; i += 4)
    {
        __m256d v = _mm256_loadu_pd(data + i);
        vacc = _mm256_add_pd(vacc, v);
    }
    // Horizontal sum across 4 lanes
    __m128d lo = _mm256_castpd256_pd128(vacc);
    __m128d hi = _mm256_extractf128_pd(vacc, 1);
    lo = _mm_add_pd(lo, hi);
    lo = _mm_hadd_pd(lo, lo);
    sum = _mm_cvtsd_f64(lo);
    for (; i < n; ++i)
        sum += data[i];
    return sum;
}

double reduce_max(const double *data, std::size_t n)
{
    if (n == 0)
        return 0.0;
    std::size_t i = 0;
    __m256d vmax = _mm256_set1_pd(data[0]);
    for (; i + 4 <= n; i += 4)
    {
        __m256d v = _mm256_loadu_pd(data + i);
        vmax = _mm256_max_pd(vmax, v);
    }
    double tmp[4];
    _mm256_storeu_pd(tmp, vmax);
    double m = tmp[0];
    for (int k = 1; k < 4; ++k)
        if (tmp[k] > m)
            m = tmp[k];
    for (; i < n; ++i)
        if (data[i] > m)
            m = data[i];
    return m;
}

double reduce_min(const double *data, std::size_t n)
{
    if (n == 0)
        return 0.0;
    std::size_t i = 0;
    __m256d vmin = _mm256_set1_pd(data[0]);
    for (; i + 4 <= n; i += 4)
    {
        __m256d v = _mm256_loadu_pd(data + i);
        vmin = _mm256_min_pd(vmin, v);
    }
    double tmp[4];
    _mm256_storeu_pd(tmp, vmin);
    double m = tmp[0];
    for (int k = 1; k < 4; ++k)
        if (tmp[k] < m)
            m = tmp[k];
    for (; i < n; ++i)
        if (data[i] < m)
            m = data[i];
    return m;
}

double dot(const double *a, const double *b, std::size_t n)
{
    std::size_t i = 0;
    __m256d vacc = _mm256_setzero_pd();
    for (; i + 4 <= n; i += 4)
    {
        __m256d va = _mm256_loadu_pd(a + i);
        __m256d vb = _mm256_loadu_pd(b + i);
        vacc = _mm256_fmadd_pd(va, vb, vacc);
    }
    __m128d lo = _mm256_castpd256_pd128(vacc);
    __m128d hi = _mm256_extractf128_pd(vacc, 1);
    lo = _mm_add_pd(lo, hi);
    lo = _mm_hadd_pd(lo, lo);
    double sum = _mm_cvtsd_f64(lo);
    for (; i < n; ++i)
        sum += a[i] * b[i];
    return sum;
}

double norm2(const double *vec, std::size_t n)
{
    std::size_t i = 0;
    __m256d vacc = _mm256_setzero_pd();
    for (; i + 4 <= n; i += 4)
    {
        __m256d v = _mm256_loadu_pd(vec + i);
        vacc = _mm256_fmadd_pd(v, v, vacc);
    }
    __m128d lo = _mm256_castpd256_pd128(vacc);
    __m128d hi = _mm256_extractf128_pd(vacc, 1);
    lo = _mm_add_pd(lo, hi);
    lo = _mm_hadd_pd(lo, lo);
    double sum = _mm_cvtsd_f64(lo);
    for (; i < n; ++i)
        sum += vec[i] * vec[i];
    return std::sqrt(sum);
}

double sqdiff_sum(const double *a, const double *b, std::size_t n)
{
    std::size_t i = 0;
    __m256d vacc = _mm256_setzero_pd();
    for (; i + 4 <= n; i += 4)
    {
        __m256d va = _mm256_loadu_pd(a + i);
        __m256d vb = _mm256_loadu_pd(b + i);
        __m256d diff = _mm256_sub_pd(va, vb);
        vacc = _mm256_fmadd_pd(diff, diff, vacc);
    }
    __m128d lo = _mm256_castpd256_pd128(vacc);
    __m128d hi = _mm256_extractf128_pd(vacc, 1);
    lo = _mm_add_pd(lo, hi);
    lo = _mm_hadd_pd(lo, lo);
    double sum = _mm_cvtsd_f64(lo);
    for (; i < n; ++i)
    {
        double d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

// Element-wise

void abs(double *data, std::size_t n)
{
    std::size_t i = 0;
    __m256d vsign = _mm256_set1_pd(-0.0);
    for (; i + 4 <= n; i += 4)
    {
        __m256d v = _mm256_loadu_pd(data + i);
        _mm256_storeu_pd(data + i, _mm256_andnot_pd(vsign, v));
    }
    for (; i < n; ++i)
        data[i] = std::abs(data[i]);
}

void neg(double *data, std::size_t n)
{
    std::size_t i = 0;
    __m256d vsign = _mm256_set1_pd(-0.0);
    for (; i + 4 <= n; i += 4)
    {
        __m256d v = _mm256_loadu_pd(data + i);
        _mm256_storeu_pd(data + i, _mm256_xor_pd(vsign, v));
    }
    for (; i < n; ++i)
        data[i] = -data[i];
}

void sqrt(double *data, std::size_t n)
{
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4)
    {
        __m256d v = _mm256_loadu_pd(data + i);
        _mm256_storeu_pd(data + i, _mm256_sqrt_pd(v));
    }
    for (; i < n; ++i)
        data[i] = std::sqrt(data[i]);
}

void exp(double *data, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        data[i] = std::exp(data[i]);
}

void log(double *data, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        data[i] = std::log(data[i]);
}

void relu(double *data, std::size_t n)
{
    std::size_t i = 0;
    __m256d vzero = _mm256_setzero_pd();
    for (; i + 4 <= n; i += 4)
    {
        __m256d v = _mm256_loadu_pd(data + i);
        _mm256_storeu_pd(data + i, _mm256_max_pd(v, vzero));
    }
    for (; i < n; ++i)
        if (data[i] < 0.0)
            data[i] = 0.0;
}

void sigmoid(double *data, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        data[i] = 1.0 / (1.0 + std::exp(-data[i]));
}

void tanh(double *data, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        data[i] = std::tanh(data[i]);
}

void min(double *a, const double *b, std::size_t n)
{
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4)
    {
        __m256d va = _mm256_loadu_pd(a + i);
        __m256d vb = _mm256_loadu_pd(b + i);
        _mm256_storeu_pd(a + i, _mm256_min_pd(va, vb));
    }
    for (; i < n; ++i)
        a[i] = a[i] < b[i] ? a[i] : b[i];
}

void max(double *a, const double *b, std::size_t n)
{
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4)
    {
        __m256d va = _mm256_loadu_pd(a + i);
        __m256d vb = _mm256_loadu_pd(b + i);
        _mm256_storeu_pd(a + i, _mm256_max_pd(va, vb));
    }
    for (; i < n; ++i)
        a[i] = a[i] > b[i] ? a[i] : b[i];
}

void clamp(double *data, double lo, double hi, std::size_t n)
{
    std::size_t i = 0;
    __m256d vlo = _mm256_set1_pd(lo);
    __m256d vhi = _mm256_set1_pd(hi);
    for (; i + 4 <= n; i += 4)
    {
        __m256d v = _mm256_loadu_pd(data + i);
        v = _mm256_max_pd(v, vlo);
        v = _mm256_min_pd(v, vhi);
        _mm256_storeu_pd(data + i, v);
    }
    for (; i < n; ++i)
    {
        if (data[i] < lo)
            data[i] = lo;
        if (data[i] > hi)
            data[i] = hi;
    }
}

void add_f32(float *a, const float *b, std::size_t n)
{
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8)
    {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(a + i, _mm256_add_ps(va, vb));
    }
    for (; i < n; ++i)
        a[i] += b[i];
}

void fmad_f32(const float *a, const float *b, float *c, std::size_t n)
{
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8)
    {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vc = _mm256_loadu_ps(c + i);
        _mm256_storeu_ps(c + i, _mm256_fmadd_ps(va, vb, vc));
    }
    for (; i < n; ++i)
        c[i] += a[i] * b[i];
}

float reduce_sum_f32(const float *data, std::size_t n)
{
    float sum = 0.0f;
    std::size_t i = 0;
    __m256 vacc = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8)
    {
        __m256 v = _mm256_loadu_ps(data + i);
        vacc = _mm256_add_ps(vacc, v);
    }
    __m128 lo = _mm256_castps256_ps128(vacc);
    __m128 hi = _mm256_extractf128_ps(vacc, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    sum = _mm_cvtss_f32(lo);
    for (; i < n; ++i)
        sum += data[i];
    return sum;
}

float dot_f32(const float *a, const float *b, std::size_t n)
{
    std::size_t i = 0;
    __m256 vacc = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8)
    {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        vacc = _mm256_fmadd_ps(va, vb, vacc);
    }
    __m128 lo = _mm256_castps256_ps128(vacc);
    __m128 hi = _mm256_extractf128_ps(vacc, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    float sum = _mm_cvtss_f32(lo);
    for (; i < n; ++i)
        sum += a[i] * b[i];
    return sum;
}

void sub_f32(float *a, const float *b, std::size_t n)
{
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8)
    {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(a + i, _mm256_sub_ps(va, vb));
    }
    for (; i < n; ++i)
        a[i] -= b[i];
}

void mul_f32(float *a, const float *b, std::size_t n)
{
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8)
    {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(a + i, _mm256_mul_ps(va, vb));
    }
    for (; i < n; ++i)
        a[i] *= b[i];
}

void scale_f32(float *a, float alpha, std::size_t n)
{
    std::size_t i = 0;
    __m256 va = _mm256_set1_ps(alpha);
    for (; i + 8 <= n; i += 8)
    {
        __m256 v = _mm256_loadu_ps(a + i);
        _mm256_storeu_ps(a + i, _mm256_mul_ps(v, va));
    }
    for (; i < n; ++i)
        a[i] *= alpha;
}

void axpy_f32(float alpha, const float *x, float *y, std::size_t n)
{
    std::size_t i = 0;
    __m256 va = _mm256_set1_ps(alpha);
    for (; i + 8 <= n; i += 8)
    {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vy = _mm256_loadu_ps(y + i);
        _mm256_storeu_ps(y + i, _mm256_fmadd_ps(va, vx, vy));
    }
    for (; i < n; ++i)
        y[i] += alpha * x[i];
}

float reduce_max_f32(const float *data, std::size_t n)
{
    if (n == 0)
        return 0.0f;
    std::size_t i = 0;
    __m256 vmax = _mm256_set1_ps(data[0]);
    for (; i + 8 <= n; i += 8)
    {
        __m256 v = _mm256_loadu_ps(data + i);
        vmax = _mm256_max_ps(vmax, v);
    }
    float tmp[8];
    _mm256_storeu_ps(tmp, vmax);
    float m = tmp[0];
    for (int k = 1; k < 8; ++k)
        if (tmp[k] > m)
            m = tmp[k];
    for (; i < n; ++i)
        if (data[i] > m)
            m = data[i];
    return m;
}

float reduce_min_f32(const float *data, std::size_t n)
{
    if (n == 0)
        return 0.0f;
    std::size_t i = 0;
    __m256 vmin = _mm256_set1_ps(data[0]);
    for (; i + 8 <= n; i += 8)
    {
        __m256 v = _mm256_loadu_ps(data + i);
        vmin = _mm256_min_ps(vmin, v);
    }
    float tmp[8];
    _mm256_storeu_ps(tmp, vmin);
    float m = tmp[0];
    for (int k = 1; k < 8; ++k)
        if (tmp[k] < m)
            m = tmp[k];
    for (; i < n; ++i)
        if (data[i] < m)
            m = data[i];
    return m;
}

float norm2_f32(const float *vec, std::size_t n)
{
    std::size_t i = 0;
    __m256 vacc = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8)
    {
        __m256 v = _mm256_loadu_ps(vec + i);
        vacc = _mm256_fmadd_ps(v, v, vacc);
    }
    __m128 lo = _mm256_castps256_ps128(vacc);
    __m128 hi = _mm256_extractf128_ps(vacc, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    float sum = _mm_cvtss_f32(lo);
    for (; i < n; ++i)
        sum += vec[i] * vec[i];
    return std::sqrt(sum);
}

float sqdiff_sum_f32(const float *a, const float *b, std::size_t n)
{
    std::size_t i = 0;
    __m256 vacc = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8)
    {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        vacc = _mm256_fmadd_ps(diff, diff, vacc);
    }
    __m128 lo = _mm256_castps256_ps128(vacc);
    __m128 hi = _mm256_extractf128_ps(vacc, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    float sum = _mm_cvtss_f32(lo);
    for (; i < n; ++i)
    {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

float euclidean_f32(const float *a, const float *b, std::size_t n)
{
    return std::sqrt(sqdiff_sum_f32(a, b, n));
}

float cosine_f32(const float *a, const float *b, std::size_t n)
{
    float dot_val = dot_f32(a, b, n);
    float na = norm2_f32(a, n);
    float nb = norm2_f32(b, n);
    if (na == 0.0f || nb == 0.0f)
        return 1.0f;
    float cos_sim = dot_val / (na * nb);
    if (cos_sim < -1.0f)
        cos_sim = -1.0f;
    if (cos_sim > 1.0f)
        cos_sim = 1.0f;
    return 1.0f - cos_sim;
}

void neg_f32(float *data, std::size_t n)
{
    std::size_t i = 0;
    __m256 vsign = _mm256_set1_ps(-0.0f);
    for (; i + 8 <= n; i += 8)
    {
        __m256 v = _mm256_loadu_ps(data + i);
        _mm256_storeu_ps(data + i, _mm256_xor_ps(vsign, v));
    }
    for (; i < n; ++i)
        data[i] = -data[i];
}

void sqrt_f32(float *data, std::size_t n)
{
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8)
    {
        __m256 v = _mm256_loadu_ps(data + i);
        _mm256_storeu_ps(data + i, _mm256_sqrt_ps(v));
    }
    for (; i < n; ++i)
        data[i] = std::sqrt(data[i]);
}

void exp_f32(float *data, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        data[i] = std::exp(data[i]);
}

void log_f32(float *data, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        data[i] = std::log(data[i]);
}

void sigmoid_f32(float *data, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        data[i] = 1.0f / (1.0f + std::exp(-data[i]));
}

void tanh_f32(float *data, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        data[i] = std::tanh(data[i]);
}

void abs_f32(float *data, std::size_t n)
{
    std::size_t i = 0;
    __m256 vsign = _mm256_set1_ps(-0.0f);
    for (; i + 8 <= n; i += 8)
    {
        __m256 v = _mm256_loadu_ps(data + i);
        _mm256_storeu_ps(data + i, _mm256_andnot_ps(vsign, v));
    }
    for (; i < n; ++i)
        data[i] = std::abs(data[i]);
}

void relu_f32(float *data, std::size_t n)
{
    std::size_t i = 0;
    __m256 vzero = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8)
    {
        __m256 v = _mm256_loadu_ps(data + i);
        _mm256_storeu_ps(data + i, _mm256_max_ps(v, vzero));
    }
    for (; i < n; ++i)
        if (data[i] < 0.0f)
            data[i] = 0.0f;
}

void min_f32(float *a, const float *b, std::size_t n)
{
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8)
    {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(a + i, _mm256_min_ps(va, vb));
    }
    for (; i < n; ++i)
        a[i] = a[i] < b[i] ? a[i] : b[i];
}

void max_f32(float *a, const float *b, std::size_t n)
{
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8)
    {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(a + i, _mm256_max_ps(va, vb));
    }
    for (; i < n; ++i)
        a[i] = a[i] > b[i] ? a[i] : b[i];
}

void clamp_f32(float *data, float lo, float hi, std::size_t n)
{
    std::size_t i = 0;
    __m256 vlo = _mm256_set1_ps(lo);
    __m256 vhi = _mm256_set1_ps(hi);
    for (; i + 8 <= n; i += 8)
    {
        __m256 v = _mm256_loadu_ps(data + i);
        v = _mm256_max_ps(v, vlo);
        v = _mm256_min_ps(v, vhi);
        _mm256_storeu_ps(data + i, v);
    }
    for (; i < n; ++i)
    {
        if (data[i] < lo)
            data[i] = lo;
        if (data[i] > hi)
            data[i] = hi;
    }
}

void gemv_f32(float alpha, const float *A, const float *x, float beta, float *y, std::size_t m,
              std::size_t n)
{
    if (beta != 1.0f)
        for (std::size_t i = 0; i < m; ++i)
            y[i] *= beta;
    for (std::size_t i = 0; i < m; ++i)
    {
        __m256 vacc = _mm256_setzero_ps();
        std::size_t j = 0;
        for (; j + 8 <= n; j += 8)
        {
            __m256 va = _mm256_loadu_ps(A + i * n + j);
            __m256 vx = _mm256_loadu_ps(x + j);
            vacc = _mm256_fmadd_ps(va, vx, vacc);
        }
        __m128 lo = _mm256_castps256_ps128(vacc);
        __m128 hi = _mm256_extractf128_ps(vacc, 1);
        lo = _mm_add_ps(lo, hi);
        lo = _mm_hadd_ps(lo, lo);
        lo = _mm_hadd_ps(lo, lo);
        float sum = _mm_cvtss_f32(lo);
        for (; j < n; ++j)
            sum += A[i * n + j] * x[j];
        y[i] += alpha * sum;
    }
}

void ger_f32(float alpha, const float *x, const float *y, float *A, std::size_t m, std::size_t n)
{
    for (std::size_t i = 0; i < m; ++i)
    {
        __m256 vx = _mm256_set1_ps(alpha * x[i]);
        std::size_t j = 0;
        for (; j + 8 <= n; j += 8)
        {
            __m256 vy = _mm256_loadu_ps(y + j);
            __m256 va = _mm256_loadu_ps(A + i * n + j);
            _mm256_storeu_ps(A + i * n + j, _mm256_fmadd_ps(vx, vy, va));
        }
        for (; j < n; ++j)
            A[i * n + j] += alpha * x[i] * y[j];
    }
}

// BLAS

void gemv(double alpha, const double *A, const double *x, double beta, double *y, std::size_t m,
          std::size_t n)
{
    if (beta != 1.0)
        for (std::size_t i = 0; i < m; ++i)
            y[i] *= beta;

    for (std::size_t i = 0; i < m; ++i)
    {
        __m256d vacc = _mm256_setzero_pd();
        std::size_t j = 0;
        for (; j + 4 <= n; j += 4)
        {
            __m256d va = _mm256_loadu_pd(A + i * n + j);
            __m256d vx = _mm256_loadu_pd(x + j);
            vacc = _mm256_fmadd_pd(va, vx, vacc);
        }
        double tmp[4];
        _mm256_storeu_pd(tmp, vacc);
        double sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];
        for (; j < n; ++j)
            sum += A[i * n + j] * x[j];
        y[i] += alpha * sum;
    }
}

void ger(double alpha, const double *x, const double *y, double *A, std::size_t m, std::size_t n)
{
    for (std::size_t i = 0; i < m; ++i)
    {
        __m256d vx = _mm256_set1_pd(alpha * x[i]);
        std::size_t j = 0;
        for (; j + 4 <= n; j += 4)
        {
            __m256d vy = _mm256_loadu_pd(y + j);
            __m256d va = _mm256_loadu_pd(A + i * n + j);
            _mm256_storeu_pd(A + i * n + j, _mm256_fmadd_pd(vx, vy, va));
        }
        for (; j < n; ++j)
            A[i * n + j] += alpha * x[i] * y[j];
    }
}

// Float16 AVX2 implementations (F16C)
// Load 8 half values as __m128i -> _mm256_cvtph_ps -> compute -> _mm256_cvtps_ph -> store

void add_f16(half *a, const half *b, std::size_t n)
{
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8)
    {
        __m128i va_half = _mm_loadu_si128(reinterpret_cast<const __m128i *>(a + i));
        __m128i vb_half = _mm_loadu_si128(reinterpret_cast<const __m128i *>(b + i));
        __m256 va = _mm256_cvtph_ps(va_half);
        __m256 vb = _mm256_cvtph_ps(vb_half);
        __m256 vr = _mm256_add_ps(va, vb);
        __m128i vr_half = _mm256_cvtps_ph(vr, 0);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(a + i), vr_half);
    }
    for (; i < n; ++i)
        a[i] = float_to_half(half_to_float(a[i]) + half_to_float(b[i]));
}

void sub_f16(half *a, const half *b, std::size_t n)
{
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8)
    {
        __m128i va_half = _mm_loadu_si128(reinterpret_cast<const __m128i *>(a + i));
        __m128i vb_half = _mm_loadu_si128(reinterpret_cast<const __m128i *>(b + i));
        __m256 va = _mm256_cvtph_ps(va_half);
        __m256 vb = _mm256_cvtph_ps(vb_half);
        __m256 vr = _mm256_sub_ps(va, vb);
        __m128i vr_half = _mm256_cvtps_ph(vr, 0);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(a + i), vr_half);
    }
    for (; i < n; ++i)
        a[i] = float_to_half(half_to_float(a[i]) - half_to_float(b[i]));
}

void mul_f16(half *a, const half *b, std::size_t n)
{
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8)
    {
        __m128i va_half = _mm_loadu_si128(reinterpret_cast<const __m128i *>(a + i));
        __m128i vb_half = _mm_loadu_si128(reinterpret_cast<const __m128i *>(b + i));
        __m256 va = _mm256_cvtph_ps(va_half);
        __m256 vb = _mm256_cvtph_ps(vb_half);
        __m256 vr = _mm256_mul_ps(va, vb);
        __m128i vr_half = _mm256_cvtps_ph(vr, 0);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(a + i), vr_half);
    }
    for (; i < n; ++i)
        a[i] = float_to_half(half_to_float(a[i]) * half_to_float(b[i]));
}

void scale_f16(half *a, half alpha, std::size_t n)
{
    float fa = half_to_float(alpha);
    __m256 vfa = _mm256_set1_ps(fa);
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8)
    {
        __m128i va_half = _mm_loadu_si128(reinterpret_cast<const __m128i *>(a + i));
        __m256 va = _mm256_cvtph_ps(va_half);
        __m256 vr = _mm256_mul_ps(va, vfa);
        __m128i vr_half = _mm256_cvtps_ph(vr, 0);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(a + i), vr_half);
    }
    for (; i < n; ++i)
        a[i] = float_to_half(half_to_float(a[i]) * fa);
}

void axpy_f16(half alpha, const half *x, half *y, std::size_t n)
{
    float fa = half_to_float(alpha);
    __m256 vfa = _mm256_set1_ps(fa);
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8)
    {
        __m128i vx_half = _mm_loadu_si128(reinterpret_cast<const __m128i *>(x + i));
        __m128i vy_half = _mm_loadu_si128(reinterpret_cast<const __m128i *>(y + i));
        __m256 vx = _mm256_cvtph_ps(vx_half);
        __m256 vy = _mm256_cvtph_ps(vy_half);
        __m256 vr = _mm256_fmadd_ps(vfa, vx, vy);
        __m128i vr_half = _mm256_cvtps_ph(vr, 0);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(y + i), vr_half);
    }
    for (; i < n; ++i)
        y[i] = float_to_half(half_to_float(y[i]) + fa * half_to_float(x[i]));
}

void fmad_f16(const half *a, const half *b, half *c, std::size_t n)
{
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8)
    {
        __m128i va_half = _mm_loadu_si128(reinterpret_cast<const __m128i *>(a + i));
        __m128i vb_half = _mm_loadu_si128(reinterpret_cast<const __m128i *>(b + i));
        __m128i vc_half = _mm_loadu_si128(reinterpret_cast<const __m128i *>(c + i));
        __m256 va = _mm256_cvtph_ps(va_half);
        __m256 vb = _mm256_cvtph_ps(vb_half);
        __m256 vc = _mm256_cvtph_ps(vc_half);
        __m256 vr = _mm256_fmadd_ps(va, vb, vc);
        __m128i vr_half = _mm256_cvtps_ph(vr, 0);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(c + i), vr_half);
    }
    for (; i < n; ++i)
        c[i] = float_to_half(half_to_float(c[i]) + half_to_float(a[i]) * half_to_float(b[i]));
}

float reduce_sum_f16(const half *data, std::size_t n)
{
    float sum = 0.0f;
    std::size_t i = 0;
    __m256 vacc = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8)
    {
        __m128i v_half = _mm_loadu_si128(reinterpret_cast<const __m128i *>(data + i));
        __m256 v = _mm256_cvtph_ps(v_half);
        vacc = _mm256_add_ps(vacc, v);
    }
    __m128 lo = _mm256_castps256_ps128(vacc);
    __m128 hi = _mm256_extractf128_ps(vacc, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    sum = _mm_cvtss_f32(lo);
    for (; i < n; ++i)
        sum += half_to_float(data[i]);
    return sum;
}

float reduce_max_f16(const half *data, std::size_t n)
{
    if (n == 0)
        return 0.0f;
    // Use pure scalar for n < 8 to avoid reading past the buffer
    if (n < 8)
    {
        float m = half_to_float(data[0]);
        for (std::size_t i = 1; i < n; ++i)
        {
            float v = half_to_float(data[i]);
            if (v > m)
                m = v;
        }
        return m;
    }
    std::size_t i = 0;
    __m128i v0_half = _mm_loadu_si128(reinterpret_cast<const __m128i *>(data));
    __m256 vmax = _mm256_cvtph_ps(v0_half);
    for (; i + 8 <= n; i += 8)
    {
        __m128i v_half = _mm_loadu_si128(reinterpret_cast<const __m128i *>(data + i));
        __m256 v = _mm256_cvtph_ps(v_half);
        vmax = _mm256_max_ps(vmax, v);
    }
    float tmp[8];
    _mm256_storeu_ps(tmp, vmax);
    float m = tmp[0];
    for (int k = 1; k < 8; ++k)
        if (tmp[k] > m)
            m = tmp[k];
    for (; i < n; ++i)
    {
        float v = half_to_float(data[i]);
        if (v > m)
            m = v;
    }
    return m;
}

float reduce_min_f16(const half *data, std::size_t n)
{
    if (n == 0)
        return 0.0f;
    // Use pure scalar for n < 8 to avoid reading past the buffer
    if (n < 8)
    {
        float m = half_to_float(data[0]);
        for (std::size_t i = 1; i < n; ++i)
        {
            float v = half_to_float(data[i]);
            if (v < m)
                m = v;
        }
        return m;
    }
    std::size_t i = 0;
    __m128i v0_half = _mm_loadu_si128(reinterpret_cast<const __m128i *>(data));
    __m256 vmin = _mm256_cvtph_ps(v0_half);
    for (; i + 8 <= n; i += 8)
    {
        __m128i v_half = _mm_loadu_si128(reinterpret_cast<const __m128i *>(data + i));
        __m256 v = _mm256_cvtph_ps(v_half);
        vmin = _mm256_min_ps(vmin, v);
    }
    float tmp[8];
    _mm256_storeu_ps(tmp, vmin);
    float m = tmp[0];
    for (int k = 1; k < 8; ++k)
        if (tmp[k] < m)
            m = tmp[k];
    for (; i < n; ++i)
    {
        float v = half_to_float(data[i]);
        if (v < m)
            m = v;
    }
    return m;
}

float dot_f16(const half *a, const half *b, std::size_t n)
{
    std::size_t i = 0;
    __m256 vacc = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8)
    {
        __m128i va_half = _mm_loadu_si128(reinterpret_cast<const __m128i *>(a + i));
        __m128i vb_half = _mm_loadu_si128(reinterpret_cast<const __m128i *>(b + i));
        __m256 va = _mm256_cvtph_ps(va_half);
        __m256 vb = _mm256_cvtph_ps(vb_half);
        vacc = _mm256_fmadd_ps(va, vb, vacc);
    }
    __m128 lo = _mm256_castps256_ps128(vacc);
    __m128 hi = _mm256_extractf128_ps(vacc, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    float sum = _mm_cvtss_f32(lo);
    for (; i < n; ++i)
        sum += half_to_float(a[i]) * half_to_float(b[i]);
    return sum;
}

float norm2_f16(const half *vec, std::size_t n)
{
    std::size_t i = 0;
    __m256 vacc = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8)
    {
        __m128i v_half = _mm_loadu_si128(reinterpret_cast<const __m128i *>(vec + i));
        __m256 v = _mm256_cvtph_ps(v_half);
        vacc = _mm256_fmadd_ps(v, v, vacc);
    }
    __m128 lo = _mm256_castps256_ps128(vacc);
    __m128 hi = _mm256_extractf128_ps(vacc, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    float sum = _mm_cvtss_f32(lo);
    for (; i < n; ++i)
    {
        float v = half_to_float(vec[i]);
        sum += v * v;
    }
    return std::sqrt(sum);
}

void neg_f16(half *data, std::size_t n)
{
    std::size_t i = 0;
    __m256 vsign = _mm256_set1_ps(-0.0f);
    for (; i + 8 <= n; i += 8)
    {
        __m128i v_half = _mm_loadu_si128(reinterpret_cast<const __m128i *>(data + i));
        __m256 v = _mm256_cvtph_ps(v_half);
        __m256 vr = _mm256_xor_ps(v, vsign);
        __m128i vr_half = _mm256_cvtps_ph(vr, 0);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(data + i), vr_half);
    }
    for (; i < n; ++i)
        data[i] = float_to_half(-half_to_float(data[i]));
}

void sqrt_f16(half *data, std::size_t n)
{
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8)
    {
        __m128i v_half = _mm_loadu_si128(reinterpret_cast<const __m128i *>(data + i));
        __m256 v = _mm256_cvtph_ps(v_half);
        __m256 vr = _mm256_sqrt_ps(v);
        __m128i vr_half = _mm256_cvtps_ph(vr, 0);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(data + i), vr_half);
    }
    for (; i < n; ++i)
        data[i] = float_to_half(std::sqrt(half_to_float(data[i])));
}

void exp_f16(half *data, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        data[i] = float_to_half(std::exp(half_to_float(data[i])));
}

void log_f16(half *data, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        data[i] = float_to_half(std::log(half_to_float(data[i])));
}

void relu_f16(half *data, std::size_t n)
{
    std::size_t i = 0;
    __m256 vzero = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8)
    {
        __m128i v_half = _mm_loadu_si128(reinterpret_cast<const __m128i *>(data + i));
        __m256 v = _mm256_cvtph_ps(v_half);
        __m256 vr = _mm256_max_ps(v, vzero);
        __m128i vr_half = _mm256_cvtps_ph(vr, 0);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(data + i), vr_half);
    }
    for (; i < n; ++i)
    {
        float v = half_to_float(data[i]);
        data[i] = float_to_half(v < 0.0f ? 0.0f : v);
    }
}

void sigmoid_f16(half *data, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
    {
        float v = half_to_float(data[i]);
        data[i] = float_to_half(1.0f / (1.0f + std::exp(-v)));
    }
}

void tanh_f16(half *data, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        data[i] = float_to_half(std::tanh(half_to_float(data[i])));
}

void abs_f16(half *data, std::size_t n)
{
    std::size_t i = 0;
    __m256 vsign = _mm256_set1_ps(-0.0f);
    for (; i + 8 <= n; i += 8)
    {
        __m128i v_half = _mm_loadu_si128(reinterpret_cast<const __m128i *>(data + i));
        __m256 v = _mm256_cvtph_ps(v_half);
        __m256 vr = _mm256_andnot_ps(vsign, v);
        __m128i vr_half = _mm256_cvtps_ph(vr, 0);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(data + i), vr_half);
    }
    for (; i < n; ++i)
        data[i] = float_to_half(std::abs(half_to_float(data[i])));
}

void min_f16(half *a, const half *b, std::size_t n)
{
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8)
    {
        __m128i va_half = _mm_loadu_si128(reinterpret_cast<const __m128i *>(a + i));
        __m128i vb_half = _mm_loadu_si128(reinterpret_cast<const __m128i *>(b + i));
        __m256 va = _mm256_cvtph_ps(va_half);
        __m256 vb = _mm256_cvtph_ps(vb_half);
        __m256 vr = _mm256_min_ps(va, vb);
        __m128i vr_half = _mm256_cvtps_ph(vr, 0);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(a + i), vr_half);
    }
    for (; i < n; ++i)
    {
        float fa = half_to_float(a[i]);
        float fb = half_to_float(b[i]);
        a[i] = float_to_half(fa < fb ? fa : fb);
    }
}

void max_f16(half *a, const half *b, std::size_t n)
{
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8)
    {
        __m128i va_half = _mm_loadu_si128(reinterpret_cast<const __m128i *>(a + i));
        __m128i vb_half = _mm_loadu_si128(reinterpret_cast<const __m128i *>(b + i));
        __m256 va = _mm256_cvtph_ps(va_half);
        __m256 vb = _mm256_cvtph_ps(vb_half);
        __m256 vr = _mm256_max_ps(va, vb);
        __m128i vr_half = _mm256_cvtps_ph(vr, 0);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(a + i), vr_half);
    }
    for (; i < n; ++i)
    {
        float fa = half_to_float(a[i]);
        float fb = half_to_float(b[i]);
        a[i] = float_to_half(fa > fb ? fa : fb);
    }
}

void clamp_f16(half *data, half lo, half hi, std::size_t n)
{
    float flo = half_to_float(lo);
    float fhi = half_to_float(hi);
    __m256 vlo = _mm256_set1_ps(flo);
    __m256 vhi = _mm256_set1_ps(fhi);
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8)
    {
        __m128i v_half = _mm_loadu_si128(reinterpret_cast<const __m128i *>(data + i));
        __m256 v = _mm256_cvtph_ps(v_half);
        v = _mm256_max_ps(v, vlo);
        v = _mm256_min_ps(v, vhi);
        __m128i vr_half = _mm256_cvtps_ph(v, 0);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(data + i), vr_half);
    }
    for (; i < n; ++i)
    {
        float v = half_to_float(data[i]);
        if (v < flo)
            v = flo;
        if (v > fhi)
            v = fhi;
        data[i] = float_to_half(v);
    }
}

float sqdiff_sum_f16(const half *a, const half *b, std::size_t n)
{
    std::size_t i = 0;
    __m256 vacc = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8)
    {
        __m128i va_half = _mm_loadu_si128(reinterpret_cast<const __m128i *>(a + i));
        __m128i vb_half = _mm_loadu_si128(reinterpret_cast<const __m128i *>(b + i));
        __m256 va = _mm256_cvtph_ps(va_half);
        __m256 vb = _mm256_cvtph_ps(vb_half);
        __m256 diff = _mm256_sub_ps(va, vb);
        vacc = _mm256_fmadd_ps(diff, diff, vacc);
    }
    __m128 lo = _mm256_castps256_ps128(vacc);
    __m128 hi = _mm256_extractf128_ps(vacc, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    float sum = _mm_cvtss_f32(lo);
    for (; i < n; ++i)
    {
        float d = half_to_float(a[i]) - half_to_float(b[i]);
        sum += d * d;
    }
    return sum;
}

void gemv_f16(half alpha, const half *A, const half *x, half beta, half *y, std::size_t m,
              std::size_t n)
{
    float fa = half_to_float(alpha);
    float fb = half_to_float(beta);
    if (fb != 1.0f)
        for (std::size_t i = 0; i < m; ++i)
            y[i] = float_to_half(half_to_float(y[i]) * fb);
    for (std::size_t i = 0; i < m; ++i)
    {
        float sum = 0.0f;
        for (std::size_t j = 0; j < n; ++j)
            sum += half_to_float(A[i * n + j]) * half_to_float(x[j]);
        y[i] = float_to_half(half_to_float(y[i]) + fa * sum);
    }
}

void ger_f16(half alpha, const half *x, const half *y, half *A, std::size_t m, std::size_t n)
{
    float fa = half_to_float(alpha);
    for (std::size_t i = 0; i < m; ++i)
    {
        float xi = half_to_float(x[i]);
        for (std::size_t j = 0; j < n; ++j)
            A[i * n + j] =
                float_to_half(half_to_float(A[i * n + j]) + fa * xi * half_to_float(y[j]));
    }
}

void quantize_f16(const half *input, std::size_t n, int bits, std::uint8_t *output)
{
    if (n == 0)
        return;
    float scale = static_cast<float>((1 << bits) - 1);
    for (std::size_t i = 0; i < n; ++i)
    {
        float v = half_to_float(input[i]);
        if (v < 0.0f)
            v = 0.0f;
        if (v > 1.0f)
            v = 1.0f;
        output[i] = static_cast<std::uint8_t>(v * scale + 0.5f);
    }
}

void dequantize_f16(const std::uint8_t *input, std::size_t n, int bits, half *output)
{
    if (n == 0)
        return;
    float inv = 1.0f / static_cast<float>((1 << bits) - 1);
    for (std::size_t i = 0; i < n; ++i)
        output[i] = float_to_half(static_cast<float>(input[i]) * inv);
}

} // namespace nerve::simd::avx2

#pragma GCC pop_options

extern "C" void nerve_simd_assign_avx2(nerve::simd::SimdDispatchTable *table)
{
    using namespace nerve::simd::avx2;
    table->memcpy = memcpy;
    table->memset = memset;
    table->add = add;
    table->sub = sub;
    table->mul = mul;
    table->scale = scale;
    table->axpy = axpy;
    table->fmad = fmad;
    table->reduce_sum = reduce_sum;
    table->reduce_max = reduce_max;
    table->reduce_min = reduce_min;
    table->dot = dot;
    table->norm2 = norm2;
    table->sqdiff_sum = sqdiff_sum;
    table->abs = abs;
    table->neg = neg;
    table->sqrt = sqrt;
    table->exp = exp;
    table->log = log;
    table->relu = relu;
    table->sigmoid = sigmoid;
    table->tanh = tanh;
    table->min = min;
    table->max = max;
    table->clamp = clamp;
    table->fmad_f32 = fmad_f32;
    table->add_f32 = add_f32;
    table->sub_f32 = sub_f32;
    table->mul_f32 = mul_f32;
    table->scale_f32 = scale_f32;
    table->axpy_f32 = axpy_f32;
    table->reduce_sum_f32 = reduce_sum_f32;
    table->reduce_max_f32 = reduce_max_f32;
    table->reduce_min_f32 = reduce_min_f32;
    table->dot_f32 = dot_f32;
    table->norm2_f32 = norm2_f32;
    table->neg_f32 = neg_f32;
    table->sqrt_f32 = sqrt_f32;
    table->exp_f32 = exp_f32;
    table->log_f32 = log_f32;
    table->sigmoid_f32 = sigmoid_f32;
    table->tanh_f32 = tanh_f32;
    table->sqdiff_sum_f32 = sqdiff_sum_f32;
    table->euclidean_f32 = euclidean_f32;
    table->cosine_f32 = cosine_f32;
    table->abs_f32 = abs_f32;
    table->relu_f32 = relu_f32;
    table->min_f32 = min_f32;
    table->max_f32 = max_f32;
    table->clamp_f32 = clamp_f32;
    table->gemv_f32 = gemv_f32;
    table->ger_f32 = ger_f32;
    table->gemv = gemv;
    table->ger = ger;
    table->add_f16 = add_f16;
    table->sub_f16 = sub_f16;
    table->mul_f16 = mul_f16;
    table->scale_f16 = scale_f16;
    table->axpy_f16 = axpy_f16;
    table->fmad_f16 = fmad_f16;
    table->reduce_sum_f16 = reduce_sum_f16;
    table->reduce_max_f16 = reduce_max_f16;
    table->reduce_min_f16 = reduce_min_f16;
    table->dot_f16 = dot_f16;
    table->norm2_f16 = norm2_f16;
    table->neg_f16 = neg_f16;
    table->sqrt_f16 = sqrt_f16;
    table->exp_f16 = exp_f16;
    table->log_f16 = log_f16;
    table->relu_f16 = relu_f16;
    table->sigmoid_f16 = sigmoid_f16;
    table->tanh_f16 = tanh_f16;
    table->abs_f16 = abs_f16;
    table->min_f16 = min_f16;
    table->max_f16 = max_f16;
    table->clamp_f16 = clamp_f16;
    table->sqdiff_sum_f16 = sqdiff_sum_f16;
    table->gemv_f16 = gemv_f16;
    table->ger_f16 = ger_f16;
    table->quantize_f16 = quantize_f16;
    table->dequantize_f16 = dequantize_f16;
}
