#include "nerve/simd/simd_arithmetic.hpp"
#include "nerve/simd/simd_blas.hpp"
#include "nerve/simd/simd_compare.hpp"
#include "nerve/simd/simd_distance.hpp"
#include "nerve/simd/simd_elementwise.hpp"
#include "nerve/simd/simd_memory.hpp"
#include "nerve/simd/simd_nn.hpp"
#include "nerve/simd/simd_quantize.hpp"
#include "nerve/simd/simd_reduce.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

// Scalar reference implementations (golden values)
//
// Note: Manual byte loops are used for memcpy/memset instead of
// std::memcpy/std::memset to avoid GCC 13/14 -Wstringop-overflow false
// positives where the optimizer over-approximates the dynamic size
// parameter and warns the memset bound (SIZE_MAX & ~15) exceeds the
// maximum object size.

namespace ref
{

void memcpy(void *dst, const void *src, std::size_t bytes)
{
    auto *d = static_cast<std::uint8_t *>(dst);
    const auto *s = static_cast<const std::uint8_t *>(src);
    for (std::size_t i = 0; i < bytes; ++i)
        d[i] = s[i];
}

void memset(void *dst, int value, std::size_t bytes)
{
    auto *d = static_cast<std::uint8_t *>(dst);
    auto v = static_cast<std::uint8_t>(value);
    for (std::size_t i = 0; i < bytes; ++i)
        d[i] = v;
}

void add(double *a, const double *b, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        a[i] += b[i];
}

void sub(double *a, const double *b, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        a[i] -= b[i];
}

void mul(double *a, const double *b, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        a[i] *= b[i];
}

void scale(double *a, double alpha, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        a[i] *= alpha;
}

void axpy(double alpha, const double *x, double *y, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        y[i] += alpha * x[i];
}

void fmad(const double *a, const double *b, double *c, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        c[i] += a[i] * b[i];
}

double reduce_sum(const double *data, std::size_t n)
{
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        s += data[i];
    return s;
}

double reduce_max(const double *data, std::size_t n)
{
    if (n == 0)
        return 0.0;
    double v = data[0];
    for (std::size_t i = 1; i < n; ++i)
        if (data[i] > v)
            v = data[i];
    return v;
}

double reduce_min(const double *data, std::size_t n)
{
    if (n == 0)
        return 0.0;
    double v = data[0];
    for (std::size_t i = 1; i < n; ++i)
        if (data[i] < v)
            v = data[i];
    return v;
}

double dot(const double *a, const double *b, std::size_t n)
{
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        s += a[i] * b[i];
    return s;
}

double norm2(const double *vec, std::size_t n)
{
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        s += vec[i] * vec[i];
    return std::sqrt(s);
}

double sqdiff_sum(const double *a, const double *b, std::size_t n)
{
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i)
    {
        double d = a[i] - b[i];
        s += d * d;
    }
    return s;
}

void abs(double *data, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        data[i] = std::abs(data[i]);
}

void neg(double *data, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        data[i] = -data[i];
}

void sqrt(double *data, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
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
    for (std::size_t i = 0; i < n; ++i)
        data[i] = (data[i] < 0.0) ? 0.0 : data[i];
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
    for (std::size_t i = 0; i < n; ++i)
        a[i] = (b[i] < a[i]) ? b[i] : a[i];
}

void max(double *a, const double *b, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        a[i] = (b[i] > a[i]) ? b[i] : a[i];
}

void clamp(double *data, double lo, double hi, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
    {
        if (data[i] < lo)
            data[i] = lo;
        else if (data[i] > hi)
            data[i] = hi;
    }
}

void gemv(double alpha, const double *A, const double *x, double beta, double *y, std::size_t m,
          std::size_t n)
{
    for (std::size_t i = 0; i < m; ++i)
    {
        double sum = 0.0;
        for (std::size_t j = 0; j < n; ++j)
            sum += A[i * n + j] * x[j];
        y[i] = alpha * sum + beta * y[i];
    }
}

void ger(double alpha, const double *x, const double *y, double *A, std::size_t m, std::size_t n)
{
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j)
            A[i * n + j] += alpha * x[i] * y[j];
}

double euclidean(const double *a, const double *b, std::size_t dim)
{
    return std::sqrt(sqdiff_sum(a, b, dim));
}

double manhattan(const double *a, const double *b, std::size_t dim)
{
    double s = 0.0;
    for (std::size_t i = 0; i < dim; ++i)
        s += std::abs(a[i] - b[i]);
    return s;
}

double cosine(const double *a, const double *b, std::size_t dim)
{
    double d = dot(a, b, dim);
    double na = norm2(a, dim);
    double nb = norm2(b, dim);
    if (na == 0.0 || nb == 0.0)
        return 1.0;
    double cos_sim = d / (na * nb);
    if (cos_sim < -1.0)
        cos_sim = -1.0;
    if (cos_sim > 1.0)
        cos_sim = 1.0;
    return 1.0 - cos_sim;
}

void batchnorm(double *data, std::size_t n, double mean, double std_inv)
{
    for (std::size_t i = 0; i < n; ++i)
        data[i] = (data[i] - mean) * std_inv;
}

void softmax(double *data, std::size_t n)
{
    if (n == 0)
        return;
    double mx = data[0];
    for (std::size_t i = 1; i < n; ++i)
        if (data[i] > mx)
            mx = data[i];
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i)
    {
        data[i] = std::exp(data[i] - mx);
        sum += data[i];
    }
    double inv = 1.0 / sum;
    for (std::size_t i = 0; i < n; ++i)
        data[i] *= inv;
}

void quantize(const double *input, std::size_t n, int bits, std::uint8_t *output)
{
    double scale = static_cast<double>((1 << bits) - 1);
    for (std::size_t i = 0; i < n; ++i)
    {
        double clipped = std::max(0.0, std::min(input[i], 1.0));
        output[i] = static_cast<std::uint8_t>(clipped * scale + 0.5);
    }
}

void dequantize(const std::uint8_t *input, std::size_t n, int bits, double *output)
{
    double inv = 1.0 / static_cast<double>((1 << bits) - 1);
    for (std::size_t i = 0; i < n; ++i)
        output[i] = static_cast<double>(input[i]) * inv;
}

// Float32 reference implementations

void add_f32(float *a, const float *b, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        a[i] += b[i];
}

void sub_f32(float *a, const float *b, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        a[i] -= b[i];
}

void mul_f32(float *a, const float *b, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        a[i] *= b[i];
}

void scale_f32(float *a, float alpha, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        a[i] *= alpha;
}

void axpy_f32(float alpha, const float *x, float *y, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        y[i] += alpha * x[i];
}

float reduce_sum_f32(const float *data, std::size_t n)
{
    float s = 0.0f;
    for (std::size_t i = 0; i < n; ++i)
        s += data[i];
    return s;
}

float reduce_max_f32(const float *data, std::size_t n)
{
    if (n == 0)
        return 0.0f;
    float v = data[0];
    for (std::size_t i = 1; i < n; ++i)
        if (data[i] > v)
            v = data[i];
    return v;
}

float dot_f32(const float *a, const float *b, std::size_t n)
{
    float s = 0.0f;
    for (std::size_t i = 0; i < n; ++i)
        s += a[i] * b[i];
    return s;
}

float norm2_f32(const float *vec, std::size_t n)
{
    float s = 0.0f;
    for (std::size_t i = 0; i < n; ++i)
        s += vec[i] * vec[i];
    return std::sqrt(s);
}

void gemv_f32(float alpha, const float *A, const float *x, float beta, float *y, std::size_t m,
              std::size_t n)
{
    for (std::size_t i = 0; i < m; ++i)
    {
        float sum = 0.0f;
        for (std::size_t j = 0; j < n; ++j)
            sum += A[i * n + j] * x[j];
        y[i] = alpha * sum + beta * y[i];
    }
}

void ger_f32(float alpha, const float *x, const float *y, float *A, std::size_t m, std::size_t n)
{
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j)
            A[i * n + j] += alpha * x[i] * y[j];
}

void abs_f32(float *data, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        data[i] = std::abs(data[i]);
}

void relu_f32(float *data, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        data[i] = (data[i] < 0.0f) ? 0.0f : data[i];
}

void min_f32(float *a, const float *b, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        a[i] = (b[i] < a[i]) ? b[i] : a[i];
}

void max_f32(float *a, const float *b, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        a[i] = (b[i] > a[i]) ? b[i] : a[i];
}

void clamp_f32(float *data, float lo, float hi, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
    {
        if (data[i] < lo)
            data[i] = lo;
        else if (data[i] > hi)
            data[i] = hi;
    }
}

float sqdiff_sum_f32(const float *a, const float *b, std::size_t n)
{
    float s = 0.0f;
    for (std::size_t i = 0; i < n; ++i)
    {
        float d = a[i] - b[i];
        s += d * d;
    }
    return s;
}

float euclidean_f32(const float *a, const float *b, std::size_t n)
{
    return std::sqrt(sqdiff_sum_f32(a, b, n));
}

float cosine_f32(const float *a, const float *b, std::size_t n)
{
    float d = dot_f32(a, b, n);
    float na = norm2_f32(a, n);
    float nb = norm2_f32(b, n);
    if (na == 0.0f || nb == 0.0f)
        return 1.0f;
    float cos_sim = d / (na * nb);
    if (cos_sim < -1.0f)
        cos_sim = -1.0f;
    if (cos_sim > 1.0f)
        cos_sim = 1.0f;
    return 1.0f - cos_sim;
}

void fmad_f32(const float *a, const float *b, float *c, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        c[i] += a[i] * b[i];
}

float reduce_min_f32(const float *data, std::size_t n)
{
    if (n == 0)
        return 0.0f;
    float v = data[0];
    for (std::size_t i = 1; i < n; ++i)
        if (data[i] < v)
            v = data[i];
    return v;
}

void neg_f32(float *data, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        data[i] = -data[i];
}

void sqrt_f32(float *data, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
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

void axpy_f16(nerve::simd::half alpha, const nerve::simd::half *x, nerve::simd::half *y,
              std::size_t n)
{
    float fa = nerve::simd::half_to_float(alpha);
    for (std::size_t i = 0; i < n; ++i)
        y[i] = nerve::simd::float_to_half(nerve::simd::half_to_float(y[i]) +
                                          fa * nerve::simd::half_to_float(x[i]));
}

void fmad_f16(const nerve::simd::half *a, const nerve::simd::half *b, nerve::simd::half *c,
              std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        c[i] = nerve::simd::float_to_half(nerve::simd::half_to_float(c[i]) +
                                          nerve::simd::half_to_float(a[i]) *
                                              nerve::simd::half_to_float(b[i]));
}

void sqrt_f16(nerve::simd::half *data, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        data[i] = nerve::simd::float_to_half(std::sqrt(nerve::simd::half_to_float(data[i])));
}

void exp_f16(nerve::simd::half *data, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        data[i] = nerve::simd::float_to_half(std::exp(nerve::simd::half_to_float(data[i])));
}

void log_f16(nerve::simd::half *data, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        data[i] = nerve::simd::float_to_half(std::log(nerve::simd::half_to_float(data[i])));
}

void min_f16(nerve::simd::half *a, const nerve::simd::half *b, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
    {
        float va = nerve::simd::half_to_float(a[i]);
        float vb = nerve::simd::half_to_float(b[i]);
        a[i] = nerve::simd::float_to_half(vb < va ? vb : va);
    }
}

void max_f16(nerve::simd::half *a, const nerve::simd::half *b, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
    {
        float va = nerve::simd::half_to_float(a[i]);
        float vb = nerve::simd::half_to_float(b[i]);
        a[i] = nerve::simd::float_to_half(vb > va ? vb : va);
    }
}

void clamp_f16(nerve::simd::half *data, nerve::simd::half lo, nerve::simd::half hi, std::size_t n)
{
    float flo = nerve::simd::half_to_float(lo);
    float fhi = nerve::simd::half_to_float(hi);
    for (std::size_t i = 0; i < n; ++i)
    {
        float v = nerve::simd::half_to_float(data[i]);
        if (v < flo)
            v = flo;
        else if (v > fhi)
            v = fhi;
        data[i] = nerve::simd::float_to_half(v);
    }
}

void sigmoid_f16(nerve::simd::half *data, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
    {
        float v = nerve::simd::half_to_float(data[i]);
        data[i] = nerve::simd::float_to_half(1.0f / (1.0f + std::exp(-v)));
    }
}

void tanh_f16(nerve::simd::half *data, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        data[i] = nerve::simd::float_to_half(std::tanh(nerve::simd::half_to_float(data[i])));
}

float sqdiff_sum_f16(const nerve::simd::half *a, const nerve::simd::half *b, std::size_t n)
{
    float s = 0.0f;
    for (std::size_t i = 0; i < n; ++i)
    {
        float d = nerve::simd::half_to_float(a[i]) - nerve::simd::half_to_float(b[i]);
        s += d * d;
    }
    return s;
}

float manhattan_f16(const nerve::simd::half *a, const nerve::simd::half *b, std::size_t n)
{
    float s = 0.0f;
    for (std::size_t i = 0; i < n; ++i)
    {
        float d = nerve::simd::half_to_float(a[i]) - nerve::simd::half_to_float(b[i]);
        s += (d < 0.0f) ? -d : d;
    }
    return s;
}

float euclidean_f16(const nerve::simd::half *a, const nerve::simd::half *b, std::size_t n)
{
    return std::sqrt(sqdiff_sum_f16(a, b, n));
}

float cosine_f16(const nerve::simd::half *a, const nerve::simd::half *b, std::size_t n)
{
    float d = 0.0f, na = 0.0f, nb = 0.0f;
    for (std::size_t i = 0; i < n; ++i)
    {
        float fa = nerve::simd::half_to_float(a[i]);
        float fb = nerve::simd::half_to_float(b[i]);
        d += fa * fb;
        na += fa * fa;
        nb += fb * fb;
    }
    na = std::sqrt(na);
    nb = std::sqrt(nb);
    if (na == 0.0f || nb == 0.0f)
        return 1.0f;
    float cos_sim = d / (na * nb);
    if (cos_sim < -1.0f)
        cos_sim = -1.0f;
    if (cos_sim > 1.0f)
        cos_sim = 1.0f;
    return 1.0f - cos_sim;
}

float norm2_f16(const nerve::simd::half *vec, std::size_t n)
{
    float s = 0.0f;
    for (std::size_t i = 0; i < n; ++i)
    {
        float v = nerve::simd::half_to_float(vec[i]);
        s += v * v;
    }
    return std::sqrt(s);
}

float reduce_max_f16(const nerve::simd::half *data, std::size_t n)
{
    if (n == 0)
        return 0.0f;
    float v = nerve::simd::half_to_float(data[0]);
    for (std::size_t i = 1; i < n; ++i)
    {
        float x = nerve::simd::half_to_float(data[i]);
        if (x > v)
            v = x;
    }
    return v;
}

float reduce_min_f16(const nerve::simd::half *data, std::size_t n)
{
    if (n == 0)
        return 0.0f;
    float v = nerve::simd::half_to_float(data[0]);
    for (std::size_t i = 1; i < n; ++i)
    {
        float x = nerve::simd::half_to_float(data[i]);
        if (x < v)
            v = x;
    }
    return v;
}

void quantize_f16(const nerve::simd::half *input, std::size_t n, int bits, std::uint8_t *output)
{
    if (n == 0)
        return;
    float scale = static_cast<float>((1 << bits) - 1);
    for (std::size_t i = 0; i < n; ++i)
    {
        float v = nerve::simd::half_to_float(input[i]);
        if (v < 0.0f)
            v = 0.0f;
        if (v > 1.0f)
            v = 1.0f;
        output[i] = static_cast<std::uint8_t>(v * scale + 0.5f);
    }
}

void dequantize_f16(const std::uint8_t *input, std::size_t n, int bits, nerve::simd::half *output)
{
    if (n == 0)
        return;
    float inv = 1.0f / static_cast<float>((1 << bits) - 1);
    for (std::size_t i = 0; i < n; ++i)
        output[i] = nerve::simd::float_to_half(static_cast<float>(input[i]) * inv);
}

void batchnorm_f16(nerve::simd::half *data, std::size_t n, float mean, float std_inv)
{
    for (std::size_t i = 0; i < n; ++i)
        data[i] =
            nerve::simd::float_to_half((nerve::simd::half_to_float(data[i]) - mean) * std_inv);
}

void softmax_f16(nerve::simd::half *data, std::size_t n)
{
    if (n == 0)
        return;
    float mx = nerve::simd::half_to_float(data[0]);
    for (std::size_t i = 1; i < n; ++i)
    {
        float v = nerve::simd::half_to_float(data[i]);
        if (v > mx)
            mx = v;
    }
    float sum = 0.0f;
    for (std::size_t i = 0; i < n; ++i)
    {
        float v = std::exp(nerve::simd::half_to_float(data[i]) - mx);
        data[i] = nerve::simd::float_to_half(v);
        sum += v;
    }
    float inv = 1.0f / sum;
    for (std::size_t i = 0; i < n; ++i)
        data[i] = nerve::simd::float_to_half(nerve::simd::half_to_float(data[i]) * inv);
}

} // namespace ref

// Test helpers

namespace
{

constexpr double kTol = 1e-12;
constexpr double kRelTol = 1e-10;
// Slightly larger tolerance for transcendental functions where target-flag-
// dependent std::exp/log implementations may differ by a few ULPs.
constexpr double kRelTolTranscendental = 1e-8;

// Random seeding is deterministic -- uses a simple LCG so no external dep
static double rand_double(std::uint64_t &seed)
{
    seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
    std::uint64_t hi = seed >> 11;
    std::uint64_t lo = seed & 0x7FFFFFFFFFFFULL;
    return static_cast<double>(hi) / static_cast<double>(lo + 1);
}

static double rand_range(std::uint64_t &seed, double lo, double hi)
{
    return lo + rand_double(seed) * (hi - lo);
}

std::vector<double> make_rand_vec(std::size_t n, std::uint64_t seed, double lo = -10.0,
                                  double hi = 10.0)
{
    std::vector<double> v(n);
    for (std::size_t i = 0; i < n; ++i)
        v[i] = rand_range(seed, lo, hi);
    return v;
}

bool approx(double a, double b)
{
    // Handle infinity identity (both +inf, both -inf)
    if (a == b)
        return true;
    if (!std::isfinite(a) || !std::isfinite(b))
        return std::isinf(a) && std::isinf(b) && ((a < 0.0) == (b < 0.0));
    double diff = std::abs(a - b);
    double scale = std::max(1.0, std::max(std::abs(a), std::abs(b)));
    return diff <= kRelTol * scale;
}

bool approx_lax(double a, double b)
{
    if (a == b)
        return true;
    if (!std::isfinite(a) || !std::isfinite(b))
        return std::isinf(a) && std::isinf(b) && ((a < 0.0) == (b < 0.0));
    double diff = std::abs(a - b);
    double scale = std::max(1.0, std::max(std::abs(a), std::abs(b)));
    return diff <= kRelTolTranscendental * scale;
}

bool all_approx_lax(const double *simd, const double *ref, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        if (!approx_lax(simd[i], ref[i]))
            return false;
    return true;
}

bool approx_abs(double a, double b)
{
    return std::abs(a - b) <= kTol;
}

bool all_approx(const double *simd, const double *ref, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        if (!approx(simd[i], ref[i]))
            return false;
    return true;
}

bool all_eq_byte(const std::uint8_t *simd, const std::uint8_t *ref, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        if (simd[i] != ref[i])
            return false;
    return true;
}

// Shorthand to run a binary-op test for various sizes
template <typename SimdFn, typename RefFn>
bool test_binary_op(const char * /*name*/, SimdFn simd_fn, RefFn ref_fn,
                    const std::vector<std::size_t> &sizes, std::uint64_t seed = 42)
{
    for (auto n : sizes)
    {
        auto a = make_rand_vec(n, seed++, -5.0, 5.0);
        auto b = make_rand_vec(n, seed++, -5.0, 5.0);
        auto c_simd = a;
        auto c_ref = a;
        simd_fn(c_simd.data(), b.data(), n);
        ref_fn(c_ref.data(), b.data(), n);
        if (!all_approx(c_simd.data(), c_ref.data(), n))
        {
            std::cerr << "  FAIL at n=" << n << "\n";
            return false;
        }
    }
    return true;
}

template <typename SimdFn, typename RefFn>
bool test_unary_op(const char * /*name*/, SimdFn simd_fn, RefFn ref_fn,
                   const std::vector<std::size_t> &sizes, std::uint64_t seed = 42)
{
    for (auto n : sizes)
    {
        auto data_simd = make_rand_vec(n, seed++, -5.0, 5.0);
        auto data_ref = data_simd;
        simd_fn(data_simd.data(), n);
        ref_fn(data_ref.data(), n);
        if (!all_approx(data_simd.data(), data_ref.data(), n))
        {
            std::cerr << "  FAIL at n=" << n << "\n";
            return false;
        }
    }
    return true;
}

} // namespace

// Individual test cases

bool test_memcpy()
{
    const std::size_t sizes[] = {0, 1, 3, 7, 16, 64, 71, 127, 1024};
    for (auto bytes : sizes)
    {
        std::vector<char> src(bytes + 32, 0);
        std::vector<char> dst_simd(bytes + 32, 0xAB);
        std::vector<char> dst_ref(bytes + 32, 0xAB);
        for (std::size_t i = 0; i < bytes; ++i)
            src[i] = static_cast<char>((i * 7 + 13) & 0xFF);
        nerve::simd::simd_memcpy(dst_simd.data(), src.data(), bytes);
        ref::memcpy(dst_ref.data(), src.data(), bytes);
        if (std::memcmp(dst_simd.data(), dst_ref.data(), bytes + 32) != 0)
        {
            std::cerr << "  FAIL memcpy at bytes=" << bytes << "\n";
            return false;
        }
    }
    return true;
}

bool test_memset()
{
    const std::size_t sizes[] = {0, 1, 3, 7, 16, 64, 71, 127, 1024};
    const int values[] = {0, 0xFF, 0xAB, 0x00, 0x80};
    for (auto bytes : sizes)
    {
        for (auto val : values)
        {
            std::vector<char> buf_simd(bytes + 16, 0xCD);
            std::vector<char> buf_ref(bytes + 16, 0xCD);
            nerve::simd::simd_memset(buf_simd.data(), val, bytes);
            ref::memset(buf_ref.data(), val, bytes);
            if (std::memcmp(buf_simd.data(), buf_ref.data(), bytes + 16) != 0)
            {
                std::cerr << "  FAIL memset at bytes=" << bytes << " val=" << val << "\n";
                return false;
            }
        }
    }
    return true;
}

bool test_memcpy_aligned()
{
    const std::size_t sizes[] = {0, 1, 3, 7, 16, 64, 67, 127, 128, 129, 256};
    for (auto bytes : sizes)
    {
        alignas(64) char src[512];
        alignas(64) char dst_simd[512] = {};
        alignas(64) char dst_ref[512] = {};
        for (std::size_t i = 0; i < bytes; ++i)
            src[i] = static_cast<char>((i * 7 + 13) & 0xFF);
        nerve::simd::simd_memcpy_aligned(dst_simd, src, bytes);
        ref::memcpy(dst_ref, src, bytes);
        if (std::memcmp(dst_simd, dst_ref, bytes) != 0)
        {
            std::cerr << "  FAIL memcpy_aligned at bytes=" << bytes << "\n";
            return false;
        }
    }
    return true;
}

bool test_add()
{
    const std::vector<std::size_t> sizes = {0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 33, 100};
    return test_binary_op("add", nerve::simd::simd_add, ref::add, sizes);
}

bool test_sub()
{
    const std::vector<std::size_t> sizes = {0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 33, 100};
    return test_binary_op("sub", nerve::simd::simd_sub, ref::sub, sizes);
}

bool test_mul()
{
    const std::vector<std::size_t> sizes = {0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 33, 100};
    return test_binary_op("mul", nerve::simd::simd_mul, ref::mul, sizes);
}

bool test_scale()
{
    const std::vector<std::size_t> sizes = {0, 1, 3, 7, 16, 31, 32, 100};
    const double alphas[] = {0.0, 1.0, -1.0, 2.5, 0.5, -3.0};
    std::uint64_t seed = 42;
    for (auto n : sizes)
    {
        for (auto alpha : alphas)
        {
            auto data_simd = make_rand_vec(n, seed++, -5.0, 5.0);
            auto data_ref = data_simd;
            nerve::simd::simd_scale(data_simd.data(), alpha, n);
            ref::scale(data_ref.data(), alpha, n);
            if (!all_approx(data_simd.data(), data_ref.data(), n))
            {
                std::cerr << "  FAIL scale at n=" << n << " alpha=" << alpha << "\n";
                return false;
            }
        }
    }
    return true;
}

bool test_axpy()
{
    const std::vector<std::size_t> sizes = {0, 1, 3, 7, 16, 31, 32, 33, 100};
    const double alphas[] = {0.0, 1.0, -1.0, 2.5, -0.5};
    std::uint64_t seed = 42;
    for (auto n : sizes)
    {
        for (auto alpha : alphas)
        {
            auto x_simd = make_rand_vec(n, seed++, -5.0, 5.0);
            auto x_ref = x_simd;
            auto y_simd = make_rand_vec(n, seed++, -5.0, 5.0);
            auto y_ref = y_simd;
            nerve::simd::simd_axpy(alpha, x_simd.data(), y_simd.data(), n);
            ref::axpy(alpha, x_ref.data(), y_ref.data(), n);
            if (!all_approx(y_simd.data(), y_ref.data(), n))
            {
                std::cerr << "  FAIL axpy at n=" << n << " alpha=" << alpha << "\n";
                return false;
            }
        }
    }
    return true;
}

bool test_fmad()
{
    const std::vector<std::size_t> sizes = {0, 1, 3, 7, 16, 31, 32, 33, 100};
    for (auto n : sizes)
    {
        auto a = make_rand_vec(n, 42);
        auto b = make_rand_vec(n, 84);
        auto c_simd = make_rand_vec(n, 126);
        auto c_ref = c_simd;
        nerve::simd::simd_fmad(a.data(), b.data(), c_simd.data(), n);
        ref::fmad(a.data(), b.data(), c_ref.data(), n);
        if (!all_approx(c_simd.data(), c_ref.data(), n))
        {
            std::cerr << "  FAIL fmad at n=" << n << "\n";
            return false;
        }
    }
    return true;
}

bool test_reduce_sum()
{
    const std::vector<std::size_t> sizes = {0,  1,  2,  3,  4,  5,  7,   8,
                                            15, 16, 17, 31, 32, 33, 100, 1000};
    for (auto n : sizes)
    {
        auto data = make_rand_vec(n, 42);
        double simd_val = nerve::simd::simd_reduce_sum(data.data(), n);
        double ref_val = ref::reduce_sum(data.data(), n);
        if (!approx(simd_val, ref_val))
        {
            std::cerr << "  FAIL reduce_sum at n=" << n << " simd=" << simd_val
                      << " ref=" << ref_val << "\n";
            return false;
        }
    }
    return true;
}

bool test_reduce_max()
{
    const std::vector<std::size_t> sizes = {0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 33, 100};
    for (auto n : sizes)
    {
        auto data = make_rand_vec(n, 42);
        double simd_val = nerve::simd::simd_reduce_max(data.data(), n);
        double ref_val = ref::reduce_max(data.data(), n);
        if (!approx(simd_val, ref_val))
        {
            std::cerr << "  FAIL reduce_max at n=" << n << " simd=" << simd_val
                      << " ref=" << ref_val << "\n";
            return false;
        }
    }
    return true;
}

bool test_reduce_min()
{
    const std::vector<std::size_t> sizes = {0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 33, 100};
    for (auto n : sizes)
    {
        auto data = make_rand_vec(n, 42);
        double simd_val = nerve::simd::simd_reduce_min(data.data(), n);
        double ref_val = ref::reduce_min(data.data(), n);
        if (!approx(simd_val, ref_val))
        {
            std::cerr << "  FAIL reduce_min at n=" << n << " simd=" << simd_val
                      << " ref=" << ref_val << "\n";
            return false;
        }
    }
    return true;
}

bool test_dot()
{
    const std::vector<std::size_t> sizes = {0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 33, 100};
    for (auto n : sizes)
    {
        auto a = make_rand_vec(n, 42);
        auto b = make_rand_vec(n, 84);
        double simd_val = nerve::simd::simd_dot(a.data(), b.data(), n);
        double ref_val = ref::dot(a.data(), b.data(), n);
        if (!approx(simd_val, ref_val))
        {
            std::cerr << "  FAIL dot at n=" << n << " simd=" << simd_val << " ref=" << ref_val
                      << "\n";
            return false;
        }
    }
    return true;
}

bool test_norm2()
{
    const std::vector<std::size_t> sizes = {0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 33, 100};
    for (auto n : sizes)
    {
        auto vec = make_rand_vec(n, 42);
        double simd_val = nerve::simd::simd_norm2(vec.data(), n);
        double ref_val = ref::norm2(vec.data(), n);
        if (!approx(simd_val, ref_val))
        {
            std::cerr << "  FAIL norm2 at n=" << n << " simd=" << simd_val << " ref=" << ref_val
                      << "\n";
            return false;
        }
    }
    return true;
}

bool test_sqdiff_sum()
{
    const std::vector<std::size_t> sizes = {0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 33, 100};
    for (auto n : sizes)
    {
        auto a = make_rand_vec(n, 42);
        auto b = make_rand_vec(n, 84);
        double simd_val = nerve::simd::simd_sqdiff_sum(a.data(), b.data(), n);
        double ref_val = ref::sqdiff_sum(a.data(), b.data(), n);
        if (!approx(simd_val, ref_val))
        {
            std::cerr << "  FAIL sqdiff_sum at n=" << n << " simd=" << simd_val
                      << " ref=" << ref_val << "\n";
            return false;
        }
    }
    return true;
}

bool test_abs()
{
    const std::vector<std::size_t> sizes = {0, 1, 3, 7, 16, 31, 32, 33, 100};
    return test_unary_op("abs", nerve::simd::simd_abs, ref::abs, sizes);
}

bool test_neg()
{
    const std::vector<std::size_t> sizes = {0, 1, 3, 7, 16, 31, 32, 33, 100};
    return test_unary_op("neg", nerve::simd::simd_neg, ref::neg, sizes);
}

bool test_sqrt()
{
    // Use non-negative values
    const std::vector<std::size_t> sizes = {0, 1, 3, 7, 16, 31, 32, 33, 100};
    for (auto n : sizes)
    {
        auto data_simd = make_rand_vec(n, 42, 0.01, 100.0);
        auto data_ref = data_simd;
        nerve::simd::simd_sqrt(data_simd.data(), n);
        ref::sqrt(data_ref.data(), n);
        if (!all_approx(data_simd.data(), data_ref.data(), n))
        {
            std::cerr << "  FAIL sqrt at n=" << n << "\n";
            return false;
        }
    }
    return true;
}

bool test_exp()
{
    // Use slightly larger tolerance for exp due to target-flag-dependent
    // std::exp implementations that may differ by a few ULPs.
    const std::vector<std::size_t> sizes = {0, 1, 3, 7, 16, 31, 32, 33, 100};
    for (auto n : sizes)
    {
        auto data_simd = make_rand_vec(n, 42, -10.0, 10.0);
        auto data_ref = data_simd;
        nerve::simd::simd_exp(data_simd.data(), n);
        ref::exp(data_ref.data(), n);
        if (!all_approx_lax(data_simd.data(), data_ref.data(), n))
        {
            std::cerr << "  FAIL exp at n=" << n << "\n";
            return false;
        }
    }
    return true;
}

bool test_log()
{
    // Use positive values only
    const std::vector<std::size_t> sizes = {0, 1, 3, 7, 16, 31, 32, 33, 100};
    for (auto n : sizes)
    {
        auto data_simd = make_rand_vec(n, 42, 0.001, 100.0);
        auto data_ref = data_simd;
        nerve::simd::simd_log(data_simd.data(), n);
        ref::log(data_ref.data(), n);
        if (!all_approx(data_simd.data(), data_ref.data(), n))
        {
            std::cerr << "  FAIL log at n=" << n << "\n";
            return false;
        }
    }
    return true;
}

bool test_relu()
{
    const std::vector<std::size_t> sizes = {0, 1, 3, 7, 16, 31, 32, 33, 100};
    return test_unary_op("relu", nerve::simd::simd_relu, ref::relu, sizes);
}

bool test_sigmoid()
{
    const std::vector<std::size_t> sizes = {0, 1, 3, 7, 16, 31, 32, 33, 100};
    for (auto n : sizes)
    {
        auto data_simd = make_rand_vec(n, 42, -10.0, 10.0);
        auto data_ref = data_simd;
        nerve::simd::simd_sigmoid(data_simd.data(), n);
        ref::sigmoid(data_ref.data(), n);
        if (!all_approx(data_simd.data(), data_ref.data(), n))
        {
            std::cerr << "  FAIL sigmoid at n=" << n << "\n";
            return false;
        }
    }
    return true;
}

bool test_tanh()
{
    const std::vector<std::size_t> sizes = {0, 1, 3, 7, 16, 31, 32, 33, 100};
    return test_unary_op("tanh", nerve::simd::simd_tanh, ref::tanh, sizes);
}

bool test_min()
{
    const std::vector<std::size_t> sizes = {0, 1, 3, 7, 16, 31, 32, 33, 100};
    for (auto n : sizes)
    {
        std::uint64_t seed = 42;
        auto a = make_rand_vec(n, seed++);
        auto b = make_rand_vec(n, seed++);
        auto c_simd = a;
        auto c_ref = a;
        nerve::simd::simd_min(c_simd.data(), b.data(), n);
        ref::min(c_ref.data(), b.data(), n);
        if (!all_approx(c_simd.data(), c_ref.data(), n))
        {
            std::cerr << "  FAIL min at n=" << n << "\n";
            return false;
        }
    }
    return true;
}

bool test_max()
{
    const std::vector<std::size_t> sizes = {0, 1, 3, 7, 16, 31, 32, 33, 100};
    for (auto n : sizes)
    {
        std::uint64_t seed = 42;
        auto a = make_rand_vec(n, seed++);
        auto b = make_rand_vec(n, seed++);
        auto c_simd = a;
        auto c_ref = a;
        nerve::simd::simd_max(c_simd.data(), b.data(), n);
        ref::max(c_ref.data(), b.data(), n);
        if (!all_approx(c_simd.data(), c_ref.data(), n))
        {
            std::cerr << "  FAIL max at n=" << n << "\n";
            return false;
        }
    }
    return true;
}

bool test_clamp()
{
    const std::vector<std::size_t> sizes = {0, 1, 3, 7, 16, 31, 32, 33, 100};
    const double bounds[][2] = {{0.0, 1.0}, {-2.0, 2.0}, {-5.0, 5.0}, {-10.0, 10.0}};
    for (auto n : sizes)
    {
        auto data_simd = make_rand_vec(n, 42, -15.0, 15.0);
        auto data_ref = data_simd;
        for (auto [lo, hi] : bounds)
        {
            nerve::simd::simd_clamp(data_simd.data(), lo, hi, n);
            ref::clamp(data_ref.data(), lo, hi, n);
            if (!all_approx(data_simd.data(), data_ref.data(), n))
            {
                std::cerr << "  FAIL clamp at n=" << n << " lo=" << lo << " hi=" << hi << "\n";
                return false;
            }
        }
    }
    return true;
}

bool test_gemv()
{
    const std::size_t dims[][2] = {{1, 1}, {1, 4}, {3, 3}, {4, 2}, {5, 8}, {8, 8}, {10, 10}};
    for (auto [m, n] : dims)
    {
        std::uint64_t seed = 42;
        auto A = make_rand_vec(m * n, seed++, -5.0, 5.0);
        auto x = make_rand_vec(n, seed++, -5.0, 5.0);
        auto y_simd = make_rand_vec(m, seed++, -5.0, 5.0);
        auto y_ref = y_simd;
        double alpha = 1.5;
        double beta = 0.5;
        nerve::simd::simd_gemv(alpha, A.data(), x.data(), beta, y_simd.data(), m, n);
        ref::gemv(alpha, A.data(), x.data(), beta, y_ref.data(), m, n);
        if (!all_approx(y_simd.data(), y_ref.data(), m))
        {
            std::cerr << "  FAIL gemv at m=" << m << " n=" << n << "\n";
            return false;
        }
    }
    return true;
}

bool test_ger()
{
    const std::size_t dims[][2] = {{1, 1}, {1, 4}, {3, 3}, {4, 2}, {5, 8}, {8, 8}, {10, 10}};
    for (auto [m, n] : dims)
    {
        std::uint64_t seed = 42;
        auto A_simd = make_rand_vec(m * n, seed++, -5.0, 5.0);
        auto A_ref = A_simd;
        auto x = make_rand_vec(m, seed++, -5.0, 5.0);
        auto y = make_rand_vec(n, seed++, -5.0, 5.0);
        double alpha = 2.0;
        nerve::simd::simd_ger(alpha, x.data(), y.data(), A_simd.data(), m, n);
        ref::ger(alpha, x.data(), y.data(), A_ref.data(), m, n);
        if (!all_approx(A_simd.data(), A_ref.data(), m * n))
        {
            std::cerr << "  FAIL ger at m=" << m << " n=" << n << "\n";
            return false;
        }
    }
    return true;
}

bool test_euclidean()
{
    const std::vector<std::size_t> dims = {1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 100};
    for (auto dim : dims)
    {
        auto a = make_rand_vec(dim, 42);
        auto b = make_rand_vec(dim, 84);
        double simd_val = nerve::simd::simd_euclidean(a.data(), b.data(), dim);
        double ref_val = ref::euclidean(a.data(), b.data(), dim);
        if (!approx(simd_val, ref_val))
        {
            std::cerr << "  FAIL euclidean at dim=" << dim << " simd=" << simd_val
                      << " ref=" << ref_val << "\n";
            return false;
        }
    }
    return true;
}

bool test_manhattan()
{
    const std::vector<std::size_t> dims = {1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 100};
    for (auto dim : dims)
    {
        auto a = make_rand_vec(dim, 42);
        auto b = make_rand_vec(dim, 84);
        double simd_val = nerve::simd::simd_manhattan(a.data(), b.data(), dim);
        double ref_val = ref::manhattan(a.data(), b.data(), dim);
        if (!approx(simd_val, ref_val))
        {
            std::cerr << "  FAIL manhattan at dim=" << dim << " simd=" << simd_val
                      << " ref=" << ref_val << "\n";
            return false;
        }
    }
    return true;
}

bool test_cosine()
{
    const std::vector<std::size_t> dims = {1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 100};
    for (auto dim : dims)
    {
        auto a = make_rand_vec(dim, 42);
        auto b = make_rand_vec(dim, 84);
        double simd_val = nerve::simd::simd_cosine(a.data(), b.data(), dim);
        double ref_val = ref::cosine(a.data(), b.data(), dim);
        if (!approx(simd_val, ref_val))
        {
            std::cerr << "  FAIL cosine at dim=" << dim << " simd=" << simd_val
                      << " ref=" << ref_val << "\n";
            return false;
        }
    }
    return true;
}

bool test_cosine_zero_vector()
{
    // Cosine distance with zero vectors should return 1.0
    const double zero[] = {0.0, 0.0, 0.0};
    const double unit[] = {1.0, 0.0, 0.0};
    double simd_val = nerve::simd::simd_cosine(zero, unit, 3);
    double ref_val = ref::cosine(zero, unit, 3);
    if (!approx(simd_val, ref_val) || !approx(simd_val, 1.0))
    {
        std::cerr << "  FAIL cosine_zero_vector simd=" << simd_val << " ref=" << ref_val << "\n";
        return false;
    }
    // Both zero -> dot=0, na=nb=0 -> return 1.0
    simd_val = nerve::simd::simd_cosine(zero, zero, 3);
    ref_val = ref::cosine(zero, zero, 3);
    if (!approx(simd_val, ref_val) || !approx(simd_val, 1.0))
    {
        std::cerr << "  FAIL cosine_both_zero simd=" << simd_val << " ref=" << ref_val << "\n";
        return false;
    }
    return true;
}

bool test_batchnorm()
{
    const std::vector<std::size_t> sizes = {0, 1, 3, 7, 16, 31, 32, 33, 100};
    const double means[] = {0.0, 1.0, -1.0, 5.0};
    const double std_invs[] = {1.0, 2.0, 0.5, -1.0};
    for (auto n : sizes)
    {
        for (auto mean : means)
        {
            for (auto si : std_invs)
            {
                auto data_simd = make_rand_vec(n, 42);
                auto data_ref = data_simd;
                nerve::simd::simd_batchnorm(data_simd.data(), n, mean, si);
                ref::batchnorm(data_ref.data(), n, mean, si);
                if (!all_approx(data_simd.data(), data_ref.data(), n))
                {
                    std::cerr << "  FAIL batchnorm at n=" << n << " mean=" << mean
                              << " std_inv=" << si << "\n";
                    return false;
                }
            }
        }
    }
    return true;
}

bool test_softmax()
{
    const std::vector<std::size_t> sizes = {0, 1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 100};
    for (auto n : sizes)
    {
        auto data_simd = make_rand_vec(n, 42, -10.0, 10.0);
        auto data_ref = data_simd;
        nerve::simd::simd_softmax(data_simd.data(), n);
        ref::softmax(data_ref.data(), n);
        if (!all_approx(data_simd.data(), data_ref.data(), n))
        {
            std::cerr << "  FAIL softmax at n=" << n << "\n";
            return false;
        }
        // Verify sum ~= 1.0
        double sum = 0.0;
        for (std::size_t i = 0; i < n; ++i)
            sum += data_simd[i];
        if (n > 0 && std::abs(sum - 1.0) > 1e-10)
        {
            std::cerr << "  FAIL softmax sum at n=" << n << " sum=" << sum << "\n";
            return false;
        }
    }
    return true;
}

bool test_quantize()
{
    const std::vector<std::size_t> sizes = {0, 1, 3, 7, 16, 31, 32, 33, 100};
    const int bits[] = {1, 2, 4, 8};
    for (auto n : sizes)
    {
        for (auto b : bits)
        {
            // Input must be in [0, 1]
            auto input = make_rand_vec(n, 42, 0.0, 1.0);
            std::vector<std::uint8_t> out_simd(n, 0);
            std::vector<std::uint8_t> out_ref(n, 0);
            nerve::simd::simd_quantize(input.data(), n, b, out_simd.data());
            ref::quantize(input.data(), n, b, out_ref.data());
            if (!all_eq_byte(out_simd.data(), out_ref.data(), n))
            {
                std::cerr << "  FAIL quantize at n=" << n << " bits=" << b << "\n";
                return false;
            }
        }
    }
    return true;
}

bool test_dequantize()
{
    const std::vector<std::size_t> sizes = {0, 1, 3, 7, 16, 31, 32, 33, 100};
    const int bits[] = {1, 2, 4, 8};
    for (auto n : sizes)
    {
        for (auto b : bits)
        {
            double scale = static_cast<double>((1 << b) - 1);
            std::vector<std::uint8_t> quant(n);
            for (std::size_t i = 0; i < n; ++i)
                quant[i] = static_cast<std::uint8_t>(static_cast<std::uint64_t>(i * 7 + 13) %
                                                     (static_cast<std::uint64_t>(scale) + 1));
            std::vector<double> deq_simd(n, 0.0);
            std::vector<double> deq_ref(n, 0.0);
            nerve::simd::simd_dequantize(quant.data(), n, b, deq_simd.data());
            ref::dequantize(quant.data(), n, b, deq_ref.data());
            if (!all_approx(deq_simd.data(), deq_ref.data(), n))
            {
                std::cerr << "  FAIL dequantize at n=" << n << " bits=" << b << "\n";
                return false;
            }
        }
    }
    return true;
}

bool test_zero_size_all_primitives()
{
    // All primitives should handle n=0 gracefully
    double dummy_a[] = {0.0};
    double dummy_b[] = {0.0};
    double dummy_c[] = {0.0};
    char dummy_mem[1] = {0};
    std::uint8_t dummy_quant[1] = {0};

    nerve::simd::simd_add(dummy_a, dummy_b, 0);
    nerve::simd::simd_sub(dummy_a, dummy_b, 0);
    nerve::simd::simd_mul(dummy_a, dummy_b, 0);
    nerve::simd::simd_scale(dummy_a, 2.0, 0);
    nerve::simd::simd_axpy(1.0, dummy_a, dummy_b, 0);
    nerve::simd::simd_fmad(dummy_a, dummy_b, dummy_c, 0);
    nerve::simd::simd_abs(dummy_a, 0);
    nerve::simd::simd_neg(dummy_a, 0);
    nerve::simd::simd_sqrt(dummy_a, 0);
    nerve::simd::simd_exp(dummy_a, 0);
    nerve::simd::simd_log(dummy_a, 0);
    nerve::simd::simd_relu(dummy_a, 0);
    nerve::simd::simd_sigmoid(dummy_a, 0);
    nerve::simd::simd_tanh(dummy_a, 0);
    nerve::simd::simd_min(dummy_a, dummy_b, 0);
    nerve::simd::simd_max(dummy_a, dummy_b, 0);
    nerve::simd::simd_clamp(dummy_a, -1.0, 1.0, 0);
    nerve::simd::simd_memcpy(dummy_mem, dummy_mem, 0);
    nerve::simd::simd_memset(dummy_mem, 0, 0);
    // gemv/ger check zero-size: m=0 or n=0
    nerve::simd::simd_gemv(1.0, dummy_a, dummy_a, 0.0, dummy_c, 0, 3);
    nerve::simd::simd_gemv(1.0, dummy_a, dummy_a, 0.0, dummy_c, 3, 0);
    nerve::simd::simd_ger(1.0, dummy_a, dummy_b, dummy_a, 0, 3);
    nerve::simd::simd_ger(1.0, dummy_a, dummy_b, dummy_a, 3, 0);

    // Reduce operations on zero-size: should return identity
    if (nerve::simd::simd_reduce_sum(dummy_a, 0) != 0.0)
        return false;
    if (nerve::simd::simd_dot(dummy_a, dummy_b, 0) != 0.0)
        return false;
    // norm2, sqdiff_sum on zero-size
    if (nerve::simd::simd_norm2(dummy_a, 0) != 0.0)
        return false;
    if (nerve::simd::simd_sqdiff_sum(dummy_a, dummy_b, 0) != 0.0)
        return false;

    // softmax on zero-size should be a no-op
    nerve::simd::simd_softmax(dummy_a, 0);

    // Quantize/dequantize zero-size
    nerve::simd::simd_quantize(dummy_a, 0, 8, dummy_quant);
    nerve::simd::simd_dequantize(dummy_quant, 0, 8, dummy_a);

    // Cosine distance zero-dim
    if (nerve::simd::simd_euclidean(dummy_a, dummy_b, 0) != 0.0)
        return false;
    if (nerve::simd::simd_manhattan(dummy_a, dummy_b, 0) != 0.0)
        return false;
    // cosine(0-dim) -- dot=0, norm2=0 for both -> 1.0
    double cos0 = nerve::simd::simd_cosine(dummy_a, dummy_b, 0);
    if (std::abs(cos0 - 1.0) > 1e-15)
        return false;

    return true;
}

// Float32 tests

bool test_add_f32()
{
    const std::vector<std::size_t> sizes = {0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 33, 100};
    for (auto n : sizes)
    {
        auto a_d = make_rand_vec(n, 42, -5.0f, 5.0f);
        auto b_d = make_rand_vec(n, 84, -5.0f, 5.0f);
        std::vector<float> a_simd(n), b_simd(n), a_ref(n), b_ref(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            a_simd[i] = static_cast<float>(a_d[i]);
            b_simd[i] = static_cast<float>(b_d[i]);
            a_ref[i] = static_cast<float>(a_d[i]);
            b_ref[i] = static_cast<float>(b_d[i]);
        }
        nerve::simd::simd_add_f32(a_simd.data(), b_simd.data(), n);
        ref::add_f32(a_ref.data(), b_ref.data(), n);
        for (std::size_t i = 0; i < n; ++i)
        {
            if (std::abs(a_simd[i] - a_ref[i]) > 1e-6f)
            {
                std::cerr << "  FAIL add_f32 at n=" << n << " i=" << i << " simd=" << a_simd[i]
                          << " ref=" << a_ref[i] << "\n";
                return false;
            }
        }
    }
    return true;
}

bool test_reduce_sum_f32()
{
    const std::vector<std::size_t> sizes = {0,  1,  2,  3,  4,  5,  7,   8,
                                            15, 16, 17, 31, 32, 33, 100, 1000};
    for (auto n : sizes)
    {
        auto d = make_rand_vec(n, 42, -5.0f, 5.0f);
        std::vector<float> data(n);
        for (std::size_t i = 0; i < n; ++i)
            data[i] = static_cast<float>(d[i]);
        float simd_val = nerve::simd::simd_reduce_sum_f32(data.data(), n);
        float ref_val = ref::reduce_sum_f32(data.data(), n);
        if (std::abs(simd_val - ref_val) >
            1e-4f * std::max(1.0f, std::max(std::abs(simd_val), std::abs(ref_val))))
        {
            std::cerr << "  FAIL reduce_sum_f32 at n=" << n << " simd=" << simd_val
                      << " ref=" << ref_val << "\n";
            return false;
        }
    }
    return true;
}

bool test_dot_f32()
{
    const std::vector<std::size_t> sizes = {0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 33, 100};
    for (auto n : sizes)
    {
        auto a_d = make_rand_vec(n, 42, -5.0f, 5.0f);
        auto b_d = make_rand_vec(n, 84, -5.0f, 5.0f);
        std::vector<float> a(n), b(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            a[i] = static_cast<float>(a_d[i]);
            b[i] = static_cast<float>(b_d[i]);
        }
        float simd_val = nerve::simd::simd_dot_f32(a.data(), b.data(), n);
        float ref_val = ref::dot_f32(a.data(), b.data(), n);
        if (std::abs(simd_val - ref_val) >
            1e-4f * std::max(1.0f, std::max(std::abs(simd_val), std::abs(ref_val))))
        {
            std::cerr << "  FAIL dot_f32 at n=" << n << " simd=" << simd_val << " ref=" << ref_val
                      << "\n";
            return false;
        }
    }
    return true;
}

bool test_sub_f32()
{
    const std::vector<std::size_t> sizes = {0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 33, 100};
    for (auto n : sizes)
    {
        auto a_d = make_rand_vec(n, 42, -5.0f, 5.0f);
        auto b_d = make_rand_vec(n, 84, -5.0f, 5.0f);
        std::vector<float> a_simd(n), b_simd(n), a_ref(n), b_ref(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            a_simd[i] = static_cast<float>(a_d[i]);
            b_simd[i] = static_cast<float>(b_d[i]);
            a_ref[i] = static_cast<float>(a_d[i]);
            b_ref[i] = static_cast<float>(b_d[i]);
        }
        nerve::simd::simd_sub_f32(a_simd.data(), b_simd.data(), n);
        ref::sub_f32(a_ref.data(), b_ref.data(), n);
        for (std::size_t i = 0; i < n; ++i)
        {
            if (std::abs(a_simd[i] - a_ref[i]) > 1e-6f)
            {
                std::cerr << "  FAIL sub_f32 at n=" << n << " i=" << i << " simd=" << a_simd[i]
                          << " ref=" << a_ref[i] << "\n";
                return false;
            }
        }
    }
    return true;
}

bool test_mul_f32()
{
    const std::vector<std::size_t> sizes = {0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 33, 100};
    for (auto n : sizes)
    {
        auto a_d = make_rand_vec(n, 42, -5.0f, 5.0f);
        auto b_d = make_rand_vec(n, 84, -5.0f, 5.0f);
        std::vector<float> a_simd(n), b_simd(n), a_ref(n), b_ref(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            a_simd[i] = static_cast<float>(a_d[i]);
            b_simd[i] = static_cast<float>(b_d[i]);
            a_ref[i] = static_cast<float>(a_d[i]);
            b_ref[i] = static_cast<float>(b_d[i]);
        }
        nerve::simd::simd_mul_f32(a_simd.data(), b_simd.data(), n);
        ref::mul_f32(a_ref.data(), b_ref.data(), n);
        for (std::size_t i = 0; i < n; ++i)
        {
            if (std::abs(a_simd[i] - a_ref[i]) > 1e-6f)
            {
                std::cerr << "  FAIL mul_f32 at n=" << n << " i=" << i << " simd=" << a_simd[i]
                          << " ref=" << a_ref[i] << "\n";
                return false;
            }
        }
    }
    return true;
}

bool test_scale_f32()
{
    const std::vector<std::size_t> sizes = {0, 1, 3, 7, 16, 31, 32, 100};
    const float alphas[] = {0.0f, 1.0f, -1.0f, 2.5f, 0.5f, -3.0f};
    for (auto n : sizes)
    {
        for (auto alpha : alphas)
        {
            auto d = make_rand_vec(n, 42, -5.0f, 5.0f);
            std::vector<float> data_simd(n), data_ref(n);
            for (std::size_t i = 0; i < n; ++i)
            {
                data_simd[i] = static_cast<float>(d[i]);
                data_ref[i] = static_cast<float>(d[i]);
            }
            nerve::simd::simd_scale_f32(data_simd.data(), alpha, n);
            ref::scale_f32(data_ref.data(), alpha, n);
            for (std::size_t i = 0; i < n; ++i)
            {
                if (std::abs(data_simd[i] - data_ref[i]) > 1e-6f)
                {
                    std::cerr << "  FAIL scale_f32 at n=" << n << " alpha=" << alpha << " i=" << i
                              << " simd=" << data_simd[i] << " ref=" << data_ref[i] << "\n";
                    return false;
                }
            }
        }
    }
    return true;
}

bool test_axpy_f32()
{
    const std::vector<std::size_t> sizes = {0, 1, 3, 7, 16, 31, 32, 33, 100};
    const float alphas[] = {0.0f, 1.0f, -1.0f, 2.5f, -0.5f};
    for (auto n : sizes)
    {
        for (auto alpha : alphas)
        {
            auto x_d = make_rand_vec(n, 42, -5.0f, 5.0f);
            auto y_d = make_rand_vec(n, 84, -5.0f, 5.0f);
            std::vector<float> x_simd(n), y_simd(n), x_ref(n), y_ref(n);
            for (std::size_t i = 0; i < n; ++i)
            {
                x_simd[i] = static_cast<float>(x_d[i]);
                y_simd[i] = static_cast<float>(y_d[i]);
                x_ref[i] = static_cast<float>(x_d[i]);
                y_ref[i] = static_cast<float>(y_d[i]);
            }
            nerve::simd::simd_axpy_f32(alpha, x_simd.data(), y_simd.data(), n);
            ref::axpy_f32(alpha, x_ref.data(), y_ref.data(), n);
            for (std::size_t i = 0; i < n; ++i)
            {
                if (std::abs(y_simd[i] - y_ref[i]) > 1e-6f)
                {
                    std::cerr << "  FAIL axpy_f32 at n=" << n << " alpha=" << alpha << " i=" << i
                              << " simd=" << y_simd[i] << " ref=" << y_ref[i] << "\n";
                    return false;
                }
            }
        }
    }
    return true;
}

bool test_reduce_max_f32()
{
    const std::vector<std::size_t> sizes = {0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 33, 100};
    for (auto n : sizes)
    {
        auto d = make_rand_vec(n, 42, -5.0f, 5.0f);
        std::vector<float> data(n);
        for (std::size_t i = 0; i < n; ++i)
            data[i] = static_cast<float>(d[i]);
        float simd_val = nerve::simd::simd_reduce_max_f32(data.data(), n);
        float ref_val = ref::reduce_max_f32(data.data(), n);
        if (std::abs(simd_val - ref_val) > 1e-6f)
        {
            std::cerr << "  FAIL reduce_max_f32 at n=" << n << " simd=" << simd_val
                      << " ref=" << ref_val << "\n";
            return false;
        }
    }
    return true;
}

bool test_norm2_f32()
{
    const std::vector<std::size_t> sizes = {0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 33, 100};
    for (auto n : sizes)
    {
        auto d = make_rand_vec(n, 42, -5.0f, 5.0f);
        std::vector<float> data(n);
        for (std::size_t i = 0; i < n; ++i)
            data[i] = static_cast<float>(d[i]);
        float simd_val = nerve::simd::simd_norm2_f32(data.data(), n);
        float ref_val = ref::norm2_f32(data.data(), n);
        if (std::abs(simd_val - ref_val) > 1e-6f)
        {
            std::cerr << "  FAIL norm2_f32 at n=" << n << " simd=" << simd_val << " ref=" << ref_val
                      << "\n";
            return false;
        }
    }
    return true;
}

bool test_abs_f32()
{
    const std::vector<std::size_t> sizes = {0, 1, 3, 7, 16, 31, 32, 33, 100};
    for (auto n : sizes)
    {
        auto d = make_rand_vec(n, 42, -5.0f, 5.0f);
        std::vector<float> data_simd(n), data_ref(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            data_simd[i] = static_cast<float>(d[i]);
            data_ref[i] = static_cast<float>(d[i]);
        }
        nerve::simd::simd_abs_f32(data_simd.data(), n);
        ref::abs_f32(data_ref.data(), n);
        for (std::size_t i = 0; i < n; ++i)
        {
            if (std::abs(data_simd[i] - data_ref[i]) > 1e-6f)
            {
                std::cerr << "  FAIL abs_f32 at n=" << n << " i=" << i << " simd=" << data_simd[i]
                          << " ref=" << data_ref[i] << "\n";
                return false;
            }
        }
    }
    return true;
}

bool test_relu_f32()
{
    const std::vector<std::size_t> sizes = {0, 1, 3, 7, 16, 31, 32, 33, 100};
    for (auto n : sizes)
    {
        auto d = make_rand_vec(n, 42, -5.0f, 5.0f);
        std::vector<float> data_simd(n), data_ref(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            data_simd[i] = static_cast<float>(d[i]);
            data_ref[i] = static_cast<float>(d[i]);
        }
        nerve::simd::simd_relu_f32(data_simd.data(), n);
        ref::relu_f32(data_ref.data(), n);
        for (std::size_t i = 0; i < n; ++i)
        {
            if (std::abs(data_simd[i] - data_ref[i]) > 1e-6f)
            {
                std::cerr << "  FAIL relu_f32 at n=" << n << " i=" << i << " simd=" << data_simd[i]
                          << " ref=" << data_ref[i] << "\n";
                return false;
            }
        }
    }
    return true;
}

bool test_min_f32()
{
    const std::vector<std::size_t> sizes = {0, 1, 3, 7, 16, 31, 32, 33, 100};
    for (auto n : sizes)
    {
        auto a_d = make_rand_vec(n, 42, -5.0f, 5.0f);
        auto b_d = make_rand_vec(n, 84, -5.0f, 5.0f);
        std::vector<float> a_simd(n), b_simd(n), a_ref(n), b_ref(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            a_simd[i] = static_cast<float>(a_d[i]);
            b_simd[i] = static_cast<float>(b_d[i]);
            a_ref[i] = static_cast<float>(a_d[i]);
            b_ref[i] = static_cast<float>(b_d[i]);
        }
        nerve::simd::simd_min_f32(a_simd.data(), b_simd.data(), n);
        ref::min_f32(a_ref.data(), b_ref.data(), n);
        for (std::size_t i = 0; i < n; ++i)
        {
            if (std::abs(a_simd[i] - a_ref[i]) > 1e-6f)
            {
                std::cerr << "  FAIL min_f32 at n=" << n << " i=" << i << " simd=" << a_simd[i]
                          << " ref=" << a_ref[i] << "\n";
                return false;
            }
        }
    }
    return true;
}

bool test_max_f32()
{
    const std::vector<std::size_t> sizes = {0, 1, 3, 7, 16, 31, 32, 33, 100};
    for (auto n : sizes)
    {
        auto a_d = make_rand_vec(n, 42, -5.0f, 5.0f);
        auto b_d = make_rand_vec(n, 84, -5.0f, 5.0f);
        std::vector<float> a_simd(n), b_simd(n), a_ref(n), b_ref(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            a_simd[i] = static_cast<float>(a_d[i]);
            b_simd[i] = static_cast<float>(b_d[i]);
            a_ref[i] = static_cast<float>(a_d[i]);
            b_ref[i] = static_cast<float>(b_d[i]);
        }
        nerve::simd::simd_max_f32(a_simd.data(), b_simd.data(), n);
        ref::max_f32(a_ref.data(), b_ref.data(), n);
        for (std::size_t i = 0; i < n; ++i)
        {
            if (std::abs(a_simd[i] - a_ref[i]) > 1e-6f)
            {
                std::cerr << "  FAIL max_f32 at n=" << n << " i=" << i << " simd=" << a_simd[i]
                          << " ref=" << a_ref[i] << "\n";
                return false;
            }
        }
    }
    return true;
}

bool test_clamp_f32()
{
    const std::vector<std::size_t> sizes = {0, 1, 3, 7, 16, 31, 32, 33, 100};
    const float bounds[][2] = {{0.0f, 1.0f}, {-2.0f, 2.0f}, {-5.0f, 5.0f}, {-10.0f, 10.0f}};
    for (auto n : sizes)
    {
        auto d = make_rand_vec(n, 42, -15.0f, 15.0f);
        std::vector<float> data_simd(n), data_ref(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            data_simd[i] = static_cast<float>(d[i]);
            data_ref[i] = static_cast<float>(d[i]);
        }
        for (auto [lo, hi] : bounds)
        {
            auto copy_simd = data_simd;
            auto copy_ref = data_ref;
            nerve::simd::simd_clamp_f32(copy_simd.data(), lo, hi, n);
            ref::clamp_f32(copy_ref.data(), lo, hi, n);
            for (std::size_t i = 0; i < n; ++i)
            {
                if (std::abs(copy_simd[i] - copy_ref[i]) > 1e-6f)
                {
                    std::cerr << "  FAIL clamp_f32 at n=" << n << " lo=" << lo << " hi=" << hi
                              << " i=" << i << " simd=" << copy_simd[i] << " ref=" << copy_ref[i]
                              << "\n";
                    return false;
                }
            }
        }
    }
    return true;
}

bool test_gemv_f32()
{
    const std::size_t dims[][2] = {{1, 1}, {1, 4}, {3, 3}, {4, 2}, {5, 8}, {8, 8}, {10, 10}};
    for (auto [m, n] : dims)
    {
        auto A_d = make_rand_vec(m * n, 42, -5.0f, 5.0f);
        auto x_d = make_rand_vec(n, 84, -5.0f, 5.0f);
        auto y_d = make_rand_vec(m, 126, -5.0f, 5.0f);
        std::vector<float> A_simd(m * n), A_ref(m * n), x(n), y_simd(m), y_ref(m);
        for (std::size_t i = 0; i < m * n; ++i)
            A_simd[i] = A_ref[i] = static_cast<float>(A_d[i]);
        for (std::size_t i = 0; i < n; ++i)
            x[i] = static_cast<float>(x_d[i]);
        for (std::size_t i = 0; i < m; ++i)
            y_simd[i] = y_ref[i] = static_cast<float>(y_d[i]);
        float alpha = 1.5f;
        float beta = 0.5f;
        nerve::simd::simd_gemv_f32(alpha, A_simd.data(), x.data(), beta, y_simd.data(), m, n);
        ref::gemv_f32(alpha, A_ref.data(), x.data(), beta, y_ref.data(), m, n);
        for (std::size_t i = 0; i < m; ++i)
        {
            if (std::abs(y_simd[i] - y_ref[i]) > 1e-6f)
            {
                std::cerr << "  FAIL gemv_f32 at m=" << m << " n=" << n << " i=" << i
                          << " simd=" << y_simd[i] << " ref=" << y_ref[i] << "\n";
                return false;
            }
        }
    }
    return true;
}

bool test_ger_f32()
{
    const std::size_t dims[][2] = {{1, 1}, {1, 4}, {3, 3}, {4, 2}, {5, 8}, {8, 8}, {10, 10}};
    for (auto [m, n] : dims)
    {
        auto A_d = make_rand_vec(m * n, 42, -5.0f, 5.0f);
        auto x_d = make_rand_vec(m, 84, -5.0f, 5.0f);
        auto y_d = make_rand_vec(n, 126, -5.0f, 5.0f);
        std::vector<float> A_simd(m * n), A_ref(m * n), x(m), y(n);
        for (std::size_t i = 0; i < m * n; ++i)
            A_simd[i] = A_ref[i] = static_cast<float>(A_d[i]);
        for (std::size_t i = 0; i < m; ++i)
            x[i] = static_cast<float>(x_d[i]);
        for (std::size_t i = 0; i < n; ++i)
            y[i] = static_cast<float>(y_d[i]);
        float alpha = 2.0f;
        nerve::simd::simd_ger_f32(alpha, x.data(), y.data(), A_simd.data(), m, n);
        ref::ger_f32(alpha, x.data(), y.data(), A_ref.data(), m, n);
        for (std::size_t i = 0; i < m * n; ++i)
        {
            if (std::abs(A_simd[i] - A_ref[i]) > 1e-6f)
            {
                std::cerr << "  FAIL ger_f32 at m=" << m << " n=" << n << " i=" << i
                          << " simd=" << A_simd[i] << " ref=" << A_ref[i] << "\n";
                return false;
            }
        }
    }
    return true;
}

bool test_sqdiff_sum_f32()
{
    const std::vector<std::size_t> dims = {0, 1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 100};
    for (auto dim : dims)
    {
        auto a_d = make_rand_vec(dim, 42, -5.0f, 5.0f);
        auto b_d = make_rand_vec(dim, 84, -5.0f, 5.0f);
        std::vector<float> a(dim), b(dim);
        for (std::size_t i = 0; i < dim; ++i)
        {
            a[i] = static_cast<float>(a_d[i]);
            b[i] = static_cast<float>(b_d[i]);
        }
        float simd_val = nerve::simd::simd_sqdiff_sum_f32(a.data(), b.data(), dim);
        float ref_val = ref::sqdiff_sum_f32(a.data(), b.data(), dim);
        if (std::abs(simd_val - ref_val) >
            1e-4f * std::max(1.0f, std::max(std::abs(simd_val), std::abs(ref_val))))
        {
            std::cerr << "  FAIL sqdiff_sum_f32 at dim=" << dim << " simd=" << simd_val
                      << " ref=" << ref_val << "\n";
            return false;
        }
    }
    return true;
}

bool test_euclidean_f32()
{
    const std::vector<std::size_t> dims = {0, 1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 100};
    for (auto dim : dims)
    {
        auto a_d = make_rand_vec(dim, 42, -5.0f, 5.0f);
        auto b_d = make_rand_vec(dim, 84, -5.0f, 5.0f);
        std::vector<float> a(dim), b(dim);
        for (std::size_t i = 0; i < dim; ++i)
        {
            a[i] = static_cast<float>(a_d[i]);
            b[i] = static_cast<float>(b_d[i]);
        }
        float simd_val = nerve::simd::simd_euclidean_f32(a.data(), b.data(), dim);
        float ref_val = ref::euclidean_f32(a.data(), b.data(), dim);
        if (std::abs(simd_val - ref_val) > 1e-6f)
        {
            std::cerr << "  FAIL euclidean_f32 at dim=" << dim << " simd=" << simd_val
                      << " ref=" << ref_val << "\n";
            return false;
        }
    }
    return true;
}

bool test_cosine_f32()
{
    const std::vector<std::size_t> dims = {0, 1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 100};
    for (auto dim : dims)
    {
        auto a_d = make_rand_vec(dim, 42, -5.0f, 5.0f);
        auto b_d = make_rand_vec(dim, 84, -5.0f, 5.0f);
        std::vector<float> a(dim), b(dim);
        for (std::size_t i = 0; i < dim; ++i)
        {
            a[i] = static_cast<float>(a_d[i]);
            b[i] = static_cast<float>(b_d[i]);
        }
        float simd_val = nerve::simd::simd_cosine_f32(a.data(), b.data(), dim);
        float ref_val = ref::cosine_f32(a.data(), b.data(), dim);
        if (std::abs(simd_val - ref_val) > 1e-6f)
        {
            std::cerr << "  FAIL cosine_f32 at dim=" << dim << " simd=" << simd_val
                      << " ref=" << ref_val << "\n";
            return false;
        }
    }
    return true;
}

bool test_fmad_f32()
{
    const std::vector<std::size_t> sizes = {0, 1, 3, 7, 16, 31, 32, 33, 100};
    for (auto n : sizes)
    {
        auto a_d = make_rand_vec(n, 42, -5.0f, 5.0f);
        auto b_d = make_rand_vec(n, 84, -5.0f, 5.0f);
        auto c_d = make_rand_vec(n, 126, -5.0f, 5.0f);
        std::vector<float> a(n), b(n), c_simd(n), c_ref(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            a[i] = static_cast<float>(a_d[i]);
            b[i] = static_cast<float>(b_d[i]);
            c_simd[i] = static_cast<float>(c_d[i]);
            c_ref[i] = static_cast<float>(c_d[i]);
        }
        nerve::simd::simd_fmad_f32(a.data(), b.data(), c_simd.data(), n);
        ref::fmad_f32(a.data(), b.data(), c_ref.data(), n);
        for (std::size_t i = 0; i < n; ++i)
        {
            if (std::abs(c_simd[i] - c_ref[i]) > 1e-6f)
            {
                std::cerr << "  FAIL fmad_f32 at n=" << n << " i=" << i << " simd=" << c_simd[i]
                          << " ref=" << c_ref[i] << "\n";
                return false;
            }
        }
    }
    return true;
}

bool test_reduce_min_f32()
{
    const std::vector<std::size_t> sizes = {0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 33, 100};
    for (auto n : sizes)
    {
        auto d = make_rand_vec(n, 42, -5.0f, 5.0f);
        std::vector<float> data(n);
        for (std::size_t i = 0; i < n; ++i)
            data[i] = static_cast<float>(d[i]);
        float simd_val = nerve::simd::simd_reduce_min_f32(data.data(), n);
        float ref_val = ref::reduce_min_f32(data.data(), n);
        if (std::abs(simd_val - ref_val) > 1e-6f)
        {
            std::cerr << "  FAIL reduce_min_f32 at n=" << n << " simd=" << simd_val
                      << " ref=" << ref_val << "\n";
            return false;
        }
    }
    return true;
}

bool test_neg_f32()
{
    const std::vector<std::size_t> sizes = {0, 1, 3, 7, 16, 31, 32, 33, 100};
    for (auto n : sizes)
    {
        auto d = make_rand_vec(n, 42, -5.0f, 5.0f);
        std::vector<float> data_simd(n), data_ref(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            data_simd[i] = static_cast<float>(d[i]);
            data_ref[i] = static_cast<float>(d[i]);
        }
        nerve::simd::simd_neg_f32(data_simd.data(), n);
        ref::neg_f32(data_ref.data(), n);
        for (std::size_t i = 0; i < n; ++i)
        {
            if (std::abs(data_simd[i] - data_ref[i]) > 1e-6f)
            {
                std::cerr << "  FAIL neg_f32 at n=" << n << " i=" << i << " simd=" << data_simd[i]
                          << " ref=" << data_ref[i] << "\n";
                return false;
            }
        }
    }
    return true;
}

bool test_sqrt_f32()
{
    const std::vector<std::size_t> sizes = {0, 1, 3, 7, 16, 31, 32, 33, 100};
    for (auto n : sizes)
    {
        auto d = make_rand_vec(n, 42, 0.01f, 100.0f);
        std::vector<float> data_simd(n), data_ref(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            data_simd[i] = static_cast<float>(d[i]);
            data_ref[i] = static_cast<float>(d[i]);
        }
        nerve::simd::simd_sqrt_f32(data_simd.data(), n);
        ref::sqrt_f32(data_ref.data(), n);
        for (std::size_t i = 0; i < n; ++i)
        {
            if (std::abs(data_simd[i] - data_ref[i]) > 1e-6f)
            {
                std::cerr << "  FAIL sqrt_f32 at n=" << n << " i=" << i << " simd=" << data_simd[i]
                          << " ref=" << data_ref[i] << "\n";
                return false;
            }
        }
    }
    return true;
}

bool test_exp_f32()
{
    const std::vector<std::size_t> sizes = {0, 1, 3, 7, 16, 31, 32, 33, 100};
    for (auto n : sizes)
    {
        auto d = make_rand_vec(n, 42, -10.0f, 10.0f);
        std::vector<float> data_simd(n), data_ref(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            data_simd[i] = static_cast<float>(d[i]);
            data_ref[i] = static_cast<float>(d[i]);
        }
        nerve::simd::simd_exp_f32(data_simd.data(), n);
        ref::exp_f32(data_ref.data(), n);
        for (std::size_t i = 0; i < n; ++i)
        {
            if (std::abs(data_simd[i] - data_ref[i]) >
                1e-4f * std::max(1.0f, std::max(std::abs(data_simd[i]), std::abs(data_ref[i]))))
            {
                std::cerr << "  FAIL exp_f32 at n=" << n << " i=" << i << " simd=" << data_simd[i]
                          << " ref=" << data_ref[i] << "\n";
                return false;
            }
        }
    }
    return true;
}

bool test_log_f32()
{
    const std::vector<std::size_t> sizes = {0, 1, 3, 7, 16, 31, 32, 33, 100};
    for (auto n : sizes)
    {
        auto d = make_rand_vec(n, 42, 0.001f, 100.0f);
        std::vector<float> data_simd(n), data_ref(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            data_simd[i] = static_cast<float>(d[i]);
            data_ref[i] = static_cast<float>(d[i]);
        }
        nerve::simd::simd_log_f32(data_simd.data(), n);
        ref::log_f32(data_ref.data(), n);
        for (std::size_t i = 0; i < n; ++i)
        {
            if (std::abs(data_simd[i] - data_ref[i]) > 1e-6f)
            {
                std::cerr << "  FAIL log_f32 at n=" << n << " i=" << i << " simd=" << data_simd[i]
                          << " ref=" << data_ref[i] << "\n";
                return false;
            }
        }
    }
    return true;
}

bool test_sigmoid_f32()
{
    const std::vector<std::size_t> sizes = {0, 1, 3, 7, 16, 31, 32, 33, 100};
    for (auto n : sizes)
    {
        auto d = make_rand_vec(n, 42, -10.0f, 10.0f);
        std::vector<float> data_simd(n), data_ref(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            data_simd[i] = static_cast<float>(d[i]);
            data_ref[i] = static_cast<float>(d[i]);
        }
        nerve::simd::simd_sigmoid_f32(data_simd.data(), n);
        ref::sigmoid_f32(data_ref.data(), n);
        for (std::size_t i = 0; i < n; ++i)
        {
            if (std::abs(data_simd[i] - data_ref[i]) > 1e-6f)
            {
                std::cerr << "  FAIL sigmoid_f32 at n=" << n << " i=" << i
                          << " simd=" << data_simd[i] << " ref=" << data_ref[i] << "\n";
                return false;
            }
        }
    }
    return true;
}

bool test_tanh_f32()
{
    const std::vector<std::size_t> sizes = {0, 1, 3, 7, 16, 31, 32, 33, 100};
    for (auto n : sizes)
    {
        auto d = make_rand_vec(n, 42, -5.0f, 5.0f);
        std::vector<float> data_simd(n), data_ref(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            data_simd[i] = static_cast<float>(d[i]);
            data_ref[i] = static_cast<float>(d[i]);
        }
        nerve::simd::simd_tanh_f32(data_simd.data(), n);
        ref::tanh_f32(data_ref.data(), n);
        for (std::size_t i = 0; i < n; ++i)
        {
            if (std::abs(data_simd[i] - data_ref[i]) > 1e-6f)
            {
                std::cerr << "  FAIL tanh_f32 at n=" << n << " i=" << i << " simd=" << data_simd[i]
                          << " ref=" << data_ref[i] << "\n";
                return false;
            }
        }
    }
    return true;
}

// Float16 tests

bool test_add_f16()
{
    const std::vector<std::size_t> sizes = {0, 1, 3, 7, 16, 31, 32, 33, 100};
    for (auto n : sizes)
    {
        std::vector<nerve::simd::half> a_simd(n), b_simd(n), a_ref(n), b_ref(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            float v = static_cast<float>(i * 7 % 10) * 0.1f;
            a_simd[i] = nerve::simd::float_to_half(v);
            b_simd[i] = nerve::simd::float_to_half(v * 0.5f);
            a_ref[i] = a_simd[i];
            b_ref[i] = b_simd[i];
        }
        nerve::simd::simd_add_f16(a_simd.data(), b_simd.data(), n);
        // Reference: convert to float, add, convert back
        for (std::size_t i = 0; i < n; ++i)
            a_ref[i] = nerve::simd::float_to_half(nerve::simd::half_to_float(a_ref[i]) +
                                                  nerve::simd::half_to_float(b_ref[i]));
        for (std::size_t i = 0; i < n; ++i)
        {
            if (std::abs(nerve::simd::half_to_float(a_simd[i]) -
                         nerve::simd::half_to_float(a_ref[i])) > 1e-3f)
            {
                std::cerr << "  FAIL add_f16 at n=" << n << " i=" << i << "\n";
                return false;
            }
        }
    }
    return true;
}

bool test_sub_f16()
{
    const std::vector<std::size_t> sizes = {0, 1, 3, 7, 16, 31, 32, 33, 100};
    for (auto n : sizes)
    {
        std::vector<nerve::simd::half> a_simd(n), b_simd(n), a_ref(n), b_ref(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            float v = static_cast<float>(i * 7 % 10) * 0.1f;
            a_simd[i] = nerve::simd::float_to_half(v);
            b_simd[i] = nerve::simd::float_to_half(v * 0.3f);
            a_ref[i] = a_simd[i];
            b_ref[i] = b_simd[i];
        }
        nerve::simd::simd_sub_f16(a_simd.data(), b_simd.data(), n);
        for (std::size_t i = 0; i < n; ++i)
            a_ref[i] = nerve::simd::float_to_half(nerve::simd::half_to_float(a_ref[i]) -
                                                  nerve::simd::half_to_float(b_ref[i]));
        for (std::size_t i = 0; i < n; ++i)
        {
            if (std::abs(nerve::simd::half_to_float(a_simd[i]) -
                         nerve::simd::half_to_float(a_ref[i])) > 1e-3f)
            {
                std::cerr << "  FAIL sub_f16 at n=" << n << " i=" << i << "\n";
                return false;
            }
        }
    }
    return true;
}

bool test_mul_f16()
{
    const std::vector<std::size_t> sizes = {0, 1, 3, 7, 16, 31, 32, 33, 100};
    for (auto n : sizes)
    {
        std::vector<nerve::simd::half> a_simd(n), b_simd(n), a_ref(n), b_ref(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            float v = static_cast<float>(i * 7 % 10) * 0.1f;
            a_simd[i] = nerve::simd::float_to_half(v);
            b_simd[i] = nerve::simd::float_to_half(v * 0.5f);
            a_ref[i] = a_simd[i];
            b_ref[i] = b_simd[i];
        }
        nerve::simd::simd_mul_f16(a_simd.data(), b_simd.data(), n);
        for (std::size_t i = 0; i < n; ++i)
            a_ref[i] = nerve::simd::float_to_half(nerve::simd::half_to_float(a_ref[i]) *
                                                  nerve::simd::half_to_float(b_ref[i]));
        for (std::size_t i = 0; i < n; ++i)
        {
            if (std::abs(nerve::simd::half_to_float(a_simd[i]) -
                         nerve::simd::half_to_float(a_ref[i])) > 1e-3f)
            {
                std::cerr << "  FAIL mul_f16 at n=" << n << " i=" << i << "\n";
                return false;
            }
        }
    }
    return true;
}

bool test_scale_f16()
{
    const std::vector<std::size_t> sizes = {0, 1, 3, 7, 16, 31, 32, 33, 100};
    const float alphas[] = {0.0f, 0.5f, 1.0f, -1.0f, 2.0f};
    for (auto n : sizes)
    {
        for (auto alpha_f : alphas)
        {
            nerve::simd::half alpha = nerve::simd::float_to_half(alpha_f);
            std::vector<nerve::simd::half> data_simd(n), data_ref(n);
            for (std::size_t i = 0; i < n; ++i)
            {
                float v = static_cast<float>(i * 7 % 10) * 0.1f;
                data_simd[i] = nerve::simd::float_to_half(v);
                data_ref[i] = data_simd[i];
            }
            nerve::simd::simd_scale_f16(data_simd.data(), alpha, n);
            float fa = nerve::simd::half_to_float(alpha);
            for (std::size_t i = 0; i < n; ++i)
                data_ref[i] =
                    nerve::simd::float_to_half(nerve::simd::half_to_float(data_ref[i]) * fa);
            for (std::size_t i = 0; i < n; ++i)
            {
                if (std::abs(nerve::simd::half_to_float(data_simd[i]) -
                             nerve::simd::half_to_float(data_ref[i])) > 1e-3f)
                {
                    std::cerr << "  FAIL scale_f16 at n=" << n << " alpha=" << alpha_f << " i=" << i
                              << "\n";
                    return false;
                }
            }
        }
    }
    return true;
}

bool test_reduce_sum_f16()
{
    const std::vector<std::size_t> sizes = {0, 1, 3, 7, 16, 31, 32, 33, 100};
    for (auto n : sizes)
    {
        std::vector<nerve::simd::half> data(n);
        float ref_sum = 0.0f;
        for (std::size_t i = 0; i < n; ++i)
        {
            float v = static_cast<float>(i * 7 % 10) * 0.1f;
            data[i] = nerve::simd::float_to_half(v);
            ref_sum += v;
        }
        float simd_val = nerve::simd::simd_reduce_sum_f16(data.data(), n);
        if (std::abs(simd_val - ref_sum) > 0.1f * std::max(1.0f, ref_sum))
        {
            std::cerr << "  FAIL reduce_sum_f16 at n=" << n << " simd=" << simd_val
                      << " ref=" << ref_sum << "\n";
            return false;
        }
    }
    return true;
}

bool test_dot_f16()
{
    const std::vector<std::size_t> sizes = {0, 1, 3, 7, 16, 31, 32, 33, 100};
    for (auto n : sizes)
    {
        std::vector<nerve::simd::half> a(n), b(n);
        float ref_dot = 0.0f;
        for (std::size_t i = 0; i < n; ++i)
        {
            float va = static_cast<float>(i * 7 % 10) * 0.1f;
            float vb = static_cast<float>(i * 3 % 10) * 0.1f;
            a[i] = nerve::simd::float_to_half(va);
            b[i] = nerve::simd::float_to_half(vb);
            ref_dot += va * vb;
        }
        float simd_val = nerve::simd::simd_dot_f16(a.data(), b.data(), n);
        if (std::abs(simd_val - ref_dot) > 0.1f * std::max(1.0f, ref_dot))
        {
            std::cerr << "  FAIL dot_f16 at n=" << n << " simd=" << simd_val << " ref=" << ref_dot
                      << "\n";
            return false;
        }
    }
    return true;
}

bool test_neg_f16()
{
    const std::vector<std::size_t> sizes = {0, 1, 3, 7, 16, 31, 32, 33, 100};
    for (auto n : sizes)
    {
        std::vector<nerve::simd::half> data_simd(n), data_ref(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            float v = static_cast<float>(i * 7 % 10) * 0.1f;
            data_simd[i] = nerve::simd::float_to_half(v);
            data_ref[i] = data_simd[i];
        }
        nerve::simd::simd_neg_f16(data_simd.data(), n);
        for (std::size_t i = 0; i < n; ++i)
            data_ref[i] = nerve::simd::float_to_half(-nerve::simd::half_to_float(data_ref[i]));
        for (std::size_t i = 0; i < n; ++i)
        {
            if (std::abs(nerve::simd::half_to_float(data_simd[i]) -
                         nerve::simd::half_to_float(data_ref[i])) > 1e-3f)
            {
                std::cerr << "  FAIL neg_f16 at n=" << n << " i=" << i << "\n";
                return false;
            }
        }
    }
    return true;
}

bool test_abs_f16()
{
    const std::vector<std::size_t> sizes = {0, 1, 3, 7, 16, 31, 32, 33, 100};
    for (auto n : sizes)
    {
        std::vector<nerve::simd::half> data_simd(n), data_ref(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            float v = (static_cast<float>(i * 7 % 10) - 5.0f) * 0.1f;
            data_simd[i] = nerve::simd::float_to_half(v);
            data_ref[i] = data_simd[i];
        }
        nerve::simd::simd_abs_f16(data_simd.data(), n);
        for (std::size_t i = 0; i < n; ++i)
            data_ref[i] =
                nerve::simd::float_to_half(std::abs(nerve::simd::half_to_float(data_ref[i])));
        for (std::size_t i = 0; i < n; ++i)
        {
            if (std::abs(nerve::simd::half_to_float(data_simd[i]) -
                         nerve::simd::half_to_float(data_ref[i])) > 1e-3f)
            {
                std::cerr << "  FAIL abs_f16 at n=" << n << " i=" << i << "\n";
                return false;
            }
        }
    }
    return true;
}

bool test_relu_f16()
{
    const std::vector<std::size_t> sizes = {0, 1, 3, 7, 16, 31, 32, 33, 100};
    for (auto n : sizes)
    {
        std::vector<nerve::simd::half> data_simd(n), data_ref(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            float v = (static_cast<float>(i * 7 % 10) - 5.0f) * 0.1f;
            data_simd[i] = nerve::simd::float_to_half(v);
            data_ref[i] = data_simd[i];
        }
        nerve::simd::simd_relu_f16(data_simd.data(), n);
        for (std::size_t i = 0; i < n; ++i)
        {
            float v = nerve::simd::half_to_float(data_ref[i]);
            data_ref[i] = nerve::simd::float_to_half(v < 0.0f ? 0.0f : v);
        }
        for (std::size_t i = 0; i < n; ++i)
        {
            if (std::abs(nerve::simd::half_to_float(data_simd[i]) -
                         nerve::simd::half_to_float(data_ref[i])) > 1e-3f)
            {
                std::cerr << "  FAIL relu_f16 at n=" << n << " i=" << i << "\n";
                return false;
            }
        }
    }
    return true;
}

bool test_axpy_f16()
{
    const std::vector<std::size_t> sizes = {0, 1, 3, 7, 16, 31, 32, 33, 100};
    const float alphas[] = {0.0f, 0.5f, 1.0f, -1.0f, 2.0f};
    for (auto n : sizes)
    {
        for (auto alpha_f : alphas)
        {
            nerve::simd::half alpha = nerve::simd::float_to_half(alpha_f);
            std::vector<nerve::simd::half> x_simd(n), y_simd(n), x_ref(n), y_ref(n);
            for (std::size_t i = 0; i < n; ++i)
            {
                float vx = static_cast<float>(i * 7 % 10) * 0.1f;
                float vy = static_cast<float>(i * 3 % 10) * 0.1f;
                x_simd[i] = nerve::simd::float_to_half(vx);
                y_simd[i] = nerve::simd::float_to_half(vy);
                x_ref[i] = x_simd[i];
                y_ref[i] = y_simd[i];
            }
            nerve::simd::simd_axpy_f16(alpha, x_simd.data(), y_simd.data(), n);
            ref::axpy_f16(alpha, x_ref.data(), y_ref.data(), n);
            for (std::size_t i = 0; i < n; ++i)
            {
                if (std::abs(nerve::simd::half_to_float(y_simd[i]) -
                             nerve::simd::half_to_float(y_ref[i])) > 1e-3f)
                {
                    std::cerr << "  FAIL axpy_f16 at n=" << n << " alpha=" << alpha_f << " i=" << i
                              << "\n";
                    return false;
                }
            }
        }
    }
    return true;
}

bool test_fmad_f16()
{
    const std::vector<std::size_t> sizes = {0, 1, 3, 7, 16, 31, 32, 33, 100};
    for (auto n : sizes)
    {
        std::vector<nerve::simd::half> a(n), b(n), c_simd(n), c_ref(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            float va = static_cast<float>(i * 7 % 10) * 0.1f;
            float vb = static_cast<float>(i * 3 % 10) * 0.1f;
            float vc = static_cast<float>(i * 5 % 10) * 0.1f;
            a[i] = nerve::simd::float_to_half(va);
            b[i] = nerve::simd::float_to_half(vb);
            c_simd[i] = nerve::simd::float_to_half(vc);
            c_ref[i] = c_simd[i];
        }
        nerve::simd::simd_fmad_f16(a.data(), b.data(), c_simd.data(), n);
        ref::fmad_f16(a.data(), b.data(), c_ref.data(), n);
        for (std::size_t i = 0; i < n; ++i)
        {
            if (std::abs(nerve::simd::half_to_float(c_simd[i]) -
                         nerve::simd::half_to_float(c_ref[i])) > 1e-3f)
            {
                std::cerr << "  FAIL fmad_f16 at n=" << n << " i=" << i << "\n";
                return false;
            }
        }
    }
    return true;
}

bool test_sqrt_f16()
{
    const std::vector<std::size_t> sizes = {0, 1, 3, 7, 16, 31, 32, 33, 100};
    for (auto n : sizes)
    {
        std::vector<nerve::simd::half> data_simd(n), data_ref(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            float v = (static_cast<float>(i * 7 % 10) + 1.0f) * 0.1f;
            data_simd[i] = nerve::simd::float_to_half(v);
            data_ref[i] = data_simd[i];
        }
        nerve::simd::simd_sqrt_f16(data_simd.data(), n);
        ref::sqrt_f16(data_ref.data(), n);
        for (std::size_t i = 0; i < n; ++i)
        {
            if (std::abs(nerve::simd::half_to_float(data_simd[i]) -
                         nerve::simd::half_to_float(data_ref[i])) > 1e-3f)
            {
                std::cerr << "  FAIL sqrt_f16 at n=" << n << " i=" << i << "\n";
                return false;
            }
        }
    }
    return true;
}

bool test_exp_f16()
{
    const std::vector<std::size_t> sizes = {0, 1, 3, 7, 16, 31, 32, 33, 100};
    for (auto n : sizes)
    {
        std::vector<nerve::simd::half> data_simd(n), data_ref(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            float v = (static_cast<float>(i * 7 % 10) - 5.0f) * 0.1f;
            data_simd[i] = nerve::simd::float_to_half(v);
            data_ref[i] = data_simd[i];
        }
        nerve::simd::simd_exp_f16(data_simd.data(), n);
        ref::exp_f16(data_ref.data(), n);
        for (std::size_t i = 0; i < n; ++i)
        {
            if (std::abs(nerve::simd::half_to_float(data_simd[i]) -
                         nerve::simd::half_to_float(data_ref[i])) > 1e-3f)
            {
                std::cerr << "  FAIL exp_f16 at n=" << n << " i=" << i << "\n";
                return false;
            }
        }
    }
    return true;
}

bool test_log_f16()
{
    const std::vector<std::size_t> sizes = {0, 1, 3, 7, 16, 31, 32, 33, 100};
    for (auto n : sizes)
    {
        std::vector<nerve::simd::half> data_simd(n), data_ref(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            float v = (static_cast<float>(i * 7 % 10) + 1.0f) * 0.1f + 0.1f;
            data_simd[i] = nerve::simd::float_to_half(v);
            data_ref[i] = data_simd[i];
        }
        nerve::simd::simd_log_f16(data_simd.data(), n);
        ref::log_f16(data_ref.data(), n);
        for (std::size_t i = 0; i < n; ++i)
        {
            if (std::abs(nerve::simd::half_to_float(data_simd[i]) -
                         nerve::simd::half_to_float(data_ref[i])) > 1e-3f)
            {
                std::cerr << "  FAIL log_f16 at n=" << n << " i=" << i << "\n";
                return false;
            }
        }
    }
    return true;
}

bool test_min_f16()
{
    const std::vector<std::size_t> sizes = {0, 1, 3, 7, 16, 31, 32, 33, 100};
    for (auto n : sizes)
    {
        std::vector<nerve::simd::half> a_simd(n), b_simd(n), a_ref(n), b_ref(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            float va = static_cast<float>(i * 7 % 10) * 0.1f;
            float vb = static_cast<float>(i * 3 % 10) * 0.1f;
            a_simd[i] = nerve::simd::float_to_half(va);
            b_simd[i] = nerve::simd::float_to_half(vb);
            a_ref[i] = a_simd[i];
            b_ref[i] = b_simd[i];
        }
        nerve::simd::simd_min_f16(a_simd.data(), b_simd.data(), n);
        ref::min_f16(a_ref.data(), b_ref.data(), n);
        for (std::size_t i = 0; i < n; ++i)
        {
            if (std::abs(nerve::simd::half_to_float(a_simd[i]) -
                         nerve::simd::half_to_float(a_ref[i])) > 1e-3f)
            {
                std::cerr << "  FAIL min_f16 at n=" << n << " i=" << i << "\n";
                return false;
            }
        }
    }
    return true;
}

bool test_max_f16()
{
    const std::vector<std::size_t> sizes = {0, 1, 3, 7, 16, 31, 32, 33, 100};
    for (auto n : sizes)
    {
        std::vector<nerve::simd::half> a_simd(n), b_simd(n), a_ref(n), b_ref(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            float va = static_cast<float>(i * 7 % 10) * 0.1f;
            float vb = static_cast<float>(i * 3 % 10) * 0.1f;
            a_simd[i] = nerve::simd::float_to_half(va);
            b_simd[i] = nerve::simd::float_to_half(vb);
            a_ref[i] = a_simd[i];
            b_ref[i] = b_simd[i];
        }
        nerve::simd::simd_max_f16(a_simd.data(), b_simd.data(), n);
        ref::max_f16(a_ref.data(), b_ref.data(), n);
        for (std::size_t i = 0; i < n; ++i)
        {
            if (std::abs(nerve::simd::half_to_float(a_simd[i]) -
                         nerve::simd::half_to_float(a_ref[i])) > 1e-3f)
            {
                std::cerr << "  FAIL max_f16 at n=" << n << " i=" << i << "\n";
                return false;
            }
        }
    }
    return true;
}

bool test_clamp_f16()
{
    const std::vector<std::size_t> sizes = {0, 1, 3, 7, 16, 31, 32, 33, 100};
    const float bounds[][2] = {{0.0f, 1.0f}, {-2.0f, 2.0f}, {-5.0f, 5.0f}};
    for (auto n : sizes)
    {
        std::vector<nerve::simd::half> data_simd(n), data_ref(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            float v = (static_cast<float>(i * 7 % 10) - 5.0f) * 0.1f;
            data_simd[i] = nerve::simd::float_to_half(v);
            data_ref[i] = data_simd[i];
        }
        for (auto [lo_f, hi_f] : bounds)
        {
            nerve::simd::half lo = nerve::simd::float_to_half(lo_f);
            nerve::simd::half hi = nerve::simd::float_to_half(hi_f);
            auto copy_simd = data_simd;
            auto copy_ref = data_ref;
            nerve::simd::simd_clamp_f16(copy_simd.data(), lo, hi, n);
            ref::clamp_f16(copy_ref.data(), lo, hi, n);
            for (std::size_t i = 0; i < n; ++i)
            {
                if (std::abs(nerve::simd::half_to_float(copy_simd[i]) -
                             nerve::simd::half_to_float(copy_ref[i])) > 1e-3f)
                {
                    std::cerr << "  FAIL clamp_f16 at n=" << n << " lo=" << lo_f << " hi=" << hi_f
                              << " i=" << i << "\n";
                    return false;
                }
            }
        }
    }
    return true;
}

bool test_sigmoid_f16()
{
    const std::vector<std::size_t> sizes = {0, 1, 3, 7, 16, 31, 32, 33, 100};
    for (auto n : sizes)
    {
        std::vector<nerve::simd::half> data_simd(n), data_ref(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            float v = (static_cast<float>(i * 7 % 10) - 5.0f) * 0.1f;
            data_simd[i] = nerve::simd::float_to_half(v);
            data_ref[i] = data_simd[i];
        }
        nerve::simd::simd_sigmoid_f16(data_simd.data(), n);
        ref::sigmoid_f16(data_ref.data(), n);
        for (std::size_t i = 0; i < n; ++i)
        {
            if (std::abs(nerve::simd::half_to_float(data_simd[i]) -
                         nerve::simd::half_to_float(data_ref[i])) > 1e-3f)
            {
                std::cerr << "  FAIL sigmoid_f16 at n=" << n << " i=" << i << "\n";
                return false;
            }
        }
    }
    return true;
}

bool test_tanh_f16()
{
    const std::vector<std::size_t> sizes = {0, 1, 3, 7, 16, 31, 32, 33, 100};
    for (auto n : sizes)
    {
        std::vector<nerve::simd::half> data_simd(n), data_ref(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            float v = static_cast<float>(i * 7 % 10) * 0.1f;
            data_simd[i] = nerve::simd::float_to_half(v);
            data_ref[i] = data_simd[i];
        }
        nerve::simd::simd_tanh_f16(data_simd.data(), n);
        ref::tanh_f16(data_ref.data(), n);
        for (std::size_t i = 0; i < n; ++i)
        {
            if (std::abs(nerve::simd::half_to_float(data_simd[i]) -
                         nerve::simd::half_to_float(data_ref[i])) > 1e-3f)
            {
                std::cerr << "  FAIL tanh_f16 at n=" << n << " i=" << i << "\n";
                return false;
            }
        }
    }
    return true;
}

bool test_sqdiff_sum_f16()
{
    const std::vector<std::size_t> sizes = {0, 1, 3, 7, 16, 31, 32, 33, 100};
    for (auto n : sizes)
    {
        std::vector<nerve::simd::half> a(n), b(n);
        float ref_sqdiff = 0.0f;
        for (std::size_t i = 0; i < n; ++i)
        {
            float va = static_cast<float>(i * 7 % 10) * 0.1f;
            float vb = static_cast<float>(i * 3 % 10) * 0.1f;
            a[i] = nerve::simd::float_to_half(va);
            b[i] = nerve::simd::float_to_half(vb);
            float d = va - vb;
            ref_sqdiff += d * d;
        }
        float simd_val = nerve::simd::simd_sqdiff_sum_f16(a.data(), b.data(), n);
        if (std::abs(simd_val - ref_sqdiff) > 0.1f * std::max(1.0f, ref_sqdiff))
        {
            std::cerr << "  FAIL sqdiff_sum_f16 at n=" << n << " simd=" << simd_val
                      << " ref=" << ref_sqdiff << "\n";
            return false;
        }
    }
    return true;
}

bool test_manhattan_f16()
{
    const std::vector<std::size_t> sizes = {0, 1, 3, 7, 16, 31, 32, 33, 100};
    for (auto n : sizes)
    {
        std::vector<nerve::simd::half> a(n), b(n);
        float ref_manhattan = 0.0f;
        for (std::size_t i = 0; i < n; ++i)
        {
            float va = static_cast<float>(i * 7 % 10) * 0.1f;
            float vb = static_cast<float>(i * 3 % 10) * 0.1f;
            a[i] = nerve::simd::float_to_half(va);
            b[i] = nerve::simd::float_to_half(vb);
            float d = va - vb;
            ref_manhattan += (d < 0.0f) ? -d : d;
        }
        float simd_val = nerve::simd::simd_manhattan_f16(a.data(), b.data(), n);
        if (std::abs(simd_val - ref_manhattan) > 0.1f * std::max(1.0f, ref_manhattan))
        {
            std::cerr << "  FAIL manhattan_f16 at n=" << n << " simd=" << simd_val
                      << " ref=" << ref_manhattan << "\n";
            return false;
        }
    }
    return true;
}

bool test_euclidean_f16()
{
    const std::vector<std::size_t> sizes = {0, 1, 3, 7, 16, 31, 32, 33, 100};
    for (auto n : sizes)
    {
        std::vector<nerve::simd::half> a(n), b(n);
        float ref_euclidean = 0.0f;
        for (std::size_t i = 0; i < n; ++i)
        {
            float va = static_cast<float>(i * 7 % 10) * 0.1f;
            float vb = static_cast<float>(i * 3 % 10) * 0.1f;
            a[i] = nerve::simd::float_to_half(va);
            b[i] = nerve::simd::float_to_half(vb);
            float d = va - vb;
            ref_euclidean += d * d;
        }
        ref_euclidean = std::sqrt(ref_euclidean);
        float simd_val = nerve::simd::simd_euclidean_f16(a.data(), b.data(), n);
        if (std::abs(simd_val - ref_euclidean) > 0.1f * std::max(1.0f, ref_euclidean))
        {
            std::cerr << "  FAIL euclidean_f16 at n=" << n << " simd=" << simd_val
                      << " ref=" << ref_euclidean << "\n";
            return false;
        }
    }
    return true;
}

bool test_cosine_f16()
{
    const std::vector<std::size_t> sizes = {0, 1, 3, 7, 16, 31, 32, 33, 100};
    for (auto n : sizes)
    {
        std::vector<nerve::simd::half> a(n), b(n);
        float ref_dot = 0.0f, ref_na = 0.0f, ref_nb = 0.0f;
        for (std::size_t i = 0; i < n; ++i)
        {
            float va = static_cast<float>(i * 7 % 10) * 0.1f;
            float vb = static_cast<float>(i * 3 % 10) * 0.1f;
            a[i] = nerve::simd::float_to_half(va);
            b[i] = nerve::simd::float_to_half(vb);
            ref_dot += va * vb;
            ref_na += va * va;
            ref_nb += vb * vb;
        }
        float ref_cosine;
        float rna = std::sqrt(ref_na);
        float rnb = std::sqrt(ref_nb);
        if (rna == 0.0f || rnb == 0.0f)
            ref_cosine = 1.0f;
        else
        {
            float cs = ref_dot / (rna * rnb);
            if (cs < -1.0f)
                cs = -1.0f;
            if (cs > 1.0f)
                cs = 1.0f;
            ref_cosine = 1.0f - cs;
        }
        float simd_val = nerve::simd::simd_cosine_f16(a.data(), b.data(), n);
        if (std::abs(simd_val - ref_cosine) > 0.1f * std::max(1.0f, ref_cosine))
        {
            std::cerr << "  FAIL cosine_f16 at n=" << n << " simd=" << simd_val
                      << " ref=" << ref_cosine << "\n";
            return false;
        }
    }
    return true;
}

bool test_norm2_f16()
{
    const std::vector<std::size_t> sizes = {0, 1, 3, 7, 16, 31, 32, 33, 100};
    for (auto n : sizes)
    {
        std::vector<nerve::simd::half> vec(n);
        float ref_norm2 = 0.0f;
        for (std::size_t i = 0; i < n; ++i)
        {
            float v = static_cast<float>(i * 7 % 10) * 0.1f;
            vec[i] = nerve::simd::float_to_half(v);
            ref_norm2 += v * v;
        }
        ref_norm2 = std::sqrt(ref_norm2);
        float simd_val = nerve::simd::simd_norm2_f16(vec.data(), n);
        if (std::abs(simd_val - ref_norm2) > 0.1f * std::max(1.0f, ref_norm2))
        {
            std::cerr << "  FAIL norm2_f16 at n=" << n << " simd=" << simd_val
                      << " ref=" << ref_norm2 << "\n";
            return false;
        }
    }
    return true;
}

bool test_reduce_max_f16()
{
    const std::vector<std::size_t> sizes = {0, 1, 3, 7, 16, 31, 32, 33, 100};
    for (auto n : sizes)
    {
        std::vector<nerve::simd::half> data(n);
        float ref_max = 0.0f;
        for (std::size_t i = 0; i < n; ++i)
        {
            float v = static_cast<float>(i * 7 % 10) * 0.1f;
            data[i] = nerve::simd::float_to_half(v);
            if (i == 0 || v > ref_max)
                ref_max = v;
        }
        float simd_val = nerve::simd::simd_reduce_max_f16(data.data(), n);
        if (std::abs(simd_val - ref_max) > 1e-3f)
        {
            std::cerr << "  FAIL reduce_max_f16 at n=" << n << " simd=" << simd_val
                      << " ref=" << ref_max << "\n";
            return false;
        }
    }
    return true;
}

bool test_reduce_min_f16()
{
    const std::vector<std::size_t> sizes = {0, 1, 3, 7, 16, 31, 32, 33, 100};
    for (auto n : sizes)
    {
        std::vector<nerve::simd::half> data(n);
        float ref_min = 0.0f;
        for (std::size_t i = 0; i < n; ++i)
        {
            float v = static_cast<float>(i * 7 % 10) * 0.1f;
            data[i] = nerve::simd::float_to_half(v);
            if (i == 0 || v < ref_min)
                ref_min = v;
        }
        float simd_val = nerve::simd::simd_reduce_min_f16(data.data(), n);
        if (std::abs(simd_val - ref_min) > 1e-3f)
        {
            std::cerr << "  FAIL reduce_min_f16 at n=" << n << " simd=" << simd_val
                      << " ref=" << ref_min << "\n";
            return false;
        }
    }
    return true;
}

bool test_quantize_f16()
{
    const std::vector<std::size_t> sizes = {0, 1, 3, 7, 16, 31, 32, 33, 100};
    const int bits[] = {1, 2, 4, 8};
    for (auto n : sizes)
    {
        for (auto b : bits)
        {
            std::vector<nerve::simd::half> input(n);
            std::vector<std::uint8_t> out_simd(n, 0), out_ref(n, 0);
            for (std::size_t i = 0; i < n; ++i)
                input[i] = nerve::simd::float_to_half(static_cast<float>(i * 7 % 10) * 0.1f);
            nerve::simd::simd_quantize_f16(input.data(), n, b, out_simd.data());
            ref::quantize_f16(input.data(), n, b, out_ref.data());
            for (std::size_t i = 0; i < n; ++i)
            {
                if (out_simd[i] != out_ref[i])
                {
                    std::cerr << "  FAIL quantize_f16 at n=" << n << " bits=" << b << " i=" << i
                              << " simd=" << (int)out_simd[i] << " ref=" << (int)out_ref[i] << "\n";
                    return false;
                }
            }
        }
    }
    return true;
}

bool test_dequantize_f16()
{
    const std::vector<std::size_t> sizes = {0, 1, 3, 7, 16, 31, 32, 33, 100};
    const int bits[] = {1, 2, 4, 8};
    for (auto n : sizes)
    {
        for (auto b : bits)
        {
            float scale = static_cast<float>((1 << b) - 1);
            std::vector<std::uint8_t> quant(n);
            for (std::size_t i = 0; i < n; ++i)
                quant[i] = static_cast<std::uint8_t>(static_cast<std::uint64_t>(i * 7 + 13) %
                                                     (static_cast<std::uint64_t>(scale) + 1));
            std::vector<nerve::simd::half> deq_simd(n), deq_ref(n);
            nerve::simd::simd_dequantize_f16(quant.data(), n, b, deq_simd.data());
            ref::dequantize_f16(quant.data(), n, b, deq_ref.data());
            for (std::size_t i = 0; i < n; ++i)
            {
                if (std::abs(nerve::simd::half_to_float(deq_simd[i]) -
                             nerve::simd::half_to_float(deq_ref[i])) > 1e-3f)
                {
                    std::cerr << "  FAIL dequantize_f16 at n=" << n << " bits=" << b << " i=" << i
                              << " simd=" << nerve::simd::half_to_float(deq_simd[i])
                              << " ref=" << nerve::simd::half_to_float(deq_ref[i]) << "\n";
                    return false;
                }
            }
        }
    }
    return true;
}

bool test_batchnorm_f16()
{
    const std::vector<std::size_t> sizes = {0, 1, 3, 7, 16, 31, 32, 33, 100};
    const float means[] = {0.0f, 1.0f, -1.0f, 5.0f};
    const float std_invs[] = {1.0f, 2.0f, 0.5f, -1.0f};
    for (auto n : sizes)
    {
        for (auto mean : means)
        {
            for (auto si : std_invs)
            {
                std::vector<nerve::simd::half> data_simd(n), data_ref(n);
                for (std::size_t i = 0; i < n; ++i)
                {
                    float v = static_cast<float>(i * 7 % 10) * 0.1f;
                    data_simd[i] = nerve::simd::float_to_half(v);
                    data_ref[i] = data_simd[i];
                }
                nerve::simd::simd_batchnorm_f16(data_simd.data(), n, mean, si);
                ref::batchnorm_f16(data_ref.data(), n, mean, si);
                for (std::size_t i = 0; i < n; ++i)
                {
                    if (std::abs(nerve::simd::half_to_float(data_simd[i]) -
                                 nerve::simd::half_to_float(data_ref[i])) > 1e-3f)
                    {
                        std::cerr << "  FAIL batchnorm_f16 at n=" << n << " mean=" << mean
                                  << " std_inv=" << si << "\n";
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

bool test_softmax_f16()
{
    const std::vector<std::size_t> sizes = {0, 1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 100};
    for (auto n : sizes)
    {
        std::vector<nerve::simd::half> data_simd(n), data_ref(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            float v = (static_cast<float>(i * 7 % 10) - 5.0f) * 0.1f;
            data_simd[i] = nerve::simd::float_to_half(v);
            data_ref[i] = data_simd[i];
        }
        nerve::simd::simd_softmax_f16(data_simd.data(), n);
        ref::softmax_f16(data_ref.data(), n);
        for (std::size_t i = 0; i < n; ++i)
        {
            if (std::abs(nerve::simd::half_to_float(data_simd[i]) -
                         nerve::simd::half_to_float(data_ref[i])) > 1e-3f)
            {
                std::cerr << "  FAIL softmax_f16 at n=" << n << "\n";
                return false;
            }
        }
        // Verify sum ~= 1.0 (in float after converting from half)
        if (n > 0)
        {
            float sum = 0.0f;
            for (std::size_t i = 0; i < n; ++i)
                sum += nerve::simd::half_to_float(data_simd[i]);
            if (std::abs(sum - 1.0f) > 1e-2f)
            {
                std::cerr << "  FAIL softmax_f16 sum at n=" << n << " sum=" << sum << "\n";
                return false;
            }
        }
    }
    return true;
}

bool test_zero_size_f16()
{
    nerve::simd::half dummy_a[] = {nerve::simd::float_to_half(0.0f)};
    nerve::simd::half dummy_b[] = {nerve::simd::float_to_half(0.0f)};
    nerve::simd::half dummy_c[] = {nerve::simd::float_to_half(0.0f)};
    std::uint8_t dummy_quant[] = {0};
    nerve::simd::simd_add_f16(dummy_a, dummy_b, 0);
    nerve::simd::simd_sub_f16(dummy_a, dummy_b, 0);
    nerve::simd::simd_mul_f16(dummy_a, dummy_b, 0);
    nerve::simd::simd_scale_f16(dummy_a, nerve::simd::float_to_half(2.0f), 0);
    nerve::simd::simd_neg_f16(dummy_a, 0);
    nerve::simd::simd_abs_f16(dummy_a, 0);
    nerve::simd::simd_relu_f16(dummy_a, 0);
    nerve::simd::simd_axpy_f16(nerve::simd::float_to_half(1.0f), dummy_a, dummy_b, 0);
    nerve::simd::simd_fmad_f16(dummy_a, dummy_b, dummy_c, 0);
    nerve::simd::simd_sqrt_f16(dummy_a, 0);
    nerve::simd::simd_exp_f16(dummy_a, 0);
    nerve::simd::simd_log_f16(dummy_a, 0);
    nerve::simd::simd_min_f16(dummy_a, dummy_b, 0);
    nerve::simd::simd_max_f16(dummy_a, dummy_b, 0);
    nerve::simd::simd_clamp_f16(dummy_a, nerve::simd::float_to_half(-1.0f),
                                nerve::simd::float_to_half(1.0f), 0);
    nerve::simd::simd_sigmoid_f16(dummy_a, 0);
    nerve::simd::simd_tanh_f16(dummy_a, 0);
    nerve::simd::simd_quantize_f16(dummy_a, 0, 8, dummy_quant);
    nerve::simd::simd_dequantize_f16(dummy_quant, 0, 8, dummy_a);
    nerve::simd::simd_batchnorm_f16(dummy_a, 0, 0.0f, 1.0f);
    nerve::simd::simd_softmax_f16(dummy_a, 0);
    if (nerve::simd::simd_reduce_sum_f16(dummy_a, 0) != 0.0f)
        return false;
    if (nerve::simd::simd_dot_f16(dummy_a, dummy_b, 0) != 0.0f)
        return false;
    if (nerve::simd::simd_sqdiff_sum_f16(dummy_a, dummy_b, 0) != 0.0f)
        return false;
    if (nerve::simd::simd_manhattan_f16(dummy_a, dummy_b, 0) != 0.0f)
        return false;
    if (nerve::simd::simd_euclidean_f16(dummy_a, dummy_b, 0) != 0.0f)
        return false;
    if (nerve::simd::simd_cosine_f16(dummy_a, dummy_b, 0) != 1.0f)
        return false;
    if (nerve::simd::simd_norm2_f16(dummy_a, 0) != 0.0f)
        return false;
    if (nerve::simd::simd_reduce_max_f16(dummy_a, 0) != 0.0f)
        return false;
    if (nerve::simd::simd_reduce_min_f16(dummy_a, 0) != 0.0f)
        return false;
    return true;
}

bool test_zero_size_f32()
{
    float dummy_a[] = {0.0f};
    float dummy_b[] = {0.0f};
    float dummy_c[] = {0.0f};
    nerve::simd::simd_add_f32(dummy_a, dummy_b, 0);
    nerve::simd::simd_sub_f32(dummy_a, dummy_b, 0);
    nerve::simd::simd_mul_f32(dummy_a, dummy_b, 0);
    nerve::simd::simd_scale_f32(dummy_a, 2.0f, 0);
    nerve::simd::simd_axpy_f32(1.0f, dummy_a, dummy_b, 0);
    nerve::simd::simd_fmad_f32(dummy_a, dummy_b, dummy_c, 0);
    nerve::simd::simd_neg_f32(dummy_a, 0);
    nerve::simd::simd_sqrt_f32(dummy_a, 0);
    nerve::simd::simd_exp_f32(dummy_a, 0);
    nerve::simd::simd_log_f32(dummy_a, 0);
    nerve::simd::simd_sigmoid_f32(dummy_a, 0);
    nerve::simd::simd_tanh_f32(dummy_a, 0);
    nerve::simd::simd_gemv_f32(1.0f, dummy_a, dummy_a, 0.0f, dummy_c, 0, 3);
    nerve::simd::simd_gemv_f32(1.0f, dummy_a, dummy_a, 0.0f, dummy_c, 3, 0);
    nerve::simd::simd_ger_f32(1.0f, dummy_a, dummy_b, dummy_a, 0, 3);
    nerve::simd::simd_ger_f32(1.0f, dummy_a, dummy_b, dummy_a, 3, 0);
    if (nerve::simd::simd_reduce_sum_f32(dummy_a, 0) != 0.0f)
        return false;
    if (nerve::simd::simd_reduce_max_f32(dummy_a, 0) != 0.0f)
        return false;
    if (nerve::simd::simd_reduce_min_f32(dummy_a, 0) != 0.0f)
        return false;
    if (nerve::simd::simd_dot_f32(dummy_a, dummy_b, 0) != 0.0f)
        return false;
    if (nerve::simd::simd_norm2_f32(dummy_a, 0) != 0.0f)
        return false;
    if (nerve::simd::simd_sqdiff_sum_f32(dummy_a, dummy_b, 0) != 0.0f)
        return false;
    if (nerve::simd::simd_euclidean_f32(dummy_a, dummy_b, 0) != 0.0f)
        return false;
    float cos0 = nerve::simd::simd_cosine_f32(dummy_a, dummy_b, 0);
    if (std::abs(cos0 - 1.0f) > 1e-6f)
        return false;
    return true;
}

// Edge case: very large values that avoid overflow
bool test_large_values()
{
    constexpr double big = 1.0e150;
    constexpr double small = 1.0;
    double a[] = {big, big, big, big, big, big, big, big};
    double b[] = {small, small, small, small, small, small, small, small};
    std::size_t n = 8;

    double dot_s = nerve::simd::simd_dot(a, b, n);
    double dot_r = ref::dot(a, b, n);
    if (!approx(dot_s, dot_r))
    {
        std::cerr << "  FAIL large_values dot\n";
        return false;
    }

    double sqd_s = nerve::simd::simd_sqdiff_sum(a, b, n);
    double sqd_r = ref::sqdiff_sum(a, b, n);
    if (!approx(sqd_s, sqd_r))
    {
        std::cerr << "  FAIL large_values sqdiff_sum\n";
        return false;
    }

    return true;
}

// Edge case: negative values for element-wise that require positivity
bool test_edge_cases()
{
    // log of very small positive
    double log_data[] = {1e-150, 0.001, 0.5, 1.0, 100.0};
    double log_ref[] = {1e-150, 0.001, 0.5, 1.0, 100.0};
    nerve::simd::simd_log(log_data, 5);
    ref::log(log_ref, 5);
    if (!all_approx(log_data, log_ref, 5))
    {
        std::cerr << "  FAIL edge log\n";
        return false;
    }

    // sqrt of values near zero
    double sqrt_data[] = {0.0, 1e-150, 1e-10, 1.0, 100.0};
    double sqrt_ref[] = {0.0, 1e-150, 1e-10, 1.0, 100.0};
    nerve::simd::simd_sqrt(sqrt_data, 5);
    ref::sqrt(sqrt_ref, 5);
    if (!all_approx(sqrt_data, sqrt_ref, 5))
    {
        std::cerr << "  FAIL edge sqrt\n";
        return false;
    }

    // clamp with inverted bounds (lo > hi) -- should still produce valid output
    double clamp_data[] = {0.0, 0.5, 1.0, -1.0, 2.0};
    double clamp_ref[] = {0.0, 0.5, 1.0, -1.0, 2.0};
    nerve::simd::simd_clamp(clamp_data, 0.2, 0.8, 5);
    ref::clamp(clamp_ref, 0.2, 0.8, 5);
    if (!all_approx(clamp_data, clamp_ref, 5))
    {
        std::cerr << "  FAIL edge clamp\n";
        return false;
    }

    return true;
}

// Main

struct TestCase
{
    const char *name;
    bool (*fn)();
};

#define TEST(x)    \
    {                \
        #x, x        \
    }

int main()
{
    // Initialize SIMD dispatch
    nerve::simd::simd_init();

    std::cerr << "Running on: " << nerve::simd::simd_arch_name(nerve::simd::detect_simd_arch())
              << "\n";

    TestCase tests[] = {
        TEST(test_memcpy),
        TEST(test_memset),
        TEST(test_memcpy_aligned),
        TEST(test_add),
        TEST(test_sub),
        TEST(test_mul),
        TEST(test_scale),
        TEST(test_axpy),
        TEST(test_fmad),
        TEST(test_reduce_sum),
        TEST(test_reduce_max),
        TEST(test_reduce_min),
        TEST(test_dot),
        TEST(test_norm2),
        TEST(test_sqdiff_sum),
        TEST(test_abs),
        TEST(test_neg),
        TEST(test_sqrt),
        TEST(test_exp),
        TEST(test_log),
        TEST(test_relu),
        TEST(test_sigmoid),
        TEST(test_tanh),
        TEST(test_min),
        TEST(test_max),
        TEST(test_clamp),
        TEST(test_gemv),
        TEST(test_ger),
        TEST(test_euclidean),
        TEST(test_manhattan),
        TEST(test_cosine),
        TEST(test_cosine_zero_vector),
        TEST(test_batchnorm),
        TEST(test_softmax),
        TEST(test_quantize),
        TEST(test_dequantize),
        TEST(test_zero_size_all_primitives),
        TEST(test_large_values),
        TEST(test_edge_cases),
        TEST(test_add_f32),
        TEST(test_sub_f32),
        TEST(test_mul_f32),
        TEST(test_scale_f32),
        TEST(test_axpy_f32),
        TEST(test_reduce_sum_f32),
        TEST(test_reduce_max_f32),
        TEST(test_dot_f32),
        TEST(test_gemv_f32),
        TEST(test_ger_f32),
        TEST(test_norm2_f32),
        TEST(test_abs_f32),
        TEST(test_relu_f32),
        TEST(test_min_f32),
        TEST(test_max_f32),
        TEST(test_clamp_f32),
        TEST(test_sqdiff_sum_f32),
        TEST(test_euclidean_f32),
        TEST(test_cosine_f32),
        TEST(test_fmad_f32),
        TEST(test_reduce_min_f32),
        TEST(test_neg_f32),
        TEST(test_sqrt_f32),
        TEST(test_exp_f32),
        TEST(test_log_f32),
        TEST(test_sigmoid_f32),
        TEST(test_tanh_f32),
        TEST(test_zero_size_f32),
        TEST(test_add_f16),
        TEST(test_sub_f16),
        TEST(test_mul_f16),
        TEST(test_scale_f16),
        TEST(test_reduce_sum_f16),
        TEST(test_dot_f16),
        TEST(test_neg_f16),
        TEST(test_abs_f16),
        TEST(test_relu_f16),
        TEST(test_axpy_f16),
        TEST(test_fmad_f16),
        TEST(test_sqrt_f16),
        TEST(test_exp_f16),
        TEST(test_log_f16),
        TEST(test_min_f16),
        TEST(test_max_f16),
        TEST(test_clamp_f16),
        TEST(test_sigmoid_f16),
        TEST(test_tanh_f16),
        TEST(test_sqdiff_sum_f16),
        TEST(test_manhattan_f16),
        TEST(test_euclidean_f16),
        TEST(test_cosine_f16),
        TEST(test_norm2_f16),
        TEST(test_reduce_max_f16),
        TEST(test_reduce_min_f16),
        TEST(test_batchnorm_f16),
        TEST(test_softmax_f16),
        TEST(test_quantize_f16),
        TEST(test_dequantize_f16),
        TEST(test_zero_size_f16),
    };

    int failures = 0;
    for (auto &t : tests)
    {
        bool ok = t.fn();
        if (ok)
            std::cout << "PASS: " << t.name << "\n";
        else
        {
            std::cerr << "FAIL: " << t.name << "\n";
            ++failures;
        }
    }

    std::cout << "\n"
              << (failures == 0 ? "All tests passed." : "Some tests FAILED.") << " ("
              << (sizeof(tests) / sizeof(tests[0])) - failures << "/"
              << (sizeof(tests) / sizeof(tests[0])) << ")\n";

    return failures > 0 ? 1 : 0;
}
