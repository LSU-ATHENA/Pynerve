#include "nerve/simd/simd_base.hpp"

#include <algorithm>
#include <cmath>

// Scalar memcpy/memset use manual byte loops with noipa to prevent GCC from
// recognizing the byte-fill pattern and converting it to __builtin_memset
// during optimization, which would trigger a false-positive
// -Wstringop-overflow warning (the optimizer over-approximates the dynamic
// size parameter and warns the memset bound exceeds max object size).

namespace nerve::simd
{

// Scalar implementations
namespace scalar
{

#if defined(__GNUC__)
__attribute__((noipa))
#elif defined(_MSC_VER)
__declspec(noinline)
#endif
void memcpy(void *dst, const void *src, std::size_t bytes)
{
    auto *d = static_cast<std::uint8_t *>(dst);
    const auto *s = static_cast<const std::uint8_t *>(src);
    for (std::size_t i = 0; i < bytes; ++i)
        d[i] = s[i];
}

#if defined(__GNUC__)
__attribute__((noipa))
#elif defined(_MSC_VER)
__declspec(noinline)
#endif
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
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        sum += data[i];
    return sum;
}

double reduce_max(const double *data, std::size_t n)
{
    if (n == 0)
        return 0.0;
    double m = data[0];
    for (std::size_t i = 1; i < n; ++i)
        if (data[i] > m)
            m = data[i];
    return m;
}

double reduce_min(const double *data, std::size_t n)
{
    if (n == 0)
        return 0.0;
    double m = data[0];
    for (std::size_t i = 1; i < n; ++i)
        if (data[i] < m)
            m = data[i];
    return m;
}

double dot(const double *a, const double *b, std::size_t n)
{
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        sum += a[i] * b[i];
    return sum;
}

double norm2(const double *vec, std::size_t n)
{
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        sum += vec[i] * vec[i];
    return std::sqrt(sum);
}

double sqdiff_sum(const double *a, const double *b, std::size_t n)
{
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i)
    {
        double d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
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
    for (std::size_t i = 0; i < n; ++i)
        a[i] = a[i] < b[i] ? a[i] : b[i];
}

void max(double *a, const double *b, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        a[i] = a[i] > b[i] ? a[i] : b[i];
}

void clamp(double *data, double lo, double hi, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
    {
        if (data[i] < lo)
            data[i] = lo;
        if (data[i] > hi)
            data[i] = hi;
    }
}

void add_f32(float *a, const float *b, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        a[i] += b[i];
}

void fmad_f32(const float *a, const float *b, float *c, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        c[i] += a[i] * b[i];
}

float reduce_sum_f32(const float *data, std::size_t n)
{
    float sum = 0.0f;
    for (std::size_t i = 0; i < n; ++i)
        sum += data[i];
    return sum;
}

float dot_f32(const float *a, const float *b, std::size_t n)
{
    float sum = 0.0f;
    for (std::size_t i = 0; i < n; ++i)
        sum += a[i] * b[i];
    return sum;
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

void abs_f32(float *data, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        data[i] = std::abs(data[i]);
}

void relu_f32(float *data, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        if (data[i] < 0.0f)
            data[i] = 0.0f;
}

void min_f32(float *a, const float *b, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        a[i] = a[i] < b[i] ? a[i] : b[i];
}

void max_f32(float *a, const float *b, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        a[i] = a[i] > b[i] ? a[i] : b[i];
}

void clamp_f32(float *data, float lo, float hi, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
    {
        if (data[i] < lo)
            data[i] = lo;
        if (data[i] > hi)
            data[i] = hi;
    }
}

float reduce_max_f32(const float *data, std::size_t n)
{
    if (n == 0)
        return 0.0f;
    float m = data[0];
    for (std::size_t i = 1; i < n; ++i)
        if (data[i] > m)
            m = data[i];
    return m;
}

float reduce_min_f32(const float *data, std::size_t n)
{
    if (n == 0)
        return 0.0f;
    float m = data[0];
    for (std::size_t i = 1; i < n; ++i)
        if (data[i] < m)
            m = data[i];
    return m;
}

float norm2_f32(const float *vec, std::size_t n)
{
    float sum = 0.0f;
    for (std::size_t i = 0; i < n; ++i)
        sum += vec[i] * vec[i];
    return std::sqrt(sum);
}

float sqdiff_sum_f32(const float *a, const float *b, std::size_t n)
{
    float sum = 0.0f;
    for (std::size_t i = 0; i < n; ++i)
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

void gemv_f32(float alpha, const float *A, const float *x, float beta, float *y, std::size_t m,
              std::size_t n)
{
    if (beta != 1.0f)
        for (std::size_t i = 0; i < m; ++i)
            y[i] *= beta;
    for (std::size_t i = 0; i < m; ++i)
    {
        float sum = 0.0f;
        for (std::size_t j = 0; j < n; ++j)
            sum += A[i * n + j] * x[j];
        y[i] += alpha * sum;
    }
}

void ger_f32(float alpha, const float *x, const float *y, float *A, std::size_t m, std::size_t n)
{
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j)
            A[i * n + j] += alpha * x[i] * y[j];
}

void gemv(double alpha, const double *A, const double *x, double beta, double *y, std::size_t m,
          std::size_t n)
{
    // y = beta * y
    if (beta != 1.0)
        for (std::size_t i = 0; i < m; ++i)
            y[i] *= beta;

    // y += alpha * A * x
    for (std::size_t i = 0; i < m; ++i)
    {
        double sum = 0.0;
        for (std::size_t j = 0; j < n; ++j)
            sum += A[i * n + j] * x[j];
        y[i] += alpha * sum;
    }
}

void ger(double alpha, const double *x, const double *y, double *A, std::size_t m, std::size_t n)
{
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j)
            A[i * n + j] += alpha * x[i] * y[j];
}

// Float16 scalar implementations
// All float16 operations internally convert to float32, compute, and convert back.

void add_f16(half *a, const half *b, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        a[i] = float_to_half(half_to_float(a[i]) + half_to_float(b[i]));
}

void sub_f16(half *a, const half *b, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        a[i] = float_to_half(half_to_float(a[i]) - half_to_float(b[i]));
}

void mul_f16(half *a, const half *b, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        a[i] = float_to_half(half_to_float(a[i]) * half_to_float(b[i]));
}

void scale_f16(half *a, half alpha, std::size_t n)
{
    float fa = half_to_float(alpha);
    for (std::size_t i = 0; i < n; ++i)
        a[i] = float_to_half(half_to_float(a[i]) * fa);
}

void axpy_f16(half alpha, const half *x, half *y, std::size_t n)
{
    float fa = half_to_float(alpha);
    for (std::size_t i = 0; i < n; ++i)
        y[i] = float_to_half(half_to_float(y[i]) + fa * half_to_float(x[i]));
}

void fmad_f16(const half *a, const half *b, half *c, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        c[i] = float_to_half(half_to_float(c[i]) + half_to_float(a[i]) * half_to_float(b[i]));
}

float reduce_sum_f16(const half *data, std::size_t n)
{
    float sum = 0.0f;
    for (std::size_t i = 0; i < n; ++i)
        sum += half_to_float(data[i]);
    return sum;
}

float reduce_max_f16(const half *data, std::size_t n)
{
    if (n == 0)
        return 0.0f;
    float m = half_to_float(data[0]);
    for (std::size_t i = 1; i < n; ++i)
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
    float m = half_to_float(data[0]);
    for (std::size_t i = 1; i < n; ++i)
    {
        float v = half_to_float(data[i]);
        if (v < m)
            m = v;
    }
    return m;
}

float dot_f16(const half *a, const half *b, std::size_t n)
{
    float sum = 0.0f;
    for (std::size_t i = 0; i < n; ++i)
        sum += half_to_float(a[i]) * half_to_float(b[i]);
    return sum;
}

float norm2_f16(const half *vec, std::size_t n)
{
    float sum = 0.0f;
    for (std::size_t i = 0; i < n; ++i)
    {
        float v = half_to_float(vec[i]);
        sum += v * v;
    }
    return std::sqrt(sum);
}

void neg_f16(half *data, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        data[i] = float_to_half(-half_to_float(data[i]));
}

void sqrt_f16(half *data, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
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
    for (std::size_t i = 0; i < n; ++i)
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
    for (std::size_t i = 0; i < n; ++i)
        data[i] = float_to_half(std::abs(half_to_float(data[i])));
}

void min_f16(half *a, const half *b, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
    {
        float fa = half_to_float(a[i]);
        float fb = half_to_float(b[i]);
        a[i] = float_to_half(fa < fb ? fa : fb);
    }
}

void max_f16(half *a, const half *b, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
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
    for (std::size_t i = 0; i < n; ++i)
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
    float sum = 0.0f;
    for (std::size_t i = 0; i < n; ++i)
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

} // namespace scalar

// Assignment function called by simd_init()
extern "C" void nerve_simd_assign_scalar(SimdDispatchTable *table)
{
    table->memcpy = scalar::memcpy;
    table->memset = scalar::memset;
    table->add = scalar::add;
    table->sub = scalar::sub;
    table->mul = scalar::mul;
    table->scale = scalar::scale;
    table->axpy = scalar::axpy;
    table->fmad = scalar::fmad;
    table->reduce_sum = scalar::reduce_sum;
    table->reduce_max = scalar::reduce_max;
    table->reduce_min = scalar::reduce_min;
    table->dot = scalar::dot;
    table->norm2 = scalar::norm2;
    table->sqdiff_sum = scalar::sqdiff_sum;
    table->abs = scalar::abs;
    table->neg = scalar::neg;
    table->sqrt = scalar::sqrt;
    table->exp = scalar::exp;
    table->log = scalar::log;
    table->relu = scalar::relu;
    table->sigmoid = scalar::sigmoid;
    table->tanh = scalar::tanh;
    table->min = scalar::min;
    table->max = scalar::max;
    table->clamp = scalar::clamp;
    table->fmad_f32 = scalar::fmad_f32;
    table->add_f32 = scalar::add_f32;
    table->sub_f32 = scalar::sub_f32;
    table->mul_f32 = scalar::mul_f32;
    table->scale_f32 = scalar::scale_f32;
    table->axpy_f32 = scalar::axpy_f32;
    table->reduce_sum_f32 = scalar::reduce_sum_f32;
    table->reduce_max_f32 = scalar::reduce_max_f32;
    table->reduce_min_f32 = scalar::reduce_min_f32;
    table->dot_f32 = scalar::dot_f32;
    table->norm2_f32 = scalar::norm2_f32;
    table->neg_f32 = scalar::neg_f32;
    table->sqrt_f32 = scalar::sqrt_f32;
    table->exp_f32 = scalar::exp_f32;
    table->log_f32 = scalar::log_f32;
    table->sigmoid_f32 = scalar::sigmoid_f32;
    table->tanh_f32 = scalar::tanh_f32;
    table->sqdiff_sum_f32 = scalar::sqdiff_sum_f32;
    table->euclidean_f32 = scalar::euclidean_f32;
    table->cosine_f32 = scalar::cosine_f32;
    table->abs_f32 = scalar::abs_f32;
    table->relu_f32 = scalar::relu_f32;
    table->min_f32 = scalar::min_f32;
    table->max_f32 = scalar::max_f32;
    table->clamp_f32 = scalar::clamp_f32;
    table->gemv_f32 = scalar::gemv_f32;
    table->ger_f32 = scalar::ger_f32;
    table->gemv = scalar::gemv;
    table->ger = scalar::ger;
    table->add_f16 = scalar::add_f16;
    table->sub_f16 = scalar::sub_f16;
    table->mul_f16 = scalar::mul_f16;
    table->scale_f16 = scalar::scale_f16;
    table->axpy_f16 = scalar::axpy_f16;
    table->fmad_f16 = scalar::fmad_f16;
    table->reduce_sum_f16 = scalar::reduce_sum_f16;
    table->reduce_max_f16 = scalar::reduce_max_f16;
    table->reduce_min_f16 = scalar::reduce_min_f16;
    table->dot_f16 = scalar::dot_f16;
    table->norm2_f16 = scalar::norm2_f16;
    table->neg_f16 = scalar::neg_f16;
    table->sqrt_f16 = scalar::sqrt_f16;
    table->exp_f16 = scalar::exp_f16;
    table->log_f16 = scalar::log_f16;
    table->relu_f16 = scalar::relu_f16;
    table->sigmoid_f16 = scalar::sigmoid_f16;
    table->tanh_f16 = scalar::tanh_f16;
    table->abs_f16 = scalar::abs_f16;
    table->min_f16 = scalar::min_f16;
    table->max_f16 = scalar::max_f16;
    table->clamp_f16 = scalar::clamp_f16;
    table->sqdiff_sum_f16 = scalar::sqdiff_sum_f16;
    table->gemv_f16 = scalar::gemv_f16;
    table->ger_f16 = scalar::ger_f16;
    table->quantize_f16 = scalar::quantize_f16;
    table->dequantize_f16 = scalar::dequantize_f16;
}

} // namespace nerve::simd
