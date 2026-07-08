#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <execution>
#include <iterator>
#include <memory>
#include <numeric>
#include <span>
#include <vector>

#ifdef NERVE_USE_SIMD
#include "nerve/cpu/x86_intrinsics.hpp"
#endif

#include "nerve/algorithms/knn_hnsw.hpp"

namespace nerve::algorithms
{

template <typename T>
concept Numeric = std::floating_point<T>;

template <Numeric T = float>
class DistanceMetric
{
public:
    virtual ~DistanceMetric() = default;
    virtual T compute(std::span<const T> a, std::span<const T> b) const = 0;
    virtual std::vector<T> compute_matrix(std::span<const T> points, size_t n_points,
                                          size_t dim) const = 0;
};

template <Numeric T = float>
class EuclideanMetric : public DistanceMetric<T>
{
public:
    T compute(std::span<const T> a, std::span<const T> b) const override;
    std::vector<T> compute_matrix(std::span<const T> points, size_t n_points,
                                  size_t dim) const override;

    static T compute_simd(std::span<const T> a, std::span<const T> b);
};

template <Numeric T = float>
class ManhattanMetric : public DistanceMetric<T>
{
public:
    T compute(std::span<const T> a, std::span<const T> b) const override;
    std::vector<T> compute_matrix(std::span<const T> points, size_t n_points,
                                  size_t dim) const override;
};

template <Numeric T = float>
class CosineMetric : public DistanceMetric<T>
{
public:
    T compute(std::span<const T> a, std::span<const T> b) const override;
    std::vector<T> compute_matrix(std::span<const T> points, size_t n_points,
                                  size_t dim) const override;
};

template <Numeric T = float>
class MinkowskiMetric : public DistanceMetric<T>
{
public:
    explicit MinkowskiMetric(T p = T(3))
        : p_(p)
    {}
    T compute(std::span<const T> a, std::span<const T> b) const override;
    std::vector<T> compute_matrix(std::span<const T> points, size_t n_points,
                                  size_t dim) const override;

private:
    T p_;
};

template <Numeric T = float>
class CanberraMetric : public DistanceMetric<T>
{
public:
    T compute(std::span<const T> a, std::span<const T> b) const override;
    std::vector<T> compute_matrix(std::span<const T> points, size_t n_points,
                                  size_t dim) const override;
};

template <Numeric T = float>
class BrayCurtisMetric : public DistanceMetric<T>
{
public:
    T compute(std::span<const T> a, std::span<const T> b) const override;
    std::vector<T> compute_matrix(std::span<const T> points, size_t n_points,
                                  size_t dim) const override;
};

template <Numeric T = float>
class CorrelationMetric : public DistanceMetric<T>
{
public:
    T compute(std::span<const T> a, std::span<const T> b) const override;
    std::vector<T> compute_matrix(std::span<const T> points, size_t n_points,
                                  size_t dim) const override;
};

template <Numeric T = float>
class DistanceMatrixComputer
{
public:
    struct Config
    {
        enum class Metric : std::uint8_t
        {
            EUCLIDEAN,
            MANHATTAN,
            COSINE,
            CHEBYSHEV,
            MINKOWSKI,
            CANBERRA,
            BRAYCURTIS,
            CORRELATION
        } metric = Metric::EUCLIDEAN;
        bool use_simd = true;
        bool use_openmp = true;
        size_t block_size = 64;
    };

    explicit DistanceMatrixComputer(Config config = {})
        : config_(config)
    {}

    [[nodiscard]] std::vector<T> compute(std::span<const T> points, size_t n_points,
                                         size_t dim) const;

    [[nodiscard]] std::vector<T> compute_pairwise(std::span<const T> set_a, size_t n_a,
                                                  std::span<const T> set_b, size_t n_b,
                                                  size_t dim) const;

    [[nodiscard]] std::vector<T> compute_symmetric(std::span<const T> points, size_t n_points,
                                                   size_t dim) const;

    [[nodiscard]] std::vector<T> compute_chunked(std::span<const T> points, size_t n_points,
                                                 size_t dim, size_t chunk_size) const;

private:
    Config config_;

    [[nodiscard]] std::vector<T> compute_euclidean(std::span<const T> points, size_t n_points,
                                                   size_t dim) const;

    [[nodiscard]] std::vector<T> compute_manhattan(std::span<const T> points, size_t n_points,
                                                   size_t dim) const;

    [[nodiscard]] std::vector<T> compute_euclidean_simd(std::span<const T> points, size_t n_points,
                                                        size_t dim) const;

    [[nodiscard]] std::vector<T> compute_blocked(std::span<const T> points, size_t n_points,
                                                 size_t dim, size_t block_size) const;

    [[nodiscard]] T compute_single(const T* a, const T* b, size_t dim) const;
};

template <Numeric T = float>
class KNNComputer
{
public:
    struct Config
    {
        size_t k = 5;
        enum class Algorithm : std::uint8_t
        {
            BRUTE_FORCE,
            HNSW
        } algorithm = Algorithm::BRUTE_FORCE;
        enum class Metric : std::uint8_t
        {
            EUCLIDEAN,
            MANHATTAN,
            COSINE
        } metric = Metric::EUCLIDEAN;
        bool use_openmp = true;
        int hnsw_M = 16;
        int hnsw_ef_construction = 200;
        int hnsw_ef_search = 100;
        size_t hnsw_random_seed = 42;
    };

    struct Result
    {
        std::vector<T> distances;
        std::vector<size_t> indices;
        size_t n_points = 0;
        size_t k = 0;
    };

    explicit KNNComputer(Config config = {})
        : config_(config)
    {}

    [[nodiscard]] Result compute(std::span<const T> points, size_t n_points, size_t dim) const;

    [[nodiscard]] Result compute_query(std::span<const T> reference, size_t n_ref,
                                       std::span<const T> queries, size_t n_query,
                                       size_t dim) const;

private:
    Config config_;

    [[nodiscard]] Result compute_brute_force(std::span<const T> points, size_t n_points,
                                             size_t dim) const;

    [[nodiscard]] Result compute_hnsw(std::span<const T> points, size_t n_points, size_t dim) const;

    static void find_k_smallest(std::span<T> distances, std::span<size_t> indices, size_t k);
};

template <Numeric T = float>
class SparseDistanceMatrixComputer
{
public:
    struct Config
    {
        T epsilon = T(0.1);
        size_t k_max = 50;
        enum class Mode : std::uint8_t
        {
            EPSILON_NEIGHBORHOOD,
            KNN
        } mode = Mode::EPSILON_NEIGHBORHOOD;
    };

    struct SparseMatrix
    {
        std::vector<T> values;
        std::vector<size_t> col_idx;
        std::vector<size_t> row_ptr;
        size_t n_rows = 0;
        size_t n_cols = 0;
        size_t nnz = 0;
    };

    explicit SparseDistanceMatrixComputer(Config config = {})
        : config_(config)
    {}

    [[nodiscard]] SparseMatrix compute(std::span<const T> points, size_t n_points,
                                       size_t dim) const;

    [[nodiscard]] static SparseMatrix from_dense(std::span<const T> dense, size_t n_rows,
                                                 size_t n_cols, T threshold);

    [[nodiscard]] static std::vector<T> to_dense(const SparseMatrix &sparse);

private:
    Config config_;

    [[nodiscard]] SparseMatrix compute_epsilon(std::span<const T> points, size_t n_points,
                                               size_t dim) const;

    [[nodiscard]] SparseMatrix compute_knn(std::span<const T> points, size_t n_points,
                                           size_t dim) const;
};

extern "C"
{
    void nerve_pairwise_distances_f32(const float *points, size_t n, size_t dim, float *output);
    void nerve_pairwise_distances_f64(const double *points, size_t n, size_t dim, double *output);

    void nerve_knn_f32(const float *points, size_t n, size_t dim, size_t k, float *out_distances,
                       size_t *out_indices);
    void nerve_knn_f64(const double *points, size_t n, size_t dim, size_t k, double *out_distances,
                       size_t *out_indices);
}

template <Numeric T>
void vectorized_sqrt(std::span<T> data);

template <Numeric T>
T vectorized_sum(std::span<const T> data);

template <Numeric T>
T dot_product(std::span<const T> a, std::span<const T> b);

template <Numeric T>
void vectorized_sqrt(std::span<T> data)
{
#if defined(NERVE_USE_SIMD) && defined(__AVX512F__)
    if constexpr (sizeof(T) == sizeof(float))
    {
        constexpr size_t kWidth = 16;
        size_t i = 0;
        for (; i + kWidth <= data.size(); i += kWidth)
        {
            const __m512 v = _mm512_loadu_ps(data.data() + i);
            _mm512_storeu_ps(data.data() + i, _mm512_sqrt_ps(v));
        }
        for (; i < data.size(); ++i)
        {
            data[i] = std::sqrt(data[i]);
        }
        return;
    }
    if constexpr (sizeof(T) == sizeof(double))
    {
        constexpr size_t kWidth = 8;
        size_t i = 0;
        for (; i + kWidth <= data.size(); i += kWidth)
        {
            const __m512d v = _mm512_loadu_pd(data.data() + i);
            _mm512_storeu_pd(data.data() + i, _mm512_sqrt_pd(v));
        }
        for (; i < data.size(); ++i)
        {
            data[i] = std::sqrt(data[i]);
        }
        return;
    }
#endif
    for (T &value : data)
    {
        value = std::sqrt(value);
    }
}

template <Numeric T>
T vectorized_sum(std::span<const T> data)
{
#if defined(NERVE_USE_SIMD) && defined(__AVX512F__)
    if constexpr (sizeof(T) == sizeof(float))
    {
        constexpr size_t kWidth = 16;
        __m512 acc = _mm512_setzero_ps();
        size_t i = 0;
        for (; i + kWidth <= data.size(); i += kWidth)
        {
            acc = _mm512_add_ps(acc, _mm512_loadu_ps(data.data() + i));
        }
        T sum = _mm512_reduce_add_ps(acc);
        for (; i < data.size(); ++i)
        {
            sum += data[i];
        }
        return sum;
    }
    if constexpr (sizeof(T) == sizeof(double))
    {
        constexpr size_t kWidth = 8;
        __m512d acc = _mm512_setzero_pd();
        size_t i = 0;
        for (; i + kWidth <= data.size(); i += kWidth)
        {
            acc = _mm512_add_pd(acc, _mm512_loadu_pd(data.data() + i));
        }
        T sum = _mm512_reduce_add_pd(acc);
        for (; i < data.size(); ++i)
        {
            sum += data[i];
        }
        return sum;
    }
#endif
    return std::accumulate(data.begin(), data.end(), T{0});
}

template <Numeric T>
T dot_product(std::span<const T> a, std::span<const T> b)
{
    const size_t dim = std::min(a.size(), b.size());
#if defined(NERVE_USE_SIMD) && defined(__AVX512F__)
    if constexpr (sizeof(T) == sizeof(float))
    {
        constexpr size_t kWidth = 16;
        __m512 acc = _mm512_setzero_ps();
        size_t i = 0;
        for (; i + kWidth <= dim; i += kWidth)
        {
            const __m512 va = _mm512_loadu_ps(a.data() + i);
            const __m512 vb = _mm512_loadu_ps(b.data() + i);
            acc = _mm512_fmadd_ps(va, vb, acc);
        }
        T sum = _mm512_reduce_add_ps(acc);
        for (; i < dim; ++i)
        {
            sum += a[i] * b[i];
        }
        return sum;
    }
    if constexpr (sizeof(T) == sizeof(double))
    {
        constexpr size_t kWidth = 8;
        __m512d acc = _mm512_setzero_pd();
        size_t i = 0;
        for (; i + kWidth <= dim; i += kWidth)
        {
            const __m512d va = _mm512_loadu_pd(a.data() + i);
            const __m512d vb = _mm512_loadu_pd(b.data() + i);
            acc = _mm512_fmadd_pd(va, vb, acc);
        }
        T sum = _mm512_reduce_add_pd(acc);
        for (; i < dim; ++i)
        {
            sum += a[i] * b[i];
        }
        return sum;
    }
#endif
    T sum = T{0};
    for (size_t i = 0; i < dim; ++i)
    {
        sum += a[i] * b[i];
    }
    return sum;
}

} // namespace nerve::algorithms
