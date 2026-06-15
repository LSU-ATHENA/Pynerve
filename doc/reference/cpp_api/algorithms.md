# `nerve::algorithms` -- Distance and KNN

### DistanceMetric (polymorphic)

```cpp
#include <nerve/algorithms/distance.hpp>

namespace nerve::algorithms;

template <Numeric T = float>
class DistanceMetric {
public:
    virtual ~DistanceMetric() = default;
    virtual T compute(std::span<const T> a, std::span<const T> b) const = 0;
    virtual std::vector<T> compute_matrix(
        std::span<const T> points, size_t n_points, size_t dim) const = 0;
};

// Concrete metrics
template <Numeric T = float> class EuclideanMetric;
template <Numeric T = float> class ManhattanMetric;
template <Numeric T = float> class CosineMetric;
template <Numeric T = float> class MinkowskiMetric;
template <Numeric T = float> class CanberraMetric;
template <Numeric T = float> class BrayCurtisMetric;
template <Numeric T = float> class CorrelationMetric;
```

**Cost (compute):** O(dim) per call.

**Cost (compute_matrix):** O(n^2 * dim) default; O(n^2) for Euclidean via optimized BLAS.

### DistanceMatrixComputer

```cpp
template <Numeric T = float>
class DistanceMatrixComputer {
public:
    struct Config {
        enum Metric { EUCLIDEAN, MANHATTAN, COSINE, CHEBYSHEV, MINKOWSKI,
                      CANBERRA, BRAYCURTIS, CORRELATION } metric = EUCLIDEAN;
        bool use_simd = true;
        bool use_openmp = true;
        size_t block_size = 64;
    };

    explicit DistanceMatrixComputer(Config config = {});

    std::vector<T> compute(std::span<const T> points, size_t n_points, size_t dim) const;
    std::vector<T> compute_pairwise(std::span<const T> set_a, size_t n_a,
                                    std::span<const T> set_b, size_t n_b, size_t dim) const;
    std::vector<T> compute_symmetric(std::span<const T> points, size_t n_points, size_t dim) const;
    std::vector<T> compute_chunked(std::span<const T> points, size_t n_points,
                                   size_t dim, size_t chunk_size) const;
};
```

**Cost (compute):** O(n^2 * dim) with SIMD, O(n^2 * dim / p) with OpenMP.

**Cost (compute_pairwise):** O(n_a * n_b * dim).

**Cost (compute_chunked):** O(n^2 * dim) but with O(chunk_size * n) memory footprint.

### KNNComputer

```cpp
template <Numeric T = float>
class KNNComputer {
public:
    struct Config {
        size_t k = 5;
        enum Algorithm { BRUTE_FORCE, HNSW } algorithm = BRUTE_FORCE;
        // HNSW parameters
        int hnsw_M = 16;
        int hnsw_ef_construction = 200;
        int hnsw_ef_search = 100;
    };

    struct Result {
        std::vector<T> distances;
        std::vector<size_t> indices;
        size_t n_points = 0;
        size_t k = 0;
    };

    explicit KNNComputer(Config config = {});

    Result compute(std::span<const T> points, size_t n_points, size_t dim) const;
    Result compute_query(std::span<const T> reference, size_t n_ref,
                         std::span<const T> queries, size_t n_query, size_t dim) const;
};
```

**Cost (compute, brute force):** O(n^2 * dim + n * k log n).

**Cost (compute, HNSW):** O(n * log n * dim + n * k * log n).

### SparseDistanceMatrixComputer

```cpp
template <Numeric T = float>
class SparseDistanceMatrixComputer {
public:
    struct Config {
        T epsilon = T(0.1);
        size_t k_max = 50;
        enum Mode { EPSILON_NEIGHBORHOOD, KNN } mode = EPSILON_NEIGHBORHOOD;
    };

    struct SparseMatrix {
        std::vector<T> values;
        std::vector<size_t> col_idx;
        std::vector<size_t> row_ptr;
        size_t n_rows, n_cols, nnz;
    };

    SparseMatrix compute(std::span<const T> points, size_t n_points, size_t dim) const;
    static SparseMatrix from_dense(std::span<const T> dense, size_t n_rows, size_t n_cols, T threshold);
    static std::vector<T> to_dense(const SparseMatrix& sparse);
};
```

**Cost (compute):** O(n * k_max * dim) for KNN mode; O(n^2 * dim) for epsilon_neighborhood when dense.

**Cost (from_dense):** O(n^2) to scan + O(nnz) to build CSR.

<- [C++ API Overview](index.md)
