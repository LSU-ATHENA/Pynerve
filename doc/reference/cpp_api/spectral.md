# `nerve::spectral` -- Laplacians and Eigensolvers

```cpp
#include <nerve/spectral/laplacian.hpp>
#include <nerve/spectral/eigensolver.hpp>

namespace nerve::spectral;

class Laplacian {
public:
    struct Config {
        bool normalize = false;
        double tolerance = 1e-10;
        size_t max_iterations = 1000;
    };

    explicit Laplacian(const Config& config = {});

    // Construct graph Laplacian L = D - A
    SparseMatrix constructFromEdges(
        std::span<const std::pair<Index, Index>> edges,
        std::span<const double> weights, size_t n_vertices);

    // Construct point cloud Laplacian (heat kernel)
    SparseMatrix constructFromPoints(
        std::span<const double> points, size_t n, size_t dim,
        double kernel_sigma = 1.0);

    std::vector<double> solve(const std::vector<double>& b) const;
};

class Eigensolver {
public:
    struct Config {
        size_t k = 10;           // number of eigenvalues
        enum Mode { LANCZOS, LOBPCG, KRYLOV_SHIFT } mode = LANCZOS;
        double tolerance = 1e-8;
        size_t max_iterations = 1000;
    };

    explicit Eigensolver(const Config& config = {});

    struct Result {
        std::vector<double> eigenvalues;
        std::vector<std::vector<double>> eigenvectors;
        size_t iterations;
        bool converged;
    };

    Result compute(const SparseMatrix& matrix) const;
    Result computeGeneralized(const SparseMatrix& A, const SparseMatrix& B) const;
};
```

**Cost (constructFromEdges):** O(m) where m = n_edges.

**Cost (constructFromPoints):** O(n^2 * dim) for distance computation + O(nnz) for assembly.

**Cost (Eigensolver::compute):** O(k * nnz) for Lanczos, O(k * n * log n) for LOBPCG.

<- [C++ API Overview](index.md)
