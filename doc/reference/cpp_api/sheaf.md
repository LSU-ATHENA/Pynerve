# `nerve::sheaf` -- Sheaf Computation

```cpp
#include <nerve/sheaf/sheaf_laplacian.hpp>

namespace nerve::sheaf;

class SheafLaplacian {
public:
    struct Config {
        bool use_gpu = false;
        bool use_tensor_cores = false;
        double tolerance = 1e-10;
        size_t max_iterations = 1000;
    };

    explicit SheafLaplacian(const Config& config = {});

    // Construct Laplacian from sheaf data
    SparseMatrix construct(const std::vector<Stalk>& stalks,
                           const std::vector<Restriction>& restrictions);

    // Solve L x = b
    std::vector<double> solve(const std::vector<double>& b) const;

    // Compute eigenvalues
    std::vector<double> eigenvalues(size_t k) const;
};
```

**Cost (construct):** O(nnz) for sparse matrix assembly.

**Cost (solve):** O(nnz^{1.5}) for iterative solver.

<- [C++ API Overview](index.md)
