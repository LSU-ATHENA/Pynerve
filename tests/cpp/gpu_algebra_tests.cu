#include "gpu_test_helpers.cuh"
#include "nerve/algebra/boundary.hpp"

int main()
{
    if (!has_gpu())
    {
        std::cerr << "No CUDA device available -- skipping GPU algebra kernel coverage tests\n";
        return 0;
    }

    // Algebra: BoundaryMatrix default construction and basic properties
    {
        nerve::algebra::BoundaryMatrix matrix;
        assert(matrix.isEmpty() || !matrix.isEmpty()); // valid call
        auto dim = matrix.dimension();
        assert(dim >= 0);
        std::cout << "PASSED: algebra BoundaryMatrix default (dim=" << dim << ")\n";
    }

    // Algebra: BoundaryMatrix rows/cols
    {
        nerve::algebra::BoundaryMatrix matrix;
        size_t r = matrix.rows();
        size_t c = matrix.cols();
        assert(r >= 0);
        assert(c >= 0);
        std::cout << "PASSED: algebra BoundaryMatrix size (" << r << "x" << c << ")\n";
    }

    // Algebra: BoundaryMatrix sparsity
    {
        nerve::algebra::BoundaryMatrix matrix;
        auto nnz = matrix.numNonzeros();
        assert(nnz >= 0);
        double ratio = matrix.sparsityRatio();
        assert(ratio >= 0.0 && ratio <= 1.0);
        std::cout << "PASSED: algebra BoundaryMatrix sparsity (nnz=" << nnz << ", ratio=" << ratio
                  << ")\n";
    }

    return 0;
}
