#include "gpu_test_helpers.cuh"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

// GPUDiracOperator is defined inline in dirac_operator_gpu.cu inside
// namespace nerve::spectral::gpu.  Include the whole file so that nvcc
// produces a single translation unit -- inline methods containing CUDA
// kernel launches are not emitted as linkable symbols when compiled
// with separable compilation across TUs.
#include "../../src/spectral/dirac_operator_gpu.cu"

int main()
{
    if (!has_gpu())
    {
        std::cout << "SKIP: no GPU available" << std::endl;
        return 0;
    }

    // Test: Dirac operator + Laplacian for a single-edge complex.
    //
    // Complex: 2 vertices (v0, v1), 1 edge (e01 = v0->v1).
    // max_dimension = 1, dimension_sizes = {2, 1}, total_dimension = 3.
    //
    // Orientation: d_1(e01) = v1 - v0.
    //
    // Dirac kernel convention: BOTH boundary and coboundary store the
    // LOWER-dimensional index as the CSR row.
    //   boundary_row_ptr:  rows = dim_k   elements (higher dim), cols = dim_{k-1}
    //   coboundary_row_ptr: rows = dim_k elements (lower dim),  cols = dim_{k+1}
    //
    // The kernel indexes row_ptr by local (offset-relative) row index.
    // With max_dim=1, only dim 0 accesses coboundary_row_ptr and only
    // dim 1 accesses boundary_row_ptr -- no cross-dimension collision.
    {
        const int max_dim = 1;
        const std::vector<int> dim_sizes = {2, 1};
        const int total_dim = 2 + 1;

        // boundary (d_1)
        // d_1(e01) = v1 - v0.
        // Stored with edges as rows (1 row): col(v0)=-1, col(v1)=+1.
        //
        // CRITICAL: diracOperatorKernel indexes row_ptr by per-dimension LOCAL
        // row index (row - dimension_offsets[row_dim]).  For dim 1 the single
        // edge has local_row = 0, so the boundary data MUST sit at indices
        // [0, 1] in the flat row_ptr array.
        std::vector<int> boundary_row_ptr = {0, 2, // dim 1: 1 edge,   2 nnz
                                             0, 0,
                                             0}; // dim 0: 2 vertices, 0 nnz (unused by kernel)
        std::vector<int> boundary_col_idx = {0, 1};
        std::vector<float> boundary_values = {-1.0f, 1.0f};

        // coboundary (d*_0)
        // d*_0 = transpose(d_1).
        // Stored with vertices as rows (2 rows): col(edge)=0 for both,
        // values {-1, +1} (= transpose of d_1: v0->edge, v1->edge)
        std::vector<int> coboundary_row_ptr = {0, 1, 2, // dim 0: 2 rows, 1 nnz each
                                               0, 0};   // dim 1: 1 row,  0 nnz
        std::vector<int> coboundary_col_idx = {0, 0};
        std::vector<float> coboundary_values = {-1.0f, 1.0f};

        std::cout << "PASS: built sparse CSR data for edge complex" << std::endl;

        // build Dirac + Laplacian
        nerve::spectral::gpu::GPUDiracOperator dirac(max_dim, dim_sizes);
        dirac.buildDiracOperator(boundary_row_ptr, boundary_col_idx, boundary_values,
                                 coboundary_row_ptr, coboundary_col_idx, coboundary_values);
        std::cout << "PASS: buildDiracOperator completed" << std::endl;

        dirac.computeLaplacian();
        std::cout << "PASS: computeLaplacian completed" << std::endl;

        std::vector<float> laplacian;
        dirac.getLaplacian(laplacian);

        assert(static_cast<int>(laplacian.size()) == total_dim * total_dim);

        // validate Laplacian
        const float eps = 1e-4f;

        // 1. All values must be finite
        for (float v : laplacian)
        {
            (void)v;
            assert(std::isfinite(v));
        }

        // 2. Symmetry: L^T == L
        for (int i = 0; i < total_dim; ++i)
        {
            for (int j = 0; j < total_dim; ++j)
            {
                float diff = laplacian[i * total_dim + j] - laplacian[j * total_dim + i];
                assert(std::fabs(diff) <= eps);
            }
        }
        std::cout << "PASS: Laplacian is symmetric" << std::endl;

        // 3. Block-diagonal (no cross-dimension coupling)
        // dim 0: rows 0..1, dim 1: row 2
        for (int r = 0; r < 2; ++r)
        {
            for (int c = 2; c < total_dim; ++c)
            {
                assert(std::fabs(laplacian[r * total_dim + c]) <= eps);
                assert(std::fabs(laplacian[c * total_dim + r]) <= eps);
            }
        }
        std::cout << "PASS: Laplacian is block-diagonal" << std::endl;

        // 4. dim-0 Laplacian = graph Laplacian of a single edge:
        //    L_0 = d_1 * d*_0 = [[ 1, -1],
        //                        [-1,  1]]
        float expected_l0[2][2] = {{1.0f, -1.0f}, {-1.0f, 1.0f}};
        for (int i = 0; i < 2; ++i)
        {
            for (int j = 0; j < 2; ++j)
            {
                float diff = std::fabs(laplacian[i * total_dim + j] - expected_l0[i][j]);
                assert(diff <= eps);
                (void)diff;
            }
        }
        std::cout << "PASS: dim-0 Laplacian matches graph Laplacian of an edge" << std::endl;

        // 5. dim-1 Laplacian = d*_0 * d_1 = [[2]]
        //    (single edge, degree = 2 incident vertices)
        float l1 = laplacian[2 * total_dim + 2];
        assert(std::fabs(l1 - 2.0f) <= eps);
        (void)l1;
        std::cout << "PASS: dim-1 Laplacian = " << l1 << std::endl;

        std::cout << "PASS: all dirac operator tests passed" << std::endl;
    }

    return 0;
}
