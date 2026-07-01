#include "nerve/torch/boundary_matrix.hpp"
#if __has_include(<torch/torch.h>)
#include "nerve/torch/simplex_tree.hpp"

#include <cstddef>
#include <iostream>
#include <random>
#include <vector>

#ifdef NERVE_HAS_TORCH
#include <torch/torch.h>
#endif

namespace
{


#ifdef NERVE_HAS_TORCH

using at::Tensor;
using nerve::torch::BoundaryMatrix;

bool check_construction_default()
{
    BoundaryMatrix m;
    if (m.n_rows() != 0 || m.n_cols() != 0 || m.nnz() != 0)
    {
        std::cerr << "default boundary matrix should be empty\n";
        return false;
    }
    return true;
}

bool check_construction_with_tensors()
{
    Tensor indices = at::tensor({0, 1, 2}, at::kLong);
    Tensor indptr = at::tensor({0, 1, 3}, at::kLong);
    Tensor data = at::tensor({1.0, 1.0, 1.0});
    BoundaryMatrix m(indices, indptr, data, 3, 2);
    if (m.n_rows() != 3 || m.n_cols() != 2 || m.nnz() != 3)
    {
        std::cerr << "dimensions mismatch\n";
        return false;
    }
    return true;
}

bool check_matvec()
{
    Tensor indices = at::tensor({0, 1, 1, 2}, at::kLong);
    Tensor indptr = at::tensor({0, 2, 4}, at::kLong);
    Tensor data = at::tensor({1.0, 1.0, 1.0, 1.0});
    BoundaryMatrix m(indices, indptr, data, 3, 2);
    Tensor x = at::tensor({1.0, 2.0});
    Tensor y = m.matvec(x);
    if (y.numel() != 3)
    {
        std::cerr << "matvec output size mismatch\n";
        return false;
    }
    return true;
}

bool check_matvec_transpose()
{
    Tensor indices = at::tensor({0, 1, 1, 2}, at::kLong);
    Tensor indptr = at::tensor({0, 2, 4}, at::kLong);
    Tensor data = at::tensor({1.0, 1.0, 1.0, 1.0});
    BoundaryMatrix m(indices, indptr, data, 3, 2);
    Tensor x = at::tensor({1.0, 2.0, 3.0});
    Tensor y = m.matvec_transpose(x);
    if (y.numel() != 2)
    {
        std::cerr << "matvec_transpose output size mismatch\n";
        return false;
    }
    return true;
}

bool check_to_sparse_consistency()
{
    Tensor indices = at::tensor({0, 1, 1, 2}, at::kLong);
    Tensor indptr = at::tensor({0, 2, 4}, at::kLong);
    Tensor data = at::tensor({1.0, 1.0, 1.0, 1.0});
    BoundaryMatrix m(indices, indptr, data, 3, 2);
    Tensor dense = m.to_dense();
    Tensor sparse = m.to_torch_sparse();
    Tensor dense_from_sparse = sparse.to_dense();
    if (!at::allclose(dense, dense_from_sparse))
    {
        std::cerr << "to_sparse and to_dense inconsistency\n";
        return false;
    }
    return true;
}

bool check_dimensions_consistent()
{
    Tensor indices = at::tensor({0, 1, 1, 2}, at::kLong);
    Tensor indptr = at::tensor({0, 2, 4}, at::kLong);
    Tensor data = at::tensor({1.0, 1.0, 1.0, 1.0});
    BoundaryMatrix m(indices, indptr, data, 3, 2);
    if (m.indices().numel() != m.nnz())
    {
        std::cerr << "indices count mismatch\n";
        return false;
    }
    if (m.indptr().numel() != m.n_cols() + 1)
    {
        std::cerr << "indptr size mismatch\n";
        return false;
    }
    return true;
}

#endif

} // namespace

int main()
{
#ifndef NERVE_HAS_TORCH
    std::cerr << "SKIP: NERVE_HAS_TORCH not defined\n";
    return 0;
#else
    if (!check_construction_default())
    {
        std::cerr << "FAIL: construction_default\n";
        return 1;
    }
    if (!check_construction_with_tensors())
    {
        std::cerr << "FAIL: construction_with_tensors\n";
        return 1;
    }
    if (!check_matvec())
    {
        std::cerr << "FAIL: matvec\n";
        return 1;
    }
    if (!check_matvec_transpose())
    {
        std::cerr << "FAIL: matvec_transpose\n";
        return 1;
    }
    if (!check_to_sparse_consistency())
    {
        std::cerr << "FAIL: to_sparse_consistency\n";
        return 1;
    }
    if (!check_dimensions_consistent())
    {
        std::cerr << "FAIL: dimensions_consistent\n";
        return 1;
    }
    return 0;
#endif
}
#else
int main()
{
    return 0;
}
#endif
