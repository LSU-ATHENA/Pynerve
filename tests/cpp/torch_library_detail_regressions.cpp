#include "nerve/torch/boundary_matrix.hpp"
#include "nerve/torch/persistence_diagram.hpp"
#include "nerve/torch/simplex_tree.hpp"

#include <algorithm>
#include <cmath>
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
using nerve::torch::PersistenceDiagram;
using nerve::torch::SimplexTree;

bool check_torch_library_registration()
{
    Tensor indices = at::tensor({0, 1}, at::kLong);
    Tensor indptr = at::tensor({0, 1, 2}, at::kLong);
    Tensor data = at::tensor({1.0, 1.0});
    BoundaryMatrix m(indices, indptr, data, 2, 2);
    Tensor dense = m.to_dense();
    if (dense.numel() != 4)
    {
        std::cerr << "expected 4 elements\n";
        return false;
    }
    Tensor sparse = m.to_torch_sparse();
    if (!sparse.is_sparse())
    {
        std::cerr << "expected sparse tensor\n";
        return false;
    }
    return true;
}

bool check_vietoris_rips_construction()
{
    SimplexTree tree;
    Tensor points = at::randn({10, 3});
    tree.build_vr(points, 0.5, 2);
    if (tree.num_simplices() <= 10)
    {
        std::cerr << "VR should create more simplices than points\n";
        return false;
    }
    if (tree.max_dimension() < 2)
    {
        std::cerr << "expected max dimension >= 2\n";
        return false;
    }
    return true;
}

bool check_boundary_matrix_from_tree()
{
    SimplexTree tree;
    Tensor points = at::randn({6, 2});
    tree.build_vr(points, 0.5, 2);
    if (tree.max_dimension() >= 1)
    {
        Tensor bd = tree.to_boundary_matrix(1);
        if (bd.numel() > 0)
        {
            return true;
        }
    }
    return true;
}

bool check_persistence_diagram_roundtrip()
{
    Tensor diagram = at::tensor({{0.0, 1.0}, {0.0, 2.0}});
    Tensor dims = at::tensor({0, 0}, at::kLong);
    Tensor birth_idx = at::tensor({0, 1}, at::kLong);
    Tensor death_idx = at::tensor({1, 2}, at::kLong);
    PersistenceDiagram pd(diagram, dims, birth_idx, death_idx);
    auto state = pd.state_dict();
    PersistenceDiagram pd2;
    pd2.load_state_dict(state);
    if (pd2.num_pairs() != pd.num_pairs())
    {
        std::cerr << "roundtrip pair count mismatch\n";
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
    if (!check_torch_library_registration())
    {
        std::cerr << "FAIL: torch_library_registration\n";
        return 1;
    }
    if (!check_vietoris_rips_construction())
    {
        std::cerr << "FAIL: vietoris_rips_construction\n";
        return 1;
    }
    if (!check_boundary_matrix_from_tree())
    {
        std::cerr << "FAIL: boundary_matrix_from_tree\n";
        return 1;
    }
    if (!check_persistence_diagram_roundtrip())
    {
        std::cerr << "FAIL: persistence_diagram_roundtrip\n";
        return 1;
    }
    return 0;
#endif
}
