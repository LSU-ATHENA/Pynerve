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
using nerve::torch::SimplexTree;

bool check_insert_contains_find()
{
    SimplexTree tree;
    tree.insert({0}, 0.0);
    tree.insert({1}, 1.0);
    tree.insert({0, 1}, 2.0);
    if (!tree.contains({0}))
    {
        std::cerr << "should contain vertex 0\n";
        return false;
    }
    if (!tree.contains({1}))
    {
        std::cerr << "should contain vertex 1\n";
        return false;
    }
    if (!tree.contains({0, 1}))
    {
        std::cerr << "should contain edge 0-1\n";
        return false;
    }
    if (tree.find({2}) != -1)
    {
        std::cerr << "should not find vertex 2\n";
        return false;
    }
    return true;
}

bool check_get_boundary_matrix_dimensions()
{
    SimplexTree tree;
    tree.insert({0}, 0.0);
    tree.insert({1}, 1.0);
    tree.insert({2}, 2.0);
    tree.insert({0, 1}, 3.0);
    tree.insert({1, 2}, 4.0);
    tree.insert({0, 1, 2}, 5.0);
    Tensor bd1 = tree.to_boundary_matrix(1);
    if (bd1.numel() == 0)
    {
        std::cerr << "boundary matrix for dim 1 should not be empty\n";
        return false;
    }
    Tensor bd2 = tree.to_boundary_matrix(2);
    if (bd2.numel() == 0)
    {
        std::cerr << "boundary matrix for dim 2 should not be empty\n";
        return false;
    }
    return true;
}

bool check_insert_batch()
{
    SimplexTree tree;
    Tensor simplices = at::tensor({{0}, {1}, {2}, {0, 1}, {1, 2}}, at::kLong);
    Tensor values = at::tensor({0.0, 1.0, 2.0, 3.0, 4.0});
    tree.insert_batch(simplices, values);
    if (tree.num_simplices() != 5)
    {
        std::cerr << "expected 5 simplices after batch insert\n";
        return false;
    }
    return true;
}

bool check_remove()
{
    SimplexTree tree;
    tree.insert({0}, 0.0);
    tree.insert({1}, 1.0);
    tree.insert({0, 1}, 2.0);
    tree.remove({0, 1});
    if (tree.contains({0, 1}))
    {
        std::cerr << "should not contain removed edge\n";
        return false;
    }
    if (!tree.contains({0}) || !tree.contains({1}))
    {
        std::cerr << "vertices should remain after edge removal\n";
        return false;
    }
    return true;
}

bool check_queries()
{
    SimplexTree tree;
    tree.insert({0}, 0.0);
    tree.insert({1}, 1.0);
    tree.insert({2}, 2.0);
    tree.insert({0, 1}, 3.0);
    tree.insert({0, 2}, 4.0);
    tree.insert({1, 2}, 5.0);
    tree.insert({0, 1, 2}, 6.0);
    auto dim0 = tree.get_simplices_by_dimension(0);
    if (dim0.size() != 3)
    {
        std::cerr << "expected 3 vertices\n";
        return false;
    }
    auto dim1 = tree.get_simplices_by_dimension(1);
    if (dim1.size() != 3)
    {
        std::cerr << "expected 3 edges\n";
        return false;
    }
    auto dim2 = tree.get_simplices_by_dimension(2);
    if (dim2.size() != 1)
    {
        std::cerr << "expected 1 triangle\n";
        return false;
    }
    auto cofaces = tree.get_cofaces({0});
    if (cofaces.size() < 3)
    {
        std::cerr << "expected cofaces of vertex 0\n";
        return false;
    }
    auto faces = tree.get_faces({0, 1, 2});
    if (faces.size() < 6)
    {
        std::cerr << "expected faces of triangle\n";
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
    if (!check_insert_contains_find())
    {
        std::cerr << "FAIL: insert_contains_find\n";
        return 1;
    }
    if (!check_get_boundary_matrix_dimensions())
    {
        std::cerr << "FAIL: get_boundary_matrix_dimensions\n";
        return 1;
    }
    if (!check_insert_batch())
    {
        std::cerr << "FAIL: insert_batch\n";
        return 1;
    }
    if (!check_remove())
    {
        std::cerr << "FAIL: remove\n";
        return 1;
    }
    if (!check_queries())
    {
        std::cerr << "FAIL: queries\n";
        return 1;
    }
    return 0;
#endif
}
