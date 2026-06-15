#include "nerve/torch/filtration.hpp"

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
using nerve::torch::Filtration;

bool check_insertion()
{
    Filtration f;
    f.append({0}, 0.0);
    f.append({1}, 1.0);
    f.append({0, 1}, 2.0);
    if (f.num_simplices() != 3)
    {
        std::cerr << "expected 3 simplices, got " << f.num_simplices() << "\n";
        return false;
    }
    return true;
}

bool check_contains()
{
    Filtration f;
    f.append({0}, 0.0);
    f.append({1}, 1.0);
    f.append({0, 1}, 2.0);
    if (f.num_simplices() != 3)
    {
        std::cerr << "unexpected simplex count\n";
        return false;
    }
    Tensor dims = f.get_simplex_counts();
    if (dims.numel() == 0)
    {
        std::cerr << "simplex counts should not be empty\n";
        return false;
    }
    return true;
}

bool check_boundary_matrix_dimensions()
{
    Filtration f;
    f.append({0}, 0.0);
    f.append({1}, 1.0);
    f.append({2}, 2.0);
    f.append({0, 1}, 3.0);
    f.append({1, 2}, 4.0);
    f.sort_by_dimension_and_filtration();
    if (f.max_dimension() < 1)
    {
        std::cerr << "expected max dimension >= 1\n";
        return false;
    }
    if (f.num_simplices() != 5)
    {
        std::cerr << "expected 5 simplices\n";
        return false;
    }
    return true;
}

bool check_empty_filtration()
{
    Filtration f;
    if (f.num_simplices() != 0)
    {
        std::cerr << "empty filtration should have 0 simplices\n";
        return false;
    }
    if (f.values().numel() != 0)
    {
        std::cerr << "empty filtration should have empty values\n";
        return false;
    }
    return true;
}

bool check_single_simplex()
{
    Filtration f;
    f.append({42}, 1.5);
    if (f.num_simplices() != 1)
    {
        std::cerr << "expected 1 simplex\n";
        return false;
    }
    if (std::abs(f.get_value(0) - 1.5) > 1e-10)
    {
        std::cerr << "value mismatch\n";
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
    if (!check_insertion())
    {
        std::cerr << "FAIL: insertion\n";
        return 1;
    }
    if (!check_contains())
    {
        std::cerr << "FAIL: contains\n";
        return 1;
    }
    if (!check_boundary_matrix_dimensions())
    {
        std::cerr << "FAIL: boundary_matrix_dimensions\n";
        return 1;
    }
    if (!check_empty_filtration())
    {
        std::cerr << "FAIL: empty_filtration\n";
        return 1;
    }
    if (!check_single_simplex())
    {
        std::cerr << "FAIL: single_simplex\n";
        return 1;
    }
    return 0;
#endif
}
