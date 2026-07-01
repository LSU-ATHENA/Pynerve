#include "nerve/torch/persistence_diagram.hpp"

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
using nerve::torch::PersistenceDiagram;

bool check_diagram_births_deaths()
{
    Tensor diagram = at::tensor({{0.0, 1.0}, {0.0, 2.0}, {1.0, 3.0}});
    Tensor dims = at::tensor({0, 0, 1}, at::kLong);
    Tensor birth_idx = at::tensor({0, 1, 2}, at::kLong);
    Tensor death_idx = at::tensor({1, 2, 3}, at::kLong);
    PersistenceDiagram pd(diagram, dims, birth_idx, death_idx);
    Tensor births = pd.births();
    Tensor deaths = pd.deaths();
    if (births.numel() != 3 || deaths.numel() != 3)
    {
        std::cerr << "expected 3 births and deaths\n";
        return false;
    }
    return true;
}

bool check_diagram_persistence_lengths()
{
    Tensor diagram = at::tensor({{0.0, 1.0}, {0.0, 2.0}, {1.0, 3.0}});
    Tensor dims = at::tensor({0, 0, 1}, at::kLong);
    Tensor birth_idx = at::tensor({0, 1, 2}, at::kLong);
    Tensor death_idx = at::tensor({1, 2, 3}, at::kLong);
    PersistenceDiagram pd(diagram, dims, birth_idx, death_idx);
    Tensor pl = pd.persistence_lengths();
    if (pl.numel() != 3)
    {
        std::cerr << "expected 3 persistence lengths\n";
        return false;
    }
    if ((pl < 0.0).any().item<bool>())
    {
        std::cerr << "persistence lengths should be non-negative\n";
        return false;
    }
    return true;
}

bool check_diagram_filter_by_dimension()
{
    Tensor diagram = at::tensor({{0.0, 1.0}, {0.0, 2.0}, {1.0, 3.0}});
    Tensor dims = at::tensor({0, 0, 1}, at::kLong);
    Tensor birth_idx = at::tensor({0, 1, 2}, at::kLong);
    Tensor death_idx = at::tensor({1, 2, 3}, at::kLong);
    PersistenceDiagram pd(diagram, dims, birth_idx, death_idx);
    PersistenceDiagram d0 = pd.filter_by_dimension(0);
    if (d0.num_pairs() != 2)
    {
        std::cerr << "expected 2 pairs in dim 0\n";
        return false;
    }
    PersistenceDiagram d1 = pd.filter_by_dimension(1);
    if (d1.num_pairs() != 1)
    {
        std::cerr << "expected 1 pair in dim 1\n";
        return false;
    }
    return true;
}

bool check_diagram_threshold()
{
    Tensor diagram = at::tensor({{0.0, 1.0}, {0.0, 0.5}, {0.0, 3.0}});
    Tensor dims = at::tensor({0, 0, 1}, at::kLong);
    Tensor birth_idx = at::tensor({0, 1, 2}, at::kLong);
    Tensor death_idx = at::tensor({1, 2, 3}, at::kLong);
    PersistenceDiagram pd(diagram, dims, birth_idx, death_idx);
    PersistenceDiagram filtered = pd.threshold(0.8);
    if (filtered.num_pairs() >= pd.num_pairs())
    {
        std::cerr << "threshold should reduce pair count\n";
        return false;
    }
    return true;
}

bool check_diagram_total_persistence()
{
    Tensor diagram = at::tensor({{0.0, 1.0}, {0.0, 2.0}});
    Tensor dims = at::tensor({0, 0}, at::kLong);
    Tensor birth_idx = at::tensor({0, 1}, at::kLong);
    Tensor death_idx = at::tensor({1, 2}, at::kLong);
    PersistenceDiagram pd(diagram, dims, birth_idx, death_idx);
    Tensor tp = pd.total_persistence();
    if (tp.numel() != 1)
    {
        std::cerr << "total persistence should be scalar\n";
        return false;
    }
    if (tp.item<double>() < 0.0)
    {
        std::cerr << "total persistence should be non-negative\n";
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
    if (!check_diagram_births_deaths())
    {
        std::cerr << "FAIL: diagram_births_deaths\n";
        return 1;
    }
    if (!check_diagram_persistence_lengths())
    {
        std::cerr << "FAIL: diagram_persistence_lengths\n";
        return 1;
    }
    if (!check_diagram_filter_by_dimension())
    {
        std::cerr << "FAIL: diagram_filter_by_dimension\n";
        return 1;
    }
    if (!check_diagram_threshold())
    {
        std::cerr << "FAIL: diagram_threshold\n";
        return 1;
    }
    if (!check_diagram_total_persistence())
    {
        std::cerr << "FAIL: diagram_total_persistence\n";
        return 1;
    }
    return 0;
#endif
}
