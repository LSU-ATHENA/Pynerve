#include "nerve/common/accelerated_types.hpp"
#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/core_types.hpp"
#include "nerve/torch/ml_operations.hpp"
#include "nerve/torch/persistence_diagram.hpp"
#include "test_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <random>
#include <tuple>
#include <vector>

#ifdef NERVE_HAS_TORCH
#include <torch/torch.h>
#endif

namespace
{


#ifdef NERVE_HAS_TORCH

using at::Tensor;
using nerve::torch::PersistenceDiagram;
using namespace nerve::test;

bool check_ml_total_persistence()
{
    Tensor diagram = at::tensor({{0.0, 1.0}, {0.0, 2.0}, {0.0, 3.0}});
    auto tp = nerve::torch::ml_total_persistence(diagram);
    if (tp < 0.0)
    {
        std::cerr << "total persistence should be non-negative\n";
        return false;
    }
    return true;
}

bool check_ml_total_persistence_with_dim()
{
    Tensor diagram = at::tensor({{0.0, 1.0}, {0.0, 2.0}, {1.0, 3.0}});
    Tensor dims = at::tensor({0, 0, 1});

    auto tp = nerve::torch::ml_total_persistence(diagram, 0);
    if (tp < 0.0)
    {
        std::cerr << "total persistence for dim 0 should be non-negative\n";
        return false;
    }
    return true;
}

bool check_ml_persistence_image()
{
    Tensor diagram = at::tensor({{0.0, 1.0}, {0.0, 2.0}});
    auto img = nerve::torch::ml_persistence_image(diagram, 5, 5);
    if (img.numel() != 25)
    {
        std::cerr << "expected 25 elements in 5x5 image, got " << img.numel() << "\n";
        return false;
    }
    return true;
}

bool check_ml_persistence_landscape()
{
    Tensor diagram = at::tensor({{0.0, 1.0}, {0.0, 2.0}, {0.0, 3.0}});
    auto landscape = nerve::torch::ml_persistence_landscape(diagram, 3, 20);
    if (landscape.numel() <= 0)
    {
        std::cerr << "landscape should have elements\n";
        return false;
    }
    return true;
}

bool check_ml_mean_persistence()
{
    Tensor diagram = at::tensor({{0.0, 2.0}, {0.0, 4.0}});
    auto mean = nerve::torch::ml_mean_persistence(diagram);
    if (mean < 0.0)
    {
        std::cerr << "mean persistence should be non-negative\n";
        return false;
    }
    return true;
}

bool check_ml_max_persistence()
{
    Tensor diagram = at::tensor({{0.0, 1.0}, {0.0, 5.0}, {0.0, 3.0}});
    auto mx = nerve::torch::ml_max_persistence(diagram);
    if (mx < 0.0)
    {
        std::cerr << "max persistence should be non-negative\n";
        return false;
    }
    return true;
}

bool check_persistence_diagram_construction()
{
    PersistenceDiagram pd;
    (void)pd;
    return true;
}

bool check_persistence_diagram_from_tensors()
{
    Tensor diagram = at::zeros({0, 2});
    Tensor dims = at::zeros({0}, at::kLong);
    Tensor birth = at::zeros({0}, at::kLong);
    Tensor death = at::zeros({0}, at::kLong);

    PersistenceDiagram pd(diagram, dims, birth, death);
    auto d = pd.diagram();
    if (d.size(0) != 0)
    {
        std::cerr << "expected empty diagram\n";
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
    if (!check_ml_total_persistence())
    {
        std::cerr << "FAIL: ml_total_persistence\n";
        return 1;
    }
    if (!check_ml_total_persistence_with_dim())
    {
        std::cerr << "FAIL: ml_total_persistence with dim\n";
        return 1;
    }
    if (!check_ml_persistence_image())
    {
        std::cerr << "FAIL: ml_persistence_image\n";
        return 1;
    }
    if (!check_ml_persistence_landscape())
    {
        std::cerr << "FAIL: ml_persistence_landscape\n";
        return 1;
    }
    if (!check_ml_mean_persistence())
    {
        std::cerr << "FAIL: ml_mean_persistence\n";
        return 1;
    }
    if (!check_ml_max_persistence())
    {
        std::cerr << "FAIL: ml_max_persistence\n";
        return 1;
    }
    if (!check_persistence_diagram_construction())
    {
        std::cerr << "FAIL: PersistenceDiagram construction\n";
        return 1;
    }
    if (!check_persistence_diagram_from_tensors())
    {
        std::cerr << "FAIL: PersistenceDiagram from tensors\n";
        return 1;
    }
    return 0;
#endif
}
