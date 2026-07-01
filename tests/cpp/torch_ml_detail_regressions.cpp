#include "nerve/torch/ml_operations.hpp"

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

bool check_ml_persistence_image()
{
    Tensor diagram = at::tensor({{0.0, 1.0}, {0.0, 2.0}, {0.0, 3.0}});
    Tensor img = nerve::torch::ml_persistence_image(diagram, 5, 5);
    if (img.numel() != 25)
    {
        std::cerr << "expected 25 elements in 5x5 image\n";
        return false;
    }
    if ((img < 0.0).any().item<bool>())
    {
        std::cerr << "image values should be non-negative\n";
        return false;
    }
    return true;
}

bool check_ml_persistence_landscape()
{
    Tensor diagram = at::tensor({{0.0, 1.0}, {0.0, 2.0}, {0.0, 3.0}});
    Tensor landscape = nerve::torch::ml_persistence_landscape(diagram, 3, 20);
    if (landscape.numel() != 60)
    {
        std::cerr << "expected 60 elements in 3x20 landscape\n";
        return false;
    }
    return true;
}

bool check_ml_statistics()
{
    Tensor diagram = at::tensor({{0.0, 1.0}, {0.0, 2.0}, {1.0, 3.0}});
    double tp = nerve::torch::ml_total_persistence(diagram);
    if (tp < 0.0)
    {
        std::cerr << "total persistence should be non-negative\n";
        return false;
    }
    double mp = nerve::torch::ml_mean_persistence(diagram);
    if (mp < 0.0)
    {
        std::cerr << "mean persistence should be non-negative\n";
        return false;
    }
    double mx = nerve::torch::ml_max_persistence(diagram);
    if (mx < 0.0)
    {
        std::cerr << "max persistence should be non-negative\n";
        return false;
    }
    int64_t nf = nerve::torch::ml_number_of_features(diagram);
    if (nf <= 0)
    {
        std::cerr << "should have at least one feature\n";
        return false;
    }
    return true;
}

bool check_ml_vectorization()
{
    Tensor diagram = at::tensor({{0.0, 1.0}, {0.0, 2.0}});
    Tensor features = nerve::torch::ml_extract_features(diagram, {0});
    if (features.numel() == 0)
    {
        std::cerr << "features should not be empty\n";
        return false;
    }
    auto all_stats = nerve::torch::ml_all_statistics(diagram, {0});
    if (all_stats.empty())
    {
        std::cerr << "statistics should not be empty\n";
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
    if (!check_ml_statistics())
    {
        std::cerr << "FAIL: ml_statistics\n";
        return 1;
    }
    if (!check_ml_vectorization())
    {
        std::cerr << "FAIL: ml_vectorization\n";
        return 1;
    }
    return 0;
#endif
}
