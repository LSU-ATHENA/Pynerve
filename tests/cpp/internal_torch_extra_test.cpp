#include "nerve/config.hpp"
#include "nerve/core_types.hpp"
#include "nerve/torch/detail/torch_detail.hpp"
#include "nerve/torch/ml_operations.hpp"
#include "nerve/torch/persistence_diagram.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

#ifdef NERVE_HAS_TORCH

#include <torch/torch.h>

namespace
{

bool check_autograd_operations_on_simple_tensors()
{
    at::Tensor filtration = at::tensor({0.0, 0.5, 1.0, 2.0}, at::kDouble);
    at::Tensor result = nerve::torch::ph_compute(filtration, 0);
    if (result.size(0) == 0)
    {
        std::cerr << "autograd persistence returned empty result\n";
        return false;
    }
    if (result.size(1) != 2)
    {
        std::cerr << "autograd persistence result should have 2 columns\n";
        return false;
    }
    auto accessor = result.accessor<double, 2>();
    for (int64_t i = 0; i < result.size(0); ++i)
    {
        if (!std::isfinite(accessor[i][0]) ||
            (!std::isfinite(accessor[i][1]) &&
             accessor[i][1] != std::numeric_limits<double>::infinity()))
        {
            std::cerr << "autograd persistence pair contains invalid values\n";
            return false;
        }
    }
    return true;
}

bool check_diagram_operations_births_deaths_lengths()
{
    at::Tensor diagram = at::tensor({{0.0, 1.0}, {0.5, 2.0}, {1.0, 3.0}}, at::kDouble);
    at::Tensor births = at::tensor({0.0, 0.5, 1.0}, at::kDouble);
    at::Tensor deaths = at::tensor({1.0, 2.0, 3.0}, at::kDouble);
    double d1 = nerve::torch::diagram_wasserstein(diagram, diagram, 2.0);
    if (!std::isfinite(d1) || std::abs(d1) > 1e-10)
    {
        std::cerr << "Wasserstein distance between identical diagrams should be zero\n";
        return false;
    }
    double d2 = nerve::torch::diagram_bottleneck(diagram, diagram);
    if (!std::isfinite(d2) || std::abs(d2) > 1e-10)
    {
        std::cerr << "Bottleneck distance between identical diagrams should be zero\n";
        return false;
    }
    at::Tensor landscape = nerve::torch::diagram_landscape(diagram, 5);
    if (landscape.size(0) == 0)
    {
        std::cerr << "diagram landscape is empty\n";
        return false;
    }
    at::Tensor betti = nerve::torch::diagram_betti(diagram, 0);
    if (betti.size(0) == 0)
    {
        std::cerr << "diagram Betti curve is empty\n";
        return false;
    }
    return true;
}

bool check_persistence_image_kernel_dimensions()
{
    at::Tensor diagram = at::tensor({{0.1, 0.8}, {0.3, 1.5}, {0.5, 2.0}}, at::kDouble);
    at::Tensor image =
        nerve::torch::ml_persistence_image(diagram, 10, 10, 0.5, 0.0, 1.0, 0.0, 2.0, "persistence");
    if (image.size(0) != 10 || image.size(1) != 10)
    {
        std::cerr << "persistence image has wrong dimensions: " << image.size(0) << "x"
                  << image.size(1) << "\n";
        return false;
    }
    if (!image.is_floating_point())
    {
        std::cerr << "persistence image should be floating point\n";
        return false;
    }
    auto result = image.sum().item<double>();
    if (!std::isfinite(result) || result < 0.0)
    {
        std::cerr << "persistence image values are invalid\n";
        return false;
    }
    return true;
}

bool check_statistics_mean_max_persistence_valid()
{
    at::Tensor diagram = at::tensor({{0.0, 2.0}, {0.5, 3.0}, {1.0, 5.0}}, at::kDouble);
    double mean = nerve::torch::ml_mean_persistence(diagram, 0);
    if (!std::isfinite(mean) || mean < 0.0)
    {
        std::cerr << "mean persistence is invalid\n";
        return false;
    }
    double max_val = nerve::torch::ml_max_persistence(diagram, 0);
    if (!std::isfinite(max_val) || max_val < 0.0)
    {
        std::cerr << "max persistence is invalid\n";
        return false;
    }
    if (max_val < mean)
    {
        std::cerr << "max persistence should be >= mean persistence\n";
        return false;
    }
    double entropy = nerve::torch::ml_persistence_entropy(diagram, 0);
    if (!std::isfinite(entropy) || entropy < 0.0)
    {
        std::cerr << "persistence entropy is invalid\n";
        return false;
    }
    double total = nerve::torch::ml_total_persistence(diagram, 0);
    if (!std::isfinite(total) || total < 0.0)
    {
        std::cerr << "total persistence is invalid\n";
        return false;
    }
    return true;
}

bool check_vectorization_output_dimension_matches_config()
{
    at::Tensor diagram = at::tensor({{0.0, 1.0}, {0.2, 1.5}, {0.4, 2.0}, {0.6, 2.5}}, at::kDouble);
    int64_t num_samples = 50;
    double sigma = 0.1;
    at::Tensor t_values = at::logspace(-2, 0, 5);
    at::Tensor hks = nerve::torch::ml_heat_kernel_signature(diagram, num_samples, sigma, t_values);
    if (hks.size(0) != 5 || hks.size(1) != 50)
    {
        std::cerr << "heat kernel signature wrong dimensions: " << hks.size(0) << "x" << hks.size(1)
                  << "\n";
        return false;
    }
    at::Tensor curve = nerve::torch::ml_birth_death_curve(diagram, 30, "count");
    if (curve.size(0) != 30)
    {
        std::cerr << "birth-death curve wrong dimension: " << curve.size(0) << "\n";
        return false;
    }
    return true;
}

} // namespace

int main()
{
    if (!check_autograd_operations_on_simple_tensors())
    {
        std::cerr << "FAIL: autograd operations on simple tensors\n";
        return 1;
    }
    if (!check_diagram_operations_births_deaths_lengths())
    {
        std::cerr << "FAIL: diagram operations births deaths lengths\n";
        return 1;
    }
    if (!check_persistence_image_kernel_dimensions())
    {
        std::cerr << "FAIL: persistence image kernel dimensions\n";
        return 1;
    }
    if (!check_statistics_mean_max_persistence_valid())
    {
        std::cerr << "FAIL: statistics mean max persistence valid\n";
        return 1;
    }
    if (!check_vectorization_output_dimension_matches_config())
    {
        std::cerr << "FAIL: vectorization output dimension matches config\n";
        return 1;
    }
    return 0;
}

#else

int main()
{
    return 0;
}

#endif
