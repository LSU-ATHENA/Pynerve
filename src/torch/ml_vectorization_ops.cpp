// ML vectorization operations  --  heat kernel and birth-death curve implementations.

#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace nerve::torch
{

// Forward declarations of helpers defined in ml_vectorization.cpp
at::Tensor validate_diagram_cpu(const at::Tensor &diagram);
std::pair<at::Tensor, at::Tensor> filter_finite_deaths(const at::Tensor &diagram);

at::Tensor heat_kernel_signature_impl(const at::Tensor &diagram, int64_t num_samples, double sigma,
                                      const at::Tensor &t_values)
{
    TORCH_CHECK(num_samples > 0, "num_samples must be positive");
    TORCH_CHECK(std::isfinite(sigma) && sigma > 0.0,
                "heat kernel sigma must be finite and positive");
    TORCH_CHECK(t_values.dim() == 1, "t_values must be a rank-1 tensor");
    TORCH_CHECK(t_values.is_floating_point(), "t_values must use a floating-point dtype");
    at::Tensor work_t_values = t_values.to(at::kCPU).to(at::kDouble).contiguous();
    TORCH_CHECK(work_t_values.size(0) > 0, "t_values must not be empty");
    TORCH_CHECK(at::isfinite(work_t_values).all().item<bool>(), "t_values must be finite");
    TORCH_CHECK((work_t_values > 0.0).all().item<bool>(), "t_values must be positive");
    const auto output_device = diagram.device();
    const auto output_dtype = diagram.scalar_type();
    at::Tensor work_diagram = validate_diagram_cpu(diagram);

    auto [births, deaths] = filter_finite_deaths(work_diagram);

    if (births.size(0) == 0)
    {
        int64_t num_t = work_t_values.size(0);
        return at::zeros({num_t, num_samples}, work_diagram.options())
            .to(output_device, output_dtype);
    }

    auto persistence = deaths - births;
    persistence = at::clamp(persistence, 0.0);

    double p_min = persistence.min().item<double>();
    double p_max = persistence.max().item<double>();

    if (std::abs(p_max - p_min) < 1e-8)
    {
        p_max = p_min + 1.0;
    }

    auto x = at::linspace(p_min, p_max, num_samples, work_diagram.options());
    int64_t num_t = work_t_values.size(0);

    auto signature = at::zeros({num_t, num_samples}, work_diagram.options());

    auto pers_a = persistence.accessor<double, 1>();
    auto x_a = x.accessor<double, 1>();
    auto t_a = work_t_values.accessor<double, 1>();
    auto sig_a = signature.accessor<double, 2>();

    for (int64_t ti = 0; ti < num_t; ++ti)
    {
        double t = t_a[ti];
        double effective_sigma = sigma * std::sqrt(t);

        for (int64_t i = 0; i < persistence.size(0); ++i)
        {
            double p = pers_a[i];

            for (int64_t j = 0; j < num_samples; ++j)
            {
                double diff = x_a[j] - p;
                double g = std::exp(-(diff * diff) / (2.0 * effective_sigma * effective_sigma));
                sig_a[ti][j] += g;
            }
        }
    }

    return signature.to(output_device, output_dtype);
}

at::Tensor birth_death_curve_impl(const at::Tensor &diagram, int64_t num_bins,
                                  const std::string &statistic)
{
    TORCH_CHECK(num_bins > 0, "num_bins must be positive");
    TORCH_CHECK(statistic == "count", "Unsupported birth-death curve statistic: ", statistic);
    const auto output_device = diagram.device();
    const auto output_dtype = diagram.scalar_type();
    at::Tensor work_diagram = validate_diagram_cpu(diagram);

    if (work_diagram.size(0) == 0)
    {
        return at::zeros({num_bins}, diagram.options());
    }

    auto [births, deaths] = filter_finite_deaths(work_diagram);

    if (births.size(0) == 0)
    {
        return at::zeros({num_bins}, work_diagram.options()).to(output_device, output_dtype);
    }

    double b_min = births.min().item<double>();
    double b_max = births.max().item<double>();

    if (std::abs(b_max - b_min) < 1e-8)
    {
        return at::zeros({num_bins}, work_diagram.options()).to(output_device, output_dtype);
    }

    auto hist = at::histc(births, num_bins, b_min, b_max);

    return hist.to(output_device, output_dtype);
}

} // namespace nerve::torch
