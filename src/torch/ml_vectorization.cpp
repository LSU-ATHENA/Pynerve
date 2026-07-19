// ML vectorization operations for persistence diagrams.

#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

namespace nerve::torch
{

template <typename T>
constexpr T clamp(T value, T min_val, T max_val)
{
    return std::max(min_val, std::min(value, max_val));
}

bool is_supported_weight_fn(const std::string &weight_fn)
{
    return weight_fn == "constant" || weight_fn == "linear" || weight_fn == "persistence";
}

void validate_sample_range(double x_min, double x_max)
{
    TORCH_CHECK(std::isfinite(x_min) && std::isfinite(x_max), "sample range bounds must be finite");
    TORCH_CHECK(x_min <= x_max, "sample range minimum must not exceed maximum");
}

at::Tensor validate_diagram_cpu(const at::Tensor &diagram)
{
    TORCH_CHECK(diagram.dim() == 2, "Expected 2D diagram tensor [N, 2]");
    TORCH_CHECK(diagram.size(1) >= 2, "Diagram must have at least birth and death columns");
    TORCH_CHECK(diagram.is_floating_point(), "Diagram must use a floating-point dtype");

    at::Tensor work_diagram = diagram.to(at::kCPU).to(at::kDouble).contiguous();
    if (work_diagram.size(0) == 0)
    {
        return work_diagram;
    }

    auto all_births = work_diagram.select(1, 0);
    auto all_deaths = work_diagram.select(1, 1);
    TORCH_CHECK(at::isfinite(all_births).all().item<bool>(), "Diagram births must be finite");
    TORCH_CHECK(!at::isnan(all_deaths).any().item<bool>(), "Diagram deaths must not be NaN");
    auto finite_death_mask = at::isfinite(all_deaths);
    auto positive_inf_death_mask = all_deaths == std::numeric_limits<double>::infinity();
    TORCH_CHECK((finite_death_mask | positive_inf_death_mask).all().item<bool>(),
                "Diagram deaths must be finite or positive infinity");
    if (finite_death_mask.any().item<bool>())
    {
        auto finite_births = all_births.masked_select(finite_death_mask);
        auto finite_deaths = all_deaths.masked_select(finite_death_mask);
        TORCH_CHECK((finite_deaths >= finite_births).all().item<bool>(),
                    "Diagram finite deaths must be greater than or equal to births");
    }

    return work_diagram;
}

std::pair<at::Tensor, at::Tensor> filter_finite_deaths(const at::Tensor &diagram)
{
    auto births = diagram.select(1, 0);
    auto deaths = diagram.select(1, 1);

    auto finite_mask = at::isfinite(deaths);

    if (!finite_mask.any().item<bool>())
    {
        return {at::empty({0}, births.options()), at::empty({0}, deaths.options())};
    }

    auto finite_indices = at::nonzero(finite_mask).squeeze(-1);
    return {births.index_select(0, finite_indices), deaths.index_select(0, finite_indices)};
}

at::Tensor persistence_image_impl(const at::Tensor &diagram, int64_t resolution_birth,
                                  int64_t resolution_death, double sigma, double birth_min,
                                  double birth_max, double death_min, double death_max,
                                  const std::string &weight_fn)
{
    TORCH_CHECK(resolution_birth > 0 && resolution_death > 0,
                "Persistence image resolution must be positive");
    TORCH_CHECK(std::isfinite(sigma) && sigma > 0.0,
                "Persistence image sigma must be finite and positive");
    TORCH_CHECK(std::isfinite(birth_min) && std::isfinite(birth_max) && std::isfinite(death_min) &&
                    std::isfinite(death_max),
                "Persistence image bounds must be finite");
    TORCH_CHECK(is_supported_weight_fn(weight_fn),
                "Unsupported persistence image weight function: ", weight_fn);

    const auto output_device = diagram.device();
    const auto output_dtype = diagram.scalar_type();
    at::Tensor work_diagram = validate_diagram_cpu(diagram);
    if (diagram.size(0) == 0)
    {
        return at::zeros({resolution_death, resolution_birth}, diagram.options());
    }

    // Filter infinite deaths
    auto [births, deaths] = filter_finite_deaths(work_diagram);

    if (births.size(0) == 0)
    {
        return at::zeros({resolution_death, resolution_birth}, work_diagram.options())
            .to(output_device, output_dtype);
    }

    // Auto-compute ranges if not provided
    double b_min = birth_min;
    double b_max = birth_max;
    double d_min = death_min;
    double d_max = death_max;

    if (b_min == b_max)
    {
        b_min = births.min().item<double>();
        b_max = births.max().item<double>();
        double b_pad = (b_max - b_min) * 0.1;
        if (b_pad < 1e-8)
            b_pad = 1.0;
        b_min -= b_pad;
        b_max += b_pad;
    }

    if (d_min == d_max)
    {
        d_min = deaths.min().item<double>();
        d_max = deaths.max().item<double>();
        double d_pad = (d_max - d_min) * 0.1;
        if (d_pad < 1e-8)
            d_pad = 1.0;
        d_min -= d_pad;
        d_max += d_pad;
    }

    // Create output image
    auto image = at::zeros({resolution_death, resolution_birth}, work_diagram.options());

    // Create coordinate grids
    auto x = at::linspace(b_min, b_max, resolution_birth, work_diagram.options());
    auto y = at::linspace(d_min, d_max, resolution_death, work_diagram.options());

    // Compute weights
    at::Tensor weights;
    if (weight_fn == "persistence" || weight_fn == "linear")
    {
        weights = deaths - births;
        weights = at::clamp(weights, 0.0);
    }
    else
    {
        weights = at::ones_like(births);
    }

    // Accumulate Gaussians
    auto births_a = births.accessor<double, 1>();
    auto deaths_a = deaths.accessor<double, 1>();
    auto weights_a = weights.accessor<double, 1>();
    auto image_a = image.accessor<double, 2>();

    double neg_inv_two_sigma_sq = -1.0 / (2.0 * sigma * sigma);

    // Pre-allocate buffers for batched exp (avoids per-iteration tensor creation)
    auto gauss_x = at::empty({resolution_birth}, work_diagram.options());
    auto gauss_y = at::empty({resolution_death}, work_diagram.options());

    for (int64_t i = 0; i < births.size(0); ++i)
    {
        double b = births_a[i];
        double d = deaths_a[i];
        double w = weights_a[i];

        // In-place: gauss_x = exp(-(x - b)^2 / (2*sigma^2))
        gauss_x.copy_(x);
        gauss_x -= b;
        gauss_x *= gauss_x;
        gauss_x *= neg_inv_two_sigma_sq;
        gauss_x.exp_();

        // In-place: gauss_y = exp(-(y - d)^2 / (2*sigma^2))
        gauss_y.copy_(y);
        gauss_y -= d;
        gauss_y *= gauss_y;
        gauss_y *= neg_inv_two_sigma_sq;
        gauss_y.exp_();

        auto gx_a = gauss_x.accessor<double, 1>();
        auto gy_a = gauss_y.accessor<double, 1>();

        for (int64_t yi = 0; yi < resolution_death; ++yi)
        {
            double gy = gy_a[yi];
            for (int64_t xi = 0; xi < resolution_birth; ++xi)
            {
                image_a[yi][xi] += w * gy * gx_a[xi];
            }
        }
    }

    // Flip y-axis (so birth-death line is diagonal)
    image = image.flip(0);

    // Normalize
    double sum = image.sum().item<double>();
    if (sum > 0)
    {
        image = image / sum;
    }

    return image.to(output_device, output_dtype);
}

// Persistence Landscape

at::Tensor persistence_landscape_impl(const at::Tensor &diagram, int64_t k, int64_t num_samples,
                                      double x_min, double x_max)
{
    TORCH_CHECK(k > 0, "landscape depth k must be positive");
    TORCH_CHECK(num_samples > 0, "num_samples must be positive");
    validate_sample_range(x_min, x_max);
    const auto output_device = diagram.device();
    const auto output_dtype = diagram.scalar_type();
    at::Tensor work_diagram = validate_diagram_cpu(diagram);

    if (work_diagram.size(0) == 0)
    {
        return at::zeros({k, num_samples}, diagram.options());
    }

    // Filter infinite deaths
    auto [births, deaths] = filter_finite_deaths(work_diagram);

    if (births.size(0) == 0)
    {
        return at::zeros({k, num_samples}, work_diagram.options()).to(output_device, output_dtype);
    }

    // Compute midpoints and heights
    auto midpoints = (births + deaths) / 2.0;
    auto heights = (deaths - births) / 2.0;

    // Determine range
    double xm = x_min;
    double xM = x_max;
    if (xm == xM)
    {
        xm = midpoints.min().item<double>();
        xM = midpoints.max().item<double>();
        double pad = (xM - xm) * 0.1;
        if (pad < 1e-8)
            pad = 1.0;
        xm -= pad;
        xM += pad;
    }

    // Create x samples
    auto x = at::linspace(xm, xM, num_samples, work_diagram.options());

    // Initialize landscape
    auto landscape = at::zeros({k, num_samples}, work_diagram.options());

    // Compute tent functions and take k-max
    auto mid_a = midpoints.accessor<double, 1>();
    auto heights_a = heights.accessor<double, 1>();
    auto x_a = x.accessor<double, 1>();
    auto landscape_a = landscape.accessor<double, 2>();

    // For each tent function
    for (int64_t i = 0; i < midpoints.size(0); ++i)
    {
        double m = mid_a[i];
        double h = heights_a[i];

        if (h <= 0)
            continue;

        // Compute tent values
        for (int64_t j = 0; j < num_samples; ++j)
        {
            double val = std::max(h - std::abs(x_a[j] - m), 0.0);

            // Insert into k-largest
            for (int64_t l = 0; l < k; ++l)
            {
                if (val > landscape_a[l][j])
                {
                    // Shift down
                    for (int64_t m_idx = k - 1; m_idx > l; --m_idx)
                    {
                        landscape_a[m_idx][j] = landscape_a[m_idx - 1][j];
                    }
                    landscape_a[l][j] = val;
                    break;
                }
            }
        }
    }

    return landscape.to(output_device, output_dtype);
}

// Persistence Silhouette

at::Tensor persistence_silhouette_impl(const at::Tensor &diagram, int64_t num_samples, double x_min,
                                       double x_max, const std::string &weight_fn)
{
    TORCH_CHECK(num_samples > 0, "num_samples must be positive");
    validate_sample_range(x_min, x_max);
    TORCH_CHECK(is_supported_weight_fn(weight_fn),
                "Unsupported persistence silhouette weight function: ", weight_fn);
    const auto output_device = diagram.device();
    const auto output_dtype = diagram.scalar_type();
    at::Tensor work_diagram = validate_diagram_cpu(diagram);

    if (work_diagram.size(0) == 0)
    {
        return at::zeros({num_samples}, diagram.options());
    }

    auto [births, deaths] = filter_finite_deaths(work_diagram);

    if (births.size(0) == 0)
    {
        return at::zeros({num_samples}, work_diagram.options()).to(output_device, output_dtype);
    }

    auto midpoints = (births + deaths) / 2.0;
    auto heights = (deaths - births) / 2.0;

    // Weights
    at::Tensor weights;
    if (weight_fn == "persistence" || weight_fn == "linear")
    {
        weights = heights * 2.0; // persistence = 2 * height
    }
    else
    {
        weights = at::ones_like(midpoints);
    }

    // Range
    double xm = x_min;
    double xM = x_max;
    if (xm == xM)
    {
        xm = midpoints.min().item<double>();
        xM = midpoints.max().item<double>();
        double pad = (xM - xm) * 0.1;
        if (pad < 1e-8)
            pad = 1.0;
        xm -= pad;
        xM += pad;
    }

    auto x = at::linspace(xm, xM, num_samples, work_diagram.options());

    // Compute weighted average of tents
    auto silhouette = at::zeros({num_samples}, work_diagram.options());

    auto mid_a = midpoints.accessor<double, 1>();
    auto heights_a = heights.accessor<double, 1>();
    auto weights_a = weights.accessor<double, 1>();
    auto x_a = x.accessor<double, 1>();
    auto silhouette_a = silhouette.accessor<double, 1>();

    double total_weight = 0.0;

    for (int64_t i = 0; i < midpoints.size(0); ++i)
    {
        double m = mid_a[i];
        double h = heights_a[i];
        double w = weights_a[i];

        if (h <= 0 || w <= 0)
            continue;

        total_weight += w;

        for (int64_t j = 0; j < num_samples; ++j)
        {
            double tent_val = std::max(h - std::abs(x_a[j] - m), 0.0);
            silhouette_a[j] += tent_val * w;
        }
    }

    if (total_weight > 0)
    {
        for (int64_t j = 0; j < num_samples; ++j)
        {
            silhouette_a[j] /= total_weight;
        }
    }

    return silhouette.to(output_device, output_dtype);
}

at::Tensor heat_kernel_signature_impl(const at::Tensor &diagram, int64_t num_samples, double sigma,
                                      const at::Tensor &t_values);
at::Tensor birth_death_curve_impl(const at::Tensor &diagram, int64_t num_bins,
                                  const std::string &statistic);

#include "detail/ml_vectorization_public_api.inl"

} // namespace nerve::torch
