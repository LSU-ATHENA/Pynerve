// ML Statistics Operations for Persistence Diagrams
// C++20 implementation of statistical feature extraction

#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace nerve::torch
{

namespace
{

at::Tensor validate_stat_diagram_cpu(const at::Tensor &diagram)
{
    if (!diagram.defined())
    {
        throw std::invalid_argument("diagram must be defined");
    }
    if (diagram.dim() != 2 || diagram.size(1) < 2)
    {
        throw std::invalid_argument("diagram must be a 2D tensor with at least two columns");
    }
    if (!diagram.is_floating_point())
    {
        throw std::invalid_argument("diagram must have a floating-point dtype");
    }

    at::Tensor diagram_cpu = diagram.cpu().to(at::kDouble).contiguous();
    if (diagram_cpu.size(0) == 0)
    {
        return diagram_cpu;
    }

    const auto births = diagram_cpu.select(1, 0);
    const auto deaths = diagram_cpu.select(1, 1);
    if (!at::isfinite(births).all().item<bool>())
    {
        throw std::invalid_argument("diagram births must be finite");
    }
    if (at::isnan(deaths).any().item<bool>())
    {
        throw std::invalid_argument("diagram deaths must not be NaN");
    }
    const auto finite_death_mask = at::isfinite(deaths);
    const auto positive_inf_death_mask = deaths == std::numeric_limits<double>::infinity();
    if (!(finite_death_mask | positive_inf_death_mask).all().item<bool>())
    {
        throw std::invalid_argument("diagram deaths must be finite or positive infinity");
    }
    if (finite_death_mask.any().item<bool>())
    {
        const auto finite_deaths = deaths.masked_select(finite_death_mask);
        const auto finite_births = births.masked_select(finite_death_mask);
        if (!(finite_deaths >= finite_births).all().item<bool>())
        {
            throw std::invalid_argument(
                "diagram finite deaths must be greater than or equal to births");
        }
    }
    if (diagram_cpu.size(1) >= 3)
    {
        const auto dims = diagram_cpu.select(1, 2);
        if (!at::isfinite(dims).all().item<bool>())
        {
            throw std::invalid_argument("diagram dimensions must be finite");
        }
        if (!(dims >= 0.0).all().item<bool>())
        {
            throw std::invalid_argument("diagram dimensions must be non-negative");
        }
        if (!(dims == at::floor(dims)).all().item<bool>())
        {
            throw std::invalid_argument("diagram dimensions must be integers");
        }
    }
    return diagram_cpu;
}

void validate_dim_filter(int64_t dim)
{
    if (dim < -1)
    {
        throw std::invalid_argument("dim must be -1 or non-negative");
    }
}

void validate_nonnegative_dim(int64_t dim)
{
    if (dim < 0)
    {
        throw std::invalid_argument("dims must contain only non-negative values");
    }
}

double validate_positive_finite(double value, const char *name)
{
    if (!std::isfinite(value) || value <= 0.0)
    {
        throw std::invalid_argument(std::string(name) + " must be finite and positive");
    }
    return value;
}

double validate_nonnegative_finite(double value, const char *name)
{
    if (!std::isfinite(value) || value < 0.0)
    {
        throw std::invalid_argument(std::string(name) + " must be finite and non-negative");
    }
    return value;
}

void validate_entropy_base(double base)
{
    validate_positive_finite(base, "base");
    if (base == 1.0)
    {
        throw std::invalid_argument("base must not be 1");
    }
}

} // namespace

std::pair<at::Tensor, at::Tensor> compute_persistence_values(const at::Tensor &diagram,
                                                             int64_t dim_filter = -1)
{
    validate_dim_filter(dim_filter);
    const auto diagram_cpu = validate_stat_diagram_cpu(diagram);
    auto births = diagram_cpu.select(1, 0);
    auto deaths = diagram_cpu.select(1, 1);

    at::Tensor dims;
    if (diagram_cpu.size(1) >= 3)
    {
        dims = diagram_cpu.select(1, 2);
    }

    auto finite_mask = at::isfinite(deaths);

    // Filter by dimension if specified
    if (dim_filter >= 0 && diagram_cpu.size(1) >= 3)
    {
        auto dim_mask = (dims == static_cast<double>(dim_filter));
        finite_mask = finite_mask & dim_mask;
    }

    if (!finite_mask.any().item<bool>())
    {
        return {at::empty({0}, births.options()), at::empty({0}, deaths.options())};
    }

    auto indices = at::nonzero(finite_mask).squeeze(-1);
    auto b = births.index_select(0, indices);
    auto d = deaths.index_select(0, indices);

    auto persistence = d - b;
    persistence = at::clamp(persistence, 0.0);

    return {persistence, finite_mask};
}

// Total Persistence

double ml_total_persistence(const at::Tensor &diagram, int64_t dim, double p)
{
    validate_positive_finite(p, "p");
    auto [persistence, _] = compute_persistence_values(diagram, dim);

    if (persistence.numel() == 0)
    {
        return 0.0;
    }

    if (p == 1.0)
    {
        return persistence.sum().item<double>();
    }
    else
    {
        return at::pow(persistence, p).sum().item<double>();
    }
}

// Mean Persistence

double ml_mean_persistence(const at::Tensor &diagram, int64_t dim)
{
    auto [persistence, _] = compute_persistence_values(diagram, dim);

    if (persistence.numel() == 0)
    {
        return 0.0;
    }

    return persistence.mean().item<double>();
}

// Max Persistence

double ml_max_persistence(const at::Tensor &diagram, int64_t dim)
{
    auto [persistence, _] = compute_persistence_values(diagram, dim);

    if (persistence.numel() == 0)
    {
        return 0.0;
    }

    return persistence.max().item<double>();
}

// Variance of Persistence

double ml_persistence_variance(const at::Tensor &diagram, int64_t dim)
{
    auto [persistence, _] = compute_persistence_values(diagram, dim);

    if (persistence.numel() < 2)
    {
        return 0.0;
    }

    return persistence.var().item<double>();
}

// Persistence Entropy

double ml_persistence_entropy(const at::Tensor &diagram, int64_t dim, double base)
{
    validate_entropy_base(base);
    auto [persistence, _] = compute_persistence_values(diagram, dim);

    if (persistence.numel() == 0)
    {
        return 0.0;
    }

    double total = persistence.sum().item<double>();
    if (total < 1e-8)
    {
        return 0.0;
    }

    // Normalize to probabilities
    auto probs = persistence / total;

    // Compute entropy: -sum(p * log(p))
    auto log_probs = at::log(probs) / std::log(base);
    auto entropy = -(probs * log_probs).sum();

    // Handle NaN from log(0)
    if (std::isnan(entropy.item<double>()))
    {
        // Recompute without zeros
        auto nonzero_mask = probs > 0;
        if (!nonzero_mask.any().item<bool>())
        {
            return 0.0;
        }
        auto nonzero_probs = probs.index_select(0, at::nonzero(nonzero_mask).squeeze(-1));
        auto nonzero_log = at::log(nonzero_probs) / std::log(base);
        entropy = -(nonzero_probs * nonzero_log).sum();
    }

    return entropy.item<double>();
}

// Number of Features

int64_t ml_number_of_features(const at::Tensor &diagram, int64_t dim, double min_persistence)
{
    validate_dim_filter(dim);
    validate_nonnegative_finite(min_persistence, "min_persistence");
    const auto diagram_cpu = validate_stat_diagram_cpu(diagram);
    auto births = diagram_cpu.select(1, 0);
    auto deaths = diagram_cpu.select(1, 1);

    at::Tensor mask = at::ones({diagram_cpu.size(0)}, at::dtype(at::kBool));

    // Filter by dimension
    if (dim >= 0 && diagram_cpu.size(1) >= 3)
    {
        auto dims = diagram_cpu.select(1, 2);
        mask = mask & (dims == static_cast<double>(dim));
    }

    // Filter by persistence
    auto persistence = deaths - births;
    auto finite_mask = at::isfinite(deaths);
    persistence = at::where(finite_mask, persistence,
                            at::full_like(persistence, std::numeric_limits<double>::infinity()));
    mask = mask & (persistence >= min_persistence);

    return mask.sum().item<int64_t>();
}

// Betti Curve

at::Tensor ml_betti_curve(const at::Tensor &diagram, int64_t num_samples, int64_t dim)
{
    if (num_samples <= 0)
    {
        throw std::invalid_argument("num_samples must be positive");
    }
    auto [persistence, _] = compute_persistence_values(diagram, dim);

    if (persistence.numel() == 0)
    {
        return at::zeros({num_samples}, at::dtype(at::kLong).device(diagram.device()));
    }

    double p_max = persistence.max().item<double>();
    auto thresholds = at::linspace(0, p_max, num_samples, persistence.options());

    std::vector<int64_t> betti_nums;
    auto thresholds_a = thresholds.accessor<double, 1>();
    auto pers_a = persistence.accessor<double, 1>();

    for (int64_t i = 0; i < num_samples; ++i)
    {
        double t = thresholds_a[i];
        int64_t count = 0;
        for (int64_t j = 0; j < persistence.size(0); ++j)
        {
            if (pers_a[j] > t)
            {
                ++count;
            }
        }
        betti_nums.push_back(count);
    }

    return at::tensor(betti_nums, at::dtype(at::kLong).device(diagram.device()));
}

// Amplitude

double ml_amplitude(const at::Tensor &diagram, const std::string &metric, double p, int64_t dim)
{
    validate_positive_finite(p, "p");
    auto [persistence, _] = compute_persistence_values(diagram, dim);

    if (persistence.numel() == 0)
    {
        return 0.0;
    }

    if (metric == "bottleneck")
    {
        return persistence.max().item<double>();
    }
    else if (metric == "persistence")
    {
        if (p == 1.0)
        {
            return persistence.sum().item<double>();
        }
        else
        {
            return at::pow(persistence, p).sum().item<double>();
        }
    }
    else if (metric == "wasserstein")
    {
        return persistence.sum().item<double>();
    }

    throw std::invalid_argument("unsupported amplitude metric: " + metric);
}

// All Statistics

std::vector<std::pair<std::string, double>> ml_all_statistics(const at::Tensor &diagram,
                                                              const std::vector<int64_t> &dims)
{
    std::vector<std::pair<std::string, double>> results;

    // Overall statistics (no dimension filter)
    results.push_back(
        {"num_features_total", static_cast<double>(ml_number_of_features(diagram, -1, 0.0))});
    results.push_back({"total_persistence_total", ml_total_persistence(diagram, -1, 1.0)});
    results.push_back({"mean_persistence_total", ml_mean_persistence(diagram, -1)});
    results.push_back({"max_persistence_total", ml_max_persistence(diagram, -1)});
    results.push_back(
        {"persistence_entropy_total", ml_persistence_entropy(diagram, -1, std::exp(1))});

    // Per-dimension statistics
    for (int64_t dim : dims)
    {
        validate_nonnegative_dim(dim);
        std::string dim_str = std::to_string(dim);

        results.push_back({"num_features_dim" + dim_str,
                           static_cast<double>(ml_number_of_features(diagram, dim, 0.0))});
        results.push_back(
            {"total_persistence_dim" + dim_str, ml_total_persistence(diagram, dim, 1.0)});
        results.push_back({"mean_persistence_dim" + dim_str, ml_mean_persistence(diagram, dim)});
        results.push_back({"max_persistence_dim" + dim_str, ml_max_persistence(diagram, dim)});
        results.push_back({"persistence_entropy_dim" + dim_str,
                           ml_persistence_entropy(diagram, dim, std::exp(1))});
    }

    return results;
}

// Feature Vector

at::Tensor ml_extract_features(const at::Tensor &diagram, const std::vector<int64_t> &dims)
{
    auto stats = ml_all_statistics(diagram, dims);

    std::vector<double> values;
    values.reserve(stats.size());
    for (const auto &[name, value] : stats)
    {
        values.push_back(value);
    }

    return at::tensor(values, at::dtype(at::kFloat).device(diagram.device()));
}

} // namespace nerve::torch
