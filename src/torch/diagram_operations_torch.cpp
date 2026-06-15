// PyTorch-Native Diagram Operations
// Implements persistence diagram distances using at::Tensor

#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <queue>
#include <vector>

namespace nerve::torch
{

namespace detail
{

[[nodiscard]] double hungarian_algorithm(const at::Tensor &cost_matrix)
{
    const int64_t n = cost_matrix.size(0);
    const int64_t m = cost_matrix.size(1);

    if (n == 0 || m == 0)
        return 0.0;

    auto accessor = cost_matrix.accessor<double, 2>();
    const int64_t size = std::max(n, m);
    std::vector<std::vector<double>> cost(size, std::vector<double>(size, 0.0));

    for (int64_t i = 0; i < n; ++i)
    {
        for (int64_t j = 0; j < m; ++j)
        {
            cost[i][j] = accessor[i][j];
        }
    }

    std::vector<double> u(size + 1), v(size + 1);
    std::vector<int64_t> p(size + 1), way(size + 1);

    for (int64_t i = 1; i <= size; ++i)
    {
        p[0] = i;
        int64_t j0 = 0;
        std::vector<double> minv(size + 1, std::numeric_limits<double>::infinity());
        std::vector<bool> used(size + 1, false);

        do
        {
            used[j0] = true;
            int64_t i0 = p[j0];
            double delta = std::numeric_limits<double>::infinity();
            int64_t j1 = 0;

            for (int64_t j = 1; j <= size; ++j)
            {
                if (!used[j])
                {
                    double cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
                    if (cur < minv[j])
                    {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if (minv[j] < delta)
                    {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }

            for (int64_t j = 0; j <= size; ++j)
            {
                if (used[j])
                {
                    u[p[j]] += delta;
                    v[j] -= delta;
                }
                else
                {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
        } while (p[j0] != 0);

        do
        {
            int64_t j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0);
    }

    return -v[0];
}

[[nodiscard]] double point_distance(double b1, double d1, double b2, double d2, double p)
{
    if (p == 1.0)
    {
        return std::abs(b1 - b2) + std::abs(d1 - d2);
    }
    else if (p == 2.0)
    {
        return std::sqrt(std::pow(b1 - b2, 2) + std::pow(d1 - d2, 2));
    }
    else
    {
        return std::pow(std::pow(std::abs(b1 - b2), p) + std::pow(std::abs(d1 - d2), p), 1.0 / p);
    }
}

[[nodiscard]] double diagonal_distance(double b, double d, double p)
{
    if (p == 1.0 || p == 2.0)
    {
        return std::abs(d - b) / std::sqrt(2.0);
    }
    else
    {
        return std::abs(d - b) / std::pow(2.0, 1.0 / p);
    }
}

[[nodiscard]] at::Tensor validate_diagram_cpu(const at::Tensor &diagram, const char *name)
{
    TORCH_CHECK(diagram.defined(), name, " diagram must be defined");
    TORCH_CHECK(diagram.dim() == 2 && diagram.size(1) >= 2, name,
                " diagram must be a 2D tensor with at least birth/death columns");
    TORCH_CHECK(diagram.is_floating_point(), name, " diagram must be floating point");

    at::Tensor diagram_cpu = diagram.cpu().to(at::kDouble).contiguous();
    auto accessor = diagram_cpu.accessor<double, 2>();
    for (int64_t i = 0; i < diagram_cpu.size(0); ++i)
    {
        const double birth = accessor[i][0];
        const double death = accessor[i][1];
        TORCH_CHECK(std::isfinite(birth), name, " diagram birth values must be finite");
        TORCH_CHECK(std::isfinite(death) || death == std::numeric_limits<double>::infinity(), name,
                    " diagram death values must be finite or +inf");
        TORCH_CHECK(death == std::numeric_limits<double>::infinity() || death >= birth, name,
                    " diagram death values must be >= birth values");
        if (diagram_cpu.size(1) > 2)
        {
            const double dim = accessor[i][2];
            TORCH_CHECK(std::isfinite(dim) && dim >= 0.0 && std::floor(dim) == dim, name,
                        " diagram dimension values must be non-negative integers");
        }
    }
    return diagram_cpu;
}

[[nodiscard]] at::Tensor finite_diagram_cpu(const at::Tensor &diagram_cpu)
{
    auto deaths = diagram_cpu.select(1, 1);
    auto finite_mask = at::isfinite(deaths);
    if (!finite_mask.any().item<bool>())
    {
        return at::empty({0, diagram_cpu.size(1)}, diagram_cpu.options());
    }
    auto finite_indices = at::nonzero(finite_mask).squeeze(-1);
    return diagram_cpu.index_select(0, finite_indices).contiguous();
}

} // namespace detail

[[nodiscard]] bool is_valid_diagram(const at::Tensor &diagram)
{
    return diagram.defined() && diagram.dim() == 2 && diagram.size(1) >= 2 &&
           diagram.is_floating_point();
}

// Diagram Operations (PyTorch-style naming)

double diagram_wasserstein(const at::Tensor &input, const at::Tensor &other, double p)
{
    TORCH_CHECK(std::isfinite(p) && p >= 1.0, "p must be finite and >= 1");

    at::Tensor input_cpu = detail::validate_diagram_cpu(input, "input");
    at::Tensor other_cpu = detail::validate_diagram_cpu(other, "other");
    at::Tensor d1_cpu = detail::finite_diagram_cpu(input_cpu);
    at::Tensor d2_cpu = detail::finite_diagram_cpu(other_cpu);
    const int64_t n1 = d1_cpu.size(0);
    const int64_t n2 = d2_cpu.size(0);

    if (n1 == 0 && n2 == 0)
        return 0.0;

    auto accessor1 = d1_cpu.accessor<double, 2>();
    auto accessor2 = d2_cpu.accessor<double, 2>();

    const int64_t size = n1 + n2;
    at::Tensor cost_matrix = at::full({size, size}, std::numeric_limits<double>::infinity(),
                                      at::TensorOptions().dtype(at::kDouble));
    auto cost_accessor = cost_matrix.accessor<double, 2>();

    // Point-to-point costs
    for (int64_t i = 0; i < n1; ++i)
    {
        for (int64_t j = 0; j < n2; ++j)
        {
            double dist = detail::point_distance(accessor1[i][0], accessor1[i][1], accessor2[j][0],
                                                 accessor2[j][1], p);
            cost_accessor[i][j] = std::pow(dist, p);
        }
    }

    // Point-to-diagonal costs
    for (int64_t i = 0; i < n1; ++i)
    {
        double diag_dist = detail::diagonal_distance(accessor1[i][0], accessor1[i][1], p);
        for (int64_t j = n2; j < size; ++j)
        {
            if (j - n2 == i)
            {
                cost_accessor[i][j] = std::pow(diag_dist, p);
            }
        }
    }

    for (int64_t i = n1; i < size; ++i)
    {
        for (int64_t j = 0; j < n2; ++j)
        {
            if (i - n1 == j)
            {
                double diag_dist = detail::diagonal_distance(accessor2[j][0], accessor2[j][1], p);
                cost_accessor[i][j] = std::pow(diag_dist, p);
            }
        }
    }

    // Diagonal-to-diagonal (zero cost)
    for (int64_t i = n1; i < size; ++i)
    {
        for (int64_t j = n2; j < size; ++j)
        {
            if (i - n1 == j - n2)
            {
                cost_accessor[i][j] = 0.0;
            }
        }
    }

    double total_cost = detail::hungarian_algorithm(cost_matrix);
    return std::pow(total_cost, 1.0 / p);
}

// Bottleneck Distance (PyTorch-style naming)

double diagram_bottleneck(const at::Tensor &input, const at::Tensor &other)
{
    at::Tensor input_cpu = detail::validate_diagram_cpu(input, "input");
    at::Tensor other_cpu = detail::validate_diagram_cpu(other, "other");
    at::Tensor d1_cpu = detail::finite_diagram_cpu(input_cpu);
    at::Tensor d2_cpu = detail::finite_diagram_cpu(other_cpu);
    const int64_t n1 = d1_cpu.size(0);
    const int64_t n2 = d2_cpu.size(0);

    if (n1 == 0 && n2 == 0)
        return 0.0;

    auto accessor1 = d1_cpu.accessor<double, 2>();
    auto accessor2 = d2_cpu.accessor<double, 2>();

    std::vector<double> candidates;
    candidates.reserve(n1 * n2 + n1 + n2);

    // Point-to-point distances (L_infinity)
    for (int64_t i = 0; i < n1; ++i)
    {
        for (int64_t j = 0; j < n2; ++j)
        {
            double d = std::max(std::abs(accessor1[i][0] - accessor2[j][0]),
                                std::abs(accessor1[i][1] - accessor2[j][1]));
            candidates.push_back(d);
        }
    }

    // Point-to-diagonal distances
    for (int64_t i = 0; i < n1; ++i)
    {
        double d = std::abs(accessor1[i][1] - accessor1[i][0]) / 2.0;
        candidates.push_back(d);
    }
    for (int64_t j = 0; j < n2; ++j)
    {
        double d = std::abs(accessor2[j][1] - accessor2[j][0]) / 2.0;
        candidates.push_back(d);
    }

    std::sort(candidates.begin(), candidates.end());

    // Binary search for bottleneck
    double low = 0.0;
    double high = candidates.empty() ? 0.0 : candidates.back();
    double best = high;

    const int max_iterations = 50;
    for (int iter = 0; iter < max_iterations; ++iter)
    {
        double mid = (low + high) / 2.0;

        std::vector<bool> matched2(n2, false);
        int64_t match_count = 0;

        for (int64_t i = 0; i < n1; ++i)
        {
            bool matched = false;

            for (int64_t j = 0; j < n2 && !matched; ++j)
            {
                if (matched2[j])
                    continue;

                double d = std::max(std::abs(accessor1[i][0] - accessor2[j][0]),
                                    std::abs(accessor1[i][1] - accessor2[j][1]));

                if (d <= mid)
                {
                    matched2[j] = true;
                    matched = true;
                    match_count++;
                }
            }

            if (!matched)
            {
                double d_diag = std::abs(accessor1[i][1] - accessor1[i][0]) / 2.0;
                if (d_diag <= mid)
                {
                    matched = true;
                }
            }

            if (!matched)
            {
                match_count = -1;
                break;
            }
        }

        bool valid = (match_count >= 0);
        if (valid)
        {
            for (int64_t j = 0; j < n2; ++j)
            {
                if (!matched2[j])
                {
                    double d_diag = std::abs(accessor2[j][1] - accessor2[j][0]) / 2.0;
                    if (d_diag > mid)
                    {
                        valid = false;
                        break;
                    }
                }
            }
        }

        if (valid)
        {
            best = mid;
            high = mid;
        }
        else
        {
            low = mid;
        }

        if (high - low < 1e-9)
            break;
    }

    return best;
}

// Persistence Landscape (PyTorch-style naming)

at::Tensor diagram_landscape(const at::Tensor &input, int64_t num_samples)
{
    TORCH_CHECK(num_samples > 0, "num_samples must be positive");

    at::Tensor input_cpu = detail::validate_diagram_cpu(input, "input");
    at::Tensor finite_cpu = detail::finite_diagram_cpu(input_cpu);
    const int64_t n = finite_cpu.size(0);

    if (n == 0)
    {
        return at::zeros({num_samples}, input.options());
    }

    at::Tensor births = finite_cpu.select(1, 0);
    at::Tensor deaths = finite_cpu.select(1, 1);

    double min_val = at::min(births).item<double>();
    double max_val = at::max(deaths).item<double>();

    at::Tensor x = at::linspace(min_val, max_val, num_samples, input.options());
    at::Tensor landscape = at::zeros({num_samples}, input.options());

    for (int64_t i = 0; i < n; ++i)
    {
        double b = births[i].item<double>();
        double d = deaths[i].item<double>();
        double m = (b + d) / 2.0;
        double h = (d - b) / 2.0;

        if (h <= 0)
            continue;

        at::Tensor triangle = at::clamp(h - at::abs(x - m), 0);
        landscape = at::max(landscape, triangle);
    }

    return landscape;
}

// Betti Number Extraction

at::Tensor diagram_betti(const at::Tensor &input, int64_t dim)
{
    TORCH_CHECK(dim >= 0, "dim must be non-negative");

    at::Tensor input_cpu = detail::validate_diagram_cpu(input, "input");
    const int64_t n = input_cpu.size(0);
    auto accessor = input_cpu.accessor<double, 2>();
    const bool has_dimensions = input_cpu.size(1) > 2;

    int64_t count = 0;
    for (int64_t i = 0; i < n; ++i)
    {
        if (has_dimensions && static_cast<int64_t>(accessor[i][2]) != dim)
        {
            continue;
        }
        double death = accessor[i][1];
        if (death == std::numeric_limits<double>::infinity())
        {
            count++;
        }
    }

    return at::tensor({count}, input.options().dtype(at::kLong));
}

} // namespace nerve::torch
