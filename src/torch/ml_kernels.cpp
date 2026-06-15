// ML Kernel Operations for Persistence Diagrams
// C++20 implementation of kernel methods

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

at::Tensor compute_distance_matrix(const at::Tensor &coords1, const at::Tensor &coords2, double p)
{
    // coords1: [N, 2], coords2: [M, 2]
    auto b1 = coords1.select(1, 0);
    auto d1 = coords1.select(1, 1);
    auto b2 = coords2.select(1, 0);
    auto d2 = coords2.select(1, 1);

    if (p == 1.0)
    {
        // L1 distance
        auto dist_b = at::abs(b1.unsqueeze(1) - b2.unsqueeze(0));
        auto dist_d = at::abs(d1.unsqueeze(1) - d2.unsqueeze(0));
        return dist_b + dist_d;
    }
    else if (p == 2.0)
    {
        // L2 distance
        auto dist_b = at::pow(b1.unsqueeze(1) - b2.unsqueeze(0), 2);
        auto dist_d = at::pow(d1.unsqueeze(1) - d2.unsqueeze(0), 2);
        return at::sqrt(dist_b + dist_d);
    }
    else
    {
        // General Lp
        auto dist_b = at::pow(at::abs(b1.unsqueeze(1) - b2.unsqueeze(0)), p);
        auto dist_d = at::pow(at::abs(d1.unsqueeze(1) - d2.unsqueeze(0)), p);
        return at::pow(dist_b + dist_d, 1.0 / p);
    }
}

void validate_positive_finite(double value, const char *name)
{
    if (value <= 0.0 || !std::isfinite(value))
    {
        throw std::invalid_argument(std::string(name) + " must be finite and positive");
    }
}

at::Tensor validate_kernel_diagram_cpu(const at::Tensor &diagram)
{
    if (diagram.dim() != 2)
    {
        throw std::invalid_argument("diagram must be a 2D tensor");
    }
    if (diagram.size(1) < 2)
    {
        throw std::invalid_argument("diagram must have at least birth and death columns");
    }
    if (!diagram.is_floating_point())
    {
        throw std::invalid_argument("diagram must use a floating-point dtype");
    }

    at::Tensor work = diagram.to(at::kCPU).to(at::kDouble).contiguous();
    if (work.size(0) == 0)
    {
        return work;
    }

    auto births = work.select(1, 0);
    auto deaths = work.select(1, 1);
    if (!at::isfinite(births).all().item<bool>())
    {
        throw std::invalid_argument("diagram births must be finite");
    }
    if (at::isnan(deaths).any().item<bool>())
    {
        throw std::invalid_argument("diagram deaths must not be NaN");
    }
    auto finite_death_mask = at::isfinite(deaths);
    auto positive_inf_death_mask = deaths == std::numeric_limits<double>::infinity();
    if (!(finite_death_mask | positive_inf_death_mask).all().item<bool>())
    {
        throw std::invalid_argument("diagram deaths must be finite or positive infinity");
    }
    if (finite_death_mask.any().item<bool>())
    {
        auto finite_births = births.masked_select(finite_death_mask);
        auto finite_deaths = deaths.masked_select(finite_death_mask);
        if (!(finite_deaths >= finite_births).all().item<bool>())
        {
            throw std::invalid_argument(
                "diagram finite deaths must be greater than or equal to births");
        }
    }
    return work;
}

at::Tensor validate_kernel_matrix(const at::Tensor &K)
{
    if (K.dim() != 2 || K.size(0) != K.size(1))
    {
        throw std::invalid_argument("kernel matrix must be square");
    }
    if (K.size(0) == 0)
    {
        throw std::invalid_argument("kernel matrix must be non-empty");
    }
    if (!K.is_floating_point())
    {
        throw std::invalid_argument("kernel matrix must use a floating-point dtype");
    }
    auto work = K.to(at::kCPU).to(at::kDouble).contiguous();
    if (!at::isfinite(work).all().item<bool>())
    {
        throw std::invalid_argument("kernel matrix must contain only finite values");
    }
    return work;
}

at::Tensor filter_diagram_finite(const at::Tensor &diagram)
{
    auto work = validate_kernel_diagram_cpu(diagram);
    auto deaths = work.select(1, 1);
    auto finite_mask = at::isfinite(deaths);

    if (!finite_mask.any().item<bool>())
    {
        return at::empty({0, 2}, work.options());
    }

    auto indices = at::nonzero(finite_mask).squeeze(-1);
    return work.index_select(0, indices).narrow(1, 0, 2); // Get only birth/death
}

// Gaussian Kernel

double ml_gaussian_kernel(const at::Tensor &d1, const at::Tensor &d2, double sigma,
                          const std::string &distance_metric)
{
    validate_positive_finite(sigma, "sigma");

    auto coords1 = filter_diagram_finite(d1);
    auto coords2 = filter_diagram_finite(d2);

    if (coords1.size(0) == 0 || coords2.size(0) == 0)
    {
        return 0.0;
    }

    if (distance_metric == "euclidean" || distance_metric == "l2")
    {
        // Sum of Gaussian similarities between all pairs
        auto dist_matrix = compute_distance_matrix(coords1, coords2, 2.0);
        auto similarities = at::exp(-dist_matrix.pow(2) / (2.0 * sigma * sigma));
        return similarities.sum().item<double>();
    }
    else if (distance_metric == "manhattan" || distance_metric == "l1")
    {
        auto dist_matrix = compute_distance_matrix(coords1, coords2, 1.0);
        auto similarities = at::exp(-dist_matrix.pow(2) / (2.0 * sigma * sigma));
        return similarities.sum().item<double>();
    }
    else if (distance_metric == "chebyshev" || distance_metric == "linf")
    {
        auto db = at::abs(coords1.select(1, 0).unsqueeze(1) - coords2.select(1, 0).unsqueeze(0));
        auto dd = at::abs(coords1.select(1, 1).unsqueeze(1) - coords2.select(1, 1).unsqueeze(0));
        auto dist_matrix = at::maximum(db, dd);
        auto similarities = at::exp(-dist_matrix.pow(2) / (2.0 * sigma * sigma));
        return similarities.sum().item<double>();
    }
    throw std::invalid_argument("unsupported Gaussian kernel distance metric: " + distance_metric);
}

// Persistence Scale-Space Kernel

double ml_persistence_scale_space_kernel(const at::Tensor &d1, const at::Tensor &d2, double sigma,
                                         double weight)
{
    validate_positive_finite(sigma, "sigma");
    if (!std::isfinite(weight) || weight < 0.0 || weight > 1.0)
    {
        throw std::invalid_argument("scale-space kernel weight must be finite and in [0, 1]");
    }
    auto coords1 = filter_diagram_finite(d1);
    auto coords2 = filter_diagram_finite(d2);

    if (coords1.size(0) == 0 || coords2.size(0) == 0)
    {
        return 0.0;
    }

    // Compute distances in birth-death space
    auto dist_matrix = compute_distance_matrix(coords1, coords2, 2.0);

    // Persistence as additional dimension
    auto pers1 = (coords1.select(1, 1) - coords1.select(1, 0)).unsqueeze(1);
    auto pers2 = (coords2.select(1, 1) - coords2.select(1, 0)).unsqueeze(0);
    auto pers_diff = at::pow(pers1 - pers2, 2);

    auto combined_dist = (1.0 - weight) * at::pow(dist_matrix, 2) + weight * pers_diff;

    auto similarities = at::exp(-combined_dist / (2.0 * sigma * sigma));
    return similarities.sum().item<double>();
}

// Sliced Wasserstein Kernel

double ml_sliced_wasserstein_kernel(const at::Tensor &d1, const at::Tensor &d2, int64_t num_slices,
                                    double sigma)
{
    validate_positive_finite(sigma, "sigma");
    if (num_slices <= 0)
    {
        throw std::invalid_argument("num_slices must be positive");
    }

    auto coords1 = filter_diagram_finite(d1);
    auto coords2 = filter_diagram_finite(d2);

    if (coords1.size(0) == 0 || coords2.size(0) == 0)
    {
        return 0.0;
    }

    double total_kernel = 0.0;
    const double slice_count = static_cast<double>(num_slices);

    for (int64_t i = 0; i < num_slices; ++i)
    {
        double angle = M_PI * static_cast<double>(i) / slice_count;
        double cos_a = std::cos(angle);
        double sin_a = std::sin(angle);

        // Projection direction [cos_a, sin_a]
        // Project points: coords @ direction
        auto proj1 = coords1.select(1, 0) * cos_a + coords1.select(1, 1) * sin_a;
        auto proj2 = coords2.select(1, 0) * cos_a + coords2.select(1, 1) * sin_a;

        // Sort projections
        proj1 = std::get<0>(at::sort(proj1));
        proj2 = std::get<0>(at::sort(proj2));

        // Pad to same size
        int64_t n1 = proj1.size(0);
        int64_t n2 = proj2.size(0);

        if (n1 < n2)
        {
            proj1 = at::cat({proj1, at::zeros(n2 - n1, proj1.options())});
        }
        else if (n2 < n1)
        {
            proj2 = at::cat({proj2, at::zeros(n1 - n2, proj2.options())});
        }

        // Compute 1D Wasserstein distance
        double w_dist = at::abs(proj1 - proj2).mean().item<double>();

        // Kernel for this slice
        total_kernel += std::exp(-w_dist * w_dist / (2.0 * sigma * sigma));
    }

    return total_kernel / slice_count;
}

// Persistence Fisher Kernel

double ml_persistence_fisher_kernel(const at::Tensor &d1, const at::Tensor &d2, double sigma)
{
    validate_positive_finite(sigma, "sigma");
    auto coords1 = filter_diagram_finite(d1);
    auto coords2 = filter_diagram_finite(d2);

    if (coords1.size(0) == 0 || coords2.size(0) == 0)
    {
        return 0.0;
    }

    // Persistence as weights
    auto pers1 = coords1.select(1, 1) - coords1.select(1, 0);
    auto pers2 = coords2.select(1, 1) - coords2.select(1, 0);

    pers1 = at::clamp(pers1, 0.0);
    pers2 = at::clamp(pers2, 0.0);

    // Normalize
    double sum1 = pers1.sum().item<double>();
    double sum2 = pers2.sum().item<double>();

    if (sum1 < 1e-8 || sum2 < 1e-8)
    {
        return 0.0;
    }

    auto weights1 = pers1 / sum1;
    auto weights2 = pers2 / sum2;

    // Compute Fisher-like kernel
    auto dist_matrix = compute_distance_matrix(coords1, coords2, 2.0);
    auto spatial_kernel = at::exp(-at::pow(dist_matrix, 2) / (2.0 * sigma * sigma));

    // Weight by persistence
    auto weighted_kernel = spatial_kernel * weights1.unsqueeze(1) * weights2.unsqueeze(0);

    return weighted_kernel.sum().item<double>();
}

// Linear Kernel (via vectorization)

// Forward declarations from ml_vectorization.cpp
at::Tensor ml_persistence_silhouette(const at::Tensor &diagram, int64_t num_samples, double x_min,
                                     double x_max, const std::string &weight_fn);

double ml_linear_kernel(const at::Tensor &d1, const at::Tensor &d2, int64_t num_samples)
{
    if (num_samples <= 0)
    {
        throw std::invalid_argument("num_samples must be positive");
    }
    // Use silhouette for vectorization
    auto v1 = ml_persistence_silhouette(d1, num_samples, 0.0, 0.0, "persistence");
    auto v2 = ml_persistence_silhouette(d2, num_samples, 0.0, 0.0, "persistence");

    // Dot product
    return at::dot(v1, v2).item<double>();
}

// Kernel Matrix Computation

at::Tensor ml_compute_kernel_matrix(const std::vector<at::Tensor> &diagrams,
                                    const std::string &kernel, double sigma, int64_t num_slices)
{
    if (diagrams.size() > static_cast<size_t>(std::numeric_limits<int64_t>::max()))
    {
        throw std::length_error("diagram count exceeds int64_t range");
    }
    if (kernel != "gaussian" && kernel != "pss" && kernel != "sliced_wasserstein" &&
        kernel != "fisher" && kernel != "linear")
    {
        throw std::invalid_argument("unsupported persistence diagram kernel: " + kernel);
    }
    int64_t n = static_cast<int64_t>(diagrams.size());
    auto K = at::zeros({n, n}, at::dtype(at::kDouble));

    for (int64_t i = 0; i < n; ++i)
    {
        for (int64_t j = i; j < n; ++j)
        {
            double k_val = 0.0;

            if (kernel == "gaussian")
            {
                k_val = ml_gaussian_kernel(diagrams[i], diagrams[j], sigma, "euclidean");
            }
            else if (kernel == "pss")
            {
                k_val = ml_persistence_scale_space_kernel(diagrams[i], diagrams[j], sigma, 0.5);
            }
            else if (kernel == "sliced_wasserstein")
            {
                k_val = ml_sliced_wasserstein_kernel(diagrams[i], diagrams[j], num_slices, sigma);
            }
            else if (kernel == "fisher")
            {
                k_val = ml_persistence_fisher_kernel(diagrams[i], diagrams[j], sigma);
            }
            else if (kernel == "linear")
            {
                k_val = ml_linear_kernel(diagrams[i], diagrams[j], 100);
            }

            K[i][j] = k_val;
            if (i != j)
            {
                K[j][i] = k_val;
            }
        }
    }

    return K;
}

// Normalize Kernel Matrix

at::Tensor ml_normalize_kernel_matrix(const at::Tensor &K)
{
    auto work = validate_kernel_matrix(K);
    auto raw_diag = at::diag(work);
    if (!(raw_diag > 0.0).all().item<bool>())
    {
        throw std::invalid_argument("kernel matrix diagonal must be positive");
    }
    auto diag = at::sqrt(raw_diag);
    auto K_norm = work / (diag.unsqueeze(0) * diag.unsqueeze(1));

    // Clamp to handle numerical issues
    K_norm = at::clamp(K_norm, -1.0, 1.0);

    return K_norm.to(K.device()).to(K.scalar_type());
}

// Center Kernel Matrix

at::Tensor ml_center_kernel_matrix(const at::Tensor &K)
{
    auto work = validate_kernel_matrix(K);
    int64_t n = work.size(0);
    auto one_n = at::ones({n, n}, work.options()) / n;

    auto K_centered =
        work - one_n.matmul(work) - work.matmul(one_n) + one_n.matmul(work).matmul(one_n);

    return K_centered.to(K.device()).to(K.scalar_type());
}

} // namespace nerve::torch
