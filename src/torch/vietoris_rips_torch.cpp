// PyTorch-Native Vietoris-Rips Implementation
// Fully integrated with existing Nerve VR algorithms

#include "nerve/filtration/vietoris_rips.hpp"
#include "nerve/persistence/vr/vr_fast_ops.hpp"

#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <string>
#include <tuple>
#include <vector>

namespace nerve::torch
{

// Helper: Convert at::Tensor to PointView for existing implementations

namespace detail
{

void validate_max_radius(double max_radius)
{
    TORCH_CHECK(max_radius > 0.0 && !std::isnan(max_radius),
                "max_radius must be positive and not NaN");
}

double effective_max_radius(double max_radius)
{
    return std::isinf(max_radius) ? std::numeric_limits<double>::max() : max_radius;
}

void validate_point_cloud_input(const at::Tensor &input, const char *name)
{
    TORCH_CHECK(input.defined(), name, " must be defined");
    TORCH_CHECK(input.dim() == 2, name, " must be 2D tensor [N, D]");
    TORCH_CHECK(input.size(0) > 0, name, " must contain at least one point");
    TORCH_CHECK(input.size(1) > 0, name, " must contain at least one coordinate dimension");
    TORCH_CHECK(input.is_floating_point(), name, " must be a floating-point tensor");
    TORCH_CHECK(at::isfinite(input).all().item<bool>(), name,
                " must contain only finite coordinates");
}

at::Tensor validate_distance_matrix_input(const at::Tensor &input)
{
    TORCH_CHECK(input.defined(), "input must be defined");
    TORCH_CHECK(input.dim() == 2, "input must be 2D");
    TORCH_CHECK(input.size(0) == input.size(1), "input must be square distance matrix");
    TORCH_CHECK(input.is_floating_point(), "input must be a floating-point distance matrix");
    TORCH_CHECK(!at::isnan(input).any().item<bool>(), "input must not contain NaN distances");

    at::Tensor dist_cpu = input.contiguous().cpu().to(at::kDouble);
    at::Tensor finite_mask = at::isfinite(dist_cpu);
    if (finite_mask.any().item<bool>())
    {
        at::Tensor finite_values = dist_cpu.masked_select(finite_mask);
        TORCH_CHECK((finite_values >= 0).all().item<bool>(),
                    "input distances must be non-negative");
    }
    TORCH_CHECK(at::allclose(dist_cpu, dist_cpu.transpose(0, 1), 1e-12, 1e-12, true),
                "input distance matrix must be symmetric");
    return dist_cpu;
}

core::ownership_utils::PointView tensor_to_point_view(const at::Tensor &t)
{
    TORCH_CHECK(t.dim() == 2, "Expected 2D tensor [N, D]");
    TORCH_CHECK(t.scalar_type() == at::kDouble, "Expected float64 tensor");
    TORCH_CHECK(t.device().is_cpu(), "Expected CPU tensor");
    TORCH_CHECK(t.is_contiguous(), "Expected contiguous tensor");

    return core::ownership_utils::PointView(t.data_ptr<double>(),
                                            static_cast<std::size_t>(t.numel()));
}

[[nodiscard]] at::Tensor build_distance_matrix_impl(const at::Tensor &points, double max_radius,
                                                    const std::string &metric)
{
    validate_max_radius(max_radius);
    validate_point_cloud_input(points, "points");
    const double radius = effective_max_radius(max_radius);
    const int64_t n = points.size(0);

    at::Tensor points_cpu = points.contiguous().cpu().to(at::kDouble);
    auto point_view = tensor_to_point_view(points_cpu);
    filtration::VietorisRips vr(radius);
    vr.setMaxDimension(1);
    vr.setDistanceMetric(metric);

    auto result = vr.buildFiltration(point_view, points.size(1));
    TORCH_CHECK(!result.isError(), "VR filtration build failed: ", result.error().message);

    at::Tensor dist_matrix = at::full({n, n}, std::numeric_limits<double>::infinity(),
                                      points.options().dtype(at::kDouble));

    const auto &filtration = result.value();
    for (const auto &[simplex, radius] : filtration)
    {
        if (simplex.dimension() == 1)
        {
            const auto &vertices = simplex.vertices();
            if (vertices.size() == 2)
            {
                dist_matrix[vertices[0]][vertices[1]] = radius;
                dist_matrix[vertices[1]][vertices[0]] = radius;
            }
        }
    }

    dist_matrix.diagonal(0).zero_();
    return dist_matrix;
}

[[nodiscard]] at::Tensor compute_vr_fast_impl(const at::Tensor &points, double max_radius,
                                              int64_t max_dim,
                                              persistence::VRAlgorithmSelection algorithm)
{
    persistence::VRConfig config;
    config.max_dim = static_cast<Size>(max_dim);
    config.max_radius = effective_max_radius(max_radius);
    config.algorithm = algorithm;

    core::ownership_utils::PointView point_view = tensor_to_point_view(points);
    const auto pairs = persistence::computeVrPersistenceFast(
        point_view, static_cast<Size>(points.size(1)), config);
    at::Tensor diagram =
        at::empty({static_cast<int64_t>(pairs.size()), 2}, points.options().dtype(at::kDouble));
    auto out_accessor = diagram.accessor<double, 2>();
    for (int64_t i = 0; i < static_cast<int64_t>(pairs.size()); ++i)
    {
        out_accessor[i][0] = pairs[static_cast<size_t>(i)].birth;
        out_accessor[i][1] = pairs[static_cast<size_t>(i)].death;
    }
    return diagram;
}

} // namespace detail

// VR Operations (PyTorch-style naming)

at::Tensor vr_persistence(const at::Tensor &input, int64_t max_dim);

at::Tensor vr_build(const at::Tensor &input, double max_radius)
{
    detail::validate_max_radius(max_radius);
    detail::validate_point_cloud_input(input, "input");
    at::Tensor points_cpu = input.contiguous().cpu().to(at::kDouble);
    at::Tensor distance_matrix =
        detail::build_distance_matrix_impl(points_cpu, max_radius, "euclidean");
    return distance_matrix.to(input.device());
}

at::Tensor vr_build_with_metric(const at::Tensor &input, double max_radius,
                                const std::string &metric)
{
    detail::validate_max_radius(max_radius);
    detail::validate_point_cloud_input(input, "input");
    at::Tensor points_cpu = input.contiguous().cpu().to(at::kDouble);
    at::Tensor distance_matrix = detail::build_distance_matrix_impl(points_cpu, max_radius, metric);
    return distance_matrix.to(input.device());
}

at::Tensor vr_fast(const at::Tensor &input, double max_radius, const std::string &algorithm)
{
    detail::validate_max_radius(max_radius);
    detail::validate_point_cloud_input(input, "input");
    const int64_t num_points = input.size(0);
    const int64_t dim = input.size(1);

    std::string chosen_algorithm = algorithm;
    if (algorithm == "auto")
    {
        if (num_points < 1000 && dim <= 3)
        {
            chosen_algorithm = "fast";
        }
        else if (num_points < 10000)
        {
            chosen_algorithm = "medium";
        }
        else
        {
            chosen_algorithm = "large";
        }
    }

    at::Tensor points_cpu = input.contiguous().cpu().to(at::kDouble);

    if (chosen_algorithm == "fast")
    {
        return vr_persistence(
                   detail::build_distance_matrix_impl(points_cpu, max_radius, "euclidean"), 2)
            .to(input.device());
    }
    else if (chosen_algorithm == "medium")
    {
        return vr_persistence(
                   detail::build_distance_matrix_impl(points_cpu, max_radius, "euclidean"), 3)
            .to(input.device());
    }
    else if (chosen_algorithm == "large")
    {
        return detail::compute_vr_fast_impl(points_cpu, max_radius, 3,
                                            persistence::VRAlgorithmSelection::LARGE_WITNESS)
            .to(input.device());
    }
    else
    {
        TORCH_CHECK(false, "Unknown algorithm: ", algorithm);
        return at::Tensor();
    }
}

at::Tensor vr_persistence(const at::Tensor &input, int64_t max_dim)
{
    TORCH_CHECK(max_dim >= 0 && max_dim <= 3, "max_dim must be 0-3");

    at::Tensor dist_cpu = detail::validate_distance_matrix_input(input);
    const int64_t n = dist_cpu.size(0);

    // Extract edges
    std::vector<std::tuple<int64_t, int64_t, double>> edges;
    auto accessor = dist_cpu.accessor<double, 2>();

    for (int64_t i = 0; i < n; ++i)
    {
        for (int64_t j = i + 1; j < n; ++j)
        {
            double dist = accessor[i][j];
            if (std::isfinite(dist))
            {
                edges.emplace_back(i, j, dist);
            }
        }
    }

    std::sort(edges.begin(), edges.end(),
              [](const auto &a, const auto &b) { return std::get<2>(a) < std::get<2>(b); });

    // Union-find
    std::vector<int64_t> parent(n);
    std::iota(parent.begin(), parent.end(), 0);
    std::vector<double> birth_times(n, 0.0);

    auto find = [&parent](int64_t x) {
        while (parent[x] != x)
        {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    };

    std::vector<std::pair<double, double>> pairs_0d;

    for (const auto &[i, j, dist] : edges)
    {
        int64_t pi = find(i);
        int64_t pj = find(j);

        if (pi != pj)
        {
            parent[pi] = pj;
            pairs_0d.emplace_back(birth_times[pi], dist);
        }
    }

    for (int64_t i = 0; i < n; ++i)
    {
        if (find(i) == i)
        {
            pairs_0d.emplace_back(birth_times[i], std::numeric_limits<double>::infinity());
        }
    }

    const auto num_pairs = static_cast<int64_t>(pairs_0d.size());
    at::Tensor diagram = at::empty({num_pairs, 2}, input.options().dtype(at::kDouble));
    auto out_accessor = diagram.accessor<double, 2>();

    for (int64_t pair_index = 0; pair_index < num_pairs; ++pair_index)
    {
        const auto vector_index = static_cast<size_t>(pair_index);
        out_accessor[pair_index][0] = pairs_0d[vector_index].first;
        out_accessor[pair_index][1] = pairs_0d[vector_index].second;
    }

    return diagram.to(input.device());
}

} // namespace nerve::torch
