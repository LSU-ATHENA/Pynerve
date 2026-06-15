// Simplex tree construction operations  --  VR and Witness complex builders.

#include "nerve/torch/boundary_matrix.hpp"
#include "nerve/torch/simplex_tree.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace nerve::torch
{

namespace
{

void validate_max_radius(double max_radius)
{
    TORCH_CHECK(max_radius > 0.0 && !std::isnan(max_radius),
                "max_radius must be positive and not NaN");
}

void validate_max_dim(int64_t max_dim, int64_t max_supported)
{
    TORCH_CHECK(max_dim >= 0 && max_dim <= max_supported, "max_dim must be between 0 and ",
                max_supported);
}

at::Tensor validate_point_tensor_cpu(const at::Tensor &points, const char *name)
{
    TORCH_CHECK(points.defined(), name, " must be defined");
    TORCH_CHECK(points.dim() == 2, name, " must be a 2D tensor [N, D]");
    TORCH_CHECK(points.size(0) > 0, name, " must contain at least one point");
    TORCH_CHECK(points.size(1) > 0, name, " must contain at least one coordinate dimension");
    TORCH_CHECK(points.is_floating_point(), name, " must be a floating-point tensor");
    TORCH_CHECK(at::isfinite(points).all().item<bool>(), name,
                " must contain only finite coordinates");
    return points.contiguous().cpu().to(at::kDouble);
}

} // namespace

void SimplexTree::build_vr(const at::Tensor &points, double max_radius, int64_t max_dim)
{
    validate_max_radius(max_radius);
    validate_max_dim(max_dim, 3);
    at::Tensor points_cpu = validate_point_tensor_cpu(points, "points");
    auto dist_matrix = at::cdist(points_cpu, points_cpu);
    int64_t n = points_cpu.size(0);
    if (n == 0)
    {
        return;
    }

    for (int64_t i = 0; i < n; ++i)
    {
        insert({i}, 0.0);
    }

    auto dist_cpu = dist_matrix.contiguous();
    auto accessor = dist_cpu.accessor<double, 2>();
    std::vector<std::vector<bool>> has_edge(static_cast<size_t>(n),
                                            std::vector<bool>(static_cast<size_t>(n), false));

    for (int64_t i = 0; i < n; ++i)
    {
        for (int64_t j = i + 1; j < n; ++j)
        {
            double d = accessor[i][j];
            if (d <= max_radius)
            {
                insert({i, j}, d);
                has_edge[static_cast<size_t>(i)][static_cast<size_t>(j)] = true;
                has_edge[static_cast<size_t>(j)][static_cast<size_t>(i)] = true;
            }
        }
    }

    if (max_dim > 1)
    {
        for (int64_t i = 0; i < n; ++i)
        {
            for (int64_t j = i + 1; j < n; ++j)
            {
                if (!has_edge[static_cast<size_t>(i)][static_cast<size_t>(j)])
                {
                    continue;
                }
                for (int64_t k = j + 1; k < n; ++k)
                {
                    if (!has_edge[static_cast<size_t>(i)][static_cast<size_t>(k)] ||
                        !has_edge[static_cast<size_t>(j)][static_cast<size_t>(k)])
                    {
                        continue;
                    }
                    const double f = std::max({accessor[i][j], accessor[i][k], accessor[j][k]});
                    insert({i, j, k}, f);
                }
            }
        }
    }

    if (max_dim > 2)
    {
        for (int64_t i = 0; i < n; ++i)
        {
            for (int64_t j = i + 1; j < n; ++j)
            {
                if (!has_edge[static_cast<size_t>(i)][static_cast<size_t>(j)])
                {
                    continue;
                }
                for (int64_t k = j + 1; k < n; ++k)
                {
                    if (!has_edge[static_cast<size_t>(i)][static_cast<size_t>(k)] ||
                        !has_edge[static_cast<size_t>(j)][static_cast<size_t>(k)])
                    {
                        continue;
                    }
                    for (int64_t l = k + 1; l < n; ++l)
                    {
                        if (!has_edge[static_cast<size_t>(i)][static_cast<size_t>(l)] ||
                            !has_edge[static_cast<size_t>(j)][static_cast<size_t>(l)] ||
                            !has_edge[static_cast<size_t>(k)][static_cast<size_t>(l)])
                        {
                            continue;
                        }
                        const double f = std::max({accessor[i][j], accessor[i][k], accessor[i][l],
                                                   accessor[j][k], accessor[j][l], accessor[k][l]});
                        insert({i, j, k, l}, f);
                    }
                }
            }
        }
    }
}

void SimplexTree::build_witness(const at::Tensor &points, const at::Tensor &landmarks,
                                double max_radius, int64_t max_dim)
{
    validate_max_radius(max_radius);
    validate_max_dim(max_dim, 2);
    at::Tensor points_cpu = validate_point_tensor_cpu(points, "points");
    at::Tensor landmarks_cpu = validate_point_tensor_cpu(landmarks, "landmarks");
    TORCH_CHECK(points_cpu.size(1) == landmarks_cpu.size(1),
                "points and landmarks must have the same coordinate dimension");
    const int64_t n_points = points_cpu.size(0);
    const int64_t n_landmarks = landmarks_cpu.size(0);
    if (n_points == 0 || n_landmarks == 0)
    {
        return;
    }

    for (int64_t l = 0; l < n_landmarks; ++l)
    {
        insert({l}, 0.0);
    }

    auto dist = at::cdist(points_cpu, landmarks_cpu).contiguous();
    auto acc = dist.accessor<double, 2>();
    std::vector<std::vector<bool>> witnessed(
        static_cast<size_t>(n_landmarks),
        std::vector<bool>(static_cast<size_t>(n_landmarks), false));
    std::vector<std::vector<double>> best_edge(
        static_cast<size_t>(n_landmarks),
        std::vector<double>(static_cast<size_t>(n_landmarks),
                            std::numeric_limits<double>::infinity()));

    for (int64_t p = 0; p < n_points; ++p)
    {
        for (int64_t i = 0; i < n_landmarks; ++i)
        {
            if (acc[p][i] > max_radius)
            {
                continue;
            }
            for (int64_t j = i + 1; j < n_landmarks; ++j)
            {
                if (acc[p][j] <= max_radius)
                {
                    witnessed[static_cast<size_t>(i)][static_cast<size_t>(j)] = true;
                    witnessed[static_cast<size_t>(j)][static_cast<size_t>(i)] = true;
                    const double f = std::max(acc[p][i], acc[p][j]);
                    best_edge[static_cast<size_t>(i)][static_cast<size_t>(j)] =
                        std::min(best_edge[static_cast<size_t>(i)][static_cast<size_t>(j)], f);
                    best_edge[static_cast<size_t>(j)][static_cast<size_t>(i)] =
                        best_edge[static_cast<size_t>(i)][static_cast<size_t>(j)];
                }
            }
        }
    }

    for (int64_t i = 0; i < n_landmarks; ++i)
    {
        for (int64_t j = i + 1; j < n_landmarks; ++j)
        {
            if (witnessed[static_cast<size_t>(i)][static_cast<size_t>(j)])
            {
                insert({i, j}, best_edge[static_cast<size_t>(i)][static_cast<size_t>(j)]);
            }
        }
    }

    if (max_dim > 1)
    {
        for (int64_t i = 0; i < n_landmarks; ++i)
        {
            for (int64_t j = i + 1; j < n_landmarks; ++j)
            {
                if (!witnessed[static_cast<size_t>(i)][static_cast<size_t>(j)])
                {
                    continue;
                }
                for (int64_t k = j + 1; k < n_landmarks; ++k)
                {
                    if (!witnessed[static_cast<size_t>(i)][static_cast<size_t>(k)] ||
                        !witnessed[static_cast<size_t>(j)][static_cast<size_t>(k)])
                    {
                        continue;
                    }
                    const double f =
                        std::max({best_edge[static_cast<size_t>(i)][static_cast<size_t>(j)],
                                  best_edge[static_cast<size_t>(i)][static_cast<size_t>(k)],
                                  best_edge[static_cast<size_t>(j)][static_cast<size_t>(k)]});
                    insert({i, j, k}, f);
                }
            }
        }
    }
}

} // namespace nerve::torch
