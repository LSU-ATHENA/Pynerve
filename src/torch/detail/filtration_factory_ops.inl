Filtration Filtration::from_vietoris_rips(const at::Tensor &points, double max_radius,
                                          int64_t max_dim, at::Device device)
{
    Filtration filtration;
    filtration.to(device);

    at::Tensor points_cpu = points.contiguous().cpu().to(at::kDouble);
    int64_t n = points_cpu.size(0);

    for (int64_t i = 0; i < n; ++i)
    {
        filtration.append({i}, 0.0);
    }

    auto dist_matrix = at::cdist(points_cpu, points_cpu);
    auto dist_cpu = dist_matrix.contiguous();
    auto accessor = dist_cpu.accessor<double, 2>();

    for (int64_t i = 0; i < n; ++i)
    {
        for (int64_t j = i + 1; j < n; ++j)
        {
            double d = accessor[i][j];
            if (d <= max_radius)
            {
                filtration.append({i, j}, d);
            }
        }
    }

    if (max_dim > 1)
    {
        for (int64_t i = 0; i < n; ++i)
        {
            for (int64_t j = i + 1; j < n; ++j)
            {
                if (accessor[i][j] > max_radius)
                {
                    continue;
                }
                for (int64_t k = j + 1; k < n; ++k)
                {
                    if (accessor[i][k] > max_radius || accessor[j][k] > max_radius)
                    {
                        continue;
                    }
                    const double f = std::max({accessor[i][j], accessor[i][k], accessor[j][k]});
                    filtration.append({i, j, k}, f);
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
                if (accessor[i][j] > max_radius)
                {
                    continue;
                }
                for (int64_t k = j + 1; k < n; ++k)
                {
                    if (accessor[i][k] > max_radius || accessor[j][k] > max_radius)
                    {
                        continue;
                    }
                    for (int64_t l = k + 1; l < n; ++l)
                    {
                        if (accessor[i][l] > max_radius || accessor[j][l] > max_radius ||
                            accessor[k][l] > max_radius)
                        {
                            continue;
                        }
                        const double f = std::max({accessor[i][j], accessor[i][k], accessor[i][l],
                                                   accessor[j][k], accessor[j][l], accessor[k][l]});
                        filtration.append({i, j, k, l}, f);
                    }
                }
            }
        }
    }

    return filtration;
}

Filtration Filtration::from_witness(const at::Tensor &points, const at::Tensor &landmarks,
                                    double max_radius, int64_t max_dim, at::Device device)
{
    Filtration filtration;
    filtration.to(device);

    at::Tensor points_cpu = points.contiguous().cpu().to(at::kDouble);
    at::Tensor landmarks_cpu = landmarks.contiguous().cpu().to(at::kDouble);
    const int64_t n_points = points_cpu.size(0);
    const int64_t n_landmarks = landmarks_cpu.size(0);
    if (n_points == 0 || n_landmarks == 0)
    {
        return filtration;
    }

    for (int64_t l = 0; l < n_landmarks; ++l)
    {
        filtration.append({l}, 0.0);
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
                filtration.append({i, j},
                                  best_edge[static_cast<size_t>(i)][static_cast<size_t>(j)]);
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
                    filtration.append({i, j, k}, f);
                }
            }
        }
    }

    return filtration;
}

Filtration Filtration::from_alpha(const at::Tensor &points, at::Device device)
{
    Filtration filtration;
    filtration.to(device);

    TORCH_CHECK(points.defined(), "points must be defined");
    TORCH_CHECK(points.dim() == 2, "points must be a 2D tensor [N, D]");
    TORCH_CHECK(points.size(0) > 0, "points must contain at least one point");
    TORCH_CHECK(points.size(1) > 0, "points must contain at least one coordinate dimension");
    TORCH_CHECK(points.is_floating_point(), "points must be a floating-point tensor");
    TORCH_CHECK(at::isfinite(points).all().item<bool>(),
                "points must contain only finite coordinates");

    at::Tensor points_cpu = points.contiguous().cpu().to(at::kDouble);
    const int64_t n = points_cpu.size(0);
    for (int64_t i = 0; i < n; ++i)
    {
        filtration.append({i}, 0.0);
    }

    at::Tensor dist_matrix = at::cdist(points_cpu, points_cpu).contiguous();
    auto dist = dist_matrix.accessor<double, 2>();
    for (int64_t i = 0; i < n; ++i)
    {
        for (int64_t j = i + 1; j < n; ++j)
        {
            filtration.append({i, j}, dist[i][j] * 0.5);
        }
    }

    filtration.sort_by_filtration();
    return filtration;
}
