MatrixXd
PersistenceGradient::computeGradient(const MatrixXd &points, const DifferentiableDiagram &diagram,
                                     const std::vector<std::pair<double, double>> &loss_gradients)
{
    const int n_points = points.rows();
    const int dim = points.cols();

    validateDifferentiableDiagram(diagram, "PersistenceGradient::computeGradient", false);
    for (const auto &grad : loss_gradients)
    {
        if (!std::isfinite(grad.first) || !std::isfinite(grad.second))
        {
            throw std::invalid_argument(
                "PersistenceGradient::computeGradient: loss gradients must be finite");
        }
    }

    MatrixXd point_gradients = MatrixXd::Zero(n_points, dim);

    // For each persistence pair, compute gradient contribution
    for (size_t i = 0; i < diagram.persistence_pairs.size() && i < loss_gradients.size(); ++i)
    {
        const auto &pair = diagram.persistence_pairs[i];
        const auto &grad = loss_gradients[i];

        // Skip infinite pairs (essential features)
        if (pair.second == std::numeric_limits<double>::infinity())
            continue;

        // Get simplices for this pair
        if (i < diagram.birth_simplices.size() && i < diagram.death_simplices.size())
        {
            const auto &birth_simplex = diagram.birth_simplices[i];
            const auto &death_simplex = diagram.death_simplices[i];

            // Compute gradient contribution from birth simplex
            MatrixXd birth_grad = computeSimplexGradient(points, birth_simplex, pair.first);
            for (int v = 0; v < birth_grad.rows() && v < n_points; ++v)
            {
                for (int d = 0; d < birth_grad.cols() && d < dim; ++d)
                {
                    point_gradients(v, d) += grad.first * birth_grad(v, d);
                }
            }

            // Compute gradient contribution from death simplex
            if (death_simplex.size() > 0)
            {
                MatrixXd death_grad = computeSimplexGradient(points, death_simplex, pair.second);
                for (int v = 0; v < death_grad.rows() && v < n_points; ++v)
                {
                    for (int d = 0; d < death_grad.cols() && d < dim; ++d)
                    {
                        point_gradients(v, d) += grad.second * death_grad(v, d);
                    }
                }
            }
        }
    }

    return point_gradients;
}

MatrixXd PersistenceGradient::computeSimplexGradient(const MatrixXd &points,
                                                     const VectorXi &simplex_vertices,
                                                     double filtration_value)
{
    int n_verts = simplex_vertices.size();
    int n_points = points.rows();
    int dim = points.cols();

    MatrixXd gradient = MatrixXd::Zero(n_points, dim);

    for (int i = 0; i < n_verts; ++i)
    {
        const int vi = simplex_vertices[i];
        if (vi < 0 || vi >= n_points)
        {
            continue;
        }
        for (int j = i + 1; j < n_verts; ++j)
        {
            const int vj = simplex_vertices[j];
            if (vj < 0 || vj >= n_points)
            {
                continue;
            }

            const double dist = pointDistance(points, vi, vj);
            if (dist <= GRADIENT_EPSILON || std::abs(dist - filtration_value) >= GRADIENT_EPSILON)
            {
                continue;
            }

            for (int d = 0; d < dim; ++d)
            {
                const double edge_grad = (points(vi, d) - points(vj, d)) / dist;
                gradient(vi, d) += edge_grad;
                gradient(vj, d) -= edge_grad;
            }
        }
    }

    return gradient;
}

MatrixXd PersistenceGradient::computeTopologyOptimizationGradient(
    const MatrixXd &points, const DifferentiableDiagram &current_diagram,
    const DifferentiableDiagram &target_diagram)
{
    validateDifferentiableDiagram(current_diagram,
                                  "PersistenceGradient::computeTopologyOptimizationGradient", true);
    validateDifferentiableDiagram(target_diagram,
                                  "PersistenceGradient::computeTopologyOptimizationGradient", true);

    // Match persistence pairs between current and target
    auto matches = matchPersistencePairs(current_diagram, target_diagram);

    MatrixXd gradient(points.rows(), points.cols());

    // Build loss gradients from matched pairs
    std::vector<std::pair<double, double>> loss_gradients;
    loss_gradients.reserve(current_diagram.persistence_pairs.size());

    for (size_t i = 0; i < current_diagram.persistence_pairs.size(); ++i)
    {
        double birth_grad = 0.0;
        double death_grad = 0.0;

        // Find matching target pair
        for (const auto &match : matches)
        {
            if (match.first == static_cast<int>(i))
            {
                int target_idx = match.second;
                if (target_idx >= 0 &&
                    target_idx < static_cast<int>(target_diagram.persistence_pairs.size()))
                {
                    const auto &current_pair = current_diagram.persistence_pairs[i];
                    const auto &target_pair = target_diagram.persistence_pairs[target_idx];

                    // Gradient = difference from target
                    birth_grad = 2.0 * (current_pair.first - target_pair.first);
                    death_grad = 2.0 * (current_pair.second - target_pair.second);
                }
                break;
            }
        }

        loss_gradients.emplace_back(birth_grad, death_grad);
    }

    return computeGradient(points, current_diagram, loss_gradients);
}

std::vector<std::pair<int, int>>
PersistenceGradient::matchPersistencePairs(const DifferentiableDiagram &current,
                                           const DifferentiableDiagram &target)
{
    std::vector<std::pair<int, int>> matches;

    // Greedy matching by dimension and birth time proximity
    std::vector<bool> target_matched(target.persistence_pairs.size(), false);

    for (size_t i = 0; i < current.persistence_pairs.size(); ++i)
    {
        double min_diff = std::numeric_limits<double>::max();
        int best_match = -1;

        for (size_t j = 0; j < target.persistence_pairs.size(); ++j)
        {
            if (target_matched[j])
                continue;
            if (current.dimensions[i] != target.dimensions[j])
                continue;

            double diff =
                std::abs(current.persistence_pairs[i].first - target.persistence_pairs[j].first);
            if (diff < min_diff)
            {
                min_diff = diff;
                best_match = static_cast<int>(j);
            }
        }

        if (best_match >= 0)
        {
            matches.emplace_back(static_cast<int>(i), best_match);
            target_matched[best_match] = true;
        }
    }

    return matches;
}

std::vector<std::pair<int, int>>
PersistenceGradient::hungarianAlgorithm(const std::vector<std::vector<double>> &cost,
                                        size_t n_current, size_t n_target)
{
    std::vector<std::pair<int, int>> matches;
    if (n_current == 0 || n_target == 0 || cost.empty())
    {
        return matches;
    }

    const size_t rows = std::min(n_current, cost.size());
    size_t cols = 0;
    for (size_t i = 0; i < rows; ++i)
    {
        cols = std::max(cols, cost[i].size());
    }
    cols = std::min(cols, n_target);
    if (cols == 0)
    {
        return matches;
    }

    const size_t size = std::max(rows, cols);
    double max_cost = 0.0;
    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < std::min(cols, cost[i].size()); ++j)
        {
            max_cost = std::max(max_cost, cost[i][j]);
        }
    }
    const double kPaddingCost = max_cost + 1e6;

    std::vector<std::vector<double>> padded(size, std::vector<double>(size, kPaddingCost));
    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < std::min(cols, cost[i].size()); ++j)
        {
            padded[i][j] = cost[i][j];
        }
    }

    std::vector<double> u(size + 1, 0.0), v(size + 1, 0.0);
    std::vector<size_t> p(size + 1, 0), way(size + 1, 0);

    for (size_t i = 1; i <= size; ++i)
    {
        p[0] = i;
        size_t j0 = 0;
        std::vector<double> minv(size + 1, std::numeric_limits<double>::infinity());
        std::vector<char> used(size + 1, 0);

        do
        {
            used[j0] = 1;
            const size_t i0 = p[j0];
            double delta = std::numeric_limits<double>::infinity();
            size_t j1 = 0;

            for (size_t j = 1; j <= size; ++j)
            {
                if (used[j])
                {
                    continue;
                }
                const double cur = padded[i0 - 1][j - 1] - u[i0] - v[j];
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

            for (size_t j = 0; j <= size; ++j)
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
            const size_t j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0 != 0);
    }

    for (size_t j = 1; j <= size; ++j)
    {
        const size_t i = p[j];
        if (i == 0)
        {
            continue;
        }
        const size_t row = i - 1;
        const size_t col = j - 1;
        if (row < rows && col < cols && row < n_current && col < n_target)
        {
            matches.emplace_back(static_cast<int>(row), static_cast<int>(col));
        }
    }
    std::sort(matches.begin(), matches.end());
    return matches;
}
