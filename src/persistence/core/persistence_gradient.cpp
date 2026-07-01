#include "nerve/persistence/core/ph_gradient.hpp"

#include <limits>
#include <ranges>

namespace nerve::persistence::gradient
{

constinit const double GRADIENT_EPSILON = 1e-10;

Eigen::MatrixXd
PersistenceGradient::computeGradient(const Eigen::MatrixXd &points,
                                     const DifferentiableDiagram &diagram,
                                     const std::vector<std::pair<double, double>> &loss_gradients)
{
    const Eigen::Index n_points = points.rows();
    const Eigen::Index dim = points.cols();

    Eigen::MatrixXd point_gradients = Eigen::MatrixXd::Zero(n_points, dim);

    // Compute gradient contribution from each persistence pair
    for (size_t i = 0; i < diagram.persistence_pairs.size(); ++i)
    {
        if (i >= loss_gradients.size())
            continue;

        double dL_db = loss_gradients[i].first;
        double dL_dd = loss_gradients[i].second;

        // Gradient from birth simplex
        if (diagram.birth_simplices[i].size() > 0)
        {
            Eigen::MatrixXd birth_grad = computeSimplexGradient(points, diagram.birth_simplices[i],
                                                                diagram.persistence_pairs[i].first);
            point_gradients += dL_db * birth_grad;
        }

        // Gradient from death simplex
        if (diagram.death_simplices[i].size() > 0)
        {
            Eigen::MatrixXd death_grad = computeSimplexGradient(
                points, diagram.death_simplices[i], diagram.persistence_pairs[i].second);
            point_gradients += dL_dd * death_grad;
        }
    }

    return point_gradients;
}

Eigen::MatrixXd PersistenceGradient::computeTopologyOptimizationGradient(
    const Eigen::MatrixXd &points, const DifferentiableDiagram &current_diagram,
    const DifferentiableDiagram &target_diagram)
{
    // Match persistence pairs between current and target
    auto matched_pairs = matchPersistencePairs(current_diagram, target_diagram);

    std::vector<std::pair<double, double>> loss_gradients;
    loss_gradients.reserve(current_diagram.persistence_pairs.size());

    for (const auto &[current_idx, target_idx] : matched_pairs)
    {
        if (target_idx >= 0)
        {
            // Matched pair - minimize distance
            double target_birth = target_diagram.persistence_pairs[target_idx].first;
            double target_death = target_diagram.persistence_pairs[target_idx].second;

            double current_birth = current_diagram.persistence_pairs[current_idx].first;
            double current_death = current_diagram.persistence_pairs[current_idx].second;

            loss_gradients.push_back(
                {2.0 * (current_birth - target_birth), 2.0 * (current_death - target_death)});
        }
        else
        {
            // Unmatched current pair - penalize
            loss_gradients.push_back({2.0 * current_diagram.persistence_pairs[current_idx].first,
                                      2.0 * current_diagram.persistence_pairs[current_idx].second});
        }
    }

    return computeGradient(points, current_diagram, loss_gradients);
}

Eigen::MatrixXd PersistenceGradient::computeSimplexGradient(const Eigen::MatrixXd &points,
                                                            const Eigen::VectorXi &simplex_vertices,
                                                            double filtration_value)
{
    const Eigen::Index n_points = points.rows();
    const Eigen::Index dim = points.cols();
    const Eigen::Index simplex_dim = simplex_vertices.size();

    Eigen::MatrixXd gradient = Eigen::MatrixXd::Zero(n_points, dim);

    // Find critical edges (edges with length = filtration_value)
    for (Eigen::Index i = 0; i < simplex_dim; ++i)
    {
        int vi = simplex_vertices[i];

        for (Eigen::Index j = i + 1; j < simplex_dim; ++j)
        {
            int vj = simplex_vertices[j];

            Eigen::VectorXd diff = points.row(vi) - points.row(vj);
            double dist = diff.norm();

            // Check if this edge determines the filtration value
            if (std::abs(dist - filtration_value) < GRADIENT_EPSILON)
            {
                // Gradient contribution: (x_i - x_j) / ||x_i - x_j||
                Eigen::VectorXd edge_grad = diff / (dist + GRADIENT_EPSILON);

                gradient.row(vi) += edge_grad;
                gradient.row(vj) -= edge_grad;
            }
        }
    }

    return gradient;
}

std::vector<std::pair<int, int>>
PersistenceGradient::matchPersistencePairs(const DifferentiableDiagram &current,
                                           const DifferentiableDiagram &target)
{
    size_t n_current = current.persistence_pairs.size();
    size_t n_target = target.persistence_pairs.size();
    size_t n = std::max(n_current, n_target);

    // Build cost matrix: distance between each pair
    std::vector<std::vector<double>> cost_matrix(n, std::vector<double>(n, 0.0));

    for (size_t i = 0; i < n_current; ++i)
    {
        for (size_t j = 0; j < n_target; ++j)
        {
            if (current.dimensions[i] != target.dimensions[j])
            {
                cost_matrix[i][j] = 1e10;
            }
            else
            {
                double db = current.persistence_pairs[i].first - target.persistence_pairs[j].first;
                double dd =
                    current.persistence_pairs[i].second - target.persistence_pairs[j].second;
                cost_matrix[i][j] = std::sqrt(db * db + dd * dd);
            }
        }
    }

    // Fill extra rows/columns with diagonal costs
    for (size_t i = n_current; i < n; ++i)
    {
        for (size_t j = 0; j < n_target; ++j)
        {
            cost_matrix[i][j] =
                (target.persistence_pairs[j].second - target.persistence_pairs[j].first) / 2.0;
        }
    }
    for (size_t j = n_target; j < n; ++j)
    {
        for (size_t i = 0; i < n_current; ++i)
        {
            cost_matrix[i][j] =
                (current.persistence_pairs[i].second - current.persistence_pairs[i].first) / 2.0;
        }
    }

    return hungarianAlgorithm(cost_matrix, n_current, n_target);
}

std::vector<std::pair<int, int>>
PersistenceGradient::hungarianAlgorithm(const std::vector<std::vector<double>> &cost,
                                        size_t n_current, size_t n_target)
{
    size_t n = cost.size();
    if (n > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        return {};
    }
    std::vector<double> u(n + 1), v(n + 1);
    std::vector<int> p(n + 1), way(n + 1);

    for (size_t i = 1; i <= n; ++i)
    {
        p[0] = static_cast<int>(i);
        size_t j0 = 0;
        std::vector<double> minv(n + 1, std::numeric_limits<double>::infinity());
        std::vector<bool> used(n + 1, false);

        do
        {
            used[j0] = true;
            size_t i0 = static_cast<size_t>(p[j0]);
            double delta = std::numeric_limits<double>::infinity();
            size_t j1 = 0;

            for (size_t j = 1; j <= n; ++j)
            {
                if (!used[j])
                {
                    double cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
                    if (cur < minv[j])
                    {
                        minv[j] = cur;
                        way[j] = static_cast<int>(j0);
                    }
                    if (minv[j] < delta)
                    {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }

            for (size_t j = 0; j <= n; ++j)
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
            size_t j1 = static_cast<size_t>(way[j0]);
            p[j0] = p[j1];
            j0 = j1;
        } while (j0 != 0);
    }

    // Build result
    std::vector<std::pair<int, int>> result;
    for (size_t j = 1; j <= n; ++j)
    {
        if (p[j] > 0 && static_cast<size_t>(p[j]) <= n_current && j <= n_target)
        {
            result.push_back({p[j] - 1, static_cast<int>(j) - 1});
        }
    }

    return result;
}

DifferentiableDiagram computeDifferentiable(std::span<const double> points_data, size_t n_points,
                                            size_t point_dim, double max_distance, int max_dim)
{
    if (n_points > std::numeric_limits<size_t>::max() / point_dim)
    {
        throw std::invalid_argument("computeDifferentiable: point data too large");
    }
    if (!std::isfinite(max_distance) || max_distance <= 0.0)
    {
        throw std::invalid_argument(
            "computeDifferentiable: max_distance must be finite and non-negative");
    }
    if (n_points == 0 || point_dim == 0)
    {
        return DifferentiableDiagram{};
    }

    Eigen::MatrixXd points_mat(n_points, point_dim);
    for (size_t i = 0; i < n_points; ++i)
    {
        for (size_t d = 0; d < point_dim; ++d)
        {
            points_mat(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(d)) =
                points_data[i * point_dim + d];
        }
    }

    DifferentiableDiagram result;
    double last_dist = 0.0;
    for (double r = 0.0; r <= max_distance; r += max_distance / 100.0)
    {
        (void)last_dist;
        for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(n_points); ++i)
        {
            bool has_pair = false;
            for (size_t pi = 0; pi < result.persistence_pairs.size(); ++pi)
            {
                double birth_val = result.persistence_pairs[pi].first;
                if (std::abs(birth_val - r) < 1e-6)
                {
                    has_pair = true;
                    break;
                }
            }
            if (!has_pair)
            {
                Eigen::VectorXi simplex(1);
                simplex[0] = static_cast<int>(i);
                result.persistence_pairs.push_back({r, -1.0});
                result.dimensions.push_back(0);
                result.birth_simplices.push_back(simplex);
                result.death_simplices.push_back(Eigen::VectorXi(0));
            }
        }
        last_dist = r;
    }

    return result;
}

std::vector<double> persistenceBackward(std::span<const double> points_data, size_t n_points,
                                        size_t point_dim, const DifferentiableDiagram &diagram,
                                        std::span<const double> grad_birth,
                                        std::span<const double> grad_death)
{
    if (n_points > std::numeric_limits<size_t>::max() / point_dim)
    {
        throw std::invalid_argument("persistenceBackward: point data too large");
    }
    for (double g : grad_birth)
    {
        if (!std::isfinite(g))
        {
            throw std::invalid_argument("persistenceBackward: gradients must be finite");
        }
    }

    size_t n_pairs = diagram.persistence_pairs.size();
    std::vector<double> result(n_points * point_dim, 0.0);

    for (size_t pair_idx = 0; pair_idx < n_pairs && pair_idx < grad_birth.size(); ++pair_idx)
    {
        double gb = grad_birth[pair_idx];
        if (pair_idx < diagram.birth_simplices.size() &&
            diagram.birth_simplices[pair_idx].size() > 0)
        {
            const auto &simplex = diagram.birth_simplices[pair_idx];
            long dim = simplex.size();
            if (dim > 1)
            {
                for (Eigen::Index i = 0; i < simplex.size(); ++i)
                {
                    int vi = simplex[i];
                    for (Eigen::Index j = i + 1; j < simplex.size(); ++j)
                    {
                        int vj = simplex[j];
                        for (size_t d = 0; d < point_dim; ++d)
                        {
                            double diff =
                                points_data[vi * point_dim + d] - points_data[vj * point_dim + d];
                            double dist = 0.0;
                            for (size_t k = 0; k < point_dim; ++k)
                            {
                                double dk = points_data[vi * point_dim + k] -
                                            points_data[vj * point_dim + k];
                                dist += dk * dk;
                            }
                            dist = std::sqrt(dist);
                            if (dist > 1e-10)
                            {
                                double contrib = gb * diff / dist;
                                result[vi * point_dim + d] += contrib;
                                result[vj * point_dim + d] -= contrib;
                            }
                        }
                    }
                }
            }
        }
    }

    return result;
}

StochasticPersistenceGradient::StochasticPersistenceGradient(int batch_size, double learning_rate)
    : batch_size_(batch_size)
    , learning_rate_(learning_rate)
{}

Eigen::MatrixXd StochasticPersistenceGradient::computeStochasticGradient(
    const Eigen::MatrixXd &points,
    const std::function<DifferentiableDiagram(const std::vector<int> &)> &ph_function,
    const std::vector<std::pair<double, double>> &loss_gradients)
{
    (void)ph_function;
    (void)loss_gradients;
    return Eigen::MatrixXd::Zero(points.rows(), points.cols());
}

std::vector<int> StochasticPersistenceGradient::sampleBatch(int n_points)
{
    (void)n_points;
    return {};
}

} // namespace nerve::persistence::gradient
