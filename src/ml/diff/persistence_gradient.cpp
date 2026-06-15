#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <string>
#include <tuple>
#include <vector>

#ifndef EIGEN_DONT_VECTORIZE
#define EIGEN_DONT_VECTORIZE
#endif
#include <Eigen/Dense>

namespace nerve::ml::diff
{

using MatrixXf = Eigen::MatrixXf;
using VectorXf = Eigen::VectorXf;
using EigenIndex = Eigen::Index;

namespace
{

constexpr float DISTANCE_GRADIENT_THRESHOLD = 1e-8f;
constexpr float CIRCUMRADIUS_MATCH_TOLERANCE = 1e-6f;
constexpr float AREA_DEGENERACY_THRESHOLD = 1e-8f;

struct PersistenceGradient
{
    int birth_simplex_idx = -1;
    int death_simplex_idx = -1;
    float grad_birth = 1.0f;
    float grad_death = 1.0f;
    std::vector<int> birth_vertices;
    std::vector<int> death_vertices;
};

float averageUpstream(const std::vector<float> &upstream_gradients)
{
    if (upstream_gradients.empty())
    {
        return 1.0f;
    }
    const float sum = std::accumulate(upstream_gradients.begin(), upstream_gradients.end(), 0.0f);
    return sum / static_cast<float>(upstream_gradients.size());
}

int wrapToIndex(float value, int n_points)
{
    if (n_points <= 0)
    {
        return 0;
    }
    const float wrapped = std::fmod(std::abs(value), static_cast<float>(n_points));
    return static_cast<int>(wrapped);
}

class VRGradientComputer
{
public:
    std::vector<PersistenceGradient>
    computeGradients(const std::vector<std::tuple<float, float, int>> &persistence_pairs,
                     const MatrixXf &distance_matrix,
                     const std::vector<std::vector<int>> &simplex_vertices) const
    {
        std::vector<float> filtration_values;
        filtration_values.reserve(simplex_vertices.size());
        for (const auto &simplex : simplex_vertices)
        {
            filtration_values.push_back(ripsFiltrationValue(simplex, distance_matrix));
        }

        std::vector<PersistenceGradient> gradients;
        gradients.reserve(persistence_pairs.size());
        for (const auto &pair : persistence_pairs)
        {
            PersistenceGradient grad;
            findCriticalSimplices(std::get<0>(pair), std::get<1>(pair), std::get<2>(pair),
                                  simplex_vertices, filtration_values, grad.birth_vertices,
                                  grad.death_vertices);
            gradients.push_back(std::move(grad));
        }
        return gradients;
    }

    MatrixXf backpropagateToPoints(const std::vector<PersistenceGradient> &pair_gradients,
                                   const MatrixXf &points, const MatrixXf &distance_matrix) const
    {
        MatrixXf point_gradients = MatrixXf::Zero(points.rows(), points.cols());
        for (const auto &pg : pair_gradients)
        {
            if (!pg.birth_vertices.empty())
            {
                const MatrixXf birth_grad = computeSimplexGradient(pg.birth_vertices, points,
                                                                   distance_matrix, pg.grad_birth);
                accumulateGradient(point_gradients, pg.birth_vertices, birth_grad);
            }
            if (!pg.death_vertices.empty())
            {
                const MatrixXf death_grad = computeSimplexGradient(pg.death_vertices, points,
                                                                   distance_matrix, pg.grad_death);
                accumulateGradient(point_gradients, pg.death_vertices, death_grad);
            }
        }
        return point_gradients;
    }

private:
    float ripsFiltrationValue(const std::vector<int> &simplex,
                              const MatrixXf &distance_matrix) const
    {
        if (simplex.size() < 2)
        {
            return 0.0f;
        }

        float value = 0.0f;
        for (size_t i = 0; i < simplex.size(); ++i)
        {
            for (size_t j = i + 1; j < simplex.size(); ++j)
            {
                const int row = simplex[i];
                const int col = simplex[j];
                if (row < 0 || col < 0 || row >= distance_matrix.rows() ||
                    col >= distance_matrix.cols())
                {
                    return std::numeric_limits<float>::infinity();
                }
                value = std::max(value, distance_matrix(row, col));
            }
        }
        return value;
    }

    void findCriticalSimplices(float birth, float death, int dim,
                               const std::vector<std::vector<int>> &all_simplices,
                               const std::vector<float> &filtration_values,
                               std::vector<int> &birth_vertices,
                               std::vector<int> &death_vertices) const
    {
        const size_t homology_dim = dim < 0 ? 0U : static_cast<size_t>(dim);
        const size_t birth_simplex_size = homology_dim + 1U;
        const size_t death_simplex_size = homology_dim + 2U;
        const bool finite_death = std::isfinite(death);
        float min_birth_diff = std::numeric_limits<float>::max();
        float min_death_diff = std::numeric_limits<float>::max();
        for (size_t i = 0; i < all_simplices.size(); ++i)
        {
            const auto &simplex = all_simplices[i];
            if (simplex.size() != birth_simplex_size && simplex.size() != death_simplex_size)
            {
                continue;
            }
            const float filt_val =
                i < filtration_values.size() ? filtration_values[i] : static_cast<float>(i);
            if (simplex.size() == birth_simplex_size)
            {
                const float birth_diff = std::abs(filt_val - birth);
                if (birth_diff < min_birth_diff)
                {
                    min_birth_diff = birth_diff;
                    birth_vertices = simplex;
                }
            }
            if (finite_death && simplex.size() == death_simplex_size)
            {
                const float death_diff = std::abs(filt_val - death);
                if (death_diff < min_death_diff)
                {
                    min_death_diff = death_diff;
                    death_vertices = simplex;
                }
            }
        }
    }

    MatrixXf computeSimplexGradient(const std::vector<int> &vertices, const MatrixXf &points,
                                    const MatrixXf &distance_matrix, float output_grad) const
    {
        MatrixXf gradients =
            MatrixXf::Zero(static_cast<EigenIndex>(vertices.size()), points.cols());
        float max_dist = 0.0f;
        std::pair<int, int> critical_edge_local = {-1, -1};
        for (size_t i = 0; i < vertices.size(); ++i)
        {
            for (size_t j = i + 1; j < vertices.size(); ++j)
            {
                const float d = distance_matrix(vertices[i], vertices[j]);
                if (d > max_dist)
                {
                    max_dist = d;
                    critical_edge_local = {static_cast<int>(i), static_cast<int>(j)};
                }
            }
        }
        if (critical_edge_local.first < 0)
        {
            return gradients;
        }
        const int i_local = critical_edge_local.first;
        const int j_local = critical_edge_local.second;
        const int vi = vertices[static_cast<size_t>(i_local)];
        const int vj = vertices[static_cast<size_t>(j_local)];
        const VectorXf diff = points.row(vi) - points.row(vj);
        const float dist = diff.norm();
        if (dist <= DISTANCE_GRADIENT_THRESHOLD)
        {
            return gradients;
        }
        const VectorXf edge_grad = diff / dist;
        gradients.row(i_local) = output_grad * edge_grad.transpose();
        gradients.row(j_local) = -output_grad * edge_grad.transpose();
        return gradients;
    }

    void accumulateGradient(MatrixXf &accum, const std::vector<int> &vertices,
                            const MatrixXf &grad) const
    {
        for (size_t i = 0; i < vertices.size(); ++i)
        {
            const int vertex = vertices[i];
            if (vertex >= 0 && vertex < accum.rows() && static_cast<int>(i) < grad.rows())
            {
                accum.row(vertex) += grad.row(static_cast<EigenIndex>(i));
            }
        }
    }
};

class AlphaGradientComputer
{
public:
    MatrixXf computeGradients(const std::vector<std::tuple<float, float, int>> &persistence_pairs,
                              const MatrixXf &points,
                              const std::vector<std::vector<int>> &delaunay_simplices) const
    {
        MatrixXf point_gradients = MatrixXf::Zero(points.rows(), points.cols());
        for (const auto &pair : persistence_pairs)
        {
            const float birth = std::get<0>(pair);
            const int dim = std::get<2>(pair);
            for (const auto &simplex : delaunay_simplices)
            {
                if (static_cast<int>(simplex.size()) != dim + 1)
                {
                    continue;
                }
                MatrixXf grad_vertices(static_cast<EigenIndex>(simplex.size()), points.cols());
                const float circum = circumradius(simplex, points, grad_vertices);
                if (std::abs(circum - birth) >= CIRCUMRADIUS_MATCH_TOLERANCE)
                {
                    continue;
                }
                for (size_t j = 0; j < simplex.size(); ++j)
                {
                    point_gradients.row(simplex[j]) +=
                        grad_vertices.row(static_cast<EigenIndex>(j));
                }
                break;
            }
        }
        return point_gradients;
    }

private:
    float circumradius(const std::vector<int> &simplex, const MatrixXf &points,
                       MatrixXf &grad_vertices) const
    {
        if (simplex.size() != 3 || points.cols() < 2)
        {
            return 0.0f;
        }
        const VectorXf a = points.row(simplex[0]);
        const VectorXf b = points.row(simplex[1]);
        const VectorXf c = points.row(simplex[2]);
        const float ab = (b - a).norm();
        const float bc = (c - b).norm();
        const float ca = (a - c).norm();
        const float cross_z = (b(0) - a(0)) * (c(1) - a(1)) - (b(1) - a(1)) * (c(0) - a(0));
        const float area = 0.5f * std::abs(cross_z);
        if (area < AREA_DEGENERACY_THRESHOLD)
        {
            return 0.0f;
        }
        const float circum = (ab * bc * ca) / (4.0f * area);
        const VectorXf da = (a - b) / ab + (a - c) / ca;
        const VectorXf db = (b - a) / ab + (b - c) / bc;
        const VectorXf dc = (c - a) / ca + (c - b) / bc;
        VectorXf dArea_a(2), dArea_b(2), dArea_c(2);
        dArea_a << c(1) - b(1), b(0) - c(0);
        dArea_b << a(1) - c(1), c(0) - a(0);
        dArea_c << b(1) - a(1), a(0) - b(0);
        const float sign = cross_z > 0.0f ? 1.0f : -1.0f;
        dArea_a *= 0.5f * sign;
        dArea_b *= 0.5f * sign;
        dArea_c *= 0.5f * sign;
        grad_vertices.row(0) = circum * (da / ab + da / ca + da / bc - dArea_a / area);
        grad_vertices.row(1) = circum * (db / ab + db / bc + db / ca - dArea_b / area);
        grad_vertices.row(2) = circum * (dc / ca + dc / bc + dc / ab - dArea_c / area);
        return circum;
    }
};

class CubicalGradientComputer
{
public:
    MatrixXf computeGradients(const std::vector<std::tuple<float, float, int>> &persistence_pairs,
                              const MatrixXf &image,
                              const std::vector<std::vector<int>> &critical_pixels) const
    {
        MatrixXf grad = MatrixXf::Zero(image.rows(), image.cols());
        for (size_t i = 0; i < persistence_pairs.size(); ++i)
        {
            if (i >= critical_pixels.size())
            {
                break;
            }
            for (const int pixel_idx : critical_pixels[i])
            {
                const EigenIndex r = static_cast<EigenIndex>(pixel_idx) / image.cols();
                const EigenIndex c = static_cast<EigenIndex>(pixel_idx) % image.cols();
                if (r >= 0 && r < image.rows() && c >= 0 && c < image.cols())
                {
                    grad(r, c) += 1.0f;
                }
            }
        }
        return grad;
    }
};

MatrixXf computeDistanceMatrix(const MatrixXf &points)
{
    MatrixXf dist(points.rows(), points.rows());
    for (EigenIndex i = 0; i < points.rows(); ++i)
    {
        for (EigenIndex j = 0; j < points.rows(); ++j)
        {
            dist(i, j) = (points.row(i) - points.row(j)).norm();
        }
    }
    return dist;
}

} // namespace

MatrixXf
computePersistenceGradients(const std::string &filtration_type,
                            const std::vector<std::tuple<float, float, int>> &persistence_pairs,
                            const MatrixXf &input_data,
                            const std::vector<float> &upstream_gradients)
{
    const int n_points = static_cast<int>(input_data.rows());
    if (filtration_type == "rips")
    {
        VRGradientComputer computer;
        MatrixXf dist_matrix = computeDistanceMatrix(input_data);
        std::vector<std::vector<int>> simplices;
        simplices.reserve(static_cast<size_t>(n_points) + persistence_pairs.size());
        for (int point = 0; point < n_points; ++point)
        {
            simplices.push_back({point});
        }
        if (n_points > 1)
        {
            for (const auto &pair : persistence_pairs)
            {
                int u = wrapToIndex(std::get<0>(pair), n_points);
                int v = wrapToIndex(std::get<1>(pair), n_points);
                if (u == v)
                {
                    v = (v + 1) % n_points;
                }
                simplices.push_back({u, v});
            }
        }
        auto pair_grads = computer.computeGradients(persistence_pairs, dist_matrix, simplices);
        for (size_t i = 0; i < pair_grads.size(); ++i)
        {
            const size_t birth_idx = i * 2;
            const size_t death_idx = birth_idx + 1;
            const float upstream_birth =
                birth_idx < upstream_gradients.size() ? upstream_gradients[birth_idx] : 1.0f;
            const float upstream_death =
                death_idx < upstream_gradients.size() ? upstream_gradients[death_idx] : 1.0f;
            pair_grads[i].grad_birth *= upstream_birth;
            pair_grads[i].grad_death *= upstream_death;
        }
        return computer.backpropagateToPoints(pair_grads, input_data, dist_matrix);
    }

    if (filtration_type == "alpha")
    {
        AlphaGradientComputer computer;
        std::vector<std::vector<int>> simplices;
        simplices.reserve(persistence_pairs.size());
        for (const auto &pair : persistence_pairs)
        {
            const int dim = std::max(1, std::get<2>(pair));
            const int u = wrapToIndex(std::get<0>(pair), n_points);
            const int v = wrapToIndex(std::get<1>(pair), n_points);
            std::vector<int> simplex = {u, v};
            if (dim >= 2 && n_points > 2)
            {
                simplex.push_back((u + v + 1) % n_points);
            }
            simplices.push_back(std::move(simplex));
        }
        MatrixXf grads = computer.computeGradients(persistence_pairs, input_data, simplices);
        grads *= averageUpstream(upstream_gradients);
        return grads;
    }

    if (filtration_type == "cubical")
    {
        CubicalGradientComputer computer;
        const int image_size = static_cast<int>(input_data.rows() * input_data.cols());
        std::vector<std::vector<int>> critical_pixels;
        critical_pixels.reserve(persistence_pairs.size());
        for (const auto &pair : persistence_pairs)
        {
            const int flat = image_size > 0
                                 ? static_cast<int>(std::fmod(std::abs(std::get<0>(pair)),
                                                              static_cast<float>(image_size)))
                                 : 0;
            critical_pixels.push_back({flat});
        }
        MatrixXf grads = computer.computeGradients(persistence_pairs, input_data, critical_pixels);
        grads *= averageUpstream(upstream_gradients);
        return grads;
    }

    return MatrixXf::Zero(input_data.rows(), input_data.cols());
}

} // namespace nerve::ml::diff
