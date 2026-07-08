
#include "nerve/algebra/simplex.hpp"
#include "nerve/simd/simd_base.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace nerve::algebra
{

namespace
{
constinit const double kGeometricTol = 1e-10;
constexpr double BARYCENTRIC_WEIGHT_TOLERANCE = 1e-8;
constexpr double BARYCENTRIC_SUM_TOLERANCE = 1e-6;
constexpr double BARYCENTRIC_POINT_TOLERANCE = 1e-6;

[[nodiscard]] Size inferPointDimension(const std::vector<Index> &simplex_vertices,
                                       const core::ownership_utils::PointView &coords,
                                       Size min_points)
{
    if (simplex_vertices.size() <= 2)
    {
        return coords.size() % min_points == 0 ? coords.size() / min_points : 0;
    }
    const Size min_affine_dim = simplex_vertices.size() > 1 ? simplex_vertices.size() - 1 : 1;
    Size selected_dim = 0;
    for (Size point_dim = min_affine_dim; point_dim <= coords.size(); ++point_dim)
    {
        if (coords.size() % point_dim != 0)
        {
            continue;
        }
        const Size num_points = coords.size() / point_dim;
        if (num_points < min_points)
        {
            continue;
        }
        selected_dim = point_dim;
        break;
    }
    if (selected_dim != 0)
    {
        return selected_dim;
    }
    return coords.size() % min_points == 0 ? coords.size() / min_points : 0;
}

[[nodiscard]] std::vector<std::vector<double>>
buildCoordinateTable(const std::vector<Index> &simplex_vertices,
                     const core::ownership_utils::PointView &coords)
{
    if (simplex_vertices.empty() || coords.empty())
    {
        return {};
    }
    const auto max_vertex = *std::max_element(simplex_vertices.begin(), simplex_vertices.end());
    if (max_vertex < 0)
    {
        return {};
    }
    const Size min_points = static_cast<Size>(max_vertex) + 1;
    const Size point_dim = inferPointDimension(simplex_vertices, coords, min_points);
    if (point_dim == 0 || coords.size() % point_dim != 0)
    {
        return {};
    }
    const Size num_points = coords.size() / point_dim;
    if (num_points < min_points)
    {
        return {};
    }
    std::vector<std::vector<double>> table(num_points, std::vector<double>(point_dim, 0.0));
    const double *data = coords.data();
    for (Size i = 0; i < num_points; ++i)
    {
        for (Size d = 0; d < point_dim; ++d)
        {
            if (!std::isfinite(data[i * point_dim + d]))
            {
                throw std::invalid_argument("simplex coordinates must be finite");
            }
            table[i][d] = data[i * point_dim + d];
        }
    }
    return table;
}
bool solveLinearSystem(const std::vector<std::vector<double>> &matrix,
                       const std::vector<double> &rhs, std::vector<double> *solution)
{
    const Size n = matrix.size();
    if (n == 0 || rhs.size() != n || solution == nullptr)
    {
        return false;
    }
    for (const auto &row : matrix)
    {
        if (row.size() != n)
        {
            return false;
        }
    }
    std::vector<std::vector<double>> aug(n, std::vector<double>(n + 1, 0.0));
    for (Size i = 0; i < n; ++i)
    {
        for (Size j = 0; j < n; ++j)
        {
            aug[i][j] = matrix[i][j];
        }
        aug[i][n] = rhs[i];
    }
    for (Size i = 0; i < n; ++i)
    {
        Size pivot = i;
        for (Size r = i + 1; r < n; ++r)
        {
            if (std::abs(aug[r][i]) > std::abs(aug[pivot][i]))
            {
                pivot = r;
            }
        }
        if (std::abs(aug[pivot][i]) <= kGeometricTol)
        {
            return false;
        }
        if (pivot != i)
        {
            std::swap(aug[pivot], aug[i]);
        }
        const double inv_pivot = 1.0 / aug[i][i];
        for (Size c = i; c <= n; ++c)
        {
            aug[i][c] *= inv_pivot;
        }
        for (Size r = 0; r < n; ++r)
        {
            if (r == i)
            {
                continue;
            }
            const double factor = aug[r][i];
            if (std::abs(factor) <= kGeometricTol)
            {
                continue;
            }
            for (Size c = i; c <= n; ++c)
            {
                aug[r][c] -= factor * aug[i][c];
            }
        }
    }
    solution->assign(n, 0.0);
    for (Size i = 0; i < n; ++i)
    {
        (*solution)[i] = aug[i][n];
    }
    return true;
}
std::vector<double> centroid(const std::vector<Index> &simplex_vertices,
                             const std::vector<std::vector<double>> &coord_table)
{
    if (simplex_vertices.empty())
    {
        return {};
    }
    const Size dim = coord_table[static_cast<Size>(simplex_vertices[0])].size();
    std::vector<double> center(dim, 0.0);
    for (const Index vertex : simplex_vertices)
    {
        const auto &p = coord_table[static_cast<Size>(vertex)];
        nerve::simd::simd_add(center.data(), p.data(), dim);
    }
    const double inv_count = 1.0 / static_cast<double>(simplex_vertices.size());
    for (double &value : center)
    {
        value *= inv_count;
    }
    return center;
}
bool computeBarycentricWeights(const std::vector<Index> &simplex_vertices,
                               const std::vector<std::vector<double>> &coord_table,
                               const core::ownership_utils::PointView &point,
                               std::vector<double> *weights)
{
    if (weights == nullptr || simplex_vertices.empty() || point.empty())
    {
        return false;
    }
    for (const Index vertex : simplex_vertices)
    {
        if (vertex < 0 || static_cast<Size>(vertex) >= coord_table.size())
        {
            return false;
        }
    }
    const auto &v0 = coord_table[static_cast<Size>(simplex_vertices[0])];
    const Size ambient_dim = v0.size();
    if (ambient_dim == 0 || point.size() != ambient_dim)
    {
        return false;
    }
    const Size m = simplex_vertices.size();
    if (m == 1)
    {
        for (Size d = 0; d < ambient_dim; ++d)
        {
            if (std::abs(point[d] - v0[d]) > kGeometricTol)
            {
                return false;
            }
        }
        weights->assign(1, 1.0);
        return true;
    }
    const Size vars = m - 1;
    if (ambient_dim < vars)
    {
        return false;
    }
    std::vector<std::vector<double>> ata(vars, std::vector<double>(vars, 0.0));
    std::vector<double> atb(vars, 0.0);
    for (Size r = 0; r < ambient_dim; ++r)
    {
        const double br = point[r] - v0[r];
        for (Size j = 0; j < vars; ++j)
        {
            const auto &vj = coord_table[static_cast<Size>(simplex_vertices[j + 1])];
            const double aj = vj[r] - v0[r];
            atb[j] += aj * br;
            for (Size k = 0; k < vars; ++k)
            {
                const auto &vk = coord_table[static_cast<Size>(simplex_vertices[k + 1])];
                const double ak = vk[r] - v0[r];
                ata[j][k] += aj * ak;
            }
        }
    }
    std::vector<double> x;
    if (!solveLinearSystem(ata, atb, &x))
    {
        return false;
    }
    weights->assign(m, 0.0);
    double sum_tail = 0.0;
    for (Size i = 1; i < m; ++i)
    {
        (*weights)[i] = x[i - 1];
        sum_tail += x[i - 1];
    }
    (*weights)[0] = 1.0 - sum_tail;
    return true;
}
double squaredDistance(const std::vector<double> &lhs, const std::vector<double> &rhs)
{
    if (lhs.size() != rhs.size())
    {
        throw std::invalid_argument("simplex coordinate dimensions must match");
    }
    const Size dim = lhs.size();
    double acc = nerve::simd::simd_sqdiff_sum(lhs.data(), rhs.data(), dim);
    if (!std::isfinite(acc))
    {
        throw std::overflow_error("squared distance overflowed");
    }
    return acc;
}
bool strictContractUnsatisfied(const core::DeterminismContract &contract)
{
    return contract.level == core::DeterminismLevel::STRICT &&
           !core::DeterminismEnforcer::canSatisfyContract(contract);
}
} // namespace
double Simplex::volume(const core::ownership_utils::PointView &coords,
                       const core::DeterminismContract &contract) const
{
    if (strictContractUnsatisfied(contract))
    {
        return 0.0;
    }
    const auto impl =
        static_cast<double (Simplex::*)(const core::ownership_utils::PointView &) const>(
            &Simplex::volume);
    return (this->*impl)(coords);
}
double Simplex::volume(const core::ownership_utils::PointView &coords) const
{
    const auto coord_table = buildCoordinateTable(vertices_, coords);
    if (coord_table.empty())
    {
        return 0.0;
    }
    return simplexVolume(coord_table);
}
double Simplex::barycentricCoordinate(const core::ownership_utils::PointView &point,
                                      const core::ownership_utils::PointView &coords,
                                      const core::DeterminismContract &contract) const
{
    if (strictContractUnsatisfied(contract))
    {
        return 0.0;
    }
    const auto impl = static_cast<double (Simplex::*)(
        const core::ownership_utils::PointView &, const core::ownership_utils::PointView &) const>(
        &Simplex::barycentricCoordinate);
    return (this->*impl)(point, coords);
}
double Simplex::barycentricCoordinate(const core::ownership_utils::PointView &point,
                                      const core::ownership_utils::PointView &coords) const
{
    const auto coord_table = buildCoordinateTable(vertices_, coords);
    if (coord_table.empty())
    {
        return 0.0;
    }
    std::vector<double> weights;
    if (!computeBarycentricWeights(vertices_, coord_table, point, &weights) || weights.empty())
    {
        return 0.0;
    }
    return weights[0];
}
std::vector<double> Simplex::circumcenter(const core::ownership_utils::PointView &coords,
                                          const core::DeterminismContract &contract) const
{
    if (strictContractUnsatisfied(contract))
    {
        return {};
    }
    const auto impl = static_cast<std::vector<double> (Simplex::*)(
        const core::ownership_utils::PointView &) const>(&Simplex::circumcenter);
    return (this->*impl)(coords);
}
std::vector<double> Simplex::circumcenter(const core::ownership_utils::PointView &coords) const
{
    const auto coord_table = buildCoordinateTable(vertices_, coords);
    if (coord_table.empty() || vertices_.empty())
    {
        return {};
    }
    for (const Index vertex : vertices_)
    {
        if (vertex < 0 || static_cast<Size>(vertex) >= coord_table.size())
        {
            return {};
        }
    }
    const auto &v0 = coord_table[static_cast<Size>(vertices_[0])];
    const Size ambient_dim = v0.size();
    if (ambient_dim == 0)
    {
        return {};
    }
    if (vertices_.size() == 1)
    {
        return v0;
    }
    const Size equations = vertices_.size() - 1;
    std::vector<std::vector<double>> a(equations, std::vector<double>(ambient_dim, 0.0));
    std::vector<double> b(equations, 0.0);
    double norm_v0 = 0.0;
    for (Size d = 0; d < ambient_dim; ++d)
    {
        norm_v0 += v0[d] * v0[d];
    }
    for (Size i = 0; i < equations; ++i)
    {
        const auto &vi = coord_table[static_cast<Size>(vertices_[i + 1])];
        double norm_vi = 0.0;
        for (Size d = 0; d < ambient_dim; ++d)
        {
            a[i][d] = 2.0 * (vi[d] - v0[d]);
            norm_vi += vi[d] * vi[d];
        }
        b[i] = norm_vi - norm_v0;
    }
    std::vector<std::vector<double>> ata(ambient_dim, std::vector<double>(ambient_dim, 0.0));
    std::vector<double> atb(ambient_dim, 0.0);
    for (Size r = 0; r < equations; ++r)
    {
        for (Size i = 0; i < ambient_dim; ++i)
        {
            atb[i] += a[r][i] * b[r];
            for (Size j = 0; j < ambient_dim; ++j)
            {
                ata[i][j] += a[r][i] * a[r][j];
            }
        }
    }
    std::vector<double> center;
    if (!solveLinearSystem(ata, atb, &center))
    {
        return centroid(vertices_, coord_table);
    }
    return center;
}
double Simplex::circumradius(const core::ownership_utils::PointView &coords,
                             const core::DeterminismContract &contract) const
{
    if (strictContractUnsatisfied(contract))
    {
        return 0.0;
    }
    const auto impl =
        static_cast<double (Simplex::*)(const core::ownership_utils::PointView &) const>(
            &Simplex::circumradius);
    return (this->*impl)(coords);
}
double Simplex::circumradius(const core::ownership_utils::PointView &coords) const
{
    const auto coord_table = buildCoordinateTable(vertices_, coords);
    if (coord_table.empty() || vertices_.empty())
    {
        return 0.0;
    }
    const auto circumcenter_impl = static_cast<std::vector<double> (Simplex::*)(
        const core::ownership_utils::PointView &) const>(&Simplex::circumcenter);
    const auto center = (this->*circumcenter_impl)(coords);
    if (center.empty())
    {
        return 0.0;
    }
    const auto &v0 = coord_table[static_cast<Size>(vertices_[0])];
    const double radius_sq = squaredDistance(center, v0);
    if (!std::isfinite(radius_sq))
    {
        throw std::overflow_error("circumradius calculation produced a non-finite value");
    }
    return std::sqrt(radius_sq);
}
bool Simplex::containsPoint(const core::ownership_utils::PointView &point,
                            const core::ownership_utils::PointView &coords,
                            const core::DeterminismContract &contract) const
{
    if (strictContractUnsatisfied(contract))
    {
        return false;
    }
    const auto impl = static_cast<bool (Simplex::*)(
        const core::ownership_utils::PointView &, const core::ownership_utils::PointView &) const>(
        &Simplex::containsPoint);
    return (this->*impl)(point, coords);
}
bool Simplex::containsPoint(const core::ownership_utils::PointView &point,
                            const core::ownership_utils::PointView &coords) const
{
    const auto coord_table = buildCoordinateTable(vertices_, coords);
    if (coord_table.empty())
    {
        return false;
    }
    std::vector<double> weights;
    if (!computeBarycentricWeights(vertices_, coord_table, point, &weights))
    {
        return false;
    }
    double sum = 0.0;
    for (double weight : weights)
    {
        if (weight < -BARYCENTRIC_WEIGHT_TOLERANCE || weight > 1.0 + BARYCENTRIC_WEIGHT_TOLERANCE)
        {
            return false;
        }
        sum += weight;
    }
    if (std::abs(sum - 1.0) > BARYCENTRIC_SUM_TOLERANCE)
    {
        return false;
    }
    const Size dim = point.size();
    std::vector<double> reconstructed(dim, 0.0);
    for (Size i = 0; i < vertices_.size(); ++i)
    {
        const auto &vertex = coord_table[static_cast<Size>(vertices_[i])];
        for (Size d = 0; d < dim; ++d)
        {
            reconstructed[d] += weights[i] * vertex[d];
        }
    }
    for (Size d = 0; d < dim; ++d)
    {
        if (std::abs(reconstructed[d] - point[d]) > BARYCENTRIC_POINT_TOLERANCE)
        {
            return false;
        }
    }
    return true;
}
} // namespace nerve::algebra
