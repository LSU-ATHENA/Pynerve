// Delaunay3D Implementation
// 3D Delaunay triangulation using a deterministic Bowyer-Watson pipeline.

#include "nerve/persistence/core/flood_complex.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace nerve::persistence
{

namespace
{

constexpr int DELAUNAY_MIN_TETRAHEDRON_POINTS = 4;
constexpr int DELAUNAY_BOWYER_WATSON_THRESHOLD = 100;
constexpr double CIRCUMSPHERE_DEGENERACY_EPS = 1e-12;
constexpr double CIRCUMSPHERE_CONTAINS_EPS = 1e-10;

struct FaceKey
{
    std::array<int, 3> vertices{};

    bool operator==(const FaceKey &other) const = default;
};

struct FaceKeyHash
{
    std::size_t operator()(const FaceKey &key) const noexcept
    {
        std::size_t h = 0;
        for (const int v : key.vertices)
        {
            h ^= std::hash<int>{}(v) + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        return h;
    }
};

struct TetKeyHash
{
    std::size_t operator()(const std::array<int, 4> &key) const noexcept
    {
        std::size_t h = 0;
        for (const int v : key)
        {
            h ^= std::hash<int>{}(v) + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        return h;
    }
};

struct Circumsphere
{
    std::array<double, 3> center{};
    double radius_sq = 0.0;
};

FaceKey makeFaceKey(int a, int b, int c)
{
    FaceKey key{{a, b, c}};
    std::sort(key.vertices.begin(), key.vertices.end());
    return key;
}

double determinant3x3(const std::array<std::array<double, 3>, 3> &m)
{
    return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
           m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
           m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
}

bool solve3x3(const std::array<std::array<double, 3>, 3> &a, const std::array<double, 3> &b,
              std::array<double, 3> *x)
{
    const double det_a = determinant3x3(a);
    if (std::abs(det_a) < CIRCUMSPHERE_DEGENERACY_EPS || x == nullptr)
    {
        return false;
    }

    std::array<std::array<double, 3>, 3> mx = a;
    std::array<std::array<double, 3>, 3> my = a;
    std::array<std::array<double, 3>, 3> mz = a;
    for (int r = 0; r < 3; ++r)
    {
        mx[r][0] = b[r];
        my[r][1] = b[r];
        mz[r][2] = b[r];
    }

    (*x)[0] = determinant3x3(mx) / det_a;
    (*x)[1] = determinant3x3(my) / det_a;
    (*x)[2] = determinant3x3(mz) / det_a;
    return true;
}

double squaredDistance(const std::array<double, 3> &p, const std::array<double, 3> &q)
{
    const double dx = p[0] - q[0];
    const double dy = p[1] - q[1];
    const double dz = p[2] - q[2];
    return dx * dx + dy * dy + dz * dz;
}

std::array<double, 3> point3(const IndexedPoint &p)
{
    return {p.coords[0], p.coords[1], p.coords[2]};
}

std::optional<Circumsphere> computeCircumsphere(const IndexedPoint &a, const IndexedPoint &b,
                                                const IndexedPoint &c, const IndexedPoint &d)
{
    const auto pa = point3(a);
    const auto pb = point3(b);
    const auto pc = point3(c);
    const auto pd = point3(d);

    std::array<std::array<double, 3>, 3> lhs{{
        {{2.0 * (pb[0] - pa[0]), 2.0 * (pb[1] - pa[1]), 2.0 * (pb[2] - pa[2])}},
        {{2.0 * (pc[0] - pa[0]), 2.0 * (pc[1] - pa[1]), 2.0 * (pc[2] - pa[2])}},
        {{2.0 * (pd[0] - pa[0]), 2.0 * (pd[1] - pa[1]), 2.0 * (pd[2] - pa[2])}},
    }};

    const auto norm_sq = [](const std::array<double, 3> &p) {
        return p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
    };
    const std::array<double, 3> rhs{
        norm_sq(pb) - norm_sq(pa),
        norm_sq(pc) - norm_sq(pa),
        norm_sq(pd) - norm_sq(pa),
    };

    std::array<double, 3> solved_center{};
    if (!solve3x3(lhs, rhs, &solved_center))
    {
        return std::nullopt;
    }

    return Circumsphere{solved_center, squaredDistance(solved_center, pa)};
}

} // namespace

Delaunay3D::Delaunay3D() = default;

std::vector<Delaunay3D::Tetrahedron> Delaunay3D::compute(const std::vector<IndexedPoint> &points)
{
    if (points.size() < DELAUNAY_MIN_TETRAHEDRON_POINTS)
    {
        return {};
    }
    if (points.size() <= DELAUNAY_BOWYER_WATSON_THRESHOLD)
    {
        return bowyerWatson(points);
    }
    return divideAndConquer(points);
}

std::vector<Delaunay3D::Tetrahedron>
Delaunay3D::bowyerWatson(const std::vector<IndexedPoint> &points)
{
    std::vector<Tetrahedron> tetrahedra;
    tetrahedra.reserve(points.size() * 8);

    double min_x = points[0].coords[0];
    double max_x = points[0].coords[0];
    double min_y = points[0].coords[1];
    double max_y = points[0].coords[1];
    double min_z = points[0].coords[2];
    double max_z = points[0].coords[2];
    for (const auto &p : points)
    {
        min_x = std::min(min_x, p.coords[0]);
        max_x = std::max(max_x, p.coords[0]);
        min_y = std::min(min_y, p.coords[1]);
        max_y = std::max(max_y, p.coords[1]);
        min_z = std::min(min_z, p.coords[2]);
        max_z = std::max(max_z, p.coords[2]);
    }

    const double dx = max_x - min_x;
    const double dy = max_y - min_y;
    const double dz = max_z - min_z;
    const double max_dim = std::max({dx, dy, dz, 1.0});
    const double cx = (min_x + max_x) * 0.5;
    const double cy = (min_y + max_y) * 0.5;
    const double cz = (min_z + max_z) * 0.5;

    std::vector<IndexedPoint> working_points = points;
    const int super_v0 = static_cast<int>(working_points.size());
    working_points.push_back({{cx - 20.0 * max_dim, cy - 20.0 * max_dim, cz - 20.0 * max_dim}, -1});
    const int super_v1 = static_cast<int>(working_points.size());
    working_points.push_back({{cx + 20.0 * max_dim, cy, cz - 20.0 * max_dim}, -2});
    const int super_v2 = static_cast<int>(working_points.size());
    working_points.push_back({{cx, cy + 20.0 * max_dim, cz + 20.0 * max_dim}, -3});
    const int super_v3 = static_cast<int>(working_points.size());
    working_points.push_back({{cx, cy - 20.0 * max_dim, cz + 20.0 * max_dim}, -4});

    Tetrahedron super_tet{};
    super_tet.v[0] = super_v0;
    super_tet.v[1] = super_v1;
    super_tet.v[2] = super_v2;
    super_tet.v[3] = super_v3;
    super_tet.valid = true;
    tetrahedra.push_back(super_tet);

    for (size_t point_idx = 0; point_idx < points.size(); ++point_idx)
    {
        std::vector<size_t> bad_indices;
        bad_indices.reserve(tetrahedra.size());
        for (size_t t = 0; t < tetrahedra.size(); ++t)
        {
            const Tetrahedron &tet = tetrahedra[t];
            if (!tet.valid)
            {
                continue;
            }
            if (pointInCircumsphere(working_points[point_idx], working_points[tet.v[0]],
                                    working_points[tet.v[1]], working_points[tet.v[2]],
                                    working_points[tet.v[3]]))
            {
                bad_indices.push_back(t);
            }
        }
        if (bad_indices.empty())
        {
            continue;
        }

        std::unordered_map<FaceKey, int, FaceKeyHash> face_counts;
        face_counts.reserve(bad_indices.size() * 4);
        for (const size_t idx : bad_indices)
        {
            Tetrahedron &tet = tetrahedra[idx];
            tet.valid = false;
            ++face_counts[makeFaceKey(tet.v[0], tet.v[1], tet.v[2])];
            ++face_counts[makeFaceKey(tet.v[0], tet.v[1], tet.v[3])];
            ++face_counts[makeFaceKey(tet.v[0], tet.v[2], tet.v[3])];
            ++face_counts[makeFaceKey(tet.v[1], tet.v[2], tet.v[3])];
        }

        std::vector<Tetrahedron> retained;
        retained.reserve(tetrahedra.size() + face_counts.size());
        for (const auto &tet : tetrahedra)
        {
            if (tet.valid)
            {
                retained.push_back(tet);
            }
        }
        tetrahedra.swap(retained);

        for (const auto &[face, count] : face_counts)
        {
            if (count != 1)
            {
                continue;
            }
            Tetrahedron new_tet{};
            new_tet.v[0] = face.vertices[0];
            new_tet.v[1] = face.vertices[1];
            new_tet.v[2] = face.vertices[2];
            new_tet.v[3] = static_cast<int>(point_idx);
            new_tet.valid = true;
            tetrahedra.push_back(new_tet);
        }
    }

    std::vector<Tetrahedron> result;
    result.reserve(tetrahedra.size());
    std::unordered_set<std::array<int, 4>, TetKeyHash> seen;
    seen.reserve(tetrahedra.size());
    for (const auto &tet : tetrahedra)
    {
        if (!tet.valid)
        {
            continue;
        }
        if (tet.v[0] >= super_v0 || tet.v[1] >= super_v0 || tet.v[2] >= super_v0 ||
            tet.v[3] >= super_v0)
        {
            continue;
        }

        std::array<int, 4> key{{tet.v[0], tet.v[1], tet.v[2], tet.v[3]}};
        std::sort(key.begin(), key.end());
        if (!seen.insert(key).second)
        {
            continue;
        }

        Tetrahedron canonical{};
        canonical.v[0] = key[0];
        canonical.v[1] = key[1];
        canonical.v[2] = key[2];
        canonical.v[3] = key[3];
        canonical.valid = true;
        result.push_back(canonical);
    }
    return result;
}

std::vector<Delaunay3D::Tetrahedron>
Delaunay3D::divideAndConquer(const std::vector<IndexedPoint> &points)
{
    // Deterministic implementation: Bowyer-Watson remains the canonical production
    // path until a specialized parallel divide-and-conquer variant is enabled.
    return bowyerWatson(points);
}

bool Delaunay3D::pointInCircumsphere(const IndexedPoint &p, const IndexedPoint &a,
                                     const IndexedPoint &b, const IndexedPoint &c,
                                     const IndexedPoint &d)
{
    const auto sphere = computeCircumsphere(a, b, c, d);
    if (!sphere.has_value())
    {
        return false;
    }
    const double point_sq = squaredDistance(point3(p), sphere->center);
    const double tolerance = std::max(1.0, sphere->radius_sq) * CIRCUMSPHERE_CONTAINS_EPS;
    return point_sq <= sphere->radius_sq + tolerance;
}

} // namespace nerve::persistence
