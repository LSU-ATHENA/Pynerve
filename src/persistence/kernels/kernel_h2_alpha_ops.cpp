//
// Uses Delaunay/Alpha complex pathways for H2 computation.
// Based on: "Distributed Persistent Homology for 2D Alpha Complexes"
// (arXiv:2403.00445)
//
// Key Innovation:
// - 2D point clouds can use Delaunay triangulation (planar graph)
// - Alpha complex is a subset of Delaunay with O(n) simplices
// - VR can require higher-order simplex growth on dense inputs
// - H2-specific callers can use the sparse alpha complex metadata without
//   fabricating planar triangles as void classes.
#include "nerve/persistence/kernels/kernel_h2_alpha_ops.hpp"
#include "nerve/persistence/utils/bit_parallel_z2.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ranges>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace nerve::persistence::h2
{
namespace
{
// Geometric tolerance for Delaunay circumcircle calculations
constexpr double GEOMETRIC_TOLERANCE = 1e-10;
// Delaunay Super-Triangle Constants (Bowyer-Watson algorithm)
constexpr double SUPER_TRIANGLE_X_MULTIPLIER_LEFT = 20.0;   // Left x offset multiplier
constexpr double SUPER_TRIANGLE_X_MULTIPLIER_RIGHT = 20.0;  // Right x offset multiplier
constexpr double SUPER_TRIANGLE_Y_MULTIPLIER_BOTTOM = 10.0; // Bottom y offset multiplier
constexpr double SUPER_TRIANGLE_Y_MULTIPLIER_TOP = 20.0;    // Top y offset multiplier
constexpr int SUPER_TRI_VERTEX_1_INDEX = -1;                // First super-triangle vertex index
constexpr int SUPER_TRI_VERTEX_2_INDEX = -2;                // Second super-triangle vertex index
constexpr int SUPER_TRI_VERTEX_3_INDEX = -3;                // Third super-triangle vertex index
// 2D point for Delaunay
struct Point2D
{
    double x = 0.0;
    double y = 0.0;
    int index = 0;
};
// Triangle in Delaunay triangulation
struct DelaunayTriangle
{
    int v1 = 0;
    int v2 = 0;
    int v3 = 0; // Vertex indices
    double circumradius = 0.0;
    Point2D circumcenter;
    bool is_valid = false;
};
// Edge for lookup
struct Edge
{
    int v1, v2;
    bool operator==(const Edge &other) const
    {
        return (v1 == other.v1 && v2 == other.v2) || (v1 == other.v2 && v2 == other.v1);
    }
};
struct EdgeHash
{
    size_t operator()(const Edge &e) const
    {
        int a = std::min(e.v1, e.v2);
        int b = std::max(e.v1, e.v2);
        return std::hash<int>()(a) ^ (std::hash<int>()(b) << 1);
    }
};
// Simple Delaunay triangulation using Bowyer-Watson algorithm
// Optimized for 2D point clouds
std::vector<DelaunayTriangle> computeDelaunay2D(const std::vector<Point2D> &points)
{
    std::vector<DelaunayTriangle> triangles;
    if (points.size() < 3)
        return triangles;
    // Compute bounding box
    double min_x = points[0].x, max_x = points[0].x;
    double min_y = points[0].y, max_y = points[0].y;
    for (const auto &p : points)
    {
        min_x = std::min(min_x, p.x);
        max_x = std::max(max_x, p.x);
        min_y = std::min(min_y, p.y);
        max_y = std::max(max_y, p.y);
    }
    // Create super-triangle that contains all points
    double dx = max_x - min_x;
    double dy = max_y - min_y;
    double max_delta = std::max({dx, dy, 1.0});
    double mid_x = (min_x + max_x) / 2.0;
    double mid_y = (min_y + max_y) / 2.0;
    Point2D super_p1{mid_x - SUPER_TRIANGLE_X_MULTIPLIER_LEFT * max_delta,
                     mid_y - SUPER_TRIANGLE_Y_MULTIPLIER_BOTTOM * max_delta,
                     SUPER_TRI_VERTEX_1_INDEX};
    Point2D super_p2{mid_x + SUPER_TRIANGLE_X_MULTIPLIER_RIGHT * max_delta,
                     mid_y - SUPER_TRIANGLE_Y_MULTIPLIER_BOTTOM * max_delta,
                     SUPER_TRI_VERTEX_2_INDEX};
    Point2D super_p3{mid_x, mid_y + SUPER_TRIANGLE_Y_MULTIPLIER_TOP * max_delta,
                     SUPER_TRI_VERTEX_3_INDEX};
    // Start with super-triangle
    DelaunayTriangle super_tri{};
    super_tri.v1 = SUPER_TRI_VERTEX_1_INDEX;
    super_tri.v2 = SUPER_TRI_VERTEX_2_INDEX;
    super_tri.v3 = SUPER_TRI_VERTEX_3_INDEX;
    super_tri.is_valid = true;
    triangles.push_back(super_tri);
    // Bowyer-Watson: Insert points one by one
    for (const auto &p : points)
    {
        std::vector<int> bad_triangles;
        // Find triangles whose circumcircle contains p
        for (int i = 0; i < static_cast<int>(triangles.size()); ++i)
        {
            if (!triangles[i].is_valid)
                continue;
            // Get vertices
            int tv1 = triangles[i].v1;
            int tv2 = triangles[i].v2;
            int tv3 = triangles[i].v3;
            const Point2D &a = (tv1 == SUPER_TRI_VERTEX_1_INDEX)   ? super_p1
                               : (tv1 == SUPER_TRI_VERTEX_2_INDEX) ? super_p2
                               : (tv1 == SUPER_TRI_VERTEX_3_INDEX) ? super_p3
                                                                   : points[tv1];
            const Point2D &b = (tv2 == SUPER_TRI_VERTEX_1_INDEX)   ? super_p1
                               : (tv2 == SUPER_TRI_VERTEX_2_INDEX) ? super_p2
                               : (tv2 == SUPER_TRI_VERTEX_3_INDEX) ? super_p3
                                                                   : points[tv2];
            const Point2D &c = (tv3 == SUPER_TRI_VERTEX_1_INDEX)   ? super_p1
                               : (tv3 == SUPER_TRI_VERTEX_2_INDEX) ? super_p2
                               : (tv3 == SUPER_TRI_VERTEX_3_INDEX) ? super_p3
                                                                   : points[tv3];
            // Compute circumcircle
            double d = 2 * (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y));
            if (std::abs(d) < GEOMETRIC_TOLERANCE)
                continue;
            double ux =
                ((a.x * a.x + a.y * a.y) * (b.y - c.y) + (b.x * b.x + b.y * b.y) * (c.y - a.y) +
                 (c.x * c.x + c.y * c.y) * (a.y - b.y)) /
                d;
            double uy =
                ((a.x * a.x + a.y * a.y) * (c.x - b.x) + (b.x * b.x + b.y * b.y) * (a.x - c.x) +
                 (c.x * c.x + c.y * c.y) * (b.x - a.x)) /
                d;
            double radius_sq = (ux - a.x) * (ux - a.x) + (uy - a.y) * (uy - a.y);
            double dist_sq = (ux - p.x) * (ux - p.x) + (uy - p.y) * (uy - p.y);
            if (dist_sq < radius_sq)
            {
                bad_triangles.push_back(i);
            }
        }
        // Find boundary of polygonal hole
        std::unordered_set<Edge, EdgeHash> polygon;
        for (int tri_idx : bad_triangles)
        {
            Edge edges[3] = {{triangles[tri_idx].v1, triangles[tri_idx].v2},
                             {triangles[tri_idx].v2, triangles[tri_idx].v3},
                             {triangles[tri_idx].v3, triangles[tri_idx].v1}};
            for (const auto &e : edges)
            {
                Edge reversed{e.v2, e.v1};
                if (polygon.count(reversed))
                {
                    polygon.erase(reversed);
                }
                else
                {
                    polygon.insert(e);
                }
            }
            triangles[tri_idx].is_valid = false;
        }
        // Create new triangles from p to polygon boundary
        for (const auto &e : polygon)
        {
            DelaunayTriangle new_tri{};
            new_tri.v1 = p.index;
            new_tri.v2 = e.v1;
            new_tri.v3 = e.v2;
            new_tri.is_valid = true;
            triangles.push_back(new_tri);
        }
    }
    // Remove triangles with super-triangle vertices
    for (auto &tri : triangles)
    {
        if (tri.v1 < 0 || tri.v2 < 0 || tri.v3 < 0)
        {
            tri.is_valid = false;
        }
    }
    // Compact
    std::vector<DelaunayTriangle> result;
    for (const auto &tri : triangles)
    {
        if (tri.is_valid)
        {
            result.push_back(tri);
        }
    }
    return result;
}
// Build Alpha complex from Delaunay (filter by radius)
AlphaComplex buildAlphaComplex(const std::vector<Point2D> &points,
                               const std::vector<DelaunayTriangle> &delaunay, double max_radius)
{
    AlphaComplex alpha;
    alpha.max_radius = max_radius;
    // Add vertices (0-simplices)
    for (const auto &p : points)
    {
        Simplex s;
        s.vertices = {p.index};
        s.dimension = 0;
        s.filtration_value = 0.0;
        alpha.simplices.push_back(s);
    }
    // Add edges (1-simplices) from Delaunay
    std::unordered_set<Edge, EdgeHash> edge_set;
    for (const auto &tri : delaunay)
    {
        Edge e1{tri.v1, tri.v2};
        Edge e2{tri.v2, tri.v3};
        Edge e3{tri.v3, tri.v1};
        edge_set.insert(e1);
        edge_set.insert(e2);
        edge_set.insert(e3);
    }
    for (const auto &e : edge_set)
    {
        // Compute edge length
        const auto &p1 = points[e.v1];
        const auto &p2 = points[e.v2];
        double length = std::sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
        if (length <= max_radius)
        {
            Simplex s;
            s.vertices = {e.v1, e.v2};
            std::ranges::sort(s.vertices);
            s.dimension = 1;
            s.filtration_value = length;
            alpha.simplices.push_back(s);
        }
    }
    alpha.num_edges = static_cast<int>(edge_set.size());
    // Add triangles (2-simplices) from Delaunay
    for (const auto &tri : delaunay)
    {
        // Compute circumradius (filtration value for triangle)
        const auto &a = points[tri.v1];
        const auto &b = points[tri.v2];
        const auto &c = points[tri.v3];
        double d = 2 * (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y));
        if (std::abs(d) < GEOMETRIC_TOLERANCE)
            continue;
        double ux = ((a.x * a.x + a.y * a.y) * (b.y - c.y) + (b.x * b.x + b.y * b.y) * (c.y - a.y) +
                     (c.x * c.x + c.y * c.y) * (a.y - b.y)) /
                    d;
        double uy = ((a.x * a.x + a.y * a.y) * (c.x - b.x) + (b.x * b.x + b.y * b.y) * (a.x - c.x) +
                     (c.x * c.x + c.y * c.y) * (b.x - a.x)) /
                    d;
        double circumradius = std::sqrt((ux - a.x) * (ux - a.x) + (uy - a.y) * (uy - a.y));
        if (circumradius <= max_radius)
        {
            Simplex s;
            s.vertices = {tri.v1, tri.v2, tri.v3};
            std::ranges::sort(s.vertices);
            s.dimension = 2;
            s.filtration_value = circumradius;
            alpha.simplices.push_back(s);
        }
    }
    alpha.num_triangles = static_cast<int>(delaunay.size());
    // Sort by filtration value
    std::ranges::sort(alpha.simplices, {}, &Simplex::filtration_value);
    return alpha;
}
} // namespace
// Main H2 computation via Alpha complex
H2Result computeH2AlphaComplex(const std::vector<double> &points_data, size_t point_dim,
                               size_t num_points, const H2Config &config)
{
    H2Result result{};
    result.config = config;
    // Only works for 2D point clouds
    if (point_dim != 2 && !config.force_alpha_complex)
    {
        result.error = "Alpha complex requires 2D points (or force_alpha_complex=true)";
        return result;
    }
    if (point_dim == 0)
    {
        result.error = "Alpha complex requires at least one coordinate per point";
        return result;
    }
    if (points_data.size() / point_dim < num_points)
    {
        result.error = "Alpha complex input does not contain enough point coordinates";
        return result;
    }
    auto start_total = std::chrono::high_resolution_clock::now();
    // Convert to 2D points
    std::vector<Point2D> points;
    points.reserve(num_points);
    for (size_t i = 0; i < num_points; ++i)
    {
        Point2D p;
        p.x = points_data[i * point_dim];
        p.y = (point_dim > 1) ? points_data[i * point_dim + 1] : 0.0;
        p.index = static_cast<int>(i);
        points.push_back(p);
    }
    // Compute Delaunay triangulation
    auto start_delaunay = std::chrono::high_resolution_clock::now();
    auto delaunay = computeDelaunay2D(points);
    auto end_delaunay = std::chrono::high_resolution_clock::now();
    result.delaunay_time_ms =
        std::chrono::duration<double, std::milli>(end_delaunay - start_delaunay).count();
    result.num_delaunay_triangles = static_cast<int>(delaunay.size());
    // Build Alpha complex
    auto start_alpha = std::chrono::high_resolution_clock::now();
    auto alpha = buildAlphaComplex(points, delaunay, config.max_radius);
    auto end_alpha = std::chrono::high_resolution_clock::now();
    result.alpha_build_time_ms =
        std::chrono::duration<double, std::milli>(end_alpha - start_alpha).count();
    result.num_alpha_simplices = static_cast<int>(alpha.simplices.size());
    // Compute H2 persistence on Alpha complex.
    auto start_persist = std::chrono::high_resolution_clock::now();
    // A planar 2D alpha complex has triangles as 2-chains, not as essential H2
    // classes. Without a closed 2-cycle in a 3D complex, this path emits no H2
    // persistence pairs.
    auto end_persist = std::chrono::high_resolution_clock::now();
    result.computation_time_ms =
        std::chrono::duration<double, std::milli>(end_persist - start_persist).count();
    auto end_total = std::chrono::high_resolution_clock::now();
    result.total_time_ms =
        std::chrono::duration<double, std::milli>(end_total - start_total).count();
    // Estimate sparsification
    // VR would have O(n^3) triangles, Alpha has O(n)
    double n = static_cast<double>(num_points);
    double vr_triangles = n * n * n / 6.0; // Approximate
    double alpha_triangles = static_cast<double>(delaunay.size());
    result.sparsification_ratio = alpha_triangles > 0.0 ? vr_triangles / alpha_triangles : 0.0;
    return result;
}
// Get optimal H2 config
H2Config getOptimalH2Config(size_t num_points, size_t point_dim, double max_radius)
{
    H2Config config;
    config.max_radius = max_radius;
    // Use Alpha complex for 2D
    if (point_dim == 2)
    {
        config.use_alpha_complex = true;
        config.force_alpha_complex = false;
    }
    else
    {
        config.use_alpha_complex = false;
    }
    (void)num_points;
    config.use_bit_parallel = false;
    return config;
}
// Estimate H2 speedup
H2SpeedupEstimate estimateH2Speedup(size_t num_points, size_t point_dim)
{
    H2SpeedupEstimate estimate;
    if (point_dim == 2)
    {
        // Alpha complex: O(n) vs VR: O(n^3)
        double n = static_cast<double>(num_points);
        double vr_complexity = n * n * n;
        double alpha_complexity = n;                                // Linear
        estimate.sparsification = vr_complexity / alpha_complexity; // n^2 factor
        estimate.delaunay_speedup = 1.0;                         // Delaunay construction overhead
        estimate.total_speedup = estimate.sparsification / 10.0; // Conservative
        // Cap at reasonable values
        estimate.total_speedup = std::min(estimate.total_speedup, 100.0);
        estimate.total_speedup = std::max(estimate.total_speedup, 5.0);
    }
    else
    {
        // For non-2D, fall back to standard cohomology
        estimate.sparsification = 1.0;
        estimate.delaunay_speedup = 1.0;
        estimate.total_speedup = 2.5; // Standard cohomology gain
    }
    return estimate;
}
} // namespace nerve::persistence::h2
