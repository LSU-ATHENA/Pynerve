#include "nerve/algebra/boundary.hpp"
#include "nerve/core.hpp"
#include "nerve/filtration/vr_runtime.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <limits>
#include <mutex>
#include <random>
#include <ranges>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <vector>

using nerve::algebra::Point;

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __x86_64__
#include "nerve/cpu/x86_intrinsics.hpp"
#endif

namespace nerve::filtration::vr::parallel
{
namespace
{

constexpr size_t kMaxSpatialGridCells = 8'000'000;

void validatePoints(const std::vector<Point> &points)
{
    for (const auto &point : points)
    {
        if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z))
        {
            throw std::invalid_argument("points must have finite coordinates");
        }
    }
}

size_t checkedCellCount(int nx, int ny, int nz)
{
    const auto x = static_cast<size_t>(nx);
    const auto y = static_cast<size_t>(ny);
    const auto z = static_cast<size_t>(nz);
    if (x == 0 || y == 0 || z == 0 || x > kMaxSpatialGridCells / y ||
        x * y > kMaxSpatialGridCells / z)
    {
        throw std::invalid_argument("spatial grid would allocate too many cells");
    }
    return x * y * z;
}

} // namespace

class SpatialGrid
{
public:
    SpatialGrid(const std::vector<Point> &points, float radius)
        : points_(points)
    {
        if (points.empty())
        {
            cell_size_ = 1.0f;
            nx_ = ny_ = nz_ = 1;
            min_x_ = min_y_ = min_z_ = 0.0f;
            cells_.resize(1);
            return;
        }

        float min_x = std::numeric_limits<float>::infinity();
        float max_x = -std::numeric_limits<float>::infinity();
        float min_y = min_x, max_y = max_x;
        float min_z = min_x, max_z = max_x;

        for (const auto &p : points)
        {
            min_x = std::min(min_x, p.x);
            max_x = std::max(max_x, p.x);
            min_y = std::min(min_y, p.y);
            max_y = std::max(max_y, p.y);
            min_z = std::min(min_z, p.z);
            max_z = std::max(max_z, p.z);
        }

        cell_size_ = std::max(radius, 1.0e-6f);

        nx_ = std::max(1, static_cast<int>((max_x - min_x) / cell_size_) + 1);
        ny_ = std::max(1, static_cast<int>((max_y - min_y) / cell_size_) + 1);
        nz_ = std::max(1, static_cast<int>((max_z - min_z) / cell_size_) + 1);

        min_x_ = min_x;
        min_y_ = min_y;
        min_z_ = min_z;

        cells_.resize(checkedCellCount(nx_, ny_, nz_));
        for (size_t i = 0; i < points.size(); ++i)
        {
            int cx = static_cast<int>((points[i].x - min_x) / cell_size_);
            int cy = static_cast<int>((points[i].y - min_y) / cell_size_);
            int cz = static_cast<int>((points[i].z - min_z) / cell_size_);

            cx = std::clamp(cx, 0, nx_ - 1);
            cy = std::clamp(cy, 0, ny_ - 1);
            cz = std::clamp(cz, 0, nz_ - 1);

            int cell_idx = cx + nx_ * (cy + ny_ * cz);
            cells_[cell_idx].push_back(static_cast<int>(i));
        }
    }

    std::vector<int> getNeighbors(int point_idx) const
    {
        std::vector<int> neighbors;
        const Point &p = points_[point_idx];

        int cx = static_cast<int>((p.x - min_x_) / cell_size_);
        int cy = static_cast<int>((p.y - min_y_) / cell_size_);
        int cz = static_cast<int>((p.z - min_z_) / cell_size_);

        for (int dx = -1; dx <= 1; ++dx)
        {
            for (int dy = -1; dy <= 1; ++dy)
            {
                for (int dz = -1; dz <= 1; ++dz)
                {
                    int nx = cx + dx;
                    int ny = cy + dy;
                    int nz = cz + dz;

                    if (nx < 0 || nx >= nx_ || ny < 0 || ny >= ny_ || nz < 0 || nz >= nz_)
                        continue;

                    int cell_idx = nx + nx_ * (ny + ny_ * nz);
                    const auto &cell = cells_[cell_idx];
                    neighbors.insert(neighbors.end(), cell.begin(), cell.end());
                }
            }
        }

        return neighbors;
    }

private:
    float cell_size_;
    float min_x_, min_y_, min_z_;
    int nx_, ny_, nz_;
    const std::vector<Point> &points_;
    std::vector<std::vector<int>> cells_;
};

class LockFreeEdgeList
{
public:
    explicit LockFreeEdgeList(size_t initial_capacity = 1024)
        : edges_(initial_capacity)
        , count_{0}
    {}

    void addEdge(int u, int v, float dist)
    {
        size_t idx = count_.fetch_add(1, std::memory_order_relaxed);

        if (idx >= edges_.size())
        {
            size_t new_size = edges_.empty() ? 1024 : edges_.size() * 2;
            if (new_size < idx + 1)
            {
                new_size = idx + 1;
            }

            std::lock_guard<std::mutex> lock(resize_mutex_);
            if (idx >= edges_.size())
            {
                edges_.resize(new_size);
            }
        }

        edges_[idx] = {u, v, dist};
    }

    size_t size() const { return count_.load(std::memory_order_relaxed); }

    const Edge *data() const { return edges_.data(); }
    Edge *data() { return edges_.data(); }

    void resize(size_t n) { edges_.resize(n); }

private:
    std::vector<Edge> edges_;
    std::atomic<size_t> count_;
    mutable std::mutex resize_mutex_;
};

inline float fastDistance(const Point &a, const Point &b)
{
#ifdef __AVX512F__
    __m512 va = _mm512_set_ps(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, a.z, a.y, a.x, 0);
    __m512 vb = _mm512_set_ps(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, b.z, b.y, b.x, 0);
    __m512 diff = _mm512_sub_ps(va, vb);
    __m512 sq = _mm512_mul_ps(diff, diff);
    return _mm512_reduce_add_ps(sq);
#elif defined(__AVX__)
    __m256 va = _mm256_set_ps(0, 0, 0, 0, 0, a.z, a.y, a.x);
    __m256 vb = _mm256_set_ps(0, 0, 0, 0, 0, b.z, b.y, b.x);
    __m256 diff = _mm256_sub_ps(va, vb);
    __m256 sq = _mm256_mul_ps(diff, diff);
    __m256 sum = _mm256_hadd_ps(sq, sq);
    sum = _mm256_hadd_ps(sum, sum);
    float result[8];
    _mm256_storeu_ps(result, sum);
    return result[0] + result[4];
#else
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    float dz = a.z - b.z;
#ifdef __GNUC__
    __builtin_prefetch(&a, 0, 1);
    __builtin_prefetch(&b, 0, 1);
#endif
    return dx * dx + dy * dy + dz * dz;
#endif
}

std::vector<Edge> parallelEdgeDetection(const std::vector<Point> &points, float threshold,
                                        int num_threads)
{
    if (!std::isfinite(threshold) || threshold < 0.0f)
    {
        throw std::invalid_argument("threshold must be finite and non-negative");
    }
    validatePoints(points);
    if (points.empty())
    {
        return {};
    }

#ifdef _OPENMP
    if (num_threads > 0)
    {
        omp_set_num_threads(num_threads);
    }
#endif

    SpatialGrid grid(points, threshold);

    size_t n = points.size();
    const size_t max_edges =
        n > std::numeric_limits<size_t>::max() / 20 ? std::numeric_limits<size_t>::max() : n * 20;
    const float thresh_sq = threshold * threshold;

    int num_threads_actual = 1;
#ifdef _OPENMP
    num_threads_actual = std::max(1, omp_get_max_threads());
#endif

    std::vector<LockFreeEdgeList> thread_edges(num_threads_actual);
    for (auto &el : thread_edges)
    {
        el.resize(max_edges / num_threads_actual + 1000);
    }

#pragma omp parallel
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif

#pragma omp for schedule(dynamic, 64)
        for (int i = 0; i < static_cast<int>(n); ++i)
        {
            auto neighbors = grid.getNeighbors(i);

            for (int j : neighbors)
            {
                if (j <= i)
                    continue;

                float dist_sq = fastDistance(points[i], points[j]);

                if (dist_sq <= thresh_sq)
                {
                    thread_edges[tid].addEdge(i, j, std::sqrt(dist_sq));
                }
            }
        }
    }

    std::vector<Edge> all_edges;
    size_t total = 0;
    for (const auto &el : thread_edges)
    {
        total += el.size();
    }
    all_edges.reserve(total);

    for (auto &el : thread_edges)
    {
        size_t count = el.size();
        const Edge *data = el.data();
        all_edges.insert(all_edges.end(), data, data + count);
    }

    std::sort(all_edges.begin(), all_edges.end());
    all_edges.erase(std::unique(all_edges.begin(), all_edges.end(),
                                [](const Edge &lhs, const Edge &rhs) {
                                    return lhs.u == rhs.u && lhs.v == rhs.v;
                                }),
                    all_edges.end());

    return all_edges;
}

VRComplex buildParallelVRComplex(const std::vector<Point> &points, float max_distance, int max_dim)
{
    if (!std::isfinite(max_distance) || max_distance < 0.0f)
    {
        throw std::invalid_argument("max_distance must be finite and non-negative");
    }
    if (max_dim < 0)
    {
        throw std::invalid_argument("max_dim must be non-negative");
    }

    VRComplex complex;
    if (max_dim == 0 || points.empty())
    {
        validatePoints(points);
        return complex;
    }

    complex.edges = parallelEdgeDetection(points, max_distance);

    if (max_dim >= 2)
    {
        std::unordered_map<int, std::vector<int>> adjacency;

        for (const auto &e : complex.edges)
        {
            adjacency[e.u].push_back(e.v);
            adjacency[e.v].push_back(e.u);
        }

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 64)
#endif
            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(adjacency.bucket_count()); ++i)
        {
            for (auto it = adjacency.begin(i); it != adjacency.end(i); ++it)
            {
                auto &neighbors = it->second;
                std::sort(neighbors.begin(), neighbors.end());
            }
        }

        std::vector<std::vector<std::vector<int>>> thread_triangles;
        int num_threads = 1;
#ifdef _OPENMP
        num_threads = omp_get_max_threads();
#endif
        thread_triangles.resize(num_threads);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 128)
#endif
            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(complex.edges.size()); ++i)
        {
            const auto &e = complex.edges[i];
            int u = e.u, v = e.v;

            const auto &u_neighbors = adjacency[u];
            const auto &v_neighbors = adjacency[v];

            int thread_id = 0;
#ifdef _OPENMP
            thread_id = omp_get_thread_num();
#endif

            const auto *small_list = &u_neighbors;
            const auto *large_list = &v_neighbors;
            if (u_neighbors.size() > v_neighbors.size())
            {
                std::swap(small_list, large_list);
            }

            for (int w : *small_list)
            {
                if (w > v && std::binary_search(large_list->begin(), large_list->end(), w))
                {
                    thread_triangles[thread_id].push_back({u, v, w});
                }
            }
        }

        size_t total_triangles = 0;
        for (const auto &local : thread_triangles)
        {
            total_triangles += local.size();
        }
        complex.triangles.reserve(total_triangles);

        for (const auto &local : thread_triangles)
        {
            complex.triangles.insert(complex.triangles.end(), local.begin(), local.end());
        }
        std::ranges::sort(complex.triangles);
        complex.triangles.erase(std::unique(complex.triangles.begin(), complex.triangles.end()),
                                complex.triangles.end());
    }

    return complex;
}

VRBenchmark benchmarkParallelVR(int n_points, float threshold, int num_threads)
{
    if (n_points < 0 || !std::isfinite(threshold) || threshold < 0.0f || num_threads < 0)
    {
        throw std::invalid_argument("benchmarkParallelVR received an invalid argument");
    }

    VRBenchmark bench;
    bench.num_points = n_points;
    bench.num_threads = num_threads > 0 ? num_threads : 4;

    std::vector<Point> points;
    points.reserve(n_points);
    std::mt19937 rng(0x56525048U);
    std::uniform_real_distribution<float> unit(0.0f, 1.0f);
    for (int i = 0; i < n_points; ++i)
    {
        points.push_back({unit(rng), unit(rng), unit(rng)});
    }

    auto start_seq = std::chrono::high_resolution_clock::now();
    std::vector<Edge> edges_seq;
    for (int i = 0; i < n_points; ++i)
    {
        for (int j = i + 1; j < n_points; ++j)
        {
            float dist = std::sqrt(fastDistance(points[i], points[j]));
            if (dist <= threshold)
            {
                edges_seq.push_back({i, j, dist});
            }
        }
    }
    auto end_seq = std::chrono::high_resolution_clock::now();
    bench.sequential_time_ms =
        std::chrono::duration<double, std::milli>(end_seq - start_seq).count();

    auto start_par = std::chrono::high_resolution_clock::now();
    auto edges_par = parallelEdgeDetection(points, threshold, num_threads);
    auto end_par = std::chrono::high_resolution_clock::now();
    bench.parallel_time_ms = std::chrono::duration<double, std::milli>(end_par - start_par).count();

    bench.speedup =
        bench.parallel_time_ms > 0.0 ? bench.sequential_time_ms / bench.parallel_time_ms : 0.0;
    bench.num_edges = edges_par.size();

    return bench;
}

} // namespace nerve::filtration::vr::parallel
