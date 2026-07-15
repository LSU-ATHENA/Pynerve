
#pragma once
#include "math/field_matrix.hpp"
#include "math/finite_field.hpp"
#include "math/matrix_reduction.hpp"
#include "math/precision_manager.hpp"
#include "nerve/algebra/boundary.hpp"
#include "nerve/algebra/complex.hpp"
#include "nerve/error/error_registry.hpp"
#include "nerve/persistence/vr/vr_fast_ops.hpp"
#include "persistence/mathematically_correct_reduction.hpp"
#include "threading/concurrent_vector.hpp"
#include "threading/thread_pool.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <future>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
namespace nerve::persistence
{
using ::nerve::math::FiniteField;

class FieldAwareVietorisRips
{
private:
    int field_characteristic_ = 2;
    std::unique_ptr<threading::ThreadPool> thread_pool_;
    template <typename Target, typename Source>
    static error::Result<Target> forwardError(const error::Result<Source> &result)
    {
        return error::Result<Target>::err(static_cast<error::TDAErrorCode>(result.error().value()),
                                          std::string(result.detail()), result.where());
    }
    template <typename Field>
    struct Edge
    {
        int u;
        int v;
        double filtration;
        Field coefficient;
        Edge(int u_val, int v_val, double filtration_value, Field coefficient_value)
            : u(u_val)
            , v(v_val)
            , filtration(filtration_value)
            , coefficient(coefficient_value)
        {}
    };
    template <typename Field>
    struct Triangle
    {
        int a;
        int b;
        int c;
        double filtration;
        Field coefficient;
        int e0;
        int e1;
        int e2;
        Triangle(int a_val, int b_val, int c_val, double filtration_value, Field coefficient_value,
                 int e0_idx, int e1_idx, int e2_idx)
            : a(a_val)
            , b(b_val)
            , c(c_val)
            , filtration(filtration_value)
            , coefficient(coefficient_value)
            , e0(e0_idx)
            , e1(e1_idx)
            , e2(e2_idx)
        {}
    };
    template <typename Field>
    error::Result<std::vector<Edge<Field>>>
    computeEdgesWithField(core::BufferView<const double> points, Size point_dim,
                          double max_radius_sq)
    {
        try
        {
            const Size n = points.size() / point_dim;
            if (n == 0)
            {
                return error::Result<std::vector<Edge<Field>>>::ok({});
            }
            threading::ConcurrentVector<Edge<Field>> edges;
            const Size numThreads = thread_pool_ ? thread_pool_->numThreads() : 1;
            const Size chunk_count = std::max<Size>(1, std::min(numThreads, n));
            const Size chunk_size = (n + chunk_count - 1) / chunk_count;
            auto compute_range = [&points, point_dim, n, max_radius_sq, &edges](Size start,
                                                                                Size end) {
                for (Size i = start; i < end; ++i)
                {
                    const double *pi = points.data() + i * point_dim;
                    for (Size j = i + 1; j < n; ++j)
                    {
                        const double *pj = points.data() + j * point_dim;
                        double dist_sq = 0.0;
                        for (Size k = 0; k < point_dim; ++k)
                        {
                            double diff = pi[k] - pj[k];
                            if (!std::isfinite(diff))
                            {
                                throw std::overflow_error(
                                    "field-aware edge distance difference overflow");
                            }
                            const double contribution = diff * diff;
                            if (!std::isfinite(contribution))
                            {
                                throw std::overflow_error(
                                    "field-aware edge distance contribution overflow");
                            }
                            dist_sq += contribution;
                            if (!std::isfinite(dist_sq))
                            {
                                throw std::overflow_error(
                                    "field-aware edge distance accumulation overflow");
                            }
                        }
                        if (dist_sq <= max_radius_sq)
                        {
                            double dist = std::sqrt(dist_sq);
                            const int u = static_cast<int>(i);
                            const int v = static_cast<int>(j);
                            edges.push_back(Edge<Field>(u, v, dist, Field::one()));
                        }
                    }
                }
            };
            if (!thread_pool_ || chunk_count == 1)
            {
                compute_range(0, n);
                return error::Result<std::vector<Edge<Field>>>::ok(edges.copy());
            }
            std::vector<std::future<void>> futures;
            futures.reserve(chunk_count);
            for (Size t = 0; t < chunk_count; ++t)
            {
                const Size start = t * chunk_size;
                const Size end = std::min(start + chunk_size, n);
                if (start >= end)
                {
                    continue;
                }
                futures.push_back(thread_pool_->submit(compute_range, start, end));
            }
            for (auto &future : futures)
            {
                future.get();
            }
            return error::Result<std::vector<Edge<Field>>>::ok(edges.copy());
        }
        catch (const std::exception &e)
        {
            return error::Result<std::vector<Edge<Field>>>::err(
                error::TDAErrorCode::InvalidFieldOperation,
                std::string("Edge computation with field failed: ") + e.what());
        }
    }
    template <typename Field>
    error::Result<std::vector<Triangle<Field>>>
    computeTrianglesWithField(const std::vector<Edge<Field>> &edges,
                              const std::vector<std::vector<uint64_t>> &neighborBits, Size n,
                              double max_radius)
    {
        try
        {
            if (edges.empty())
            {
                return error::Result<std::vector<Triangle<Field>>>::ok({});
            }
            threading::ConcurrentVector<Triangle<Field>> triangles;
            const Size numThreads = thread_pool_ ? thread_pool_->numThreads() : 1;
            const Size chunk_count = std::max<Size>(1, std::min(numThreads, edges.size()));
            const Size chunk_size = (edges.size() + chunk_count - 1) / chunk_count;
            auto compute_range = [this, &edges, &neighborBits, &triangles, n,
                                  max_radius](Size start, Size end) {
                for (Size idx = start; idx < end; ++idx)
                {
                    const auto &e = edges[idx];
                    int i = e.u;
                    int j = e.v;
                    for (int k = j + 1; k < static_cast<int>(n); ++k)
                    {
                        if (isNeighbor(i, k, neighborBits) && isNeighbor(j, k, neighborBits))
                        {
                            int e0_idx = findEdgeIndex(i, j, edges);
                            int e1_idx = findEdgeIndex(i, k, edges);
                            int e2_idx = findEdgeIndex(j, k, edges);
                            if (e0_idx < 0 || e1_idx < 0 || e2_idx < 0)
                            {
                                continue;
                            }
                            const double filtration =
                                std::max({e.filtration, edges[static_cast<Size>(e1_idx)].filtration,
                                          edges[static_cast<Size>(e2_idx)].filtration});
                            if (filtration <= max_radius)
                            {
                                triangles.push_back(Triangle<Field>(
                                    i, j, k, filtration, Field::one(), e0_idx, e1_idx, e2_idx));
                            }
                        }
                    }
                }
            };
            if (!thread_pool_ || chunk_count == 1)
            {
                compute_range(0, edges.size());
                return error::Result<std::vector<Triangle<Field>>>::ok(triangles.copy());
            }
            std::vector<std::future<void>> futures;
            futures.reserve(chunk_count);
            for (Size t = 0; t < chunk_count; ++t)
            {
                const Size start = t * chunk_size;
                const Size end = std::min(start + chunk_size, edges.size());
                if (start >= end)
                {
                    continue;
                }
                futures.push_back(thread_pool_->submit(compute_range, start, end));
            }
            for (auto &future : futures)
            {
                future.get();
            }
            return error::Result<std::vector<Triangle<Field>>>::ok(triangles.copy());
        }
        catch (const std::exception &e)
        {
            return error::Result<std::vector<Triangle<Field>>>::err(
                error::TDAErrorCode::InvalidFieldOperation,
                std::string("Triangle computation with field failed: ") + e.what());
        }
    }
    bool isNeighbor(int i, int j, const std::vector<std::vector<uint64_t>> &neighborBits) const
    {
        if (i < 0 || j < 0 || i >= static_cast<int>(neighborBits.size()) ||
            j >= static_cast<int>(neighborBits.size()))
        {
            return false;
        }
        const auto &bits_i = neighborBits[static_cast<Size>(i)];
        const auto &bits_j = neighborBits[static_cast<Size>(j)];
        if (bits_i.empty() || bits_j.empty())
        {
            return false;
        }
        Size block = static_cast<Size>(j) / 64;
        Size bit = static_cast<Size>(j) % 64;
        if (block >= bits_i.size())
        {
            return false;
        }
        return (bits_i[block] & (1ULL << bit)) != 0;
    }
    template <typename Field>
    int findEdgeIndex(int i, int j, const std::vector<Edge<Field>> &edges) const
    {
        for (Size idx = 0; idx < edges.size(); ++idx)
        {
            const auto &e = edges[idx];
            if ((e.u == i && e.v == j) || (e.u == j && e.v == i))
            {
                return static_cast<int>(idx);
            }
        }
        return -1;
    }
    template <typename Field>
    error::Result<std::vector<Pair>>
    computeVrPersistenceFastWithField(core::BufferView<const double> points, Size point_dim,
                                      const VRConfig &config)
    {
        try
        {
            if (point_dim == 0)
            {
                return error::Result<std::vector<Pair>>::err(error::TDAErrorCode::InvalidDimension,
                                                             "Point dimension must be non-zero");
            }
            if (points.size() % point_dim != 0)
            {
                return error::Result<std::vector<Pair>>::err(
                    error::TDAErrorCode::InvalidInput,
                    "Point coordinate buffer size must be divisible by point "
                    "dimension");
            }
            if (!std::isfinite(config.max_radius) || config.max_radius < 0.0)
            {
                const char *invalid_radius_message =
                    R"(Field-aware Vietoris-Rips requires a finite non-negative max radius)";
                return error::Result<std::vector<Pair>>::err(error::TDAErrorCode::InvalidInput,
                                                             invalid_radius_message);
            }
            const double max_radius_sq = config.max_radius * config.max_radius;
            if (!std::isfinite(max_radius_sq))
            {
                return error::Result<std::vector<Pair>>::err(
                    error::TDAErrorCode::InvalidInput,
                    "Field-aware Vietoris-Rips max radius square exceeds supported range");
            }
            const Size n = points.size() / point_dim;
            if (n == 0)
            {
                return error::Result<std::vector<Pair>>::ok({});
            }
            for (Size idx = 0; idx < points.size(); ++idx)
            {
                if (!std::isfinite(points[idx]))
                {
                    return error::Result<std::vector<Pair>>::err(
                        error::TDAErrorCode::InvalidInput,
                        "Field-aware Vietoris-Rips point coordinates must be finite");
                }
            }
            if (n > static_cast<Size>(std::numeric_limits<int>::max()))
            {
                return error::Result<std::vector<Pair>>::err(
                    error::TDAErrorCode::InvalidInput,
                    "Field-aware Vietoris-Rips supports at most INT_MAX points");
            }
            auto edges_result = computeEdgesWithField<Field>(points, point_dim, max_radius_sq);
            if (edges_result.isErr())
            {
                return forwardError<std::vector<Pair>>(edges_result);
            }
            auto edges = edges_result.value();
            std::sort(edges.begin(), edges.end(), [](const Edge<Field> &a, const Edge<Field> &b) {
                if (a.filtration != b.filtration)
                    return a.filtration < b.filtration;
                if (a.u != b.u)
                    return a.u < b.u;
                return a.v < b.v;
            });
            const Size blocks = (n + 63) / 64;
            std::vector<std::vector<uint64_t>> neighborBits(n, std::vector<uint64_t>(blocks, 0));
            for (const auto &e : edges)
            {
                Size iu = static_cast<Size>(e.u);
                Size iv = static_cast<Size>(e.v);
                setBit(neighborBits[iu], iv);
                setBit(neighborBits[iv], iu);
            }
            std::vector<Triangle<Field>> triangles;
            if (config.max_dim >= 2)
            {
                auto triangles_result =
                    computeTrianglesWithField<Field>(edges, neighborBits, n, config.max_radius);
                if (triangles_result.isErr())
                {
                    return forwardError<std::vector<Pair>>(triangles_result);
                }
                triangles = triangles_result.value();
            }
            std::sort(triangles.begin(), triangles.end(),
                      [](const Triangle<Field> &a, const Triangle<Field> &b) {
                          if (a.filtration != b.filtration)
                              return a.filtration < b.filtration;
                          if (a.a != b.a)
                              return a.a < b.a;
                          if (a.b != b.b)
                              return a.b < b.b;
                          return a.c < b.c;
                      });
            auto boundary_matrix_result =
                createBoundaryMatrixWithField<Field>(edges, triangles, n, config.max_dim);
            if (boundary_matrix_result.isErr())
            {
                return forwardError<std::vector<Pair>>(boundary_matrix_result);
            }
            const auto &boundary_matrix = boundary_matrix_result.value();
            return computePersistenceFromBoundaryMatrix<Field>(boundary_matrix);
        }
        catch (const std::exception &e)
        {
            return error::Result<std::vector<Pair>>::err(
                error::TDAErrorCode::InvalidFieldOperation,
                std::string("VR persistence computation with field failed: ") + e.what());
        }
    }
    void setBit(std::vector<uint64_t> &bits, Size idx) const
    {
        bits[idx / 64] |= (1ULL << (idx % 64));
    }
    template <typename Field>
    error::Result<algebra::BoundaryMatrix>
    createBoundaryMatrixWithField(const std::vector<Edge<Field>> &edges,
                                  const std::vector<Triangle<Field>> &triangles, Size n,
                                  Size max_dim)
    {
        try
        {
            algebra::SimplicialComplex complex;
            for (Size v = 0; v < n; ++v)
            {
                complex.addSimplexWithFiltration(algebra::Simplex({static_cast<Index>(v)}), 0.0);
            }
            for (const auto &edge : edges)
            {
                complex.addSimplexWithFiltration(
                    algebra::Simplex({static_cast<Index>(edge.u), static_cast<Index>(edge.v)}),
                    edge.filtration);
            }
            if (max_dim >= 2)
            {
                for (const auto &tri : triangles)
                {
                    complex.addSimplexWithFiltration(
                        algebra::Simplex({static_cast<Index>(tri.a), static_cast<Index>(tri.b),
                                          static_cast<Index>(tri.c)}),
                        tri.filtration);
                }
            }
            algebra::BoundaryMatrix matrix(complex);
            return error::Result<algebra::BoundaryMatrix>::ok(std::move(matrix));
        }
        catch (const std::exception &e)
        {
            return error::Result<algebra::BoundaryMatrix>::err(
                error::TDAErrorCode::InvalidFieldOperation,
                std::string("Boundary matrix creation failed: ") + e.what());
        }
    }
    template <typename Field>
    error::Result<std::vector<Pair>>
    computePersistenceFromBoundaryMatrix(const algebra::BoundaryMatrix &boundary_matrix)
    {
        try
        {
            auto reduction = MathematicallyCorrectReduction(boundary_matrix);
            auto field_result = reduction.setFieldCharacteristic(field_characteristic_);
            if (field_result.isErr())
            {
                return forwardError<std::vector<Pair>>(field_result);
            }
            auto compute_result = reduction.compute();
            if (compute_result.isErr())
            {
                return forwardError<std::vector<Pair>>(compute_result);
            }
            return error::Result<std::vector<Pair>>::ok(reduction.getPersistencePairs());
        }
        catch (const std::exception &e)
        {
            return error::Result<std::vector<Pair>>::err(
                error::TDAErrorCode::InvalidFieldOperation,
                std::string("Persistence computation failed: ") + e.what());
        }
    }

public:
    FieldAwareVietorisRips()
    {
        auto pool_result = threading::makeThreadPool();
        if (pool_result.isOk())
        {
            thread_pool_ = std::move(pool_result.value());
        }
    }
    explicit FieldAwareVietorisRips(int numThreads)
    {
        auto pool_result = threading::makeThreadPool(numThreads);
        if (pool_result.isOk())
        {
            thread_pool_ = std::move(pool_result.value());
        }
    }
    error::Result<void> setFieldCharacteristic(int characteristic)
    {
        if (characteristic < 2)
        {
            return error::Result<void>::err(error::TDAErrorCode::InvalidFieldOperation,
                                            "Field characteristic must be at least 2");
        }
        if (!isPrime(characteristic))
        {
            return error::Result<void>::err(error::TDAErrorCode::InvalidFieldOperation,
                                            "Field characteristic must be prime");
        }
        field_characteristic_ = characteristic;
        return error::Result<void>::ok();
    }
    error::Result<std::vector<Pair>> computeVrPersistenceFast(core::BufferView<const double> points,
                                                              Size point_dim,
                                                              const VRConfig &config)
    {
        switch (field_characteristic_)
        {
            case 2:
                return computeVrPersistenceFastWithField<FiniteField<2>>(points, point_dim, config);
            case 3:
                return computeVrPersistenceFastWithField<FiniteField<3>>(points, point_dim, config);
            case 5:
                return computeVrPersistenceFastWithField<FiniteField<5>>(points, point_dim, config);
            default:
                return error::Result<std::vector<Pair>>::err(
                    error::TDAErrorCode::InvalidFieldOperation,
                    "Unsupported field characteristic for compile-time finite-field "
                    "backend");
        }
    }
    int getFieldCharacteristic() const { return field_characteristic_; }
    size_t getNumThreads() const { return thread_pool_ ? thread_pool_->numThreads() : 1; }

private:
    static bool isPrime(int n)
    {
        if (n <= 1)
            return false;
        if (n <= 3)
            return true;
        if (n % 2 == 0 || n % 3 == 0)
            return false;
        for (int i = 5; i * i <= n; i += 6)
        {
            if (n % i == 0 || n % (i + 2) == 0)
            {
                return false;
            }
        }
        return true;
    }
};
inline error::Result<std::unique_ptr<FieldAwareVietorisRips>> makeFieldAwareVietorisRips()
{
    try
    {
        auto vr = std::make_unique<FieldAwareVietorisRips>();
        return error::Result<std::unique_ptr<FieldAwareVietorisRips>>::ok(std::move(vr));
    }
    catch (const std::exception &e)
    {
        return error::Result<std::unique_ptr<FieldAwareVietorisRips>>::err(
            error::TDAErrorCode::AllocationFailed,
            std::string("Failed to create field-aware Vietoris-Rips: ") + e.what());
    }
}
inline error::Result<std::unique_ptr<FieldAwareVietorisRips>>
makeFieldAwareVietorisRips(int numThreads)
{
    try
    {
        auto vr = std::make_unique<FieldAwareVietorisRips>(numThreads);
        return error::Result<std::unique_ptr<FieldAwareVietorisRips>>::ok(std::move(vr));
    }
    catch (const std::exception &e)
    {
        return error::Result<std::unique_ptr<FieldAwareVietorisRips>>::err(
            error::TDAErrorCode::AllocationFailed,
            std::string("Failed to create field-aware Vietoris-Rips with threads: ") + e.what());
    }
}
} // namespace nerve::persistence
