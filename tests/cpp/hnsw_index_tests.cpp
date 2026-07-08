#include "nerve/algorithms/knn_hnsw.hpp"

#include <cassert>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace
{

std::vector<float> randomPoints(size_t n, int dim, unsigned seed = 42)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> pts(n * dim);
    for (size_t i = 0; i < n * static_cast<size_t>(dim); ++i)
    {
        pts[i] = dist(rng);
    }
    return pts;
}

float euclideanSq(const float *a, const float *b, int dim)
{
    float sum = 0.0f;
    for (int i = 0; i < dim; ++i)
    {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

} // namespace

int main()
{
    // Build index from random point cloud and verify k-NN search
    {
        const int dim = 16;
        const size_t n = 200;

        nerve::algorithms::HNSWIndex<float>::Config config;
        config.M = 16;
        config.ef_construction = 100;
        config.ef_search = 50;
        config.random_seed = 42;

        nerve::algorithms::HNSWIndex<float> index(dim, config);
        auto pts = randomPoints(n, dim);
        index.build(pts, n);
        assert(index.node_count() == n);
        assert(index.dimension() == dim);

        std::vector<float> query(dim);
        for (int i = 0; i < dim; ++i)
            query[i] = pts[i];

        auto results = index.search(query, 5);
        assert(results.size() > 0);

        float prev_dist = -1.0f;
        for (const auto &[idx, dist] : results)
        {
            assert(idx < n);
            assert(dist >= 0.0f);
            assert(dist >= prev_dist);
            prev_dist = dist;
        }

        assert(results.size() > 0);
        assert(results[0].first < n);
    }

    // Radius search  --  verify all results within radius
    {
        const int dim = 8;
        const size_t n = 100;

        nerve::algorithms::HNSWIndex<float> index(dim);
        auto pts = randomPoints(n, dim);
        index.build(pts, n);

        std::vector<float> query(dim, 0.5f);
        float radius = 2.0f;

        auto results = index.searchRadius(query, radius);
        for (const auto &[idx, dist] : results)
        {
            assert(dist <= radius * radius + 1e-4f);
            assert(idx < n);
        }

        for (size_t i = 1; i < results.size(); ++i)
        {
            assert(results[i - 1].second <= results[i].second);
        }
    }

    // Radius search with zero radius
    {
        const int dim = 4;
        const size_t n = 10;

        nerve::algorithms::HNSWIndex<float> index(dim);
        auto pts = randomPoints(n, dim);
        index.build(pts, n);

        std::vector<float> query(pts.begin(), pts.begin() + dim);
        auto results = index.searchRadius(query, 0.0f);
        assert(results.size() >= 0);
        assert(results[0].first < n);
    }

    // Empty index  --  verify graceful handling
    {
        nerve::algorithms::HNSWIndex<float> empty_index(8);
        assert(empty_index.node_count() == 0);

        std::vector<float> query(8, 0.5f);
        auto results = empty_index.search(query, 5);
        assert(results.empty());

        auto radius_results = empty_index.searchRadius(query, 1.0f);
        assert(radius_results.empty());
    }

    // Single point index  --  verify returns itself
    {
        const int dim = 3;
        const size_t n = 1;

        nerve::algorithms::HNSWIndex<float> index(dim);
        std::vector<float> pts = {1.0f, 2.0f, 3.0f};
        index.build(pts, n);

        std::vector<float> query = {1.0f, 2.0f, 3.0f};
        auto results = index.search(query, 5);
        assert(results.size() > 0);
        assert(results.size() > 0);
        assert(results[0].first < n);

        auto radius_results = index.searchRadius(query, 1.0f);
        assert(radius_results.size() == 1);
        assert(radius_results[0].first == 0);
    }

    // Batch search
    {
        const int dim = 4;
        const size_t n = 50;

        nerve::algorithms::HNSWIndex<float> index(dim);
        auto pts = randomPoints(n, dim);
        index.build(pts, n);

        auto batch_flat = index.batchSearch(pts, n, 3);
        assert(batch_flat.size() == n);
        for (const auto &results : batch_flat)
        {
            assert(results.size() > 0);
        }
    }

    // batchSearch with vector of spans
    {
        const int dim = 4;
        const size_t n = 20;

        nerve::algorithms::HNSWIndex<float> index(dim);
        auto pts = randomPoints(n, dim);
        index.build(pts, n);

        std::vector<std::span<const float>> queries;
        for (size_t i = 0; i < 5; ++i)
        {
            queries.push_back(std::span<const float>(pts.data() + i * dim, dim));
        }

        auto batch_results = index.batchSearch(queries, 3);
        assert(batch_results.size() == 5);
        for (const auto &results : batch_results)
        {
            assert(results.size() > 0);
            float prev = -1.0f;
            for (const auto &[idx, dist] : results)
            {
                assert(idx < n);
                assert(dist >= prev);
                prev = dist;
            }
        }
    }

    // Performance  --  1000 points, 128D
    {
        const int dim = 128;
        const size_t n = 1000;

        nerve::algorithms::HNSWIndex<float>::Config config;
        config.M = 16;
        config.ef_construction = 100;

        nerve::algorithms::HNSWIndex<float> index(dim, config);
        auto pts = randomPoints(n, dim, 123);
        index.build(pts, n);
        assert(index.node_count() == n);

        std::vector<float> query(pts.begin(), pts.begin() + dim);
        auto results = index.search(query, 10);
        assert(results.size() > 0);
        assert(results[0].first < n);
    }

    // Invalid dimensions throw
    {
        bool caught = false;
        try
        {
            nerve::algorithms::HNSWIndex<float>(0);
        }
        catch (const std::invalid_argument &)
        {
            caught = true;
        }
        assert(caught);
    }

    // Config validation
    {
        bool m_caught = false;
        try
        {
            nerve::algorithms::HNSWIndex<float>::Config config;
            config.M = 2;
            nerve::algorithms::HNSWIndex<float>(10, config);
        }
        catch (const std::invalid_argument &)
        {
            m_caught = true;
        }
        assert(m_caught);
    }

    // Build on data that's too small for n_points * dim
    {
        nerve::algorithms::HNSWIndex<float> index(10);
        std::vector<float> too_small(5, 0.0f);
        bool caught = false;
        try
        {
            index.build(too_small, 10);
        }
        catch (const std::runtime_error &)
        {
            caught = true;
        }
        catch (const std::invalid_argument &)
        {
            caught = true;
        }
        assert(caught);
    }

    return 0;
}
