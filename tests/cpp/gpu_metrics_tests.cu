#include "gpu_test_helpers.cuh"
#include "nerve/metrics/gpu_distances.hpp"

int main()
{
    if (!has_gpu())
    {
        std::cerr << "No CUDA device available -- skipping GPU metrics kernel coverage tests\n";
        return 0;
    }

    // Metrics: benchmarkSinkhorn
    {
        auto bench = nerve::metrics::sinkhorn::benchmarkSinkhorn(16);
        assert(bench.n == 16);
        assert(bench.exact_time_ms >= 0.0);
        assert(bench.sinkhorn_time_ms >= 0.0);
        std::cout << "PASSED: benchmarkSinkhorn (n=16, speedup=" << bench.speedup << ")\n";
    }

    // Metrics: sinkhornDiagramDistance on identical diagrams
    {
        std::vector<std::pair<float, float>> d1 = {{0.0f, 1.0f}, {0.2f, 0.8f}};
        std::vector<std::pair<float, float>> d2 = {{0.0f, 1.0f}, {0.2f, 0.8f}};
        double dist = nerve::metrics::sinkhorn::sinkhornDiagramDistance(d1, d2);
        assert(dist >= 0.0);
        std::cout << "PASSED: sinkhornDiagramDistance identical (dist=" << dist << ")\n";
    }

    // Metrics: slicedWassersteinDistance
    {
        std::vector<std::pair<float, float>> d1 = {{0.0f, 1.0f}};
        std::vector<std::pair<float, float>> d2 = {{0.1f, 0.9f}};
        double dist = nerve::metrics::sinkhorn::slicedWassersteinDistance(d1, d2, 10);
        assert(dist >= 0.0);
        std::cout << "PASSED: slicedWassersteinDistance (dist=" << dist << ")\n";
    }

    // Metrics: bottleneck distance
    {
        std::vector<std::pair<float, float>> d1 = {{0.0f, 1.0f}};
        std::vector<std::pair<float, float>> d2 = {{0.0f, 1.0f}};
        double dist = nerve::metrics::bottleneck::adaptiveBottleneckDistance(d1, d2);
        assert(dist >= 0.0);
        std::cout << "PASSED: adaptiveBottleneckDistance identical (dist=" << dist << ")\n";
    }

#if defined(__AVX512F__)
    // Metrics: benchmarkAVX512 (only when AVX512 available)
    {
        auto bench = nerve::metrics::avx512::benchmarkAVX512(8, 8, 2, 5);
        assert(bench.n1 == 8);
        assert(bench.n2 == 8);
        assert(bench.scalar_time_ms >= 0.0);
        std::cout << "PASSED: benchmarkAVX512 (8x8)\n";
    }
#endif

    return 0;
}
