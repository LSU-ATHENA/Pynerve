#include "nerve/metrics/gpu_distances.hpp"

#include <cuda_runtime.h>

#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace
{

bool check_cuda(cudaError_t code, const char *expression)
{
    if (code == cudaSuccess)
        return true;
    std::cerr << expression << " failed: " << cudaGetErrorString(code) << '\n';
    return false;
}

bool has_gpu()
{
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return err == cudaSuccess && device_count > 0;
}

bool approxEqual(double a, double b, double eps = 1e-8)
{
    return std::abs(a - b) <= eps * std::max(1.0, std::max(std::abs(a), std::abs(b)));
}

} // namespace

int main()
{
    if (!has_gpu())
    {
        std::cerr << "No CUDA device available  --  skipping GPU distance tests\n";
        return 0;
    }

    // Test 1: Bottleneck distance between two small diagrams
    {
        std::vector<std::pair<float, float>> diagram1 = {{0.0f, 0.5f}, {0.1f, 0.3f}, {0.2f, 1.0f}};

        std::vector<std::pair<float, float>> diagram2 = {
            {0.0f, 0.4f},
            {0.15f, 0.35f},
            {0.2f, 0.9f},
            {0.0f, std::numeric_limits<float>::infinity()}};

        double distance =
            nerve::metrics::bottleneck::adaptiveBottleneckDistance(diagram1, diagram2);
        assert(std::isfinite(distance));
        assert(distance >= 0.0);
    }

    // Test 2: Bottleneck  --  identical diagrams produce zero distance
    {
        std::vector<std::pair<float, float>> diagram = {{0.0f, 0.5f}, {0.1f, 0.3f}};

        double distance = nerve::metrics::bottleneck::adaptiveBottleneckDistance(diagram, diagram);
        assert(approxEqual(distance, 0.0, 1e-6));
    }

    // Test 3: Parallel bottleneck distances
    {
        std::vector<std::vector<std::pair<float, float>>> diagrams1 = {
            {{0.0f, 0.5f}, {0.1f, 0.3f}}, {{0.0f, 0.8f}, {0.2f, 0.6f}}};
        std::vector<std::vector<std::pair<float, float>>> diagrams2 = {
            {{0.0f, 0.4f}, {0.15f, 0.35f}}, {{0.05f, 0.7f}, {0.25f, 0.55f}}};

        auto distances =
            nerve::metrics::bottleneck::parallelBottleneckDistances(diagrams1, diagrams2);
        assert(distances.size() == 2);
        for (double d : distances)
        {
            assert(std::isfinite(d));
            assert(d >= 0.0);
        }
    }

    // Test 4: Sinkhorn (Wasserstein) distance between two diagrams
    {
        std::vector<std::pair<float, float>> diagram1 = {{0.0f, 0.5f}, {0.1f, 0.8f}, {0.3f, 0.6f}};

        std::vector<std::pair<float, float>> diagram2 = {
            {0.05f, 0.45f},
            {0.15f, 0.75f},
            {0.25f, 0.55f},
            {0.0f, std::numeric_limits<float>::infinity()}};

        nerve::metrics::sinkhorn::SinkhornConfig config;
        config.epsilon = 0.1;
        config.max_iterations = 50;
        config.gpu_accelerated = true;

        double distance =
            nerve::metrics::sinkhorn::sinkhornDiagramDistance(diagram1, diagram2, config);
        assert(std::isfinite(distance));
        assert(distance >= 0.0);
    }

    // Test 5: Sinkhorn  --  identical diagrams produce near-zero distance
    {
        std::vector<std::pair<float, float>> diagram = {{0.0f, 0.5f}, {0.2f, 0.7f}};

        nerve::metrics::sinkhorn::SinkhornConfig config;
        config.epsilon = 0.1;

        double distance =
            nerve::metrics::sinkhorn::sinkhornDiagramDistance(diagram, diagram, config);
        assert(distance >= 0.0);
        assert(distance < 1.0);
    }

    // Test 6: Sinkhorn  --  empty diagram
    {
        std::vector<std::pair<float, float>> empty;
        std::vector<std::pair<float, float>> nonempty = {{0.0f, 0.5f}};

        double distance = nerve::metrics::sinkhorn::sinkhornDiagramDistance(empty, nonempty);
        assert(std::isfinite(distance));
        assert(distance >= 0.0);
    }

    // Test 7: Sliced Wasserstein distance
    {
        std::vector<std::pair<float, float>> diagram1 = {{0.0f, 0.5f}, {0.1f, 0.8f}};

        std::vector<std::pair<float, float>> diagram2 = {{0.05f, 0.45f}, {0.15f, 0.75f}};

        double distance =
            nerve::metrics::sinkhorn::slicedWassersteinDistance(diagram1, diagram2, 50);
        assert(std::isfinite(distance));
        assert(distance >= 0.0);
    }

    // Test 8: Hierarchical Wasserstein
    {
        std::vector<std::pair<float, float>> diagram1 = {{0.0f, 0.5f}, {0.1f, 0.8f}, {0.3f, 0.6f}};

        std::vector<std::pair<float, float>> diagram2 = {{0.05f, 0.45f}, {0.15f, 0.75f}};

        double distance = nerve::metrics::sinkhorn::hierarchicalWasserstein(diagram1, diagram2, 2);
        assert(std::isfinite(distance));
        assert(distance >= 0.0);
    }

    // Test 9: SinkhornBenchmark
    {
        auto benchmark = nerve::metrics::sinkhorn::benchmarkSinkhorn(20);
        assert(benchmark.n == 20);
        assert(benchmark.speedup >= 0.0);
    }

    // Test 10: Bottleneck with large persistence (infinite death)
    {
        std::vector<std::pair<float, float>> diagram1 = {
            {0.0f, 0.5f}, {0.0f, std::numeric_limits<float>::infinity()}};

        std::vector<std::pair<float, float>> diagram2 = {
            {0.0f, 0.4f}, {0.01f, std::numeric_limits<float>::infinity()}};

        double distance =
            nerve::metrics::bottleneck::adaptiveBottleneckDistance(diagram1, diagram2);
        assert(std::isfinite(distance));
        assert(distance >= 0.0);
    }

    return 0;
}
