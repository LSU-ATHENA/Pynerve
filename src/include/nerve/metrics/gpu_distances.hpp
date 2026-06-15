#pragma once

#include <utility>
#include <vector>

namespace nerve::metrics::avx512
{

struct AVX512Benchmark
{
    double scalar_time_ms = 0.0;
    double avx512_time_ms = 0.0;
    double speedup = 1.0;
    int n1 = 0;
    int n2 = 0;
    int dim = 0;
};

void avx512DiagramDistanceMatrix(const float *birth1, const float *death1, int n1,
                                 const float *birth2, const float *death2, int n2,
                                 float *out_matrix);
void computeDistanceMatrixAuto(const float *birth1, const float *death1, int n1,
                               const float *birth2, const float *death2, int n2, float *out_matrix);
bool isAVX512Available();
AVX512Benchmark benchmarkAVX512(int n1, int n2, int dim, int iterations);

} // namespace nerve::metrics::avx512

namespace nerve::metrics::bottleneck
{

double adaptiveBottleneckDistance(const std::vector<std::pair<float, float>> &diagram1,
                                  const std::vector<std::pair<float, float>> &diagram2);

std::vector<double>
parallelBottleneckDistances(const std::vector<std::vector<std::pair<float, float>>> &diagrams1,
                            const std::vector<std::vector<std::pair<float, float>>> &diagrams2);

} // namespace nerve::metrics::bottleneck

namespace nerve::metrics::sinkhorn
{

struct SinkhornConfig
{
    double epsilon = 0.01;
    int max_iterations = 100;
    double tolerance = 1e-6;
    bool gpu_accelerated = true;
};

struct SinkhornBenchmark
{
    double exact_time_ms = 0.0;
    double sinkhorn_time_ms = 0.0;
    double sliced_time_ms = 0.0;
    double speedup = 1.0;
    double relative_error = 0.0;
    int n = 0;
};

double sinkhornDiagramDistance(const std::vector<std::pair<float, float>> &diagram1,
                               const std::vector<std::pair<float, float>> &diagram2,
                               const SinkhornConfig &config = SinkhornConfig{});

double slicedWassersteinDistance(const std::vector<std::pair<float, float>> &diagram1,
                                 const std::vector<std::pair<float, float>> &diagram2,
                                 int num_projections = 100);

double hierarchicalWasserstein(const std::vector<std::pair<float, float>> &diagram1,
                               const std::vector<std::pair<float, float>> &diagram2,
                               int levels = 3);

SinkhornBenchmark benchmarkSinkhorn(int n);

} // namespace nerve::metrics::sinkhorn
