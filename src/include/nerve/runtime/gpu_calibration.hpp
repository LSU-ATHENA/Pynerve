
#pragma once

#include <vector>

namespace nerve::runtime
{
namespace gpu
{

struct CalibrationBenchmark
{
    double cpu_update_ms;
    double gpu_update_ms;
    double cpu_query_ms;
    double gpu_query_ms;
    double cpu_fit_ms;
    double gpu_fit_ms;
    double speedup_update;
    double speedup_query;
    double speedup_fit;
    int num_samples;
    int num_buckets;
};

CalibrationBenchmark benchmarkCalibration(int num_samples, int num_buckets);

} // namespace gpu
} // namespace nerve::runtime
