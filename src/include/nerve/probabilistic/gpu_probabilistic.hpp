
#pragma once

#include <vector>

namespace nerve
{
namespace probabilistic
{
namespace gpu
{

struct ProbabilisticBenchmark
{
    double cpu_time_ms;
    double gpu_time_ms;
    double speedup;
    int num_samples;
    float accuracy;
};

ProbabilisticBenchmark benchmarkProbabilistic(int num_samples);

} // namespace gpu
} // namespace probabilistic
} // namespace nerve
