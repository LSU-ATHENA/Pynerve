#pragma once

namespace nerve::streaming::gpu
{

struct WindowedBenchmark
{
    double cpu_time_ms = 0.0;
    double gpu_time_ms = 0.0;
    double speedup = 1.0;
    int window_size = 0;
    int num_windows = 0;
};

WindowedBenchmark benchmarkWindowed(int window_size, int num_windows);

} // namespace nerve::streaming::gpu
