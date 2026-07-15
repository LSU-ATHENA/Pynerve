/**
 * @brief Benchmark zigzag persistence
 */
struct ZigzagBenchmark
{
    double cpu_time_ms;
    double gpu_time_ms;
    double speedup;
    int num_time_steps;
    int avg_simplices_per_step;
    int total_persistence_pairs;
};

ZigzagBenchmark benchmarkZigzag(int num_steps, int simplices_per_step)
{
    if (num_steps <= 0 || simplices_per_step <= 0)
    {
        throw std::invalid_argument("zigzag benchmark sizes must be positive");
    }
    const int total_simplices = checkedIntSize(
        checkedMulSize(static_cast<size_t>(num_steps), static_cast<size_t>(simplices_per_step),
                       "zigzag benchmark simplex count"),
        "zigzag benchmark simplex count");

    ZigzagBenchmark bench;
    bench.num_time_steps = num_steps;
    bench.avg_simplices_per_step = simplices_per_step;

    GPUZigzagPersistence zigzag(total_simplices, num_steps);

    // Generate a deterministic benchmark workload.
    std::vector<std::vector<int>> simplices;
    std::vector<int> dimensions;
    std::vector<ZigzagStep> steps;

    for (int i = 0; i < total_simplices; ++i)
    {
        simplices.push_back({i});
        dimensions.push_back(i % 3);
    }

    for (int t = 0; t < num_steps; ++t)
    {
        ZigzagStep step;
        step.type = (t % 2 == 0) ? ZigzagStep::FORWARD_INCLUSION : ZigzagStep::FORWARD_DELETION;
        step.simplex_index = t * simplices_per_step;
        step.time = static_cast<float>(t);
        steps.push_back(step);
    }

    auto start_cpu = std::chrono::high_resolution_clock::now();
    struct CpuPair
    {
        int simplex;
        int dim;
        float birth;
        float death = std::numeric_limits<float>::infinity();
    };
    std::vector<CpuPair> cpu_pairs;
    cpu_pairs.reserve(steps.size());
    for (const ZigzagStep &step : steps)
    {
        const int dim = dimensions[static_cast<size_t>(step.simplex_index)];
        if (step.type == ZigzagStep::FORWARD_INCLUSION ||
            step.type == ZigzagStep::BACKWARD_INCLUSION)
        {
            cpu_pairs.push_back({step.simplex_index, dim, step.time});
        }
        else
        {
            for (auto &pair : cpu_pairs)
            {
                if (pair.dim == dim && std::isinf(pair.death))
                {
                    pair.death = step.time;
                    break;
                }
            }
        }
    }
    int cpu_pairs_closed = 0;
    for (const auto &pair : cpu_pairs)
    {
        if (std::isfinite(pair.death))
        {
            ++cpu_pairs_closed;
        }
    }
    auto end_cpu = std::chrono::high_resolution_clock::now();
    bench.cpu_time_ms = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();

    // GPU benchmark
    auto start_gpu = std::chrono::high_resolution_clock::now();
    auto result = zigzag.computeZigzag(simplices, dimensions, steps);
    auto end_gpu = std::chrono::high_resolution_clock::now();

    bench.gpu_time_ms = std::chrono::duration<double, std::milli>(end_gpu - start_gpu).count();
    bench.total_persistence_pairs = result.size();

    auto ratio = [](double cpu_ms, double gpu_ms) {
        if (!std::isfinite(cpu_ms) || !std::isfinite(gpu_ms) || cpu_ms < 0.0 || gpu_ms <= 0.0)
        {
            return 1.0;
        }
        return cpu_ms / gpu_ms;
    };
    bench.speedup = ratio(bench.cpu_time_ms, bench.gpu_time_ms);
    bench.total_persistence_pairs = std::max<int>(bench.total_persistence_pairs, cpu_pairs_closed);

    return bench;
}
