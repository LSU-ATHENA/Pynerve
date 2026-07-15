SheafGPUBenchmark benchmarkSheafGPU(int num_stalks, int stalk_dim)
{
    if (num_stalks <= 0 || stalk_dim <= 0)
    {
        throw std::invalid_argument(
            "benchmarkSheafGPU requires positive stalk count and stalk dimension");
    }

    SheafGPUBenchmark bench;
    bench.num_stalks = num_stalks;
    bench.stalk_dim = stalk_dim;

    std::vector<float> rhs(num_stalks);
    std::vector<float> diagonal(num_stalks);
    for (int i = 0; i < num_stalks; ++i)
    {
        rhs[i] = 0.25f + static_cast<float>((i * 17) % 101) / 101.0f;
        diagonal[i] = 1.0f + static_cast<float>(i) * 0.01f;
    }

    std::vector<int> row_ptr(num_stalks + 1);
    std::vector<int> col_idx(num_stalks);
    std::vector<float> values(num_stalks);
    for (int i = 0; i < num_stalks; ++i)
    {
        row_ptr[i] = i;
        col_idx[i] = i;
        values[i] = std::sqrt(diagonal[i]);
    }
    row_ptr[num_stalks] = num_stalks;

    auto start_cpu = std::chrono::high_resolution_clock::now();
    std::vector<float> cpu_solution(num_stalks);
    for (int i = 0; i < num_stalks; ++i)
    {
        cpu_solution[i] = rhs[i] / diagonal[i];
    }
    auto end_cpu = std::chrono::high_resolution_clock::now();
    bench.cpu_time_ms = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();

    GPUSheafLaplacian gpu_solver(num_stalks, stalk_dim);
    gpu_solver.buildLaplacian(row_ptr, col_idx, values, {});

    auto start_gpu = std::chrono::high_resolution_clock::now();
    std::vector<float> gpu_solution;
    gpu_solver.solve(rhs, gpu_solution);
    auto end_gpu = std::chrono::high_resolution_clock::now();
    bench.gpu_time_ms = std::chrono::duration<double, std::milli>(end_gpu - start_gpu).count();

    bench.speedup = finiteBenchmarkSpeedup(bench.cpu_time_ms, bench.gpu_time_ms);

    return bench;
}
