DimensionConfig getOptimalDimensionConfig(
    size_t num_simplices,
    int max_dim,
    size_t num_points) {
    (void)num_simplices;
    DimensionConfig config;
    config.use_cohomology = true;
    config.use_involution = (max_dim >= 3);
    config.use_bit_parallel = false;
    config.use_clear_compress = false;
    config.use_prefetching = false;
    config.use_branchless = false;
    config.chunk_size = num_points > 32768 ? 2048 : 1024;
    config.num_threads = 0;
    return config;
}

SpecializedBenchmark benchmarkSpecialized(
    const std::vector<std::vector<int>>& simplices,
        const std::vector<double>& filtration_values,
        const std::vector<int>& dimensions,
        int max_dim) {
    SpecializedBenchmark bench{};

    if (simplices.empty() ||
        filtration_values.size() != simplices.size() ||
        dimensions.size() != simplices.size() ||
        max_dim < 0) {
        return bench;
    }

    auto config = getOptimalDimensionConfig(simplices.size(), max_dim, 0);
    auto start = std::chrono::high_resolution_clock::now();
    auto result = computeDimensionSpecialized(
        simplices, filtration_values, dimensions, max_dim, config);
    auto end = std::chrono::high_resolution_clock::now();
    bench.specialized_time_ms =
        std::chrono::duration<double, std::milli>(end - start).count();

    if (result.h12.computation_time_ms > 0.0) {
        bench.cohomology_time_ms = result.h12.computation_time_ms;
    }
    if (result.h36.computation_time_ms > 0.0) {
        bench.involuted_time_ms = result.h36.computation_time_ms;
        if (result.h36.used_bit_parallel) {
            bench.bit_parallel_time_ms = result.h36.computation_time_ms;
        }
    }
    return bench;
}

DimensionSpeedupEstimate estimateDimensionSpeedup(
    int dim,
    size_t num_simplices,
    size_t num_points) {
    (void)num_simplices;
    DimensionSpeedupEstimate estimate;

    const bool large_point_cloud = num_points > 4096;

    switch (dim) {
        case 0:
            estimate.algorithm = "Union-Find";
            estimate.speedup = 1.10;
            if (large_point_cloud) {
                estimate.speedup += 0.05;
            }
            break;
        case 1:
        case 2:
            estimate.algorithm = "Cohomology + Clearing";
            estimate.speedup = 1.20;
            if (large_point_cloud) {
                estimate.speedup += 0.05;
            }
            break;
        case 3:
        case 4:
        case 5:
        case 6:
            estimate.algorithm = "Involuted Homology";
            estimate.speedup = 1.25;
            if (large_point_cloud) {
                estimate.speedup += 0.05;
            }
            break;
        default:
            estimate.algorithm = "Standard";
            estimate.speedup = 1.0;
    }

    return estimate;
}
