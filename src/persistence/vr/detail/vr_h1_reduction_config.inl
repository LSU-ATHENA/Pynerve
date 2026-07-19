// Configuration and speedup helpers

ReducedVRH1Config getOptimalReducedVRH1Config(size_t num_points, double max_radius)
{
    ReducedVRH1Config config;
    config.max_radius = max_radius;

    // Enable all reduction passes by default; connectivity pruning
    // is deferred to larger clouds where its overhead amortizes better.
    config.use_cycle_filter = true;
    config.use_connectivity_pruning = (num_points > 1000);
    config.use_triangle_filter = true;
    config.preserve_connectivity = true;
    config.num_threads = 0; // auto
    return config;
}

ReducedVRSpeedup estimateReducedVRSpeedup(size_t num_points, double max_radius)
{
    ReducedVRSpeedup speedup;
    if (num_points == 0 || !std::isfinite(max_radius) || max_radius <= 0.0)
    {
        return speedup;
    }
    if (num_points < 1000)
    {
        speedup.edge_reduction = 1.5;
        speedup.computation_speedup = 1.3;
    }
    else if (num_points < 10000)
    {
        speedup.edge_reduction = 2.0;
        speedup.computation_speedup = 1.8;
    }
    else
    {
        speedup.edge_reduction = 2.5;
        speedup.computation_speedup = 2.2;
    }

    if (max_radius < 1.0)
    {
        speedup.edge_reduction *= 0.9;
        speedup.computation_speedup *= 0.95;
    }
    else if (max_radius > 10.0)
    {
        speedup.edge_reduction *= 1.1;
        speedup.computation_speedup *= 1.05;
    }
    speedup.memory_reduction = speedup.edge_reduction;
    speedup.total_speedup = speedup.computation_speedup;
    return speedup;
}
