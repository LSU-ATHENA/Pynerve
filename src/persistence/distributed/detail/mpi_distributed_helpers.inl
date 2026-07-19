DistributedConfig getOptimalDistributedConfig(size_t num_points, size_t point_dim, int num_ranks)
{
    DistributedConfig config;
    const auto effective_ranks = static_cast<size_t>(std::max(1, num_ranks));
    const double points_per_rank =
        static_cast<double>(num_points) / static_cast<double>(effective_ranks);

    config.max_dim = point_dim <= 1 ? 1 : 2;
    config.max_radius = points_per_rank >= 250000.0 ? 0.75 : 1.0;

    if (point_dim <= 2)
    {
        config.overlap_ratio = 0.05;
    }
    else if (point_dim == 3)
    {
        config.overlap_ratio = 0.1;
    }
    else
    {
        config.overlap_ratio = 0.15;
    }
    if (effective_ranks > 1)
    {
        config.overlap_ratio += std::min(0.10, 0.01 * std::log2(effective_ranks));
    }

    config.use_openmp = points_per_rank >= 1000.0 || effective_ranks == 1;
    config.use_cuda = false;
    return config;
}

bool shouldUseDistributed(size_t num_points, int available_cores)
{
    return num_points >= 100000 && available_cores >= 4;
}

DistributedSystemInfo getDistributedSystemInfo()
{
    DistributedSystemInfo info;

#if defined(NERVE_HAS_MPI)
    if (!mpi_initialized)
    {
        int mpi_is_initialized = 0;
        MPI_Initialized(&mpi_is_initialized);
        if (!mpi_is_initialized)
        {
            int argc = 0;
            char **argv = nullptr;
            MPI_Init(&argc, &argv);
            mpi_initialized = true;
        }
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &info.mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &info.mpi_size);

    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len = 0;
    MPI_Get_processor_name(processor_name, &name_len);
    info.processor_name = processor_name;
    info.mpi_available = true;
#else
    info.mpi_rank = 0;
    info.mpi_size = 1;
    info.mpi_available = false;
#endif

    info.num_threads = static_cast<int>(std::thread::hardware_concurrency());
    return info;
}
