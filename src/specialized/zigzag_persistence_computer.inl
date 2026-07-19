class GPUZigzagPersistence
{
public:
    GPUZigzagPersistence(int max_simplices, int max_steps)
        : max_simplices_(max_simplices)
        , max_steps_(max_steps)
    {
        if (max_simplices_ <= 0 || max_steps_ <= 0)
        {
            throw std::invalid_argument("zigzag persistence capacities must be positive");
        }

        try
        {
            const size_t simplex_count = static_cast<size_t>(max_simplices_);
            const size_t boundary_count =
                checkedMulSize(simplex_count, simplex_count, "zigzag boundary matrix count");
            allocateDevice(&d_boundary_matrix_, boundary_count, "allocate zigzag boundary matrix");
            allocateDevice(&d_simplex_dims_, simplex_count, "allocate zigzag simplex dimensions");
            allocateDevice(&d_steps_, static_cast<size_t>(max_steps_), "allocate zigzag steps");
            allocateDevice(&d_complex_simplices_, simplex_count,
                           "allocate zigzag complex simplices");
            allocateDevice(&d_complex_size_, 1, "allocate zigzag complex size");
            allocateDevice(&d_birth_indices_, simplex_count, "allocate zigzag birth indices");
            allocateDevice(&d_death_indices_, simplex_count, "allocate zigzag death indices");
            allocateDevice(&d_birth_times_, simplex_count, "allocate zigzag birth times");
            allocateDevice(&d_death_times_, simplex_count, "allocate zigzag death times");
            allocateDevice(&d_persistence_dim_, simplex_count, "allocate zigzag persistence dims");
            allocateDevice(&d_pair_count_, 1, "allocate zigzag pair count");
            checkCuda(cudaMemset(d_complex_size_, 0, sizeof(int)),
                      "initialize zigzag complex size");
        }
        catch (...)
        {
            cleanup();
            throw;
        }
    }

    ~GPUZigzagPersistence() { cleanup(); }

    /**
     * @brief Compute zigzag persistence for dynamic filtration
     */
    std::vector<std::tuple<float, float, int, float>>
    computeZigzag(const std::vector<std::vector<int>> &simplices,
                  const std::vector<int> &dimensions, const std::vector<ZigzagStep> &steps)
    {
        if (dimensions.size() != simplices.size())
        {
            throw std::invalid_argument("dimension count must match simplex count");
        }
        const int num_simplices = checkedIntSize(simplices.size(), "zigzag simplex count");
        const int num_steps = checkedIntSize(steps.size(), "zigzag step count");
        if (num_simplices > max_simplices_ || num_steps > max_steps_)
        {
            throw std::length_error("zigzag workload exceeds configured GPU capacity");
        }
        for (const ZigzagStep &step : steps)
        {
            if (step.simplex_index < 0 || step.simplex_index >= num_simplices ||
                !std::isfinite(step.time))
            {
                throw std::invalid_argument("zigzag step references an invalid simplex or time");
            }
        }
        if (num_simplices == 0 || num_steps == 0)
        {
            buildComplex(simplices, dimensions);
            return {};
        }

        buildComplex(simplices, dimensions);
        copyToDevice(d_steps_, steps.data(), steps.size(), "copy zigzag steps");
        checkCuda(cudaMemset(d_pair_count_, 0, sizeof(int)), "reset zigzag pair count");
        checkCuda(cudaMemset(d_death_indices_, -1,
                             checkedMulSize(static_cast<size_t>(max_simplices_), sizeof(int),
                                            "zigzag death index bytes")),
                  "reset zigzag death indices");

        const int blocks = checkedGridBlocks(steps.size(), "zigzag forward grid");
        zigzagForwardKernel<<<blocks, ZIGZAG_BLOCK_SIZE>>>(
            d_boundary_matrix_, d_simplex_dims_, num_simplices, d_steps_, num_steps,
            d_birth_indices_, d_death_indices_, d_birth_times_, d_death_times_, d_persistence_dim_,
            d_pair_count_, max_simplices_);
        checkCuda(cudaPeekAtLastError(), "launch zigzag forward kernel");
        checkCuda(cudaDeviceSynchronize(), "synchronize zigzag forward kernel");

        int pair_count = 0;
        copyToHost(&pair_count, d_pair_count_, 1, "copy zigzag pair count");
        pair_count = std::clamp(pair_count, 0, max_simplices_);

        std::vector<int> h_birth_idx(static_cast<size_t>(pair_count));
        std::vector<int> h_death_idx(static_cast<size_t>(pair_count));
        std::vector<float> h_birth_time(static_cast<size_t>(pair_count));
        std::vector<float> h_death_time(static_cast<size_t>(pair_count));
        std::vector<int> h_dim(static_cast<size_t>(pair_count));

        copyToHost(h_birth_idx.data(), d_birth_indices_, h_birth_idx.size(),
                   "copy zigzag birth indices");
        copyToHost(h_death_idx.data(), d_death_indices_, h_death_idx.size(),
                   "copy zigzag death indices");
        copyToHost(h_birth_time.data(), d_birth_times_, h_birth_time.size(),
                   "copy zigzag birth times");
        copyToHost(h_death_time.data(), d_death_times_, h_death_time.size(),
                   "copy zigzag death times");
        copyToHost(h_dim.data(), d_persistence_dim_, h_dim.size(), "copy zigzag dims");

        std::vector<std::tuple<float, float, int, float>> result;
        for (int i = 0; i < pair_count; ++i)
        {
            if (h_death_idx[static_cast<size_t>(i)] >= 0)
            {
                float persistence =
                    h_death_time[static_cast<size_t>(i)] - h_birth_time[static_cast<size_t>(i)];
                result.emplace_back(h_birth_time[static_cast<size_t>(i)],
                                    h_death_time[static_cast<size_t>(i)],
                                    h_dim[static_cast<size_t>(i)], persistence);
            }
        }

        return result;
    }

    /**
     * @brief Update complex incrementally
     */
    void updateComplex(const std::vector<int> &new_simplices,
                       const std::vector<int> &removed_simplices)
    {
        const int num_new = checkedIntSize(new_simplices.size(), "zigzag new simplex count");
        const int num_removed =
            checkedIntSize(removed_simplices.size(), "zigzag removed simplex count");
        const int max_op = std::max(num_new, num_removed);
        if (max_op == 0)
        {
            return;
        }

        int *d_new = nullptr;
        int *d_removed = nullptr;
        auto free_runtime = [&]() {
            if (d_new)
                cudaFree(d_new);
            if (d_removed)
                cudaFree(d_removed);
        };

        try
        {
            allocateDevice(&d_new, new_simplices.size(), "allocate zigzag new simplices");
            allocateDevice(&d_removed, removed_simplices.size(),
                           "allocate zigzag removed simplices");
            copyToDevice(d_new, new_simplices.data(), new_simplices.size(),
                         "copy zigzag new simplices");
            copyToDevice(d_removed, removed_simplices.data(), removed_simplices.size(),
                         "copy zigzag removed simplices");

            const int blocks = checkedGridBlocks(static_cast<size_t>(max_op), "zigzag update grid");
            dynamicUpdateKernel<<<blocks, ZIGZAG_BLOCK_SIZE>>>(
                d_complex_simplices_, d_complex_size_, max_simplices_, d_new, num_new, d_removed,
                num_removed);
            checkCuda(cudaPeekAtLastError(), "launch zigzag update kernel");
            checkCuda(cudaDeviceSynchronize(), "synchronize zigzag update kernel");
            free_runtime();
        }
        catch (...)
        {
            free_runtime();
            throw;
        }
    }

    /**
     * @brief Compute zigzag for time series data
     */
    std::vector<std::vector<std::tuple<float, float, int>>>
    computeTimeSeriesTopology(const std::vector<std::vector<std::vector<float>>> &time_slices,
                              float max_distance)
    {
        if (!std::isfinite(max_distance) || max_distance < 0.0f)
        {
            throw std::invalid_argument("max_distance must be finite and non-negative");
        }
        std::vector<std::vector<std::tuple<float, float, int>>> results;
        auto build_snapshot = [max_distance](const std::vector<std::vector<float>> &points) {
            if (points.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
            {
                throw std::length_error("time-slice point count exceeds int range");
            }
            std::set<std::vector<int>> snapshot;
            const float radius_sq = max_distance * max_distance;
            for (size_t i = 0; i < points.size(); ++i)
            {
                snapshot.insert({static_cast<int>(i)});
            }
            for (size_t i = 0; i < points.size(); ++i)
            {
                for (size_t j = i + 1; j < points.size(); ++j)
                {
                    const size_t dims = std::min(points[i].size(), points[j].size());
                    float dist_sq = 0.0f;
                    for (size_t d = 0; d < dims; ++d)
                    {
                        const float delta = points[i][d] - points[j][d];
                        dist_sq += delta * delta;
                    }
                    if (dist_sq <= radius_sq)
                    {
                        snapshot.insert({static_cast<int>(i), static_cast<int>(j)});
                    }
                }
            }
            return snapshot;
        };

        std::set<std::vector<int>> previous;

        for (size_t t = 0; t < time_slices.size(); ++t)
        {
            const auto current = build_snapshot(time_slices[t]);
            if (t == 0)
            {
                results.push_back({});
                previous = current;
                continue;
            }

            std::set<std::vector<int>> universe = previous;
            universe.insert(current.begin(), current.end());
            checkedIntSize(universe.size(), "zigzag universe simplex count");

            std::vector<std::vector<int>> simplices;
            std::vector<int> dimensions;
            std::map<std::vector<int>, int> simplex_to_index;
            for (const auto &simplex : universe)
            {
                const int index = checkedIntSize(simplices.size(), "zigzag simplex index");
                simplex_to_index[simplex] = index;
                simplices.push_back(simplex);
                dimensions.push_back(checkedIntSize(simplex.size(), "zigzag simplex dimension") -
                                     1);
            }

            std::vector<ZigzagStep> steps;
            for (const auto &simplex : current)
            {
                if (!previous.contains(simplex))
                {
                    steps.push_back({ZigzagStep::FORWARD_INCLUSION, simplex_to_index[simplex],
                                     static_cast<float>(t)});
                }
            }
            for (const auto &simplex : previous)
            {
                if (!current.contains(simplex))
                {
                    steps.push_back({ZigzagStep::FORWARD_DELETION, simplex_to_index[simplex],
                                     static_cast<float>(t)});
                }
            }

            auto pairs = computeZigzag(simplices, dimensions, steps);
            std::vector<std::tuple<float, float, int>> slice_result;
            slice_result.reserve(pairs.size());
            for (const auto &p : pairs)
            {
                slice_result.emplace_back(std::get<0>(p), std::get<1>(p), std::get<2>(p));
            }
            results.push_back(std::move(slice_result));
            previous = current;
        }

        return results;
    }

private:
    void buildComplex(const std::vector<std::vector<int>> &simplices,
                      const std::vector<int> &dimensions)
    {
        if (dimensions.size() != simplices.size())
        {
            throw std::invalid_argument("dimension count must match simplex count");
        }
        const int n = checkedIntSize(simplices.size(), "zigzag complex simplex count");
        if (n > max_simplices_)
        {
            throw std::length_error("zigzag complex exceeds configured capacity");
        }

        copyToDevice(d_simplex_dims_, dimensions.data(), dimensions.size(),
                     "copy zigzag simplex dimensions");

        std::vector<int> complex_simplices(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i)
        {
            complex_simplices[static_cast<size_t>(i)] = i;
        }
        copyToDevice(d_complex_simplices_, complex_simplices.data(), complex_simplices.size(),
                     "copy zigzag complex simplices");
        copyToDevice(d_complex_size_, &n, 1, "copy zigzag complex size");

        const size_t boundary_count = checkedMulSize(static_cast<size_t>(n), static_cast<size_t>(n),
                                                     "zigzag host boundary count");
        std::vector<int> h_boundary(boundary_count, 0);
        copyToDevice(d_boundary_matrix_, h_boundary.data(), h_boundary.size(),
                     "copy zigzag boundary matrix");
    }

    void cleanup() noexcept
    {
        if (d_boundary_matrix_)
            cudaFree(d_boundary_matrix_);
        if (d_simplex_dims_)
            cudaFree(d_simplex_dims_);
        if (d_steps_)
            cudaFree(d_steps_);
        if (d_complex_simplices_)
            cudaFree(d_complex_simplices_);
        if (d_complex_size_)
            cudaFree(d_complex_size_);
        if (d_birth_indices_)
            cudaFree(d_birth_indices_);
        if (d_death_indices_)
            cudaFree(d_death_indices_);
        if (d_birth_times_)
            cudaFree(d_birth_times_);
        if (d_death_times_)
            cudaFree(d_death_times_);
        if (d_persistence_dim_)
            cudaFree(d_persistence_dim_);
        if (d_pair_count_)
            cudaFree(d_pair_count_);
        d_boundary_matrix_ = nullptr;
        d_simplex_dims_ = nullptr;
        d_steps_ = nullptr;
        d_complex_simplices_ = nullptr;
        d_complex_size_ = nullptr;
        d_birth_indices_ = nullptr;
        d_death_indices_ = nullptr;
        d_birth_times_ = nullptr;
        d_death_times_ = nullptr;
        d_persistence_dim_ = nullptr;
        d_pair_count_ = nullptr;
    }

    int max_simplices_ = 0;
    int max_steps_ = 0;

    int *d_boundary_matrix_ = nullptr;
    int *d_simplex_dims_ = nullptr;
    ZigzagStep *d_steps_ = nullptr;
    int *d_complex_simplices_ = nullptr;
    int *d_complex_size_ = nullptr;

    int *d_birth_indices_ = nullptr;
    int *d_death_indices_ = nullptr;
    float *d_birth_times_ = nullptr;
    float *d_death_times_ = nullptr;
    int *d_persistence_dim_ = nullptr;
    int *d_pair_count_ = nullptr;
};
