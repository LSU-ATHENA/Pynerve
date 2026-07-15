class GPUSheafLaplacian
{
public:
    GPUSheafLaplacian(int num_stalks, int max_stalk_dim)
        : num_stalks_(num_stalks)
        , max_stalk_dim_(max_stalk_dim)
    {
        if (num_stalks_ <= 0 || max_stalk_dim_ <= 0)
        {
            throw std::invalid_argument(
                "GPUSheafLaplacian requires positive stalk count and stalk dimension");
        }

        try
        {
            allocateDevice(&d_laplacian_diag_, static_cast<size_t>(num_stalks_),
                           "allocate sheaf Laplacian diagonal");
            allocateDevice(&d_rhs_, static_cast<size_t>(num_stalks_), "allocate sheaf RHS");
            allocateDevice(&d_solution_, static_cast<size_t>(num_stalks_),
                           "allocate sheaf solution");
            checkCusparse(cusparseCreate(&cusparse_handle_),
                          "create GPUSheafLaplacian cuSPARSE handle");
        }
        catch (...)
        {
            cleanup();
            throw;
        }
    }

    ~GPUSheafLaplacian() { cleanup(); }

    [[nodiscard]] int getNumStalks() const noexcept { return num_stalks_; }

    void buildLaplacian(const std::vector<int> &coboundary_row_ptr,
                        const std::vector<int> &coboundary_col_idx,
                        const std::vector<float> &coboundary_values,
                        const std::vector<RestrictionMap> &restrictions)
    {
        if (coboundary_row_ptr.size() != static_cast<size_t>(num_stalks_) + 1)
        {
            throw std::invalid_argument("coboundary_row_ptr size must equal num_stalks + 1");
        }
        if (coboundary_col_idx.size() != coboundary_values.size())
        {
            throw std::invalid_argument("coboundary_col_idx and coboundary_values sizes differ");
        }
        requireFiniteInput(coboundary_values, "coboundary values must be finite");
        checkedIntSize(coboundary_col_idx.size(), "sheaf coboundary nnz");
        const int restriction_count =
            checkedIntSize(restrictions.size(), "sheaf restriction count");
        if (!coboundary_row_ptr.empty() &&
            coboundary_row_ptr.back() != static_cast<int>(coboundary_col_idx.size()))
        {
            throw std::invalid_argument("coboundary_row_ptr final entry must equal coboundary nnz");
        }
        for (size_t i = 1; i < coboundary_row_ptr.size(); ++i)
        {
            if (coboundary_row_ptr[i - 1] > coboundary_row_ptr[i] || coboundary_row_ptr[i] < 0)
            {
                throw std::invalid_argument(
                    "coboundary_row_ptr must be monotonic and non-negative");
            }
        }
        for (int col : coboundary_col_idx)
        {
            if (col < 0 || col >= num_stalks_)
            {
                throw std::out_of_range(
                    "coboundary_col_idx contains a stalk outside sheaf dimensions");
            }
        }

        int *d_row_ptr = nullptr;
        int *d_col_idx = nullptr;
        float *d_vals = nullptr;
        RestrictionMap *d_restrictions = nullptr;
        std::vector<int *> d_restriction_indices(restrictions.size(), nullptr);
        std::vector<float *> d_restriction_values(restrictions.size(), nullptr);

        auto free_runtime = [&]() {
            if (d_row_ptr)
                cudaFree(d_row_ptr);
            if (d_col_idx)
                cudaFree(d_col_idx);
            if (d_vals)
                cudaFree(d_vals);
            if (d_restrictions)
                cudaFree(d_restrictions);
            for (int *ptr : d_restriction_indices)
            {
                if (ptr)
                    cudaFree(ptr);
            }
            for (float *ptr : d_restriction_values)
            {
                if (ptr)
                    cudaFree(ptr);
            }
        };

        try
        {
            allocateDevice(&d_row_ptr, coboundary_row_ptr.size(), "allocate sheaf row pointer");
            allocateDevice(&d_col_idx, coboundary_col_idx.size(), "allocate sheaf column indices");
            allocateDevice(&d_vals, coboundary_values.size(), "allocate sheaf coboundary values");
            copyToDevice(d_row_ptr, coboundary_row_ptr.data(), coboundary_row_ptr.size(),
                         "copy sheaf row pointer");
            copyToDevice(d_col_idx, coboundary_col_idx.data(), coboundary_col_idx.size(),
                         "copy sheaf column indices");
            copyToDevice(d_vals, coboundary_values.data(), coboundary_values.size(),
                         "copy sheaf coboundary values");

            std::vector<RestrictionMap> device_restrictions = restrictions;
            for (size_t r = 0; r < restrictions.size(); ++r)
            {
                const RestrictionMap &map = restrictions[r];
                if (map.from_stalk < 0 || map.from_stalk >= num_stalks_ || map.to_stalk < 0 ||
                    map.to_stalk >= num_stalks_ || map.nnz < 0)
                {
                    throw std::invalid_argument(
                        "restriction map metadata is outside sheaf dimensions");
                }
                if (map.nnz > 0 && (map.indices == nullptr || map.values == nullptr))
                {
                    throw std::invalid_argument("restriction map storage is invalid");
                }
                const size_t nnz = static_cast<size_t>(map.nnz);
                requireFiniteInput(map.values, nnz, "restriction map values must be finite");
                allocateDevice(&d_restriction_indices[r], nnz,
                               "allocate sheaf restriction indices");
                allocateDevice(&d_restriction_values[r], nnz, "allocate sheaf restriction values");
                copyToDevice(d_restriction_indices[r], map.indices, nnz,
                             "copy sheaf restriction indices");
                copyToDevice(d_restriction_values[r], map.values, nnz,
                             "copy sheaf restriction values");
                device_restrictions[r].indices = d_restriction_indices[r];
                device_restrictions[r].values = d_restriction_values[r];
            }

            allocateDevice(&d_restrictions, restrictions.size(), "allocate sheaf restrictions");
            copyToDevice(d_restrictions, device_restrictions.data(), device_restrictions.size(),
                         "copy sheaf restrictions");

            const int blocks =
                checkedGridBlocks(static_cast<size_t>(num_stalks_), "sheaf Laplacian grid");
            sheafLaplacianKernel<<<blocks, BLOCK_SIZE>>>(d_row_ptr, d_col_idx, d_vals,
                                                         d_restrictions, restriction_count,
                                                         d_laplacian_diag_, num_stalks_);
            checkCuda(cudaPeekAtLastError(), "launch sheaf Laplacian kernel");
            checkCuda(cudaDeviceSynchronize(), "synchronize sheaf Laplacian kernel");
            std::vector<float> diagonal(static_cast<size_t>(num_stalks_));
            copyToHost(diagonal.data(), d_laplacian_diag_, diagonal.size(),
                       "copy sheaf Laplacian diagonal");
            requireFiniteOutput(diagonal,
                                "sheaf Laplacian construction produced non-finite values");
            laplacian_built_ = true;
            free_runtime();
        }
        catch (...)
        {
            free_runtime();
            throw;
        }
    }

    void solve(const std::vector<float> &rhs, std::vector<float> &solution)
    {
        if (rhs.size() != static_cast<size_t>(num_stalks_))
        {
            throw std::invalid_argument("rhs size must equal num_stalks");
        }
        if (!laplacian_built_)
        {
            throw std::logic_error("sheaf Laplacian must be built before solve");
        }
        requireFiniteInput(rhs, "sheaf RHS must be finite");

        copyToDevice(d_rhs_, rhs.data(), rhs.size(), "copy sheaf RHS");
        checkCuda(cudaMemset(d_solution_, 0,
                             checkedBytes(static_cast<size_t>(num_stalks_), sizeof(float),
                                          "clear sheaf solution bytes")),
                  "clear sheaf solution");

        const int blocks =
            checkedGridBlocks(static_cast<size_t>(num_stalks_), "sheaf cohomology grid");
        sheafCohomologyKernel<<<blocks, BLOCK_SIZE>>>(d_laplacian_diag_, d_rhs_, d_solution_,
                                                      num_stalks_, SHEAF_MAX_ITERATIONS);
        checkCuda(cudaPeekAtLastError(), "launch sheaf cohomology kernel");
        checkCuda(cudaDeviceSynchronize(), "synchronize sheaf cohomology kernel");

        solution.resize(static_cast<size_t>(num_stalks_));
        copyToHost(solution.data(), d_solution_, solution.size(), "copy sheaf solution");
        requireFiniteOutput(solution, "sheaf solve produced non-finite values");
    }

    void applyRestriction(const std::vector<float> &from_data, std::vector<float> &to_data,
                          const RestrictionMap &map)
    {
        if (map.nnz < 0 || ((map.indices == nullptr || map.values == nullptr) && map.nnz > 0))
        {
            throw std::invalid_argument("RestrictionMap storage is invalid");
        }
        const int from_dim = checkedIntSize(from_data.size(), "restriction input dimension");
        const int to_dim = checkedIntSize(to_data.size(), "restriction output dimension");
        if (to_dim == 0)
        {
            return;
        }
        for (int i = 0; i < map.nnz; ++i)
        {
            if (map.indices[i] < 0 || map.indices[i] >= from_dim)
            {
                throw std::out_of_range("restriction map index is outside input dimension");
            }
        }
        requireFiniteInput(from_data, "restriction input data must be finite");
        requireFiniteInput(map.values, static_cast<size_t>(map.nnz),
                           "restriction map values must be finite");

        int *d_map_indices = nullptr;
        float *d_map_values = nullptr;
        float *d_from = nullptr;
        float *d_to = nullptr;
        auto free_runtime = [&]() {
            if (d_map_indices)
                cudaFree(d_map_indices);
            if (d_map_values)
                cudaFree(d_map_values);
            if (d_from)
                cudaFree(d_from);
            if (d_to)
                cudaFree(d_to);
        };

        try
        {
            const size_t nnz = static_cast<size_t>(map.nnz);
            allocateDevice(&d_map_indices, nnz, "allocate restriction indices");
            allocateDevice(&d_map_values, nnz, "allocate restriction values");
            allocateDevice(&d_from, from_data.size(), "allocate restriction input");
            allocateDevice(&d_to, to_data.size(), "allocate restriction output");

            copyToDevice(d_map_indices, map.indices, nnz, "copy restriction indices");
            copyToDevice(d_map_values, map.values, nnz, "copy restriction values");
            copyToDevice(d_from, from_data.data(), from_data.size(), "copy restriction input");

            const int blocks = checkedGridBlocks(to_data.size(), "restriction kernel grid");
            applyRestrictionKernel<<<blocks, BLOCK_SIZE>>>(d_from, d_to, d_map_indices,
                                                           d_map_values, map.nnz, from_dim, to_dim);
            checkCuda(cudaPeekAtLastError(), "launch restriction kernel");
            checkCuda(cudaDeviceSynchronize(), "synchronize restriction kernel");

            copyToHost(to_data.data(), d_to, to_data.size(), "copy restriction output");
            requireFiniteOutput(to_data, "restriction application produced non-finite values");
            free_runtime();
        }
        catch (...)
        {
            free_runtime();
            throw;
        }
    }

private:
    void cleanup() noexcept
    {
        if (d_laplacian_diag_)
            cudaFree(d_laplacian_diag_);
        if (d_rhs_)
            cudaFree(d_rhs_);
        if (d_solution_)
            cudaFree(d_solution_);
        if (cusparse_handle_)
            cusparseDestroy(cusparse_handle_);
        d_laplacian_diag_ = nullptr;
        d_rhs_ = nullptr;
        d_solution_ = nullptr;
        cusparse_handle_ = nullptr;
    }

    int num_stalks_;
    int max_stalk_dim_;
    bool laplacian_built_ = false;

    float *d_laplacian_diag_ = nullptr;
    float *d_rhs_ = nullptr;
    float *d_solution_ = nullptr;

    cusparseHandle_t cusparse_handle_ = nullptr;
};
