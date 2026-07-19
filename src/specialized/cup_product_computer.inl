class GPUCohomologyComputer
{
public:
    GPUCohomologyComputer(int max_simplices, int max_dim)
        : max_simplices_(max_simplices)
        , max_dim_(max_dim)
    {
        if (max_simplices_ <= 0 || max_dim_ < 0)
        {
            throw std::invalid_argument("cohomology computer dimensions are invalid");
        }

        try
        {
            const size_t simplex_count = static_cast<size_t>(max_simplices_);
            const size_t matrix_count =
                checkedMulSize(simplex_count, simplex_count, "cup product coboundary matrix count");
            allocateDevice(&d_cocycle_buffer_, simplex_count, "allocate cocycle buffer");
            allocateDevice(&d_coboundary_matrix_, matrix_count, "allocate coboundary matrix");
            allocateDevice(&d_cup_product_buffer_, simplex_count, "allocate cup product buffer");
            allocateDevice(&d_temp_simplices_, simplex_count, "allocate cup temp simplices");
            allocateDevice(&d_temp_coeffs_, simplex_count, "allocate cup temp coefficients");
        }
        catch (...)
        {
            cleanup();
            throw;
        }
    }

    ~GPUCohomologyComputer() { cleanup(); }

    /**
     * @brief Compute cohomology basis
     */
    std::vector<Cochain> computeCohomologyBasis(const std::vector<std::vector<int>> &simplices,
                                                const std::vector<std::vector<int>> &coboundaries)
    {
        buildCoboundaryMatrix(simplices, coboundaries);
        return reduceCohomology();
    }

    /**
     * @brief Compute cup product of two cohomology classes
     */
    Cochain cupProduct(const Cochain &alpha, const Cochain &beta)
    {
        const int alpha_index_count =
            checkedIntSize(alpha.simplex_indices.size(), "alpha simplex index count");
        const int alpha_coeff_count =
            checkedIntSize(alpha.coefficients.size(), "alpha coefficient count");
        const int beta_index_count =
            checkedIntSize(beta.simplex_indices.size(), "beta simplex index count");
        const int beta_coeff_count =
            checkedIntSize(beta.coefficients.size(), "beta coefficient count");
        const int alpha_size =
            std::min({alpha.num_simplices, alpha_index_count, alpha_coeff_count});
        const int beta_size = std::min({beta.num_simplices, beta_index_count, beta_coeff_count});
        Cochain result;
        result.dimension = alpha.dimension + beta.dimension;
        if (alpha_size <= 0 || beta_size <= 0 || active_simplices_ <= 0)
        {
            return result;
        }

        const size_t max_result_size =
            checkedMulSize(static_cast<size_t>(alpha_size), static_cast<size_t>(beta_size),
                           "cup product result count");
        const int max_result = checkedIntSize(max_result_size, "cup product result count");

        int *d_result_simplices = nullptr;
        float *d_result_coeffs = nullptr;
        int *d_result_size = nullptr;
        int *d_alpha_simplices = nullptr;
        float *d_alpha_coeffs = nullptr;
        int *d_beta_simplices = nullptr;
        float *d_beta_coeffs = nullptr;

        auto free_runtime = [&]() {
            if (d_alpha_simplices)
                cudaFree(d_alpha_simplices);
            if (d_alpha_coeffs)
                cudaFree(d_alpha_coeffs);
            if (d_beta_simplices)
                cudaFree(d_beta_simplices);
            if (d_beta_coeffs)
                cudaFree(d_beta_coeffs);
            if (d_result_simplices)
                cudaFree(d_result_simplices);
            if (d_result_coeffs)
                cudaFree(d_result_coeffs);
            if (d_result_size)
                cudaFree(d_result_size);
        };

        try
        {
            allocateDevice(&d_result_simplices, max_result_size, "allocate cup result simplices");
            allocateDevice(&d_result_coeffs, max_result_size, "allocate cup result coefficients");
            allocateDevice(&d_result_size, 1, "allocate cup result size");
            checkCuda(cudaMemset(d_result_size, 0, sizeof(int)), "reset cup result size");

            allocateDevice(&d_alpha_simplices, static_cast<size_t>(alpha_size),
                           "allocate alpha simplices");
            allocateDevice(&d_alpha_coeffs, static_cast<size_t>(alpha_size),
                           "allocate alpha coefficients");
            copyToDevice(d_alpha_simplices, alpha.simplex_indices.data(),
                         static_cast<size_t>(alpha_size), "copy alpha simplices");
            copyToDevice(d_alpha_coeffs, alpha.coefficients.data(), static_cast<size_t>(alpha_size),
                         "copy alpha coefficients");

            allocateDevice(&d_beta_simplices, static_cast<size_t>(beta_size),
                           "allocate beta simplices");
            allocateDevice(&d_beta_coeffs, static_cast<size_t>(beta_size),
                           "allocate beta coefficients");
            copyToDevice(d_beta_simplices, beta.simplex_indices.data(),
                         static_cast<size_t>(beta_size), "copy beta simplices");
            copyToDevice(d_beta_coeffs, beta.coefficients.data(), static_cast<size_t>(beta_size),
                         "copy beta coefficients");

            const int blocks = checkedGridBlocks(max_result_size, "cup product grid");
            cupProductKernel<<<blocks, CUP_PRODUCT_BLOCK_SIZE>>>(
                d_alpha_simplices, d_alpha_coeffs, alpha_size, alpha.dimension, d_beta_simplices,
                d_beta_coeffs, beta_size, beta.dimension, d_coboundary_matrix_, max_simplices_,
                active_simplices_, d_result_simplices, d_result_coeffs, d_result_size, max_result);
            checkCuda(cudaPeekAtLastError(), "launch cup product kernel");
            checkCuda(cudaDeviceSynchronize(), "synchronize cup product kernel");

            int result_size = 0;
            copyToHost(&result_size, d_result_size, 1, "copy cup result size");
            result_size = std::clamp(result_size, 0, max_result);

            result.num_simplices = result_size;
            result.simplex_indices.resize(static_cast<size_t>(result_size));
            result.coefficients.resize(static_cast<size_t>(result_size));
            copyToHost(result.simplex_indices.data(), d_result_simplices,
                       static_cast<size_t>(result_size), "copy cup result simplices");
            copyToHost(result.coefficients.data(), d_result_coeffs,
                       static_cast<size_t>(result_size), "copy cup result coefficients");

            free_runtime();
            return result;
        }
        catch (...)
        {
            free_runtime();
            throw;
        }
    }

    /**
     * @brief Compute full cohomology ring structure
     */
    std::map<std::pair<int, int>, std::vector<Cochain>>
    computeCohomologyRing(const std::vector<Cochain> &basis)
    {
        std::map<std::pair<int, int>, std::vector<Cochain>> ring_structure;

        for (size_t i = 0; i < basis.size(); ++i)
        {
            for (size_t j = i; j < basis.size(); ++j)
            {
                Cochain product = cupProduct(basis[i], basis[j]);
                if (product.num_simplices > 0)
                {
                    auto key = std::make_pair(static_cast<int>(i), static_cast<int>(j));
                    ring_structure[key].push_back(product);
                }
            }
        }

        return ring_structure;
    }

private:
    void buildCoboundaryMatrix(const std::vector<std::vector<int>> &simplices,
                               const std::vector<std::vector<int>> &coboundaries)
    {
        const int n = checkedIntSize(simplices.size(), "cup product simplex count");
        active_simplices_ = std::min(n, max_simplices_);
        const size_t matrix_count =
            checkedMulSize(static_cast<size_t>(max_simplices_), static_cast<size_t>(max_simplices_),
                           "cup product host matrix count");
        std::vector<int> h_matrix(matrix_count, 0);

        for (size_t i = 0; i < coboundaries.size(); ++i)
        {
            if (i >= static_cast<size_t>(active_simplices_))
            {
                break;
            }
            for (int cob : coboundaries[i])
            {
                if (cob >= 0 && cob < active_simplices_)
                {
                    h_matrix[static_cast<size_t>(cob) * max_simplices_ + i] = 1;
                }
            }
        }

        copyToDevice(d_coboundary_matrix_, h_matrix.data(), h_matrix.size(),
                     "copy cup coboundary matrix");
    }

    std::vector<Cochain> reduceCohomology()
    {
        const int n = active_simplices_ > 0 ? active_simplices_ : max_simplices_;
        const int blocks = checkedGridBlocks(static_cast<size_t>(n), "cohomology reduction grid");

        int *d_pivots = nullptr;
        int *d_basis = nullptr;
        auto free_runtime = [&]() {
            if (d_pivots)
                cudaFree(d_pivots);
            if (d_basis)
                cudaFree(d_basis);
        };

        try
        {
            allocateDevice(&d_pivots, static_cast<size_t>(max_simplices_),
                           "allocate cohomology pivots");
            allocateDevice(&d_basis, static_cast<size_t>(max_simplices_),
                           "allocate cohomology basis");
            checkCuda(cudaMemset(d_basis, -1,
                                 checkedMulSize(static_cast<size_t>(max_simplices_), sizeof(int),
                                                "cohomology basis bytes")),
                      "initialize cohomology basis");

            cohomologyReductionKernel<<<blocks, CUP_PRODUCT_BLOCK_SIZE>>>(
                d_coboundary_matrix_, n, max_simplices_, n, d_pivots, d_basis);
            checkCuda(cudaPeekAtLastError(), "launch cohomology reduction kernel");
            checkCuda(cudaDeviceSynchronize(), "synchronize cohomology reduction kernel");

            std::vector<int> h_basis(static_cast<size_t>(max_simplices_));
            copyToHost(h_basis.data(), d_basis, h_basis.size(), "copy cohomology basis");

            std::vector<Cochain> basis;
            for (int i = 0; i < n; ++i)
            {
                if (h_basis[static_cast<size_t>(i)] >= 0)
                {
                    Cochain c;
                    c.dimension =
                        (n > 0) ? static_cast<int>((static_cast<float>(i) / n) * max_dim_) : 0;
                    c.num_simplices = 1;
                    c.simplex_indices = {i};
                    c.coefficients = {1.0f};
                    basis.push_back(c);
                }
            }

            free_runtime();
            return basis;
        }
        catch (...)
        {
            free_runtime();
            throw;
        }
    }

    void cleanup() noexcept
    {
        if (d_cocycle_buffer_)
            cudaFree(d_cocycle_buffer_);
        if (d_coboundary_matrix_)
            cudaFree(d_coboundary_matrix_);
        if (d_cup_product_buffer_)
            cudaFree(d_cup_product_buffer_);
        if (d_temp_simplices_)
            cudaFree(d_temp_simplices_);
        if (d_temp_coeffs_)
            cudaFree(d_temp_coeffs_);
        d_cocycle_buffer_ = nullptr;
        d_coboundary_matrix_ = nullptr;
        d_cup_product_buffer_ = nullptr;
        d_temp_simplices_ = nullptr;
        d_temp_coeffs_ = nullptr;
    }

    int max_simplices_ = 0;
    int max_dim_ = 0;
    int active_simplices_ = 0;

    float *d_cocycle_buffer_ = nullptr;
    int *d_coboundary_matrix_ = nullptr;
    float *d_cup_product_buffer_ = nullptr;
    int *d_temp_simplices_ = nullptr;
    float *d_temp_coeffs_ = nullptr;
};
