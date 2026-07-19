__device__ inline float eigensolverInfinityFloat()
{
    return __int_as_float(0x7f800000);
}

__device__ inline double eigensolverInfinityDouble()
{
    return __longlong_as_double(0x7ff0000000000000ULL);
}

__global__ void convertFP32toFP64Kernel(const float *__restrict__ input,
                                        double *__restrict__ output, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        const float value = input[idx];
        output[idx] = isfinite(value) ? static_cast<double>(value) : eigensolverInfinityDouble();
    }
}

__global__ void convertFP64toFP32Kernel(const double *__restrict__ input,
                                        float *__restrict__ output, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        constexpr double max_float = 3.40282346638528859812e38;
        const double value = input[idx];
        output[idx] = isfinite(value) && fabs(value) <= max_float ? static_cast<float>(value)
                                                                  : eigensolverInfinityFloat();
    }
}

namespace
{

bool valuesAreFinite(const std::vector<float> &values)
{
    return std::all_of(values.begin(), values.end(),
                       [](float value) { return std::isfinite(value); });
}

void requireFiniteValues(const std::vector<float> &values, const char *context)
{
    if (!valuesAreFinite(values))
    {
        throw std::invalid_argument(context);
    }
}

size_t checkedProduct(size_t lhs, size_t rhs, const char *context)
{
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs)
    {
        throw std::length_error(context);
    }
    return lhs * rhs;
}

size_t checkedByteCount(size_t count, size_t element_size, const char *context)
{
    return checkedProduct(count, element_size, context);
}

size_t checkedPositiveSquareElements(int matrix_size, const char *context)
{
    if (matrix_size <= 0)
    {
        throw std::invalid_argument(context);
    }
    const size_t n = static_cast<size_t>(matrix_size);
    return checkedProduct(n, n, context);
}

int checkedElementCountForIntKernel(size_t count, const char *context)
{
    if (count > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        throw std::length_error(context);
    }
    return static_cast<int>(count);
}

int checkedGridBlocks(size_t elements, int block_size, const char *context)
{
    const size_t blocks =
        (elements + static_cast<size_t>(block_size) - 1) / static_cast<size_t>(block_size);
    if (blocks > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        throw std::length_error(context);
    }
    return static_cast<int>(blocks);
}

void checkCuSolver(cusolverStatus_t status, const char *expression)
{
    if (status != CUSOLVER_STATUS_SUCCESS)
    {
        throw std::runtime_error(std::string(expression) + " failed with cuSOLVER status " +
                                 std::to_string(static_cast<int>(status)));
    }
}

void checkCuBlas(cublasStatus_t status, const char *expression)
{
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        throw std::runtime_error(std::string(expression) + " failed with cuBLAS status " +
                                 std::to_string(static_cast<int>(status)));
    }
}

void checkSolverInfo(int *d_info, const char *context)
{
    int info = 0;
    GPU_CHECK(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (info != 0)
    {
        throw std::runtime_error(std::string(context) + " failed with solver info " +
                                 std::to_string(info));
    }
}

} // namespace

/**
 * @brief GPU eigensolver using cuSOLVER
 */
class GPUEigensolver
{
public:
    GPUEigensolver(int matrix_size, bool use_mixed_precision = true)
        : matrix_size_(matrix_size)
        , use_mixed_precision_(use_mixed_precision)
    {
        matrix_entries_ = checkedPositiveSquareElements(
            matrix_size_, "GPU eigensolver matrix size must be positive");
        matrix_float_bytes_ =
            checkedByteCount(matrix_entries_, sizeof(float),
                             "GPU eigensolver matrix allocation exceeds host limits");
        eigenvalue_float_bytes_ =
            checkedByteCount(static_cast<size_t>(matrix_size_), sizeof(float),
                             "GPU eigensolver eigenvalue allocation exceeds host limits");
        eigenvector_float_bytes_ = matrix_float_bytes_;

        if (use_mixed_precision_)
        {
            matrix_entries_int_ = checkedElementCountForIntKernel(
                matrix_entries_, "GPU eigensolver mixed-precision matrix exceeds kernel limits");
            matrix_double_bytes_ =
                checkedByteCount(matrix_entries_, sizeof(double),
                                 "GPU eigensolver FP64 matrix allocation exceeds host limits");
            eigenvalue_double_bytes_ =
                checkedByteCount(static_cast<size_t>(matrix_size_), sizeof(double),
                                 "GPU eigensolver FP64 eigenvalue allocation exceeds host limits");
        }

        try
        {
            // Create cuSOLVER handle
            checkCuSolver(cusolverDnCreate(&cusolver_handle_), "cusolverDnCreate");
            checkCuBlas(cublasCreate(&cublas_handle_), "cublasCreate");

            // Allocate device memory
            GPU_CHECK(cudaMalloc(&d_matrix_, matrix_float_bytes_));
            GPU_CHECK(cudaMalloc(&d_eigenvalues_, eigenvalue_float_bytes_));
            GPU_CHECK(cudaMalloc(&d_eigenvectors_, eigenvector_float_bytes_));
            GPU_CHECK(cudaMalloc(&d_work_, getWorkspaceSize()));
            GPU_CHECK(cudaMalloc(&d_info_, sizeof(int)));

            // Mixed precision buffers
            if (use_mixed_precision_)
            {
                GPU_CHECK(cudaMalloc(&d_matrix_fp64_, matrix_double_bytes_));
                GPU_CHECK(cudaMalloc(&d_eigenvalues_fp64_, eigenvalue_double_bytes_));
            }
        }
        catch (...)
        {
            cleanup();
            throw;
        }
    }

    ~GPUEigensolver() { cleanup(); }

    /**
     * @brief Solve eigenvalue problem: A * v = lambda * v
     *
     * @param matrix Input symmetric matrix (column-major)
     * @param eigenvalues Output eigenvalues (ascending order)
     * @param eigenvectors Output eigenvectors (column-major)
     */
    void solve(const std::vector<float> &matrix, std::vector<float> &eigenvalues,
               std::vector<float> &eigenvectors)
    {
        if (matrix.size() != matrix_entries_)
        {
            throw std::invalid_argument("GPU eigensolver input matrix has invalid shape");
        }
        requireFiniteValues(matrix, "GPU eigensolver input matrix must be finite");

        // Copy matrix to device
        GPU_CHECK(
            cudaMemcpy(d_matrix_, matrix.data(), matrix_float_bytes_, cudaMemcpyHostToDevice));

        if (use_mixed_precision_)
        {
            solveMixedPrecision();
        }
        else
        {
            solveFP32();
        }

        // Copy results back
        eigenvalues.resize(matrix_size_);
        eigenvectors.resize(matrix_entries_);

        GPU_CHECK(cudaMemcpy(eigenvalues.data(), d_eigenvalues_, eigenvalue_float_bytes_,
                             cudaMemcpyDeviceToHost));
        GPU_CHECK(cudaMemcpy(eigenvectors.data(), d_eigenvectors_, eigenvector_float_bytes_,
                             cudaMemcpyDeviceToHost));

        if (!valuesAreFinite(eigenvalues) || !valuesAreFinite(eigenvectors))
        {
            eigenvalues.clear();
            eigenvectors.clear();
            throw std::runtime_error("GPU eigensolver produced non-finite output");
        }
    }

    /**
     * @brief Solve for k smallest eigenvalues only
     */
    void solveKSmallest(const std::vector<float> &matrix, int k, std::vector<float> &eigenvalues,
                        std::vector<float> &eigenvectors)
    {
        if (k <= 0 || k > matrix_size_)
        {
            eigenvalues.clear();
            eigenvectors.clear();
            return;
        }

        // Solve full problem then extract k smallest
        std::vector<float> all_eigenvalues;
        std::vector<float> all_eigenvectors;

        solve(matrix, all_eigenvalues, all_eigenvectors);

        // Extract k smallest (already sorted ascending by cuSOLVER)
        eigenvalues.resize(static_cast<size_t>(k));
        eigenvectors.resize(static_cast<size_t>(matrix_size_) * static_cast<size_t>(k));

        std::copy(all_eigenvalues.begin(), all_eigenvalues.begin() + k, eigenvalues.begin());

        // Copy first k eigenvectors
        for (int i = 0; i < k; ++i)
        {
            const size_t src_offset = static_cast<size_t>(i) * static_cast<size_t>(matrix_size_);
            const size_t dst_offset = src_offset;
            std::copy(all_eigenvectors.begin() + src_offset,
                      all_eigenvectors.begin() + src_offset + static_cast<size_t>(matrix_size_),
                      eigenvectors.begin() + dst_offset);
        }
    }

private:
    int matrix_size_;
    bool use_mixed_precision_;
    size_t matrix_entries_ = 0;
    int matrix_entries_int_ = 0;
    size_t matrix_float_bytes_ = 0;
    size_t eigenvalue_float_bytes_ = 0;
    size_t eigenvector_float_bytes_ = 0;
    size_t matrix_double_bytes_ = 0;
    size_t eigenvalue_double_bytes_ = 0;

    cusolverDnHandle_t cusolver_handle_ = nullptr;
    cublasHandle_t cublas_handle_ = nullptr;

    float *d_matrix_ = nullptr;
    float *d_eigenvalues_ = nullptr;
    float *d_eigenvectors_ = nullptr;
    float *d_work_ = nullptr;
    int *d_info_ = nullptr;

    double *d_matrix_fp64_ = nullptr;
    double *d_eigenvalues_fp64_ = nullptr;

    size_t getWorkspaceSize()
    {
        int lwork = 0;
        checkCuSolver(cusolverDnSsyevd_bufferSize(cusolver_handle_, CUSOLVER_EIG_MODE_VECTOR,
                                                  CUBLAS_FILL_MODE_LOWER, matrix_size_, d_matrix_,
                                                  matrix_size_, d_eigenvalues_, &lwork),
                      "cusolverDnSsyevd_bufferSize");
        if (lwork < 0)
        {
            throw std::runtime_error("GPU eigensolver workspace size is negative");
        }
        return checkedByteCount(static_cast<size_t>(lwork), sizeof(float),
                                "GPU eigensolver workspace allocation exceeds host limits");
    }

    void cleanup() noexcept
    {
        if (d_matrix_)
        {
            cudaFree(d_matrix_);
            d_matrix_ = nullptr;
        }
        if (d_eigenvalues_)
        {
            cudaFree(d_eigenvalues_);
            d_eigenvalues_ = nullptr;
        }
        if (d_eigenvectors_)
        {
            cudaFree(d_eigenvectors_);
            d_eigenvectors_ = nullptr;
        }
        if (d_work_)
        {
            cudaFree(d_work_);
            d_work_ = nullptr;
        }
        if (d_info_)
        {
            cudaFree(d_info_);
            d_info_ = nullptr;
        }
        if (d_matrix_fp64_)
        {
            cudaFree(d_matrix_fp64_);
            d_matrix_fp64_ = nullptr;
        }
        if (d_eigenvalues_fp64_)
        {
            cudaFree(d_eigenvalues_fp64_);
            d_eigenvalues_fp64_ = nullptr;
        }
        if (cusolver_handle_)
        {
            cusolverDnDestroy(cusolver_handle_);
            cusolver_handle_ = nullptr;
        }
        if (cublas_handle_)
        {
            cublasDestroy(cublas_handle_);
            cublas_handle_ = nullptr;
        }
    }

    void solveFP32()
    {
        // Standard FP32 solve using divide and conquer
        checkCuSolver(cusolverDnSsyevd(cusolver_handle_,
                                       CUSOLVER_EIG_MODE_VECTOR, // Compute eigenvectors
                                       CUBLAS_FILL_MODE_LOWER, matrix_size_, d_matrix_,
                                       matrix_size_, d_eigenvalues_, d_work_,
                                       getWorkspaceSize() / sizeof(float), d_info_),
                      "cusolverDnSsyevd");
        checkSolverInfo(d_info_, "cusolverDnSsyevd");

        // Copy eigenvectors (matrix is overwritten with eigenvectors)
        GPU_CHECK(cudaMemcpy(d_eigenvectors_, d_matrix_, eigenvector_float_bytes_,
                             cudaMemcpyDeviceToDevice));
    }

    void solveMixedPrecision()
    {
        // Convert to FP64
        int matrix_blocks =
            checkedGridBlocks(matrix_entries_, EIGENSOLVER_BLOCK_SIZE,
                              "GPU eigensolver mixed-precision grid exceeds CUDA limits");
        convertFP32toFP64Kernel<<<matrix_blocks, EIGENSOLVER_BLOCK_SIZE>>>(
            d_matrix_, d_matrix_fp64_, matrix_entries_int_);
        GPU_CHECK(cudaPeekAtLastError());

        // Solve in FP64
        int lwork = 0;
        checkCuSolver(cusolverDnDsyevd_bufferSize(
                          cusolver_handle_, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
                          matrix_size_, d_matrix_fp64_, matrix_size_, d_eigenvalues_fp64_, &lwork),
                      "cusolverDnDsyevd_bufferSize");
        if (lwork < 0)
        {
            throw std::runtime_error("GPU eigensolver FP64 workspace size is negative");
        }

        double *d_work_fp64 = nullptr;
        const size_t fp64_workspace_bytes =
            checkedByteCount(static_cast<size_t>(lwork), sizeof(double),
                             "GPU eigensolver FP64 workspace allocation exceeds host limits");
        GPU_CHECK(cudaMalloc(&d_work_fp64, fp64_workspace_bytes));

        try
        {
            checkCuSolver(cusolverDnDsyevd(cusolver_handle_, CUSOLVER_EIG_MODE_VECTOR,
                                           CUBLAS_FILL_MODE_LOWER, matrix_size_, d_matrix_fp64_,
                                           matrix_size_, d_eigenvalues_fp64_, d_work_fp64, lwork,
                                           d_info_),
                          "cusolverDnDsyevd");
            checkSolverInfo(d_info_, "cusolverDnDsyevd");
        }
        catch (...)
        {
            cudaFree(d_work_fp64);
            throw;
        }

        cudaFree(d_work_fp64);

        // Convert back to FP32
        const int eigenvalue_blocks =
            checkedGridBlocks(static_cast<size_t>(matrix_size_), EIGENSOLVER_BLOCK_SIZE,
                              "GPU eigensolver eigenvalue grid exceeds CUDA limits");
        convertFP64toFP32Kernel<<<eigenvalue_blocks, EIGENSOLVER_BLOCK_SIZE>>>(
            d_eigenvalues_fp64_, d_eigenvalues_, matrix_size_);
        GPU_CHECK(cudaPeekAtLastError());
        convertFP64toFP32Kernel<<<matrix_blocks, EIGENSOLVER_BLOCK_SIZE>>>(
            d_matrix_fp64_, d_eigenvectors_, matrix_entries_int_);
        GPU_CHECK(cudaPeekAtLastError());
    }
};
