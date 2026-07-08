#include "nerve/spectral/persistent_laplacian.hpp"
#include "persistent_laplacian_gpu_internal.hpp"

#ifdef NERVE_HAS_CUDA
#include <cublas_v2.h>
#include <cuda_runtime.h>
#endif

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cmath>
#include <vector>

namespace nerve::spectral
{

#ifdef NERVE_HAS_CUDA

void CsrStorage::release()
{
    if (d_row_offsets)
        cudaFree(d_row_offsets);
    if (d_col_indices)
        cudaFree(d_col_indices);
    if (d_values)
        cudaFree(d_values);
    d_row_offsets = nullptr;
    d_col_indices = nullptr;
    d_values = nullptr;
}

double dotProduct(const double *a, const double *b, int n)
{
    double result = 0.0;
    cublasHandle_t handle = nullptr;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS)
        return 0.0;
    cublasDdot(handle, n, a, 1, b, 1, &result);
    cublasDestroy(handle);
    return result;
}

double norm(const double *x, int n)
{
    double result = 0.0;
    cublasHandle_t handle = nullptr;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS)
        return 0.0;
    cublasDnrm2(handle, n, x, 1, &result);
    cublasDestroy(handle);
    return result;
}

CsrStorage buildCsrOnDevice(const Eigen::SparseMatrix<double> &mat)
{
    CsrStorage csr;
    csr.n = static_cast<int>(mat.rows());
    csr.nnz = static_cast<int>(mat.nonZeros());

    std::vector<int> row_offsets(csr.n + 1);
    std::vector<int> col_indices(csr.nnz);
    std::vector<double> values(csr.nnz);

    int idx = 0;
    for (int k = 0; k < mat.outerSize(); ++k)
    {
        row_offsets[k] = idx;
        for (Eigen::SparseMatrix<double>::InnerIterator it(mat, k); it; ++it)
        {
            col_indices[idx] = static_cast<int>(it.col());
            values[idx] = it.value();
            ++idx;
        }
    }
    row_offsets[csr.n] = csr.nnz;

    cudaMalloc(&csr.d_row_offsets, (csr.n + 1) * sizeof(int));
    cudaMalloc(&csr.d_col_indices, csr.nnz * sizeof(int));
    cudaMalloc(&csr.d_values, csr.nnz * sizeof(double));

    cudaMemcpy(csr.d_row_offsets, row_offsets.data(), (csr.n + 1) * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(csr.d_col_indices, col_indices.data(), csr.nnz * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(csr.d_values, values.data(), csr.nnz * sizeof(double), cudaMemcpyHostToDevice);

    return csr;
}

void lanczosIteration(const CsrStorage &csr, const double *v0, int n, int krylov_dim,
                      std::vector<double> &alpha, std::vector<double> &beta)
{
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    double *d_v_prev = nullptr;
    double *d_v_curr = nullptr;
    double *d_w = nullptr;

    cudaMalloc(&d_v_prev, n * sizeof(double));
    cudaMalloc(&d_v_curr, n * sizeof(double));
    cudaMalloc(&d_w, n * sizeof(double));

    cudaMemcpy(d_v_curr, v0, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(d_v_prev, 0, n * sizeof(double));

    alpha.resize(krylov_dim);
    beta.resize(krylov_dim + 1);
    beta[0] = norm(d_v_curr, n);
    scaleKernel<<<grid_size, block_size>>>(n, 1.0 / beta[0], d_v_curr);

    for (int j = 0; j < krylov_dim; ++j)
    {
        csrSpMVKernel<<<grid_size, block_size>>>(csr.n, csr.d_row_offsets, csr.d_col_indices,
                                                 csr.d_values, d_v_curr, d_w);

        alpha[j] = dotProduct(d_v_curr, d_w, n);
        double aj = -alpha[j];
        axpyKernel<<<grid_size, block_size>>>(n, aj, d_v_curr, d_w);

        if (j > 0)
        {
            double bj = -beta[j];
            axpyKernel<<<grid_size, block_size>>>(n, bj, d_v_prev, d_w);
        }

        beta[j + 1] = norm(d_w, n);
        if (beta[j + 1] < kOrthoTol)
            break;

        scaleKernel<<<grid_size, block_size>>>(n, 1.0 / beta[j + 1], d_w);

        double *tmp = d_v_prev;
        d_v_prev = d_v_curr;
        d_v_curr = d_w;
        d_w = tmp;
    }

    cudaFree(d_v_prev);
    cudaFree(d_v_curr);
    cudaFree(d_w);
}

SpectralDecomposition tridiagToEigenpairs(const std::vector<double> &alpha,
                                          const std::vector<double> &beta, int n, int k)
{
    SpectralDecomposition result;
    const int effective_k = std::min(k, static_cast<int>(alpha.size()));

    Eigen::MatrixXd T = Eigen::MatrixXd::Zero(effective_k, effective_k);
    for (int i = 0; i < effective_k; ++i)
    {
        T(i, i) = alpha[i];
        if (i + 1 < effective_k)
        {
            T(i, i + 1) = beta[i + 1];
            T(i + 1, i) = beta[i + 1];
        }
    }

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(T);
    if (solver.info() != Eigen::Success)
        return result;

    result.eigenvalues.resize(effective_k);
    for (int i = 0; i < effective_k; ++i)
        result.eigenvalues[i] = solver.eigenvalues()[i];

    return result;
}

#endif

} // namespace nerve::spectral
