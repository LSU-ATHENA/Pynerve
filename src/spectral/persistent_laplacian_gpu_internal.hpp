#pragma once

#include <cstddef>

namespace nerve::spectral
{

constexpr int kDefaultKrylovDim = 30;
constexpr double kOrthoTol = 1e-12;
constexpr std::size_t kBytesPerMb = 1024ULL * 1024ULL;
constexpr int kDefaultBlockSize = 256;

#ifdef NERVE_HAS_CUDA

struct CsrStorage
{
    int n = 0;
    int nnz = 0;
    int *d_row_offsets = nullptr;
    int *d_col_indices = nullptr;
    double *d_values = nullptr;

    void release();
};

__global__ void csrSpMVKernel(int n, const int *row_offsets, const int *col_indices,
                              const double *values, const double *x, double *y);

__global__ void axpyKernel(int n, double alpha, const double *x, double *y);

__global__ void scaleKernel(int n, double alpha, double *x);

__global__ void orthogonalizeKernel(int n, const double *v, double *w, double dot);

double dotProduct(const double *a, const double *b, int n);

double norm(const double *x, int n);

CsrStorage buildCsrOnDevice(const Eigen::SparseMatrix<double> &mat);

void lanczosIteration(const CsrStorage &csr, const double *v0, int n, int krylov_dim,
                      std::vector<double> &alpha, std::vector<double> &beta);

SpectralDecomposition tridiagToEigenpairs(const std::vector<double> &alpha,
                                          const std::vector<double> &beta, int n, int k);

#endif

} // namespace nerve::spectral
