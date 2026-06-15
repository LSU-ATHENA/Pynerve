#ifdef NERVE_HAS_CUDA

namespace nerve::spectral
{

__global__ void csrSpMVKernel(int n, const int *row_offsets, const int *col_indices,
                              const double *values, const double *x, double *y)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n)
        return;
    double sum = 0.0;
    for (int j = row_offsets[row]; j < row_offsets[row + 1]; ++j)
        sum += values[j] * x[col_indices[j]];
    y[row] = sum;
}

__global__ void axpyKernel(int n, double alpha, const double *x, double *y)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    y[i] += alpha * x[i];
}

__global__ void scaleKernel(int n, double alpha, double *x)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    x[i] *= alpha;
}

__global__ void orthogonalizeKernel(int n, const double *v, double *w, double dot)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    w[i] -= dot * v[i];
}

} // namespace nerve::spectral

#endif
