/**
 * @brief GPU kernel for cup product
 *
 * Computes cup product of two cochains
 * (alpha  U  beta)(sigma) = alpha(sigma|[v0,...,vp]) * beta(sigma|[vp,...,vp+q])
 */
__global__ void
cupProductKernel(const int *__restrict__ cochain1_simplices,
                 const float *__restrict__ cochain1_coeffs, int cochain1_size, int cochain1_dim,

                 const int *__restrict__ cochain2_simplices,
                 const float *__restrict__ cochain2_coeffs, int cochain2_size, int cochain2_dim,

                 const int *__restrict__ coboundary_matrix, int matrix_stride, int active_simplices,

                 int *__restrict__ result_simplices, float *__restrict__ result_coeffs,
                 int *__restrict__ result_size, int max_result_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= cochain1_size * cochain2_size)
        return;

    int i = idx / cochain2_size;
    int j = idx % cochain2_size;

    if (cochain1_dim < 0 || cochain2_dim < 0 || cochain1_dim + cochain2_dim > MAX_COHOM_DIM ||
        coboundary_matrix == nullptr || matrix_stride <= 0 || active_simplices <= 0)
    {
        return;
    }

    const int s1 = cochain1_simplices[i];
    const int s2 = cochain2_simplices[j];
    if (s1 < 0 || s1 >= active_simplices || s2 < 0 || s2 >= active_simplices)
    {
        return;
    }

    float c1 = cochain1_coeffs[i];
    float c2 = cochain2_coeffs[j];
    const float cup_coeff = c1 * c2;
    if (cup_coeff == 0.0f)
    {
        return;
    }

    /*
     * The sparse cochain representation stores simplex ids, while the
     * coboundary matrix stores coface rows by face columns. A non-zero cup
     * product must land on a simplex that contains both input faces; the kernel
     * chooses the lowest such coface to keep the operation deterministic.
     */
    int result_simplex = -1;
    for (int candidate = 0; candidate < active_simplices; ++candidate)
    {
        const bool contains_first =
            candidate == s1 || coboundary_matrix[candidate * matrix_stride + s1] != 0;
        const bool contains_second =
            candidate == s2 || coboundary_matrix[candidate * matrix_stride + s2] != 0;
        if (contains_first && contains_second)
        {
            result_simplex = candidate;
            break;
        }
    }

    if (result_simplex < 0)
    {
        return;
    }

    int result_idx = atomicAdd(result_size, 1);
    if (result_idx < max_result_size)
    {
        result_simplices[result_idx] = result_simplex;
        result_coeffs[result_idx] = cup_coeff;
    }
}

/**
 * @brief GPU kernel for cohomology basis reduction
 */
__global__ void cohomologyReductionKernel(int *__restrict__ cocycle_matrix, int num_cocycles,
                                          int matrix_stride, int num_simplices,
                                          int *__restrict__ pivot_rows,
                                          int *__restrict__ basis_indices)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col >= num_cocycles)
        return;

    // Find pivot (lowest non-zero entry)
    int pivot = -1;
    for (int row = num_simplices - 1; row >= 0; --row)
    {
        if (cocycle_matrix[row * matrix_stride + col] != 0)
        {
            pivot = row;
            break;
        }
    }

    pivot_rows[col] = pivot;

    // Atomic write to basis if unique pivot
    if (pivot >= 0)
    {
        int expected = -1;
        atomicCAS(&basis_indices[pivot], expected, col);
    }
}
