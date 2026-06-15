/**
 * @brief GPU kernel for sheaf Laplacian construction
 *
 * Builds the sheaf Laplacian matrix L = D^T D where D is the coboundary
 */
__global__ void
sheafLaplacianKernel(const int *__restrict__ coboundary_row_ptr,
                     const int *__restrict__ coboundary_col_idx,
                     const float *__restrict__ coboundary_values,
                     const RestrictionMap *__restrict__ restrictions,
                     int num_restriction_maps,
                     float *__restrict__ laplacian_values, int num_stalks) {
  int stalk_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (stalk_idx >= num_stalks)
    return;

  // Compute diagonal entry (degree)
  float degree = 0.0f;

  // Iterate through coboundary entries affecting this stalk
  int row_start = coboundary_row_ptr[stalk_idx];
  int row_end = coboundary_row_ptr[stalk_idx + 1];

  for (int j = row_start; j < row_end; ++j) {
    float val = coboundary_values[j];
    const float contribution = val * val;
    const float next = degree + contribution;
    if (!isfinite(val) || !isfinite(contribution) || !isfinite(next)) {
      degree = INFINITY;
      break;
    }
    degree = next;
  }

  // Apply restriction map contributions
  for (int r = 0; r < num_restriction_maps && isfinite(degree); ++r) {
    const RestrictionMap &rest = restrictions[r];
    if (rest.from_stalk == stalk_idx || rest.to_stalk == stalk_idx) {
      float rest_norm = 0.0f;
      for (int i = 0; i < rest.nnz; ++i) {
        const float value = rest.values[i];
        const float contribution = value * value;
        const float next = rest_norm + contribution;
        if (!isfinite(value) || !isfinite(contribution) || !isfinite(next)) {
          rest_norm = INFINITY;
          break;
        }
        rest_norm = next;
      }
      const float next_degree = degree + rest_norm;
      if (!isfinite(rest_norm) || !isfinite(next_degree)) {
        degree = INFINITY;
        break;
      }
      degree = next_degree;
    }
  }

  laplacian_values[stalk_idx] = degree;
}

/**
 * @brief Kernel for parallel stalk data operations
 */
__global__ void
stalkOperationKernel(float *__restrict__ stalk_data,
                     const float *__restrict__ input_data, int stalk_size,
                     int operation // 0=copy, 1=add, 2=scale, 3=normalize
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= stalk_size)
    return;

  switch (operation) {
  case 0: // copy
    stalk_data[idx] = isfinite(input_data[idx]) ? input_data[idx] : INFINITY;
    break;
  case 1: { // add
    const float next = stalk_data[idx] + input_data[idx];
    stalk_data[idx] =
        (isfinite(stalk_data[idx]) && isfinite(input_data[idx]) && isfinite(next)) ? next
                                                                                   : INFINITY;
    break;
  }
  case 2: { // scale
    const float next = stalk_data[idx] * input_data[idx];
    stalk_data[idx] =
        (isfinite(stalk_data[idx]) && isfinite(input_data[idx]) && isfinite(next)) ? next
                                                                                   : INFINITY;
    break;
  }
  case 3: { // normalize
    // Compute norm across block
    float val = stalk_data[idx];
    float sum_sq = val * val;
    if (!isfinite(val) || !isfinite(sum_sq)) {
      stalk_data[idx] = INFINITY;
      break;
    }

    // Warp-level reduction
    cg::thread_block_tile<32> warp =
        cg::tiled_partition<32>(cg::this_thread_block());

    for (int offset = 16; offset > 0; offset /= 2) {
      const float next = sum_sq + warp.shfl_down(sum_sq, offset);
      sum_sq = isfinite(next) ? next : INFINITY;
    }

    float norm = sqrtf(sum_sq + 1e-8f);
    stalk_data[idx] = (isfinite(norm) && norm > 0.0f) ? val / norm : INFINITY;
    break;
  }
  }
}

/**
 * @brief Kernel for restriction map application
 */
__global__ void
applyRestrictionKernel(const float *__restrict__ from_stalk_data,
                       float *__restrict__ to_stalk_data,
                       const int *__restrict__ map_indices,
                       const float *__restrict__ map_values, int map_nnz,
                       int from_dim, int to_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= to_dim)
    return;

  float result = 0.0f;

  // Sparse matrix-vector multiplication
  for (int i = 0; i < map_nnz; ++i) {
    int col = map_indices[i];
    float val = map_values[i];

    if (col >= 0 && col < from_dim) {
      const float source = from_stalk_data[col];
      const float contribution = val * source;
      const float next = result + contribution;
      if (!isfinite(val) || !isfinite(source) || !isfinite(contribution) || !isfinite(next)) {
        result = INFINITY;
        break;
      }
      result = next;
    }
  }

  to_stalk_data[idx] = result;
}

/**
 * @brief Sheaf cohomology computation kernel
 *
 * Solves L x = b for sheaf cohomology classes
 */
__global__ void sheafCohomologyKernel(const float *__restrict__ laplacian_diag,
                                      const float *__restrict__ b,
                                      float *__restrict__ x, int n,
                                      int max_iters) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= n)
    return;

  // Damped Jacobi on a diagonal preconditioner.
  // This keeps the kernel numerically stable for near-singular diagonal
  // entries.
  float x_val = x[idx];
  const float rhs = b[idx];
  const float diag =
      fabsf(laplacian_diag[idx]) > 1e-8f ? laplacian_diag[idx] : 1e-8f;
  constexpr float jacobi_omega = 0.8f;
  if (!isfinite(x_val) || !isfinite(rhs) || !isfinite(diag)) {
    x[idx] = INFINITY;
    return;
  }

  for (int iter = 0; iter < max_iters; ++iter) {
    const float residual = rhs - diag * x_val;
    const float delta = jacobi_omega * (residual / diag);
    const float next = x_val + delta;
    if (!isfinite(residual) || !isfinite(delta) || !isfinite(next)) {
      x_val = INFINITY;
      break;
    }
    x_val = next;
    if (fabsf(delta) < SHEAF_CONVERGENCE_TOLERANCE) {
      break;
    }
  }

  x[idx] = x_val;
}
