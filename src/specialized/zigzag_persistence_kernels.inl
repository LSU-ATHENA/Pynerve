
__global__ void zigzagForwardKernel(
    const int *__restrict__ boundary_matrix,
    const int *__restrict__ simplex_dims, int num_simplices,
    const ZigzagStep *__restrict__ steps, int num_steps,

    int *__restrict__ birth_indices, int *__restrict__ death_indices,
    float *__restrict__ birth_times, float *__restrict__ death_times,
    int *__restrict__ persistence_dim, int *__restrict__ pair_count,
    int max_pairs) {
  int step_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (step_idx >= num_steps)
    return;

  ZigzagStep step = steps[step_idx];
  int simplex = step.simplex_index;
  int dim = simplex_dims[simplex];

  // Track birth/death based on step type
  if (step.type == ZigzagStep::FORWARD_INCLUSION ||
      step.type == ZigzagStep::BACKWARD_INCLUSION) {
    // Simplex being added - potential birth
    int idx = atomicAdd(pair_count, 1);
    if (idx < max_pairs && idx < MAX_ZIGZAG_INTERVALS) {
      birth_indices[idx] = simplex;
      birth_times[idx] = step.time;
      persistence_dim[idx] = dim;
    }
  } else {
    // Simplex being removed - potential death
    // Find matching birth
    for (int i = 0; i < *pair_count; ++i) {
      if (death_indices[i] == -1 && persistence_dim[i] == dim) {
        // Check if simplices are related in boundary
        bool related = false;
        for (int j = 0; j < num_simplices; ++j) {
          if (boundary_matrix[simplex * num_simplices + j] &&
              boundary_matrix[birth_indices[i] * num_simplices + j]) {
            related = true;
            break;
          }
        }

        if (related) {
          death_indices[i] = simplex;
          death_times[i] = step.time;
          break;
        }
      }
    }
  }
}

/**
 * @brief GPU kernel for dynamic complex update
 */
__global__ void dynamicUpdateKernel(
    int *__restrict__ complex_simplices, int *__restrict__ complex_size,
    int max_complex_size, const int *__restrict__ new_simplices, int num_new,
    const int *__restrict__ removed_simplices, int num_removed) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Add new simplices
  if (idx < num_new) {
    int pos = atomicAdd(complex_size, 1);
    if (pos < max_complex_size) {
      complex_simplices[pos] = new_simplices[idx];
    }
  }

  // Remove simplices (mark as -1)
  if (idx < num_removed) {
    int to_remove = removed_simplices[idx];
    for (int i = 0; i < *complex_size; ++i) {
      if (complex_simplices[i] == to_remove) {
        complex_simplices[i] = -1;
      }
    }
  }
}
