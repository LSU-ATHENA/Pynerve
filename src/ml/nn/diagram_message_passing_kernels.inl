/**
 * @brief Compute birth-death distance between two persistence pairs
 */
__device__ __inline__ float birthDeathDistance(float b1, float d1, float b2,
                                               float d2) {
  float db = b1 - b2;
  float dd = d1 - d2;
  float db_sq = db * db;
  float dd_sq = dd * dd;
  float sum = db_sq + dd_sq;
  float distance = sqrtf(sum);
  return (isfinite(db) && isfinite(dd) && isfinite(db_sq) && isfinite(dd_sq) &&
          isfinite(sum) && isfinite(distance))
             ? distance
             : INFINITY;
}

/**
 * @brief Compute projection onto diagonal (birth=death line)
 */
__device__ __inline__ float diagonalDistance(float b, float d) {
  const float diff = d - b;
  const float distance = fabsf(diff) / sqrtf(2.0f);
  return (isfinite(diff) && isfinite(distance)) ? distance : INFINITY;
}

/**
 * @brief GPU kernel for computing pairwise diagram distances
 */
__global__ void diagramDistanceKernel(const float *__restrict__ diagrams,
                                      float *__restrict__ distance_matrix,
                                      int num_pairs, float bandwidth) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i >= num_pairs || j >= num_pairs)
    return;

  float b1 = diagrams[i * DIAGRAM_FIELDS_PER_PAIR + 0];
  float d1 = diagrams[i * DIAGRAM_FIELDS_PER_PAIR + 1];
  float dim1 = diagrams[i * DIAGRAM_FIELDS_PER_PAIR + 2];

  float b2 = diagrams[j * DIAGRAM_FIELDS_PER_PAIR + 0];
  float d2 = diagrams[j * DIAGRAM_FIELDS_PER_PAIR + 1];
  float dim2 = diagrams[j * DIAGRAM_FIELDS_PER_PAIR + 2];

  float dist;

  if (dim1 != dim2) {
    // Different dimensions: infinite distance (or very large)
    dist = DIFFERENT_DIMENSION_DISTANCE;
  } else if (i == j) {
    dist = 0.0f;
  } else {
    // Same dimension: birth-death distance
    dist = birthDeathDistance(b1, d1, b2, d2);

    // RBF kernel for soft adjacency
    const float denom = RBF_KERNEL_DENOMINATOR * bandwidth * bandwidth;
    const float scaled = -dist * dist / denom;
    dist = (isfinite(dist) && isfinite(denom) && denom > 0.0f && isfinite(scaled))
               ? expf(scaled)
               : INFINITY;
  }

  distance_matrix[i * num_pairs + j] = isfinite(dist) ? dist : INFINITY;
}

/**
 * @brief GPU kernel for message passing on diagram graph
 */
template <typename T>
__global__ void diagramMessagePassingKernel(const T *__restrict__ node_features,
                                            const float *__restrict__ adjacency,
                                            T *__restrict__ messages,
                                            int num_pairs, int feature_dim,
                                            bool use_attention) {
  int pair = blockIdx.x;
  int feat = threadIdx.x;

  if (pair >= num_pairs || feat >= feature_dim)
    return;

  extern __shared__ float s_adj[];

  // Load adjacency row to shared memory
  for (int i = threadIdx.x; i < num_pairs; i += blockDim.x) {
    s_adj[i] = adjacency[pair * num_pairs + i];
  }
  __syncthreads();

  // Aggregate messages from neighbors
  T sum = 0;
  float weight_sum = 0;

  for (int neighbor = 0; neighbor < num_pairs; ++neighbor) {
    float w = s_adj[neighbor];
    if (isfinite(w) && w > EPSILON) {
      const T feature = node_features[neighbor * feature_dim + feat];
      const T contribution = w * feature;
      const T next = sum + contribution;
      const float next_weight = weight_sum + w;
      if (!isfinite(static_cast<float>(feature)) || !isfinite(static_cast<float>(contribution)) ||
          !isfinite(static_cast<float>(next)) || !isfinite(next_weight)) {
        sum = static_cast<T>(INFINITY);
        break;
      }
      sum = next;
      weight_sum = next_weight;
    }
  }

  // Normalize
  if (isfinite(static_cast<float>(sum)) && weight_sum > EPSILON) {
    sum /= weight_sum;
  }

  messages[pair * feature_dim + feat] = sum;
}

/**
 * @brief GPU kernel for attention-based message passing
 */
template <typename T>
__global__ void diagramAttentionKernel(const T *__restrict__ query,
                                       const T *__restrict__ key,
                                       const T *__restrict__ value,
                                       const float *__restrict__ diagrams,
                                       float *__restrict__ attention_weights,
                                       T *__restrict__ output, int num_pairs,
                                       int feature_dim, float temperature) {
  int pair = blockIdx.x;
  if (pair >= num_pairs)
    return;

  extern __shared__ float shared[];
  float *s_scores = shared;
  float *s_partials = shared + num_pairs;
  const int tid = threadIdx.x;

  /*
   * Each block owns one persistence pair. Threads cooperate to compute
   * scaled dot-product attention scores for every neighbor, then all output
   * feature lanes reuse the same softmax weights. This avoids the older
   * per-feature score race where different lanes overwrote the shared score
   * row with incompatible values.
   */
  for (int neighbor = 0; neighbor < num_pairs; ++neighbor) {
    float dot = 0.0f;
    for (int feat = tid; feat < feature_dim; feat += blockDim.x) {
      const float q = query[pair * feature_dim + feat];
      const float k = key[neighbor * feature_dim + feat];
      const float contribution = q * k;
      const float next = dot + contribution;
      if (!isfinite(q) || !isfinite(k) || !isfinite(contribution) || !isfinite(next)) {
        dot = INFINITY;
        break;
      }
      dot = next;
    }
    s_partials[tid] = dot;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        const float next = s_partials[tid] + s_partials[tid + stride];
        s_partials[tid] = isfinite(next) ? next : INFINITY;
      }
      __syncthreads();
    }

    if (tid == 0) {
      float b1 = diagrams[pair * DIAGRAM_FIELDS_PER_PAIR + 0];
      float d1 = diagrams[pair * DIAGRAM_FIELDS_PER_PAIR + 1];
      float b2 = diagrams[neighbor * DIAGRAM_FIELDS_PER_PAIR + 0];
      float d2 = diagrams[neighbor * DIAGRAM_FIELDS_PER_PAIR + 1];
      float topo_dist = birthDeathDistance(b1, d1, b2, d2);
      const float scaled_dot = s_partials[0] / sqrtf(static_cast<float>(feature_dim));
      const float penalty = topo_dist * TOPOLOGICAL_BIAS_PENALTY;
      const float score = scaled_dot - penalty;
      s_scores[neighbor] =
          (isfinite(scaled_dot) && isfinite(penalty) && isfinite(score)) ? score : INFINITY;
    }
    __syncthreads();
  }

  if (tid == 0) {
    float max_score = -INFINITY;
    for (int i = 0; i < num_pairs; ++i) {
      max_score = fmaxf(max_score, s_scores[i]);
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < num_pairs; ++i) {
      const float normalized = (s_scores[i] - max_score) / temperature;
      const float exp_score = expf(normalized);
      const float next = sum_exp + exp_score;
      s_scores[i] =
          (isfinite(normalized) && isfinite(exp_score) && isfinite(next)) ? exp_score : INFINITY;
      sum_exp = isfinite(next) ? next : INFINITY;
    }

    const float inv_sum = isfinite(sum_exp) ? 1.0f / fmaxf(sum_exp, EPSILON) : INFINITY;
    for (int i = 0; i < num_pairs; ++i) {
      s_scores[i] *= inv_sum;
      attention_weights[pair * num_pairs + i] = s_scores[i];
    }
  }
  __syncthreads();

  for (int feat = tid; feat < feature_dim; feat += blockDim.x) {
    T out_val = 0;
    for (int i = 0; i < num_pairs; ++i) {
      const T feature = value[i * feature_dim + feat];
      const T contribution = s_scores[i] * feature;
      const T next = out_val + contribution;
      if (!isfinite(s_scores[i]) || !isfinite(static_cast<float>(feature)) ||
          !isfinite(static_cast<float>(contribution)) || !isfinite(static_cast<float>(next))) {
        out_val = static_cast<T>(INFINITY);
        break;
      }
      out_val = next;
    }
    output[pair * feature_dim + feat] = out_val;
  }
}

/**
 * @brief GPU kernel for diagram convolution
 */
template <typename T>
__global__ void diagramConvolutionKernel(const T *__restrict__ input,
                                         const float *__restrict__ kernels,
                                         const float *__restrict__ diagrams,
                                         T *__restrict__ output, int num_pairs,
                                         int in_channels, int out_channels,
                                         int kernel_size, float bandwidth) {
  int out_ch = blockIdx.x;
  int pair = blockIdx.y;

  if (out_ch >= out_channels || pair >= num_pairs)
    return;

  float b = diagrams[pair * DIAGRAM_FIELDS_PER_PAIR + 0];
  float d = diagrams[pair * DIAGRAM_FIELDS_PER_PAIR + 1];

  T sum = 0;
  bool valid = isfinite(b) && isfinite(d);

  // Convolve with kernel in birth-death plane
  for (int in_ch = 0; in_ch < in_channels; ++in_ch) {
    for (int k = 0; k < kernel_size; ++k) {
      // Compute kernel center in birth-death space
      float kb = b + (k - kernel_size / 2) * bandwidth;
      float kd = d + (k - kernel_size / 2) * bandwidth;

      // Find nearby pairs using distance threshold
      for (int other = 0; other < num_pairs; ++other) {
        float b2 = diagrams[other * DIAGRAM_FIELDS_PER_PAIR + 0];
        float d2 = diagrams[other * DIAGRAM_FIELDS_PER_PAIR + 1];

        float dist = birthDeathDistance(kb, kd, b2, d2);
        const float denom = 2.0f * bandwidth * bandwidth;
        const float scaled = -dist * dist / denom;
        float weight = (isfinite(dist) && isfinite(denom) && denom > 0.0f && isfinite(scaled))
                           ? expf(scaled)
                           : INFINITY;

        T val = input[other * in_channels + in_ch];
        T k_val = kernels[((out_ch * in_channels + in_ch) * kernel_size + k)];

        const T contribution = weight * val * k_val;
        const T next = sum + contribution;
        if (!isfinite(kb) || !isfinite(kd) || !isfinite(b2) || !isfinite(d2) ||
            !isfinite(weight) || !isfinite(static_cast<float>(val)) ||
            !isfinite(static_cast<float>(k_val)) || !isfinite(static_cast<float>(contribution)) ||
            !isfinite(static_cast<float>(next))) {
          valid = false;
          break;
        }
        sum = next;
      }
      if (!valid)
        break;
    }
    if (!valid)
      break;
  }

  output[pair * out_channels + out_ch] = valid ? sum : static_cast<T>(INFINITY);
}

__global__ void accumulateScaleKernel(const float *__restrict__ scale_messages,
                                      float *__restrict__ combined,
                                      int total_values, float weight) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total_values) {
    const float contribution = weight * scale_messages[idx];
    const float next = combined[idx] + contribution;
    combined[idx] =
        (isfinite(weight) && isfinite(scale_messages[idx]) && isfinite(contribution) &&
         isfinite(next))
            ? next
            : INFINITY;
  }
}
