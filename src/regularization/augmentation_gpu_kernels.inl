struct RNGState {
  curandState state;

  __device__ void init(unsigned int seed, int idx) {
    curand_init(seed, idx, 0, &state);
  }

  __device__ float uniform() { return curand_uniform(&state); }

  __device__ float normal(float mean, float stddev) {
    return curand_normal(&state) * stddev + mean;
  }
};

__global__ void gaussianNoiseKernel(const float *__restrict__ input,
                                    float *__restrict__ output,
                                    int num_elements, float mean, float stddev,
                                    unsigned int seed) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_elements)
    return;

  RNGState rng;
  rng.init(seed, idx);

  float noise = rng.normal(mean, stddev);
  output[idx] = input[idx] + noise;
}

__global__ void uniformNoiseKernel(const float *__restrict__ input,
                                   float *__restrict__ output, int num_elements,
                                   float min_val, float max_val,
                                   unsigned int seed) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_elements)
    return;

  RNGState rng;
  rng.init(seed, idx);

  float noise = rng.uniform() * (max_val - min_val) + min_val;
  output[idx] = input[idx] + noise;
}
__global__ void geometricTransformKernel(const float *__restrict__ input,
                                         float *__restrict__ output,
                                         int num_samples, int feature_dim,
                                         float rotation_angle,
                                         float scale_factor) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int sample = idx / feature_dim;
  int feature = idx % feature_dim;

  if (sample >= num_samples)
    return;

  if (feature < 2 && feature_dim >= 2) {
    float x = input[sample * feature_dim];
    float y = input[sample * feature_dim + 1];

    float cos_a = cosf(rotation_angle);
    float sin_a = sinf(rotation_angle);

    if (feature == 0) {
      output[idx] = scale_factor * (x * cos_a - y * sin_a);
    } else {
      output[idx] = scale_factor * (x * sin_a + y * cos_a);
    }
  } else {
    output[idx] = scale_factor * input[idx];
  }
}
__global__ void
persistenceAwareNoiseKernel(const float *__restrict__ input,
                            const float *__restrict__ persistence_values,
                            float *__restrict__ output, int num_elements,
                            float base_noise, float persistence_scale,
                            unsigned int seed) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_elements)
    return;

  RNGState rng;
  rng.init(seed, idx);

  float persistence = persistence_values[idx];
  float noise_scale = base_noise / (1.0f + persistence_scale * persistence);

  float noise = rng.normal(0.0f, noise_scale);
  output[idx] = input[idx] + noise;
}

__global__ void mixupKernel(const float *__restrict__ input1,
                            const float *__restrict__ input2,
                            float *__restrict__ output, int num_elements,
                            float alpha, unsigned int seed) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_elements)
    return;

  RNGState rng;
  rng.init(seed, idx);

  float lambda = rng.uniform() * alpha;
  output[idx] = lambda * input1[idx] + (1.0f - lambda) * input2[idx];
}

__global__ void cutoutKernel(const float *__restrict__ input,
                             float *__restrict__ output, int num_samples,
                             int feature_dim, int mask_size, float mask_value,
                             unsigned int seed) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_samples)
    return;

  RNGState rng;
  rng.init(seed, idx);

  for (int i = 0; i < feature_dim; ++i) {
    output[idx * feature_dim + i] = input[idx * feature_dim + i];
  }

  const int mask_width = mask_size < feature_dim ? mask_size : feature_dim;
  const int span = feature_dim - mask_width + 1;
  int mask_start = static_cast<int>(rng.uniform() * span);
  if (mask_start >= span) {
    mask_start = span - 1;
  }
  for (int i = 0; i < mask_width && (mask_start + i) < feature_dim; ++i) {
    output[idx * feature_dim + mask_start + i] = mask_value;
  }
}
