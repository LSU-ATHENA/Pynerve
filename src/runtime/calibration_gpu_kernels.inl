__device__ inline void atomicAddDouble(double* addr, double val) {
#if __CUDA_ARCH__ >= 600
    atomicAdd(addr, val);
#else
    unsigned long long* addr_as_ull = reinterpret_cast<unsigned long long*>(addr);
    unsigned long long old = *addr_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
#endif
}

__global__ void
aggregateBucketsKernel(const double *__restrict__ predicted_times,
                       const double *__restrict__ observed_times,
                       const int *__restrict__ bucket_ids,
                       double *__restrict__ bucket_predicted_sum,
                       double *__restrict__ bucket_observed_sum,
                       double *__restrict__ bucket_error_sum,
                       int *__restrict__ bucket_counts, int num_samples,
                       int num_buckets) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_samples)
    return;

  int bucket = bucket_ids[idx];
  if (bucket < 0 || bucket >= num_buckets)
    return;

  atomicAddDouble(&bucket_predicted_sum[bucket], predicted_times[idx]);
  atomicAddDouble(&bucket_observed_sum[bucket], observed_times[idx]);

  double error = fabs(predicted_times[idx] - observed_times[idx]);
  atomicAddDouble(&bucket_error_sum[bucket], error);

  atomicAdd(&bucket_counts[bucket], 1);
}

__global__ void computeConfidenceKernel(const double *__restrict__ errors,
                                        const int *__restrict__ bucket_counts,
                                        double *__restrict__ confidence,
                                        double *__restrict__ error_bounds,
                                        int num_buckets, int min_samples,
                                        double confidence_threshold) {
  int bucket = blockIdx.x * blockDim.x + threadIdx.x;
  if (bucket >= num_buckets)
    return;

  int count = bucket_counts[bucket];
  if (count < min_samples) {
    confidence[bucket] = 0.0;
    error_bounds[bucket] = 1.0;
    return;
  }

  double avg_error = errors[bucket] / count;
  double raw_confidence = 1.0 - avg_error;

  double sample_factor = static_cast<double>(count) / (count + 5);
  double scaled_confidence =
      fmax(0.0, fmin(1.0, raw_confidence * sample_factor));
  confidence[bucket] =
      scaled_confidence >= confidence_threshold ? scaled_confidence : 0.0;

  double z_score = 1.28;
  double estimated_std = avg_error * 1.25;
  error_bounds[bucket] = avg_error + z_score * estimated_std;
}

__global__ void modelUpdateKernel(const float *__restrict__ features,
                                  const float *__restrict__ targets,
                                  float *__restrict__ weights,
                                  float *__restrict__ bias, int num_samples,
                                  int num_features, float learning_rate,
                                  float regularization) {
  int feature = blockIdx.x * blockDim.x + threadIdx.x;
  if (feature >= num_features)
    return;

  float gradient = 0.0f;
  float bias_grad = 0.0f;

  for (int i = 0; i < num_samples; ++i) {
    float pred = *bias;
    for (int f = 0; f < num_features; ++f) {
      pred += weights[f] * features[i * num_features + f];
    }

    float error = pred - targets[i];
    gradient += error * features[i * num_features + feature];
    bias_grad += error;
  }

  gradient = (gradient / num_samples) + regularization * weights[feature];
  bias_grad /= num_samples;

  weights[feature] -= learning_rate * gradient;
  if (feature == 0) {
    *bias -= learning_rate * bias_grad;
  }
}
