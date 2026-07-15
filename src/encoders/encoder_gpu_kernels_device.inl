__device__ bool finiteLifetime(const float *__restrict__ diagram, int pair_idx, float &lifetime)
{
    const float birth = diagram[2 * pair_idx];
    const float death = diagram[2 * pair_idx + 1];
    if (!isfinite(birth))
    {
        lifetime = INFINITY;
        return false;
    }
    if (isinf(death) && death > 0.0f)
    {
        lifetime = 0.0f;
        return false;
    }
    if (!isfinite(death) || death < birth)
    {
        lifetime = INFINITY;
        return false;
    }
    const float value = death - birth;
    if (!isfinite(value) || value < 0.0f)
    {
        lifetime = INFINITY;
        return false;
    }
    lifetime = value;
    return value > 0.0f;
}

__device__ bool checkedAdd(float contribution, float &sum)
{
    const float next = sum + contribution;
    if (!isfinite(contribution) || !isfinite(next))
    {
        sum = INFINITY;
        return false;
    }
    sum = next;
    return true;
}

__device__ float computeTopologicalFeatureValue(const float *__restrict__ diagram, int max_pairs,
                                                int feature_idx)
{
    float feature_value = 0.0f;

    switch (feature_idx % 8)
    {
        case 0:
        {
            for (int i = 0; i < max_pairs; ++i)
            {
                float lifetime = 0.0f;
                if (finiteLifetime(diagram, i, lifetime) && !checkedAdd(lifetime, feature_value))
                {
                    return INFINITY;
                }
                if (isinf(lifetime))
                {
                    return INFINITY;
                }
            }
            break;
        }
        case 1:
        {
            for (int i = 0; i < max_pairs; ++i)
            {
                float lifetime = 0.0f;
                if (finiteLifetime(diagram, i, lifetime))
                {
                    feature_value = fmaxf(feature_value, lifetime);
                }
                else if (isinf(lifetime))
                {
                    return INFINITY;
                }
            }
            break;
        }
        case 2:
        {
            int count = 0;
            for (int i = 0; i < max_pairs; ++i)
            {
                float lifetime = 0.0f;
                if (finiteLifetime(diagram, i, lifetime))
                {
                    count++;
                }
                else if (isinf(lifetime))
                {
                    return INFINITY;
                }
            }
            feature_value = static_cast<float>(count);
            break;
        }
        case 3:
        {
            float sum_birth = 0.0f;
            int count = 0;
            for (int i = 0; i < max_pairs; ++i)
            {
                float lifetime = 0.0f;
                if (finiteLifetime(diagram, i, lifetime))
                {
                    if (!checkedAdd(diagram[2 * i], sum_birth))
                    {
                        return INFINITY;
                    }
                    count++;
                }
                else if (isinf(lifetime))
                {
                    return INFINITY;
                }
            }
            feature_value = (count > 0) ? sum_birth / count : 0.0f;
            break;
        }
        case 4:
        {
            float sum_death = 0.0f;
            int count = 0;
            for (int i = 0; i < max_pairs; ++i)
            {
                float lifetime = 0.0f;
                if (finiteLifetime(diagram, i, lifetime))
                {
                    if (!checkedAdd(diagram[2 * i + 1], sum_death))
                    {
                        return INFINITY;
                    }
                    count++;
                }
                else if (isinf(lifetime))
                {
                    return INFINITY;
                }
            }
            feature_value = (count > 0) ? sum_death / count : 0.0f;
            break;
        }
        case 5:
        {
            float sum_persistence = 0.0f;
            for (int i = 0; i < max_pairs; ++i)
            {
                float lifetime = 0.0f;
                if (finiteLifetime(diagram, i, lifetime) && !checkedAdd(lifetime, sum_persistence))
                {
                    return INFINITY;
                }
                if (isinf(lifetime))
                {
                    return INFINITY;
                }
            }

            float entropy = 0.0f;
            for (int i = 0; i < max_pairs; ++i)
            {
                float lifetime = 0.0f;
                if (!finiteLifetime(diagram, i, lifetime))
                {
                    if (isinf(lifetime))
                    {
                        return INFINITY;
                    }
                    continue;
                }
                float p = lifetime / (sum_persistence + 1e-8f);
                if (p > 0.0f)
                {
                    const float contribution = p * logf(p);
                    const float next = entropy - contribution;
                    if (!isfinite(p) || !isfinite(contribution) || !isfinite(next))
                    {
                        return INFINITY;
                    }
                    entropy = next;
                }
            }
            feature_value = entropy;
            break;
        }
        case 6:
        {
            int betti = 0;
            for (int i = 0; i < max_pairs; ++i)
            {
                const float birth = diagram[2 * i];
                const float death = diagram[2 * i + 1];
                if (!isfinite(birth) || (!isfinite(death) && !(isinf(death) && death > 0.0f)) ||
                    (isfinite(death) && death < birth))
                {
                    return INFINITY;
                }
                if (isinf(death) && death > 0.0f)
                {
                    betti++;
                }
            }
            feature_value = static_cast<float>(betti);
            break;
        }
        case 7:
        {
            int count = 0;
            for (int i = 0; i < max_pairs; ++i)
            {
                float persistence = 0.0f;
                if (!finiteLifetime(diagram, i, persistence))
                {
                    if (isinf(persistence))
                    {
                        return INFINITY;
                    }
                    continue;
                }
                if (persistence > 0.1f && persistence < 1.0f)
                {
                    count++;
                }
            }
            feature_value = static_cast<float>(count);
            break;
        }
    }

    return feature_value;
}

__global__ void __launch_bounds__(128)
    topologicalFeatureExtractionKernel(const float *__restrict__ persistence_diagrams,
                                       float *__restrict__ features, int batch_size,
                                       int max_pairs_per_sample, int num_features)
{
    int batch_idx = blockIdx.x;
    int feature_idx = threadIdx.x;

    if (batch_idx >= batch_size || feature_idx >= num_features)
        return;

    const float *diagram = persistence_diagrams + batch_idx * max_pairs_per_sample * 2;

    float feature_value =
        computeTopologicalFeatureValue(diagram, max_pairs_per_sample, feature_idx);
    features[batch_idx * num_features + feature_idx] = feature_value;
}

__global__ void __launch_bounds__(256)
    fusedMLPLayerKernel(const float *__restrict__ input, const float *__restrict__ weights,
                        const float *__restrict__ bias, float *__restrict__ output, int batch_size,
                        int in_features, int out_features, int activation)
{
    int batch_idx = blockIdx.y;
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx >= batch_size || out_idx >= out_features)
        return;

    float sum = bias[out_idx];

    for (int in_idx = 0; in_idx < in_features; ++in_idx)
    {
        float w = weights[in_idx * out_features + out_idx];
        float x = input[batch_idx * in_features + in_idx];
        const float contribution = w * x;
        const float next = sum + contribution;
        if (!isfinite(sum) || !isfinite(w) || !isfinite(x) || !isfinite(contribution) ||
            !isfinite(next))
        {
            sum = INFINITY;
            break;
        }
        sum = next;
    }

    float result;
    switch (activation)
    {
        case 0:
            result = fmaxf(sum, 0.0f);
            break;
        case 1:
            result = tanhf(sum);
            break;
        case 2:
            result = 1.0f / (1.0f + expf(-sum));
            break;
        default:
            result = sum;
    }

    output[batch_idx * out_features + out_idx] = isfinite(result) ? result : INFINITY;
}

__global__ void __launch_bounds__(256)
    fusedTopologicalEncoderKernel(const float *__restrict__ diagrams,
                                  const float *__restrict__ mlp_weights,
                                  const float *__restrict__ mlp_bias, float *__restrict__ output,
                                  int batch_size, int max_pairs, int topo_features,
                                  int out_features)
{
    extern __shared__ float shared_topo[];

    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;

    const float *diagram = diagrams + batch_idx * max_pairs * 2;
    for (int feature_idx = tid; feature_idx < topo_features; feature_idx += blockDim.x)
    {
        shared_topo[feature_idx] = computeTopologicalFeatureValue(diagram, max_pairs, feature_idx);
    }
    __syncthreads();

    for (int out_idx = tid; out_idx < out_features; out_idx += blockDim.x)
    {
        float sum = mlp_bias[out_idx];
        for (int feature_idx = 0; feature_idx < topo_features; ++feature_idx)
        {
            const float feature = shared_topo[feature_idx];
            const float weight = mlp_weights[feature_idx * out_features + out_idx];
            const float contribution = feature * weight;
            const float next = sum + contribution;
            if (!isfinite(feature) || !isfinite(weight) || !isfinite(contribution) ||
                !isfinite(next))
            {
                sum = INFINITY;
                break;
            }
            sum = next;
        }
        const float result = fmaxf(sum, 0.0f);
        output[batch_idx * out_features + out_idx] = isfinite(result) ? result : INFINITY;
    }
}
