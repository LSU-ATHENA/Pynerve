template <typename T>
__inline__ __device__ T warpReduceSum(T val)
{
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
    {
        val += __shfl_down_sync(FULL_WARP_MASK, val, offset);
    }
    return val;
}

template <typename T>
__inline__ __device__ T blockReduceSum(T val)
{
    static __shared__ T shared[32];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    val = warpReduceSum(val);

    if (lane == 0)
        shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : T(0);
    if (wid == 0)
        val = warpReduceSum(val);

    return val;
}

__inline__ __device__ float warpReduceMin(float val)
{
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
    {
        val = fminf(val, __shfl_down_sync(FULL_WARP_MASK, val, offset));
    }
    return val;
}

__inline__ __device__ float blockReduceMin(float val)
{
    static __shared__ float shared[32];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    val = warpReduceMin(val);
    if (lane == 0)
        shared[wid] = val;
    __syncthreads();
    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : INFINITY;
    if (wid == 0)
        val = warpReduceMin(val);
    return val;
}

__global__ void bettiCurveMSELossKernel(const float *__restrict__ target_betti,
                                        const float *__restrict__ pred_betti,
                                        float *__restrict__ loss, int num_dims)
{
    float local_loss = 0.0f;

    for (int i = threadIdx.x; i < num_dims; i += blockDim.x)
    {
        float diff = target_betti[i] - pred_betti[i];
        float contribution = diff * diff;
        float next = local_loss + contribution;
        if (!isfinite(diff) || !isfinite(contribution) || !isfinite(next))
        {
            local_loss = INFINITY;
            break;
        }
        local_loss = next;
    }

    local_loss = blockReduceSum(local_loss);

    if (threadIdx.x == 0)
    {
        atomicAdd(loss, local_loss / num_dims);
    }
}

__global__ void wassersteinLossKernel(const float *__restrict__ target_pairs,
                                      const float *__restrict__ pred_pairs,
                                      float *__restrict__ loss, int num_pairs_target,
                                      int num_pairs_pred, int p_norm)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    if (bid >= num_pairs_pred)
        return;

    float pred_birth = pred_pairs[bid * 2];
    float pred_death = pred_pairs[bid * 2 + 1];

    float min_dist = INFINITY;

    for (int i = tid; i < num_pairs_target; i += blockDim.x)
    {
        float target_birth = target_pairs[i * 2];
        float target_death = target_pairs[i * 2 + 1];

        float d_birth = pred_birth - target_birth;
        float d_death = pred_death - target_death;

        float dist = 0.0f;
        if (p_norm == 1)
        {
            dist = fabsf(d_birth) + fabsf(d_death);
        }
        else if (p_norm == 2)
        {
            dist = sqrtf(d_birth * d_birth + d_death * d_death);
        }
        else
        {
            dist = powf(powf(fabsf(d_birth), static_cast<float>(p_norm)) +
                            powf(fabsf(d_death), static_cast<float>(p_norm)),
                        1.0f / static_cast<float>(p_norm));
        }
        if (!isfinite(pred_birth) || !isfinite(pred_death) || !isfinite(target_birth) ||
            !isfinite(target_death) || !isfinite(d_birth) || !isfinite(d_death) || !isfinite(dist))
        {
            dist = INFINITY;
        }
        min_dist = fminf(min_dist, dist);
    }

    min_dist = blockReduceMin(min_dist);

    if (tid == 0)
    {
        atomicAdd(loss, min_dist);
    }
}

__global__ void persistenceEntropyKernel(const float *__restrict__ persistence_values,
                                         int num_values, float *__restrict__ entropy)
{
    extern __shared__ float s_buffer[];

    int idx = threadIdx.x;
    float sum = 0.0f;

    for (int i = idx; i < num_values; i += blockDim.x)
    {
        const float value = persistence_values[i];
        const float next = sum + value;
        if (!isfinite(value) || value < 0.0f || !isfinite(next))
        {
            sum = INFINITY;
            break;
        }
        sum = next;
    }

    sum = blockReduceSum(sum);

    __syncthreads();

    if (idx == 0)
    {
        s_buffer[0] = sum;
    }

    __syncthreads();

    float total = s_buffer[0];
    if (!isfinite(total))
    {
        if (idx == 0)
        {
            atomicAdd(entropy, INFINITY);
        }
        return;
    }
    if (total < EPSILON)
        return;

    float local_entropy = 0.0f;
    for (int i = idx; i < num_values; i += blockDim.x)
    {
        float p = persistence_values[i] / total;
        if (p > EPSILON)
        {
            const float contribution = p * logf(p);
            const float next = local_entropy - contribution;
            if (!isfinite(p) || !isfinite(contribution) || !isfinite(next))
            {
                local_entropy = INFINITY;
                break;
            }
            local_entropy = next;
        }
    }

    local_entropy = blockReduceSum(local_entropy);

    if (idx == 0)
    {
        atomicAdd(entropy, local_entropy);
    }
}

__global__ void topologySmoothnessKernel(const float *__restrict__ features,
                                         const int *__restrict__ adjacency, int num_nodes,
                                         int feature_dim, float *__restrict__ smoothness_loss)
{
    int node = blockIdx.x;
    int feat = threadIdx.x;

    if (node >= num_nodes || feat >= feature_dim)
        return;

    float node_feat = features[node * feature_dim + feat];
    float neighbor_sum = 0.0f;
    int degree = 0;

    for (int i = 0; i < num_nodes; ++i)
    {
        if (adjacency[node * num_nodes + i])
        {
            float diff = node_feat - features[i * feature_dim + feat];
            float contribution = diff * diff;
            float next = neighbor_sum + contribution;
            if (!isfinite(node_feat) || !isfinite(diff) || !isfinite(contribution) ||
                !isfinite(next))
            {
                neighbor_sum = INFINITY;
                break;
            }
            neighbor_sum = next;
            degree++;
        }
    }

    if (degree > 0 && isfinite(neighbor_sum))
    {
        neighbor_sum /= degree;
    }

    atomicAdd(smoothness_loss, neighbor_sum);
}

__global__ void combinedTopologyLossKernel(const float *__restrict__ betti_target,
                                           const float *__restrict__ betti_pred,
                                           const float *__restrict__ persistence_values,
                                           int num_dims, int num_persistence, float lambda_betti,
                                           float lambda_persistence, float *__restrict__ total_loss)
{
    extern __shared__ float s_losses[];

    float local_loss = 0.0f;
    int tid = threadIdx.x;

    for (int i = tid; i < num_dims; i += blockDim.x)
    {
        float diff = betti_target[i] - betti_pred[i];
        float contribution = lambda_betti * diff * diff;
        float next = local_loss + contribution;
        if (!isfinite(diff) || !isfinite(contribution) || !isfinite(next))
        {
            local_loss = INFINITY;
            break;
        }
        local_loss = next;
    }

    for (int i = tid; i < num_persistence; i += blockDim.x)
    {
        const float value = persistence_values[i];
        const float contribution = lambda_persistence * value;
        const float next = local_loss + contribution;
        if (!isfinite(value) || value < 0.0f || !isfinite(contribution) || !isfinite(next))
        {
            local_loss = INFINITY;
            break;
        }
        local_loss = next;
    }

    local_loss = blockReduceSum(local_loss);

    if (tid == 0)
    {
        s_losses[0] = local_loss;
    }

    __syncthreads();

    if (tid == 0)
    {
        atomicAdd(total_loss, s_losses[0]);
    }
}
