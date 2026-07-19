template <typename T>
__global__ void __launch_bounds__(256) reluKernel(T *__restrict__ data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;

    T val = data[idx];
    data[idx] = isfinite(static_cast<float>(val)) ? ((val > 0) ? val : static_cast<T>(0))
                                                  : static_cast<T>(INFINITY);
}

template <typename T>
__global__ void __launch_bounds__(256) sigmoidKernel(T *__restrict__ data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;

    T val = data[idx];
    data[idx] = isfinite(static_cast<float>(val))
                    ? static_cast<T>(1.0f / (1.0f + expf(-static_cast<float>(val))))
                    : static_cast<T>(INFINITY);
}

template <typename T>
__global__ void __launch_bounds__(256)
    addBiasKernel(T *__restrict__ data, const T *__restrict__ bias, int batch_size, int dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * dim;
    if (idx >= total)
        return;

    int feature = idx % dim;
    const T next = data[idx] + bias[feature];
    data[idx] = (isfinite(static_cast<float>(data[idx])) &&
                 isfinite(static_cast<float>(bias[feature])) && isfinite(static_cast<float>(next)))
                    ? next
                    : static_cast<T>(INFINITY);
}
