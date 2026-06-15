#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)

namespace nerve::optimization
{

constexpr int GPU_PRIMITIVES_BLOCK_SIZE = 256;
constexpr int GPU_PRIMITIVES_TILE_SIZE = 32;

__global__ void spmvKernel(float *matrix, float *vector, float *result, size_t rows, size_t cols)
{
    size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows)
    {
        double sum = 0.0;
        bool valid = true;
        for (size_t col = 0; col < cols; ++col)
        {
            const double matrix_value = static_cast<double>(matrix[row * cols + col]);
            const double vector_value = static_cast<double>(vector[col]);
            const double product = matrix_value * vector_value;
            const double next = sum + product;
            if (!isfinite(matrix_value) || !isfinite(vector_value) || !isfinite(product) ||
                !isfinite(next))
            {
                valid = false;
                break;
            }
            sum = next;
        }
        const float narrowed = static_cast<float>(sum);
        result[row] = (valid && isfinite(narrowed)) ? narrowed : 0.0f;
    }
}

__global__ void columnXorReductionKernel(const uint32_t *column, uint32_t *result, size_t size)
{
    extern __shared__ uint32_t sdata[];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < size) ? column[idx] : 0;
    __syncthreads();

    for (size_t s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] ^= sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        atomicXor(result, sdata[0]);
    }
}

__global__ void tiledSpmvKernel(const float *matrix, const float *vector, float *result,
                                size_t rows, size_t cols)
{
    __shared__ float tile[GPU_PRIMITIVES_TILE_SIZE][GPU_PRIMITIVES_TILE_SIZE];

    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t tile_col = threadIdx.x;

    double sum = 0.0;
    bool valid = true;

    for (size_t tile_start = 0; tile_start < cols; tile_start += GPU_PRIMITIVES_TILE_SIZE)
    {
        size_t col = tile_start + tile_col;
        if (row < rows && col < cols)
        {
            tile[threadIdx.y][threadIdx.x] = matrix[row * cols + col];
        }
        else
        {
            tile[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();

        for (size_t k = 0; k < GPU_PRIMITIVES_TILE_SIZE; ++k)
        {
            size_t vec_idx = tile_start + k;
            if (valid && vec_idx < cols)
            {
                const double matrix_value = static_cast<double>(tile[threadIdx.y][k]);
                const double vector_value = static_cast<double>(vector[vec_idx]);
                const double product = matrix_value * vector_value;
                const double next = sum + product;
                if (!isfinite(matrix_value) || !isfinite(vector_value) || !isfinite(product) ||
                    !isfinite(next))
                {
                    valid = false;
                    continue;
                }
                sum = next;
            }
        }
        __syncthreads();
    }

    if (row < rows)
    {
        const float narrowed = static_cast<float>(sum);
        result[row] = (valid && isfinite(narrowed)) ? narrowed : 0.0f;
    }
}

} // namespace nerve::optimization

#endif // defined(__CUDACC__)
