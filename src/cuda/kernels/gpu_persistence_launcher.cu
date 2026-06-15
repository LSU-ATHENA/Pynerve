#include "nerve/algebra/boundary.hpp"
#include "nerve/errors/detail/error_result.hpp"
#include "nerve/errors/errors.hpp"
#include "nerve/gpu/persistence_kernels.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace nerve
{
namespace gpu
{
namespace kernels
{

extern __global__ void gpuComputePivotsKernel(const int *__restrict__ col_data,
                                              const int *__restrict__ col_indices,
                                              const int *__restrict__ col_starts,
                                              int *__restrict__ pivots, int n_columns,
                                              int max_height);

extern __global__ void gpuComputePivotsCohomologyKernel(const int *__restrict__ col_data,
                                                        const int *__restrict__ col_indices,
                                                        const int *__restrict__ col_starts,
                                                        int *__restrict__ pivots, int n_columns,
                                                        int max_height);

extern __global__ void gpuAssignPivotOwnersKernel(const int *__restrict__ pivots,
                                                  int *__restrict__ low_to_col,
                                                  int *__restrict__ conflict_src,
                                                  char *__restrict__ has_conflict, int n_columns);

extern __global__ void
gpuAddColumnsKernel(int *__restrict__ col_data, int *__restrict__ col_indices,
                    const int *__restrict__ col_starts, const int *__restrict__ conflict_src,
                    const char *__restrict__ has_conflict, int n_columns, int max_height);

extern __global__ void
gpuResolveConflictsKernel(int *__restrict__ col_data, int *__restrict__ col_indices,
                          const int *__restrict__ col_starts, int *__restrict__ pivots,
                          int *__restrict__ low_to_col, int *__restrict__ changed, int n_columns,
                          int max_height, int max_passes);

extern __global__ void
gpuCohomologyReductionKernel(int *__restrict__ cob_data, int *__restrict__ cob_indices,
                             const int *__restrict__ cob_starts, int *__restrict__ pivots,
                             int *__restrict__ low_to_col, int *__restrict__ changed,
                             int n_cochains, int max_height, int max_passes);

extern __global__ void gpuClearingKernel(const int *__restrict__ pivots,
                                         const int *__restrict__ low_to_col,
                                         int *__restrict__ col_data, int *__restrict__ col_indices,
                                         const int *__restrict__ col_starts, int n_columns,
                                         int max_height);

extern __global__ void gpuExtractPairsKernel(const int *__restrict__ pivots,
                                             const int *__restrict__ low_to_col,
                                             int2 *__restrict__ pairs, int *__restrict__ pair_count,
                                             int n_columns);

extern __global__ void gpuCheckConvergenceKernel(int *__restrict__ changed,
                                                 int *__restrict__ iteration_flag);

namespace detail
{

constexpr int kDefaultBlockSize = 256;
constexpr int kDefaultMaxIterations = 1000;
constexpr int kDefaultMaxPasses = 32;
constexpr int kDefaultClearingInterval = 5;

inline int gridSize(int count, int block_size)
{
    return (count + block_size - 1) / block_size;
}

inline bool checkedProduct(std::size_t lhs, std::size_t rhs, std::size_t &out) noexcept
{
    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs)
    {
        return false;
    }
    out = lhs * rhs;
    return true;
}

inline bool checkedByteCount(std::size_t count, std::size_t element_size, std::size_t &out) noexcept
{
    return checkedProduct(count, element_size, out);
}

inline bool checkedIntSize(std::size_t value, int &out) noexcept
{
    if (value > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        return false;
    }
    out = static_cast<int>(value);
    return true;
}

inline errors::ErrorResult<void> resourceLimit(const char *msg)
{
    return errors::ErrorResult<void>::error(errors::ErrorCode::E41_RESOURCE_LIMIT, msg);
}

inline errors::ErrorResult<void> gpuOom()
{
    return errors::ErrorResult<void>::error(errors::ErrorCode::E10_GPU_OOM);
}

inline errors::ErrorResult<void> gpuLaunchFail(const char *msg)
{
    return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL, msg);
}

struct CudaDeleter
{
    void operator()(void *p) const noexcept { (void)cudaFree(p); }
};

template <typename T>
struct CudaBuffer
{
    T *ptr = nullptr;

    ~CudaBuffer() { reset(); }
    CudaBuffer() = default;

    CudaBuffer(const CudaBuffer &) = delete;
    CudaBuffer &operator=(const CudaBuffer &) = delete;
    CudaBuffer(CudaBuffer &&other) noexcept
        : ptr(other.ptr)
    {
        other.ptr = nullptr;
    }
    CudaBuffer &operator=(CudaBuffer &&other) noexcept
    {
        if (this != &other)
        {
            reset();
            ptr = other.ptr;
            other.ptr = nullptr;
        }
        return *this;
    }

    explicit operator bool() const noexcept { return ptr != nullptr; }
    T *get() noexcept { return ptr; }
    const T *get() const noexcept { return ptr; }

    void reset()
    {
        if (ptr)
        {
            (void)cudaFree(ptr);
            ptr = nullptr;
        }
    }
};

struct GpuMatrix
{
    CudaBuffer<int> data;
    CudaBuffer<int> indices;
    CudaBuffer<int> starts;
    int n_columns = 0;
    int max_height = 0;
};

struct GpuReductionState
{
    CudaBuffer<int> pivots;
    CudaBuffer<int> low_to_col;
    CudaBuffer<int> conflict_src;
    CudaBuffer<char> has_conflict;
    CudaBuffer<int> changed;
    CudaBuffer<int> iteration_flag;
    CudaBuffer<int2> pairs;
    CudaBuffer<int> pair_count;
};

} // namespace detail

class GpuPersistenceReducer
{
public:
    GpuPersistenceReducer() = default;
    ~GpuPersistenceReducer() = default;

    errors::ErrorResult<void> compute(const algebra::BoundaryMatrix &boundary_matrix,
                                      std::vector<Index> &out_pivots,
                                      std::vector<std::pair<Size, Size>> &out_pairs)
    {
        out_pivots.clear();
        out_pairs.clear();

        if (boundary_matrix.cols() == 0)
        {
            return errors::ErrorResult<void>::success();
        }

        auto convert_result = convertToGpu(boundary_matrix);
        if (convert_result.isError())
        {
            return convert_result;
        }

        auto state_result = allocateState(matrix_.n_columns);
        if (state_result.isError())
        {
            return state_result;
        }

        auto init_result = initializeState(matrix_.n_columns);
        if (init_result.isError())
        {
            return init_result;
        }

        auto reduce_result = reduceMatrix();
        if (reduce_result.isError())
        {
            return reduce_result;
        }

        auto extract_result = extractPairs(out_pivots, out_pairs);
        return extract_result;
    }

    errors::ErrorResult<void> computeCohomology(const algebra::BoundaryMatrix &boundary_matrix,
                                                std::vector<Index> &out_pivots,
                                                std::vector<std::pair<Size, Size>> &out_pairs)
    {
        out_pivots.clear();
        out_pairs.clear();

        if (boundary_matrix.cols() == 0)
        {
            return errors::ErrorResult<void>::success();
        }

        auto convert_result = convertToGpu(boundary_matrix);
        if (convert_result.isError())
        {
            return convert_result;
        }

        auto state_result = allocateState(matrix_.n_columns);
        if (state_result.isError())
        {
            return state_result;
        }

        auto init_result = initializeState(matrix_.n_columns);
        if (init_result.isError())
        {
            return init_result;
        }

        auto reduce_result = reduceMatrixCohomology();
        if (reduce_result.isError())
        {
            return reduce_result;
        }

        auto extract_result = extractPairs(out_pivots, out_pairs);
        return extract_result;
    }

    void setMaxIterations(int n) { max_iterations_ = n; }
    void setBlockSize(int b) { block_size_ = b; }
    void setMaxPasses(int p) { max_passes_ = p; }
    void setClearingInterval(int i) { clearing_interval_ = i; }
    void enableClearing(bool e) { use_clearing_ = e; }

    int maxIterations() const { return max_iterations_; }
    int blockSize() const { return block_size_; }

private:
    errors::ErrorResult<void> convertToGpu(const algebra::BoundaryMatrix &boundary_matrix)
    {
        const Size n_cols = boundary_matrix.cols();
        const Size n_rows = boundary_matrix.rows();

        int n_cols_int = 0;
        if (!detail::checkedIntSize(static_cast<std::size_t>(n_cols), n_cols_int))
        {
            return detail::resourceLimit("GPU reduction matrix columns exceed int range");
        }

        int max_entries = 0;
        for (Size col = 0; col < n_cols; ++col)
        {
            int count = 0;
            for (Size row = 0; row < n_rows; ++row)
            {
                if (boundary_matrix.getMatrixEntry(row, col) != 0.0)
                {
                    ++count;
                }
            }
            max_entries = std::max(max_entries, count);
        }

        matrix_.n_columns = n_cols_int;
        matrix_.max_height = max_entries;

        if (max_entries == 0)
        {
            matrix_.n_columns = n_cols_int;
            matrix_.max_height = 1;
            max_entries = 1;
        }

        std::size_t total_entries = 0;
        if (!detail::checkedProduct(static_cast<std::size_t>(n_cols_int),
                                    static_cast<std::size_t>(max_entries), total_entries))
        {
            return detail::resourceLimit("GPU matrix packed size overflows");
        }

        std::size_t data_bytes = 0;
        std::size_t starts_bytes = 0;
        if (!detail::checkedByteCount(total_entries, sizeof(int), data_bytes) ||
            !detail::checkedByteCount(static_cast<std::size_t>(n_cols_int), sizeof(int),
                                      starts_bytes))
        {
            return detail::resourceLimit("GPU matrix byte count overflows");
        }

        std::vector<int> hData(total_entries, 0);
        std::vector<int> hIndices(total_entries, -1);
        std::vector<int> hStarts(static_cast<std::size_t>(n_cols_int), 0);

        std::size_t pos = 0;
        for (Size col = 0; col < n_cols; ++col)
        {
            hStarts[static_cast<std::size_t>(col)] = static_cast<int>(pos);
            int row_count = 0;
            for (Size row = 0; row < n_rows && row_count < max_entries; ++row)
            {
                if (boundary_matrix.getMatrixEntry(row, col) != 0.0)
                {
                    hData[pos] = 1;
                    hIndices[pos] = static_cast<int>(row);
                    ++pos;
                    ++row_count;
                }
            }
            pos = static_cast<std::size_t>(hStarts[static_cast<std::size_t>(col)]) +
                  static_cast<std::size_t>(max_entries);
        }

        cudaError_t err;
        int *d_data = nullptr;
        int *d_indices = nullptr;
        int *d_starts = nullptr;

        err = cudaMalloc(&d_data, data_bytes);
        if (err != cudaSuccess)
        {
            return detail::gpuOom();
        }
        err = cudaMalloc(&d_indices, data_bytes);
        if (err != cudaSuccess)
        {
            cudaFree(d_data);
            return detail::gpuOom();
        }
        err = cudaMalloc(&d_starts, starts_bytes);
        if (err != cudaSuccess)
        {
            cudaFree(d_data);
            cudaFree(d_indices);
            return detail::gpuOom();
        }

        err = cudaMemcpy(d_data, hData.data(), data_bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            cudaFree(d_data);
            cudaFree(d_indices);
            cudaFree(d_starts);
            return detail::gpuLaunchFail("copy matrix data to device");
        }
        err = cudaMemcpy(d_indices, hIndices.data(), data_bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            cudaFree(d_data);
            cudaFree(d_indices);
            cudaFree(d_starts);
            return detail::gpuLaunchFail("copy matrix indices to device");
        }
        err = cudaMemcpy(d_starts, hStarts.data(), starts_bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            cudaFree(d_data);
            cudaFree(d_indices);
            cudaFree(d_starts);
            return detail::gpuLaunchFail("copy matrix starts to device");
        }

        matrix_.data.ptr = d_data;
        matrix_.indices.ptr = d_indices;
        matrix_.starts.ptr = d_starts;
        return errors::ErrorResult<void>::success();
    }

    errors::ErrorResult<void> allocateState(int n_columns)
    {
        const std::size_t n = static_cast<std::size_t>(n_columns);

        std::size_t int_bytes = 0;
        std::size_t char_bytes = 0;
        if (!detail::checkedByteCount(n, sizeof(int), int_bytes) ||
            !detail::checkedByteCount(n, sizeof(char), char_bytes))
        {
            return detail::resourceLimit("GPU reduction state size overflows");
        }

        cudaError_t err;

        err = cudaMalloc(&state_.pivots.ptr, int_bytes);
        if (err != cudaSuccess)
        {
            return detail::gpuOom();
        }
        err = cudaMalloc(&state_.low_to_col.ptr, int_bytes);
        if (err != cudaSuccess)
        {
            return detail::gpuOom();
        }
        err = cudaMalloc(&state_.conflict_src.ptr, int_bytes);
        if (err != cudaSuccess)
        {
            return detail::gpuOom();
        }
        err = cudaMalloc(&state_.has_conflict.ptr, char_bytes);
        if (err != cudaSuccess)
        {
            return detail::gpuOom();
        }
        err = cudaMalloc(&state_.changed.ptr, sizeof(int));
        if (err != cudaSuccess)
        {
            return detail::gpuOom();
        }
        err = cudaMalloc(&state_.iteration_flag.ptr, sizeof(int));
        if (err != cudaSuccess)
        {
            return detail::gpuOom();
        }
        err = cudaMalloc(&state_.pairs.ptr, int_bytes * 2);
        if (err != cudaSuccess)
        {
            return detail::gpuOom();
        }
        err = cudaMalloc(&state_.pair_count.ptr, sizeof(int));
        if (err != cudaSuccess)
        {
            return detail::gpuOom();
        }

        return errors::ErrorResult<void>::success();
    }

    errors::ErrorResult<void> initializeState(int n_columns)
    {
        const std::size_t n = static_cast<std::size_t>(n_columns);
        std::size_t int_bytes = n * sizeof(int);
        std::size_t char_bytes = n * sizeof(char);

        std::vector<int> hInit(n, -1);
        std::vector<char> hInitC(n, 0);

        cudaError_t err;
        err = cudaMemcpy(state_.pivots.ptr, hInit.data(), int_bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
            return detail::gpuLaunchFail("init pivots");
        err = cudaMemcpy(state_.low_to_col.ptr, hInit.data(), int_bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
            return detail::gpuLaunchFail("init low_to_col");
        err = cudaMemcpy(state_.conflict_src.ptr, hInit.data(), int_bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
            return detail::gpuLaunchFail("init conflict_src");
        err =
            cudaMemcpy(state_.has_conflict.ptr, hInitC.data(), char_bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
            return detail::gpuLaunchFail("init has_conflict");
        err = cudaMemset(state_.changed.ptr, 0, sizeof(int));
        if (err != cudaSuccess)
            return detail::gpuLaunchFail("init changed");
        err = cudaMemset(state_.iteration_flag.ptr, 0, sizeof(int));
        if (err != cudaSuccess)
            return detail::gpuLaunchFail("init iteration_flag");
        err = cudaMemset(state_.pair_count.ptr, 0, sizeof(int));
        if (err != cudaSuccess)
            return detail::gpuLaunchFail("init pair_count");

        return errors::ErrorResult<void>::success();
    }

    errors::ErrorResult<void> reduceMatrix()
    {
        const int n_cols = matrix_.n_columns;
        const int max_h = matrix_.max_height;
        const int blk = block_size_;
        const int cols_grid = detail::gridSize(n_cols, blk);

        for (int iter = 0; iter < max_iterations_; ++iter)
        {
            {
                gpuComputePivotsKernel<<<n_cols, blk>>>(matrix_.data.ptr, matrix_.indices.ptr,
                                                        matrix_.starts.ptr, state_.pivots.ptr,
                                                        n_cols, max_h);
                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess)
                    return detail::gpuLaunchFail("gpuComputePivotsKernel");
                err = cudaDeviceSynchronize();
                if (err != cudaSuccess)
                    return detail::gpuLaunchFail("sync gpuComputePivotsKernel");
            }

            {
                gpuAssignPivotOwnersKernel<<<cols_grid, blk>>>(
                    state_.pivots.ptr, state_.low_to_col.ptr, state_.conflict_src.ptr,
                    state_.has_conflict.ptr, n_cols);
                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess)
                    return detail::gpuLaunchFail("gpuAssignPivotOwnersKernel");
                err = cudaDeviceSynchronize();
                if (err != cudaSuccess)
                    return detail::gpuLaunchFail("sync gpuAssignPivotOwnersKernel");
            }

            {
                gpuAddColumnsKernel<<<n_cols, blk>>>(matrix_.data.ptr, matrix_.indices.ptr,
                                                     matrix_.starts.ptr, state_.conflict_src.ptr,
                                                     state_.has_conflict.ptr, n_cols, max_h);
                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess)
                    return detail::gpuLaunchFail("gpuAddColumnsKernel");
                err = cudaDeviceSynchronize();
                if (err != cudaSuccess)
                    return detail::gpuLaunchFail("sync gpuAddColumnsKernel");
            }

            if (use_clearing_ && (iter % clearing_interval_ == 0))
            {
                gpuComputePivotsKernel<<<n_cols, blk>>>(matrix_.data.ptr, matrix_.indices.ptr,
                                                        matrix_.starts.ptr, state_.pivots.ptr,
                                                        n_cols, max_h);
                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess)
                    return detail::gpuLaunchFail("gpuComputePivotsKernel clearing");
                err = cudaDeviceSynchronize();
                if (err != cudaSuccess)
                    return detail::gpuLaunchFail("sync gpuComputePivotsKernel clearing");

                gpuClearingKernel<<<cols_grid, blk>>>(state_.pivots.ptr, state_.low_to_col.ptr,
                                                      matrix_.data.ptr, matrix_.indices.ptr,
                                                      matrix_.starts.ptr, n_cols, max_h);
                err = cudaGetLastError();
                if (err != cudaSuccess)
                    return detail::gpuLaunchFail("gpuClearingKernel");
                err = cudaDeviceSynchronize();
                if (err != cudaSuccess)
                    return detail::gpuLaunchFail("sync gpuClearingKernel");
            }
        }

        return errors::ErrorResult<void>::success();
    }

    errors::ErrorResult<void> reduceMatrixCohomology()
    {
        const int n_cols = matrix_.n_columns;
        const int max_h = matrix_.max_height;
        const int blk = block_size_;
        const int cols_grid = detail::gridSize(n_cols, blk);

        for (int iter = 0; iter < max_iterations_; ++iter)
        {
            {
                gpuComputePivotsCohomologyKernel<<<n_cols, blk>>>(
                    matrix_.data.ptr, matrix_.indices.ptr, matrix_.starts.ptr, state_.pivots.ptr,
                    n_cols, max_h);
                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess)
                    return detail::gpuLaunchFail("gpuComputePivotsCohomologyKernel");
                err = cudaDeviceSynchronize();
                if (err != cudaSuccess)
                    return detail::gpuLaunchFail("sync gpuComputePivotsCohomologyKernel");
            }

            {
                gpuAssignPivotOwnersKernel<<<cols_grid, blk>>>(
                    state_.pivots.ptr, state_.low_to_col.ptr, state_.conflict_src.ptr,
                    state_.has_conflict.ptr, n_cols);
                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess)
                    return detail::gpuLaunchFail("gpuAssignPivotOwnersKernel cohom");
                err = cudaDeviceSynchronize();
                if (err != cudaSuccess)
                    return detail::gpuLaunchFail("sync gpuAssignPivotOwnersKernel cohom");
            }

            {
                gpuAddColumnsKernel<<<n_cols, blk>>>(matrix_.data.ptr, matrix_.indices.ptr,
                                                     matrix_.starts.ptr, state_.conflict_src.ptr,
                                                     state_.has_conflict.ptr, n_cols, max_h);
                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess)
                    return detail::gpuLaunchFail("gpuAddColumnsKernel cohom");
                err = cudaDeviceSynchronize();
                if (err != cudaSuccess)
                    return detail::gpuLaunchFail("sync gpuAddColumnsKernel cohom");
            }

            if (use_clearing_ && (iter % clearing_interval_ == 0))
            {
                gpuComputePivotsCohomologyKernel<<<n_cols, blk>>>(
                    matrix_.data.ptr, matrix_.indices.ptr, matrix_.starts.ptr, state_.pivots.ptr,
                    n_cols, max_h);
                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess)
                    return detail::gpuLaunchFail("gpuComputePivotsCohomologyKernel clearing");
                err = cudaDeviceSynchronize();
                if (err != cudaSuccess)
                    return detail::gpuLaunchFail("sync gpuComputePivotsCohomologyKernel clearing");

                gpuClearingKernel<<<cols_grid, blk>>>(state_.pivots.ptr, state_.low_to_col.ptr,
                                                      matrix_.data.ptr, matrix_.indices.ptr,
                                                      matrix_.starts.ptr, n_cols, max_h);
                err = cudaGetLastError();
                if (err != cudaSuccess)
                    return detail::gpuLaunchFail("gpuClearingKernel cohom");
                err = cudaDeviceSynchronize();
                if (err != cudaSuccess)
                    return detail::gpuLaunchFail("sync gpuClearingKernel cohom");
            }
        }

        return errors::ErrorResult<void>::success();
    }

    errors::ErrorResult<void> extractPairs(std::vector<Index> &out_pivots,
                                           std::vector<std::pair<Size, Size>> &out_pairs)
    {
        const int n_cols = matrix_.n_columns;
        const int max_h = matrix_.max_height;
        const int blk = block_size_;
        const int grid = detail::gridSize(n_cols, blk);

        {
            gpuComputePivotsKernel<<<n_cols, blk>>>(matrix_.data.ptr, matrix_.indices.ptr,
                                                    matrix_.starts.ptr, state_.pivots.ptr, n_cols,
                                                    max_h);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess)
                return detail::gpuLaunchFail("gpuComputePivotsKernel extract");
            err = cudaDeviceSynchronize();
            if (err != cudaSuccess)
                return detail::gpuLaunchFail("sync gpuComputePivotsKernel extract");
        }

        auto reset_result = resetPairCount();
        if (reset_result.isError())
            return reset_result;

        {
            gpuExtractPairsKernel<<<grid, blk>>>(state_.pivots.ptr, state_.low_to_col.ptr,
                                                 state_.pairs.ptr, state_.pair_count.ptr, n_cols);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess)
                return detail::gpuLaunchFail("gpuExtractPairsKernel");
            err = cudaDeviceSynchronize();
            if (err != cudaSuccess)
                return detail::gpuLaunchFail("sync gpuExtractPairsKernel");
        }

        int h_pair_count = 0;
        {
            cudaError_t err = cudaMemcpy(&h_pair_count, state_.pair_count.ptr, sizeof(int),
                                         cudaMemcpyDeviceToHost);
            if (err != cudaSuccess)
                return detail::gpuLaunchFail("read pair_count");
        }

        if (h_pair_count < 0 || h_pair_count > n_cols)
        {
            return detail::gpuLaunchFail("invalid pair count from GPU");
        }

        std::vector<int> hPivots(static_cast<std::size_t>(n_cols), -1);
        {
            std::size_t pivot_bytes = static_cast<std::size_t>(n_cols) * sizeof(int);
            cudaError_t err =
                cudaMemcpy(hPivots.data(), state_.pivots.ptr, pivot_bytes, cudaMemcpyDeviceToHost);
            if (err != cudaSuccess)
                return detail::gpuLaunchFail("read pivots");
        }

        out_pivots.resize(static_cast<std::size_t>(n_cols));
        for (int i = 0; i < n_cols; ++i)
        {
            out_pivots[static_cast<std::size_t>(i)] =
                static_cast<Index>(hPivots[static_cast<std::size_t>(i)]);
        }

        if (h_pair_count > 0)
        {
            std::vector<int2> hPairs(static_cast<std::size_t>(h_pair_count));
            std::size_t pairs_bytes = static_cast<std::size_t>(h_pair_count) * sizeof(int2);
            cudaError_t err =
                cudaMemcpy(hPairs.data(), state_.pairs.ptr, pairs_bytes, cudaMemcpyDeviceToHost);
            if (err != cudaSuccess)
                return detail::gpuLaunchFail("read pairs");

            out_pairs.reserve(static_cast<std::size_t>(h_pair_count));
            for (int i = 0; i < h_pair_count; ++i)
            {
                out_pairs.emplace_back(static_cast<Size>(hPairs[static_cast<std::size_t>(i)].x),
                                       static_cast<Size>(hPairs[static_cast<std::size_t>(i)].y));
            }
        }

        return errors::ErrorResult<void>::success();
    }

    errors::ErrorResult<void> resetPairCount()
    {
        cudaError_t err = cudaMemset(state_.pair_count.ptr, 0, sizeof(int));
        if (err != cudaSuccess)
            return detail::gpuLaunchFail("reset pair_count");
        return errors::ErrorResult<void>::success();
    }

    detail::GpuMatrix matrix_;
    detail::GpuReductionState state_;
    int max_iterations_ = detail::kDefaultMaxIterations;
    int block_size_ = detail::kDefaultBlockSize;
    int max_passes_ = detail::kDefaultMaxPasses;
    int clearing_interval_ = detail::kDefaultClearingInterval;
    bool use_clearing_ = true;
};

} // namespace kernels
} // namespace gpu
} // namespace nerve
