static errors::ErrorResult<void>
detectBirthSimplices(const algebra::BoundaryMatrix &boundary_matrix, std::vector<bool> &out_cleared,
                     size_t &out_accelerated_ops)
{
    const size_t cols = boundary_matrix.cols();
    constexpr size_t GPU_THRESHOLD = 2048;

    if (cols < GPU_THRESHOLD)
    {
        computeBirthSimplicesCpu(boundary_matrix, out_cleared, out_accelerated_ops);
        return errors::ErrorResult<void>::success();
    }

    auto result = detectBoundaryColumnsGpu(boundary_matrix, out_cleared, out_accelerated_ops,
                                           BoundaryDetectionMode::Birth);
    if (result.isError())
    {
        computeBirthSimplicesCpu(boundary_matrix, out_cleared, out_accelerated_ops);
    }
    return errors::ErrorResult<void>::success();
}

static errors::ErrorResult<void> detectEmergentPairs(const algebra::BoundaryMatrix &boundary_matrix,
                                                     std::vector<bool> &out_emergent,
                                                     size_t &out_emergent_count)
{
    const size_t cols = boundary_matrix.cols();
    constexpr size_t GPU_THRESHOLD = 2048;

    if (cols < GPU_THRESHOLD)
    {
        computeEmergentPairsCpu(boundary_matrix, out_emergent, out_emergent_count);
        return errors::ErrorResult<void>::success();
    }

    auto result = detectBoundaryColumnsGpu(boundary_matrix, out_emergent, out_emergent_count,
                                           BoundaryDetectionMode::Emergent);
    if (result.isError())
    {
        computeEmergentPairsCpu(boundary_matrix, out_emergent, out_emergent_count);
    }
    return errors::ErrorResult<void>::success();
}

private:
enum class BoundaryDetectionMode
{
    Birth,
    Emergent
};

static errors::ErrorResult<void>
detectBoundaryColumnsGpu(const algebra::BoundaryMatrix &boundary_matrix,
                         std::vector<bool> &out_flags, size_t &out_count,
                         BoundaryDetectionMode mode)
{
    const size_t cols = boundary_matrix.cols();
    out_flags.assign(cols, false);
    out_count = 0;

    if (!fitsCudaInt(cols))
    {
        return resourceLimitError();
    }

    GpuBoundaryMatrix gpu_matrix;
    auto conversion = convertToGpuFormat(boundary_matrix, gpu_matrix);
    if (conversion.isError())
    {
        return conversion;
    }

    std::size_t flag_bytes = 0;
    if (!checkedBytes(cols, sizeof(bool), flag_bytes))
    {
        cleanup(gpu_matrix);
        return resourceLimitError();
    }

    bool *d_flags = nullptr;
    int *d_count = nullptr;
    cudaStream_t stream = nullptr;
    auto free_device = [&]() {
        if (d_flags)
            cudaFree(d_flags);
        if (d_count)
            cudaFree(d_count);
        if (stream)
            cudaStreamDestroy(stream);
        cleanup(gpu_matrix);
    };

    cudaError_t status = cudaMalloc(reinterpret_cast<void **>(&d_flags), flag_bytes);
    if (status != cudaSuccess)
    {
        free_device();
        return errors::ErrorResult<void>::error(errors::ErrorCode::E10_GPU_OOM);
    }
    status = cudaMalloc(reinterpret_cast<void **>(&d_count), sizeof(int));
    if (status != cudaSuccess)
    {
        free_device();
        return errors::ErrorResult<void>::error(errors::ErrorCode::E10_GPU_OOM);
    }
    status = cudaMemset(d_count, 0, sizeof(int));
    if (status != cudaSuccess)
    {
        free_device();
        return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
    }
    status = cudaStreamCreate(&stream);
    if (status != cudaSuccess)
    {
        free_device();
        return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
    }

    try
    {
        const int n_cols = static_cast<int>(cols);
        if (mode == BoundaryDetectionMode::Birth)
        {
            launchDetectBirthSimplices(gpu_matrix.data, gpu_matrix.indices, gpu_matrix.starts,
                                       gpu_matrix.simplex_dimensions, d_flags, d_count, n_cols,
                                       gpu_matrix.max_height, stream);
        }
        else
        {
            launchDetectEmergentPairs(gpu_matrix.data, gpu_matrix.indices, gpu_matrix.starts,
                                      d_flags, d_count, n_cols, gpu_matrix.max_height, stream);
        }
    }
    catch (const std::exception &)
    {
        free_device();
        return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
    }

    status = cudaPeekAtLastError();
    if (status != cudaSuccess)
    {
        free_device();
        return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
    }

    status = cudaStreamSynchronize(stream);
    if (status != cudaSuccess)
    {
        free_device();
        return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
    }

    std::vector<unsigned char> flags(flag_bytes, 0);
    int count = 0;
    status = cudaMemcpy(flags.data(), d_flags, flag_bytes, cudaMemcpyDeviceToHost);
    if (status != cudaSuccess)
    {
        free_device();
        return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
    }
    status = cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess)
    {
        free_device();
        return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
    }

    for (size_t j = 0; j < cols; ++j)
    {
        out_flags[j] = flags[j * sizeof(bool)] != 0;
    }
    out_count = static_cast<size_t>(std::max(count, 0));
    free_device();
    return errors::ErrorResult<void>::success();
}

static void computeBirthSimplicesCpu(const algebra::BoundaryMatrix &boundary_matrix,
                                     std::vector<bool> &out_cleared, size_t &out_accelerated_ops)
{
    const size_t cols = boundary_matrix.cols();
    out_cleared.assign(cols, false);
    out_accelerated_ops = 0;

    for (size_t j = 0; j < cols; ++j)
    {
        if (boundary_matrix.getColSimplexDimension(j) == 0)
        {
            out_cleared[j] = true;
            ++out_accelerated_ops;
            continue;
        }

        bool has_boundary = false;
        for (size_t row = 0; row < boundary_matrix.rows(); ++row)
        {
            if (boundary_matrix.getMatrixEntry(row, j) != 0.0)
            {
                has_boundary = true;
                break;
            }
        }

        if (!has_boundary)
        {
            out_cleared[j] = true;
            ++out_accelerated_ops;
        }
    }
}

static void computeEmergentPairsCpu(const algebra::BoundaryMatrix &boundary_matrix,
                                    std::vector<bool> &out_emergent, size_t &out_emergent_count)
{
    const size_t cols = boundary_matrix.cols();
    out_emergent.assign(cols, false);
    out_emergent_count = 0;

    for (size_t j = 0; j < cols; ++j)
    {
        bool is_zero = true;
        for (size_t row = 0; row < boundary_matrix.rows(); ++row)
        {
            if (boundary_matrix.getMatrixEntry(row, j) != 0.0)
            {
                is_zero = false;
                break;
            }
        }

        if (is_zero)
        {
            out_emergent[j] = true;
            ++out_emergent_count;
        }
    }
}

static errors::ErrorResult<void>
buildCoboundaryMatrixGpu(const algebra::BoundaryMatrix &boundary_matrix,
                         CohomologyMatrixGPU &out_matrix)
{
    const size_t n_rows = boundary_matrix.rows();
    const size_t n_cols = boundary_matrix.cols();

    if (!fitsCudaInt(n_rows) || !fitsCudaInt(n_cols))
    {
        return resourceLimitError();
    }

    std::vector<int> rowCounts(n_rows, 0);
    int max_height = 0;

    for (size_t row = 0; row < n_rows; ++row)
    {
        int count = 0;
        for (size_t col = 0; col < n_cols; ++col)
        {
            if (boundary_matrix.getMatrixEntry(row, col) != 0.0)
            {
                ++count;
            }
        }
        rowCounts[row] = count;
        max_height = std::max(max_height, count);
    }

    size_t total_entries = 0;
    if (!checkedBytes(n_rows, static_cast<size_t>(std::max(max_height, 0)), total_entries) ||
        !fitsCudaInt(total_entries))
    {
        return resourceLimitError();
    }
    out_matrix.n_cochains = static_cast<int>(n_rows);
    out_matrix.max_height = max_height;
    out_matrix.data_size = total_entries;
    if (total_entries == 0)
    {
        return errors::ErrorResult<void>::success();
    }

    std::vector<int> hData(total_entries, 0);
    std::vector<int> hIndices(total_entries, -1);
    std::vector<int> hStarts(n_rows, 0);

    int current_pos = 0;
    for (size_t row = 0; row < n_rows; ++row)
    {
        hStarts[row] = current_pos;
        int col_idx = 0;
        for (size_t col = 0; col < n_cols; ++col)
        {
            if (boundary_matrix.getMatrixEntry(row, col) != 0.0)
            {
                hData[static_cast<size_t>(current_pos)] = 1;
                hIndices[static_cast<size_t>(current_pos)] = static_cast<int>(col);
                ++current_pos;
                ++col_idx;
                if (col_idx >= rowCounts[row])
                    break;
            }
        }
        for (int i = col_idx; i < max_height; ++i)
        {
            hIndices[static_cast<size_t>(hStarts[row] + i)] = -1;
        }
        current_pos = static_cast<int>((row + 1) * static_cast<size_t>(max_height));
    }

    std::size_t entries_bytes = 0;
    std::size_t starts_bytes = 0;
    if (!checkedBytes(total_entries, sizeof(int), entries_bytes) ||
        !checkedBytes(n_rows, sizeof(int), starts_bytes))
    {
        return resourceLimitError();
    }

    cudaError_t err = cudaMalloc(reinterpret_cast<void **>(&out_matrix.data), entries_bytes);
    if (err != cudaSuccess)
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E10_GPU_OOM);
    }

    err = cudaMalloc(reinterpret_cast<void **>(&out_matrix.indices), entries_bytes);
    if (err != cudaSuccess)
    {
        cleanup(out_matrix);
        return errors::ErrorResult<void>::error(errors::ErrorCode::E10_GPU_OOM);
    }

    err = cudaMalloc(reinterpret_cast<void **>(&out_matrix.starts), starts_bytes);
    if (err != cudaSuccess)
    {
        cleanup(out_matrix);
        return errors::ErrorResult<void>::error(errors::ErrorCode::E10_GPU_OOM);
    }

    err = cudaMemcpy(out_matrix.data, hData.data(), entries_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        cleanup(out_matrix);
        return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
    }

    err = cudaMemcpy(out_matrix.indices, hIndices.data(), entries_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        cleanup(out_matrix);
        return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
    }

    err = cudaMemcpy(out_matrix.starts, hStarts.data(), starts_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        cleanup(out_matrix);
        return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
    }

    return errors::ErrorResult<void>::success();
}

#include "gpu_cohomology_boundary_matrix.inl"
}
;
