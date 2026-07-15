static errors::ErrorResult<void> convertToGpuFormat(const algebra::BoundaryMatrix &boundary_matrix,
                                                    GpuBoundaryMatrix &out_matrix)
{
    const size_t n_cols = boundary_matrix.cols();
    const size_t n_rows = boundary_matrix.rows();

    if (!fitsCudaInt(n_cols) || !fitsCudaInt(n_rows))
    {
        return resourceLimitError();
    }

    std::vector<int> col_counts(n_cols, 0);
    std::vector<int> dimensions(n_cols, 0);
    int max_height = 0;

    for (size_t col = 0; col < n_cols; ++col)
    {
        dimensions[col] = boundary_matrix.getColSimplexDimension(col);
        int count = 0;
        for (size_t row = 0; row < n_rows; ++row)
        {
            if (boundary_matrix.getMatrixEntry(row, col) != 0.0)
            {
                ++count;
            }
        }
        col_counts[col] = count;
        max_height = std::max(max_height, count);
    }

    size_t total_entries = 0;
    if (!checkedBytes(n_cols, static_cast<size_t>(std::max(max_height, 0)), total_entries) ||
        !fitsCudaInt(total_entries))
    {
        return resourceLimitError();
    }
    out_matrix.n_simplices = static_cast<int>(n_cols);
    out_matrix.max_height = max_height;
    out_matrix.data_size = total_entries;

    if (n_cols == 0)
    {
        return errors::ErrorResult<void>::success();
    }

    std::vector<int> starts(n_cols, 0);
    std::vector<int> data(total_entries, 0);
    std::vector<int> indices(total_entries, -1);

    if (total_entries > 0)
    {
        int current_pos = 0;
        for (size_t col = 0; col < n_cols; ++col)
        {
            starts[col] = current_pos;
            int row_idx = 0;
            for (size_t row = 0; row < n_rows; ++row)
            {
                if (boundary_matrix.getMatrixEntry(row, col) != 0.0)
                {
                    data[static_cast<size_t>(current_pos)] = 1;
                    indices[static_cast<size_t>(current_pos)] = static_cast<int>(row);
                    ++current_pos;
                    ++row_idx;
                    if (row_idx >= col_counts[col])
                        break;
                }
            }
            current_pos = static_cast<int>((col + 1) * static_cast<size_t>(max_height));
        }
    }

    std::size_t starts_bytes = 0;
    std::size_t entries_bytes = 0;
    if (!checkedBytes(n_cols, sizeof(int), starts_bytes) ||
        !checkedBytes(total_entries, sizeof(int), entries_bytes))
    {
        return resourceLimitError();
    }

    cudaError_t err = cudaMalloc(reinterpret_cast<void **>(&out_matrix.starts), starts_bytes);
    if (err != cudaSuccess)
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E10_GPU_OOM);
    }
    err = cudaMemcpy(out_matrix.starts, starts.data(), starts_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        cleanup(out_matrix);
        return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
    }

    err = cudaMalloc(reinterpret_cast<void **>(&out_matrix.simplex_dimensions), starts_bytes);
    if (err != cudaSuccess)
    {
        cleanup(out_matrix);
        return errors::ErrorResult<void>::error(errors::ErrorCode::E10_GPU_OOM);
    }
    err = cudaMemcpy(out_matrix.simplex_dimensions, dimensions.data(), starts_bytes,
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        cleanup(out_matrix);
        return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
    }

    if (total_entries == 0)
    {
        return errors::ErrorResult<void>::success();
    }

    err = cudaMalloc(reinterpret_cast<void **>(&out_matrix.data), entries_bytes);
    if (err != cudaSuccess)
    {
        cleanup(out_matrix);
        return errors::ErrorResult<void>::error(errors::ErrorCode::E10_GPU_OOM);
    }
    err = cudaMemcpy(out_matrix.data, data.data(), entries_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        cleanup(out_matrix);
        return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
    }

    err = cudaMalloc(reinterpret_cast<void **>(&out_matrix.indices), entries_bytes);
    if (err != cudaSuccess)
    {
        cleanup(out_matrix);
        return errors::ErrorResult<void>::error(errors::ErrorCode::E10_GPU_OOM);
    }
    err = cudaMemcpy(out_matrix.indices, indices.data(), entries_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        cleanup(out_matrix);
        return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
    }

    return errors::ErrorResult<void>::success();
}

static void cleanup(CohomologyMatrixGPU &matrix)
{
    if (matrix.data)
        cudaFree(matrix.data);
    if (matrix.indices)
        cudaFree(matrix.indices);
    if (matrix.starts)
        cudaFree(matrix.starts);
    matrix.data = nullptr;
    matrix.indices = nullptr;
    matrix.starts = nullptr;
}

static void cleanup(GpuBoundaryMatrix &matrix)
{
    if (matrix.data)
        cudaFree(matrix.data);
    if (matrix.indices)
        cudaFree(matrix.indices);
    if (matrix.starts)
        cudaFree(matrix.starts);
    if (matrix.simplex_dimensions)
        cudaFree(matrix.simplex_dimensions);
    matrix.data = nullptr;
    matrix.indices = nullptr;
    matrix.starts = nullptr;
    matrix.simplex_dimensions = nullptr;
}
