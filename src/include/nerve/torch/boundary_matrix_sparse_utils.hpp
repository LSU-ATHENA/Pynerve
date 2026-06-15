#pragma once

#include "nerve/torch/boundary_matrix.hpp"

#include <cmath>
#include <cstdint>
#include <vector>

namespace nerve::torch::detail
{

constexpr double kBoundaryZeroTol = 1e-12;

struct EncodedSparse
{
    at::Tensor indices;
    at::Tensor indptr;
    at::Tensor data;
    int64_t nnz = 0;
};

inline EncodedSparse denseToCsc(const at::Tensor &dense_cpu)
{
    const int64_t n_rows = dense_cpu.size(0);
    const int64_t n_cols = dense_cpu.size(1);
    std::vector<int64_t> row_indices;
    std::vector<int64_t> indptr(static_cast<size_t>(n_cols + 1), 0);
    std::vector<double> values;
    auto dense_acc = dense_cpu.accessor<double, 2>();
    for (int64_t col = 0; col < n_cols; ++col)
    {
        indptr[static_cast<size_t>(col)] = static_cast<int64_t>(row_indices.size());
        for (int64_t row = 0; row < n_rows; ++row)
        {
            const double value = dense_acc[row][col];
            if (std::abs(value) > kBoundaryZeroTol)
            {
                row_indices.push_back(row);
                values.push_back(value);
            }
        }
    }
    indptr[static_cast<size_t>(n_cols)] = static_cast<int64_t>(row_indices.size());
    return {at::tensor(row_indices, at::TensorOptions().dtype(at::kLong)),
            at::tensor(indptr, at::TensorOptions().dtype(at::kLong)),
            at::tensor(values, at::TensorOptions().dtype(at::kDouble)),
            static_cast<int64_t>(values.size())};
}

inline EncodedSparse denseToCsr(const at::Tensor &dense_cpu)
{
    const int64_t n_rows = dense_cpu.size(0);
    const int64_t n_cols = dense_cpu.size(1);
    std::vector<int64_t> col_indices;
    std::vector<int64_t> indptr(static_cast<size_t>(n_rows + 1), 0);
    std::vector<double> values;
    auto dense_acc = dense_cpu.accessor<double, 2>();
    for (int64_t row = 0; row < n_rows; ++row)
    {
        indptr[static_cast<size_t>(row)] = static_cast<int64_t>(col_indices.size());
        for (int64_t col = 0; col < n_cols; ++col)
        {
            const double value = dense_acc[row][col];
            if (std::abs(value) > kBoundaryZeroTol)
            {
                col_indices.push_back(col);
                values.push_back(value);
            }
        }
    }
    indptr[static_cast<size_t>(n_rows)] = static_cast<int64_t>(col_indices.size());
    return {at::tensor(col_indices, at::TensorOptions().dtype(at::kLong)),
            at::tensor(indptr, at::TensorOptions().dtype(at::kLong)),
            at::tensor(values, at::TensorOptions().dtype(at::kDouble)),
            static_cast<int64_t>(values.size())};
}

inline EncodedSparse denseToCoo(const at::Tensor &dense_cpu)
{
    const int64_t n_rows = dense_cpu.size(0);
    const int64_t n_cols = dense_cpu.size(1);
    std::vector<int64_t> row_indices;
    std::vector<int64_t> col_indices;
    std::vector<double> values;
    auto dense_acc = dense_cpu.accessor<double, 2>();
    for (int64_t row = 0; row < n_rows; ++row)
    {
        for (int64_t col = 0; col < n_cols; ++col)
        {
            const double value = dense_acc[row][col];
            if (std::abs(value) > kBoundaryZeroTol)
            {
                row_indices.push_back(row);
                col_indices.push_back(col);
                values.push_back(value);
            }
        }
    }
    at::Tensor indices =
        at::zeros({2, static_cast<int64_t>(values.size())}, at::TensorOptions().dtype(at::kLong));
    if (!values.empty())
    {
        indices.select(0, 0).copy_(at::tensor(row_indices, at::TensorOptions().dtype(at::kLong)));
        indices.select(0, 1).copy_(at::tensor(col_indices, at::TensorOptions().dtype(at::kLong)));
    }
    return {indices, at::empty({0}, at::TensorOptions().dtype(at::kLong)),
            at::tensor(values, at::TensorOptions().dtype(at::kDouble)),
            static_cast<int64_t>(values.size())};
}

inline at::Tensor sparseToDenseCpu(BoundaryMatrix::Format format, const at::Tensor &indices,
                                   const at::Tensor &indptr, const at::Tensor &data, int64_t n_rows,
                                   int64_t n_cols)
{
    at::Tensor dense = at::zeros({n_rows, n_cols}, at::TensorOptions().dtype(at::kDouble));
    auto dense_acc = dense.accessor<double, 2>();
    if (format == BoundaryMatrix::Format::COO)
    {
        if (indices.dim() == 2 && indices.size(0) == 2 && data.dim() == 1)
        {
            auto idx_cpu = indices.cpu().to(at::kLong).contiguous();
            auto val_cpu = data.cpu().to(at::kDouble).contiguous();
            auto row_indices = idx_cpu.select(0, 0);
            auto col_indices = idx_cpu.select(0, 1);
            auto row_acc = row_indices.accessor<int64_t, 1>();
            auto col_acc = col_indices.accessor<int64_t, 1>();
            auto val_acc = val_cpu.accessor<double, 1>();
            for (int64_t i = 0; i < val_cpu.size(0); ++i)
            {
                const int64_t row = row_acc[i];
                const int64_t col = col_acc[i];
                if (row >= 0 && row < n_rows && col >= 0 && col < n_cols)
                {
                    dense_acc[row][col] = val_acc[i];
                }
            }
        }
        return dense;
    }
    if (indptr.numel() == 0 || indices.numel() == 0 || data.numel() == 0)
    {
        return dense;
    }
    auto indptr_cpu = indptr.cpu().to(at::kLong).contiguous();
    auto indices_cpu = indices.cpu().to(at::kLong).contiguous();
    auto values_cpu = data.cpu().to(at::kDouble).contiguous();
    auto indptr_acc = indptr_cpu.accessor<int64_t, 1>();
    auto idx_acc = indices_cpu.accessor<int64_t, 1>();
    auto val_acc = values_cpu.accessor<double, 1>();
    if (format == BoundaryMatrix::Format::CSC)
    {
        for (int64_t col = 0; col < n_cols; ++col)
        {
            for (int64_t i = indptr_acc[col]; i < indptr_acc[col + 1]; ++i)
            {
                const int64_t row = idx_acc[i];
                if (row >= 0 && row < n_rows)
                {
                    dense_acc[row][col] = val_acc[i];
                }
            }
        }
    }
    else
    {
        for (int64_t row = 0; row < n_rows; ++row)
        {
            for (int64_t i = indptr_acc[row]; i < indptr_acc[row + 1]; ++i)
            {
                const int64_t col = idx_acc[i];
                if (col >= 0 && col < n_cols)
                {
                    dense_acc[row][col] = val_acc[i];
                }
            }
        }
    }
    return dense;
}

inline EncodedSparse denseToFormat(const at::Tensor &dense_cpu, BoundaryMatrix::Format format)
{
    switch (format)
    {
        case BoundaryMatrix::Format::CSR:
            return denseToCsr(dense_cpu);
        case BoundaryMatrix::Format::COO:
            return denseToCoo(dense_cpu);
        case BoundaryMatrix::Format::CSC:
        default:
            return denseToCsc(dense_cpu);
    }
}

} // namespace nerve::torch::detail
