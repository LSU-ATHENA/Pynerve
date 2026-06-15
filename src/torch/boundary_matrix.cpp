#include "nerve/torch/boundary_matrix.hpp"
#include "nerve/torch/boundary_matrix_sparse_utils.hpp"
#include "nerve/torch/simplex_tree.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

namespace nerve::torch
{

BoundaryMatrix::BoundaryMatrix()
    : format_(Format::CSC)
    , n_rows_(0)
    , n_cols_(0)
    , nnz_(0)
{}

BoundaryMatrix::BoundaryMatrix(at::Tensor indices, at::Tensor indptr, at::Tensor data,
                               int64_t n_rows, int64_t n_cols, Format format)
    : format_(format)
    , indices_(std::move(indices).to(at::kLong).contiguous())
    , indptr_(std::move(indptr).to(at::kLong).contiguous())
    , data_(std::move(data).to(at::kDouble).contiguous())
    , n_rows_(n_rows)
    , n_cols_(n_cols)
    , nnz_(data_.numel())
{
    const auto state_options = at::TensorOptions().dtype(at::kLong).device(data_.device());
    pivot_rows_ = at::full({n_cols_}, -1, state_options);
    is_reduced_ = at::zeros({n_cols_}, at::TensorOptions().dtype(at::kBool).device(data_.device()));
    column_order_ = at::arange(n_cols_, state_options);
}

BoundaryMatrix::BoundaryMatrix(const SimplexTree &tree, int64_t dim, Format format)
    : format_(Format::CSC)
    , n_rows_(0)
    , n_cols_(0)
    , nnz_(0)
{
    auto dim_simplices = tree.get_simplices_by_dimension(dim);
    auto dim_minus_1_simplices = tree.get_simplices_by_dimension(dim - 1);

    n_cols_ = static_cast<int64_t>(dim_simplices.size());
    n_rows_ = static_cast<int64_t>(dim_minus_1_simplices.size());

    at::Tensor dense = at::zeros({n_rows_, n_cols_}, at::TensorOptions().dtype(at::kDouble));
    if (n_rows_ > 0 && n_cols_ > 0)
    {
        auto dense_acc = dense.accessor<double, 2>();
        for (int64_t col = 0; col < n_cols_; ++col)
        {
            const auto &simplex = dim_simplices[static_cast<size_t>(col)];
            for (size_t i = 0; i < simplex.size(); ++i)
            {
                std::vector<int64_t> face;
                face.reserve(simplex.size() - 1);
                for (size_t j = 0; j < simplex.size(); ++j)
                {
                    if (j != i)
                    {
                        face.push_back(simplex[j]);
                    }
                }
                auto it =
                    std::find(dim_minus_1_simplices.begin(), dim_minus_1_simplices.end(), face);
                if (it != dim_minus_1_simplices.end())
                {
                    const int64_t row =
                        static_cast<int64_t>(std::distance(dim_minus_1_simplices.begin(), it));
                    dense_acc[row][col] = (i % 2 == 0) ? 1.0 : -1.0;
                }
            }
        }
    }

    detail::EncodedSparse encoded = detail::denseToCsc(dense);
    indices_ = encoded.indices;
    indptr_ = encoded.indptr;
    data_ = encoded.data;
    nnz_ = encoded.nnz;

    pivot_rows_ = at::full({n_cols_}, -1, at::TensorOptions().dtype(at::kLong));
    is_reduced_ = at::zeros({n_cols_}, at::TensorOptions().dtype(at::kBool));
    column_order_ = at::arange(n_cols_, at::TensorOptions().dtype(at::kLong));

    if (format != Format::CSC)
    {
        format_ = Format::CSC;
        if (format == Format::CSR)
        {
            convert_to_csr();
        }
        else if (format == Format::COO)
        {
            convert_to_coo();
        }
    }
}

at::Tensor BoundaryMatrix::matvec(const at::Tensor &x) const
{
    TORCH_CHECK(x.dim() == 1 && x.size(0) == n_cols_, "x must be 1D tensor of size n_cols");
    at::Tensor dense = to_dense().to(x.device()).to(x.scalar_type());
    return at::matmul(dense, x);
}

at::Tensor BoundaryMatrix::matvec_transpose(const at::Tensor &x) const
{
    TORCH_CHECK(x.dim() == 1 && x.size(0) == n_rows_, "x must be 1D tensor of size n_rows");
    at::Tensor dense = to_dense().to(x.device()).to(x.scalar_type());
    return at::matmul(dense.t(), x);
}

at::Tensor BoundaryMatrix::get_column(int64_t col_idx) const
{
    TORCH_CHECK(col_idx >= 0 && col_idx < n_cols_, "col_idx out of bounds");
    at::Tensor dense = to_dense();
    return dense.select(1, col_idx).clone();
}

void BoundaryMatrix::add_column(int64_t target_col, int64_t source_col)
{
    TORCH_CHECK(target_col >= 0 && target_col < n_cols_, "target_col out of bounds");
    TORCH_CHECK(source_col >= 0 && source_col < n_cols_, "source_col out of bounds");

    at::Tensor dense_cpu =
        detail::sparseToDenseCpu(format_, indices_, indptr_, data_, n_rows_, n_cols_);
    auto dense_acc = dense_cpu.accessor<double, 2>();
    for (int64_t row = 0; row < n_rows_; ++row)
    {
        dense_acc[row][target_col] += dense_acc[row][source_col];
    }

    detail::EncodedSparse encoded = detail::denseToFormat(dense_cpu, format_);
    indices_ = encoded.indices.to(data_.device());
    indptr_ = encoded.indptr.to(data_.device());
    data_ = encoded.data.to(data_.device());
    nnz_ = encoded.nnz;

    if (pivot_rows_.defined())
    {
        pivot_rows_.index_put_({target_col}, -1);
    }
    if (is_reduced_.defined())
    {
        is_reduced_.index_put_({target_col}, false);
    }
}

void BoundaryMatrix::reduce_column(int64_t col_idx)
{
    if (is_zero_column(col_idx))
    {
        is_reduced_.index_put_({col_idx}, true);
        return;
    }

    int64_t pivot = get_pivot(col_idx);
    while (pivot != -1)
    {
        int64_t other_col = -1;
        for (int64_t j = 0; j < col_idx; ++j)
        {
            if (is_reduced_[j].item<bool>() && get_pivot(j) == pivot)
            {
                other_col = j;
                break;
            }
        }
        if (other_col == -1)
        {
            pivot_rows_.index_put_({col_idx}, pivot);
            break;
        }
        add_column(col_idx, other_col);
        pivot = get_pivot(col_idx);
    }

    is_reduced_.index_put_({col_idx}, true);
}

void BoundaryMatrix::reduce_all()
{
    for (int64_t col = 0; col < n_cols_; ++col)
    {
        reduce_column(col);
    }
}

int64_t BoundaryMatrix::get_pivot(int64_t col_idx) const
{
    if (is_reduced_.defined() && is_reduced_[col_idx].item<bool>())
    {
        return pivot_rows_[col_idx].item<int64_t>();
    }
    return find_pivot(get_column(col_idx));
}

at::Tensor BoundaryMatrix::get_pivots() const
{
    std::vector<int64_t> pivots;
    pivots.reserve(static_cast<size_t>(n_cols_));
    for (int64_t col = 0; col < n_cols_; ++col)
    {
        pivots.push_back(get_pivot(col));
    }
    return at::tensor(pivots, at::TensorOptions().dtype(at::kLong));
}

std::pair<at::Tensor, at::Tensor> BoundaryMatrix::compute_persistence_pairs() const
{
    auto reduced = *this;
    reduced.reduce_all();

    std::vector<int64_t> birth_cols;
    std::vector<int64_t> death_cols;
    for (int64_t col = 0; col < reduced.n_cols_; ++col)
    {
        const int64_t pivot = reduced.pivot_rows_[col].item<int64_t>();
        if (pivot != -1)
        {
            birth_cols.push_back(pivot);
            death_cols.push_back(col);
        }
    }
    return {at::tensor(birth_cols, at::TensorOptions().dtype(at::kLong)),
            at::tensor(death_cols, at::TensorOptions().dtype(at::kLong))};
}

int64_t BoundaryMatrix::find_pivot(const at::Tensor &column) const
{
    at::Tensor col_cpu = column.cpu().to(at::kDouble);
    auto acc = col_cpu.accessor<double, 1>();
    for (int64_t row = col_cpu.size(0) - 1; row >= 0; --row)
    {
        if (std::abs(acc[row]) > detail::kBoundaryZeroTol)
        {
            return row;
        }
    }
    return -1;
}

bool BoundaryMatrix::is_zero_column(int64_t col_idx) const
{
    at::Tensor column = get_column(col_idx);
    return at::all(at::abs(column) <= detail::kBoundaryZeroTol).item<bool>();
}

int64_t BoundaryMatrix::dimension() const
{
    at::Tensor dense_cpu =
        detail::sparseToDenseCpu(format_, indices_, indptr_, data_, n_rows_, n_cols_);
    auto dense_acc = dense_cpu.accessor<double, 2>();
    int64_t max_faces = 0;
    for (int64_t col = 0; col < n_cols_; ++col)
    {
        int64_t faces = 0;
        for (int64_t row = 0; row < n_rows_; ++row)
        {
            if (std::abs(dense_acc[row][col]) > detail::kBoundaryZeroTol)
            {
                ++faces;
            }
        }
        max_faces = std::max(max_faces, faces);
    }
    if (max_faces == 0)
    {
        return -1;
    }
    return std::max<int64_t>(0, max_faces - 1);
}

void BoundaryMatrix::to(at::Device device)
{
    indices_ = indices_.to(device);
    indptr_ = indptr_.to(device);
    data_ = data_.to(device);
    pivot_rows_ = pivot_rows_.to(device);
    is_reduced_ = is_reduced_.to(device);
    if (column_order_.defined())
    {
        column_order_ = column_order_.to(device);
    }
}

at::Tensor BoundaryMatrix::to_dense() const
{
    at::Tensor dense_cpu =
        detail::sparseToDenseCpu(format_, indices_, indptr_, data_, n_rows_, n_cols_);
    return dense_cpu.to(data_.device());
}

void BoundaryMatrix::convert_to_csr()
{
    if (format_ == Format::CSR)
    {
        return;
    }
    at::Tensor dense_cpu =
        detail::sparseToDenseCpu(format_, indices_, indptr_, data_, n_rows_, n_cols_);
    detail::EncodedSparse encoded = detail::denseToCsr(dense_cpu);
    indices_ = encoded.indices.to(data_.device());
    indptr_ = encoded.indptr.to(data_.device());
    data_ = encoded.data.to(data_.device());
    nnz_ = encoded.nnz;
    format_ = Format::CSR;
}

void BoundaryMatrix::convert_to_csc()
{
    if (format_ == Format::CSC)
    {
        return;
    }
    at::Tensor dense_cpu =
        detail::sparseToDenseCpu(format_, indices_, indptr_, data_, n_rows_, n_cols_);
    detail::EncodedSparse encoded = detail::denseToCsc(dense_cpu);
    indices_ = encoded.indices.to(data_.device());
    indptr_ = encoded.indptr.to(data_.device());
    data_ = encoded.data.to(data_.device());
    nnz_ = encoded.nnz;
    format_ = Format::CSC;
}

void BoundaryMatrix::convert_to_coo()
{
    if (format_ == Format::COO)
    {
        return;
    }
    at::Tensor dense_cpu =
        detail::sparseToDenseCpu(format_, indices_, indptr_, data_, n_rows_, n_cols_);
    detail::EncodedSparse encoded = detail::denseToCoo(dense_cpu);
    indices_ = encoded.indices.to(data_.device());
    indptr_ = encoded.indptr.to(data_.device());
    data_ = encoded.data.to(data_.device());
    nnz_ = encoded.nnz;
    format_ = Format::COO;
}

at::Tensor BoundaryMatrix::to_torch_sparse() const
{
    at::Tensor dense_cpu =
        detail::sparseToDenseCpu(format_, indices_, indptr_, data_, n_rows_, n_cols_);
    detail::EncodedSparse encoded = detail::denseToCoo(dense_cpu);
    return at::sparse_coo_tensor(encoded.indices.to(data_.device()),
                                 encoded.data.to(data_.device()), {n_rows_, n_cols_})
        .coalesce();
}

BoundaryMatrix BoundaryMatrix::from_torch_sparse(const at::Tensor &sparse_tensor, Format format)
{
    TORCH_CHECK(sparse_tensor.dim() == 2, "sparse_tensor must be 2D");
    at::Tensor dense_cpu = sparse_tensor.to_dense().cpu().to(at::kDouble);
    detail::EncodedSparse encoded = detail::denseToFormat(dense_cpu, format);
    return BoundaryMatrix(encoded.indices, encoded.indptr, encoded.data, dense_cpu.size(0),
                          dense_cpu.size(1), format);
}

BoundaryMatrix BoundaryMatrix::batch(const std::vector<BoundaryMatrix> &matrices)
{
    if (matrices.empty())
    {
        return BoundaryMatrix();
    }

    int64_t total_rows = 0;
    int64_t total_cols = 0;
    for (const auto &matrix : matrices)
    {
        total_rows += matrix.n_rows_;
        total_cols += matrix.n_cols_;
    }

    at::Tensor dense = at::zeros({total_rows, total_cols}, at::TensorOptions().dtype(at::kDouble));
    std::vector<int64_t> batch_meta;
    batch_meta.reserve(1 + 4 * matrices.size());
    batch_meta.push_back(static_cast<int64_t>(matrices.size()));

    int64_t row_offset = 0;
    int64_t col_offset = 0;
    for (const auto &matrix : matrices)
    {
        at::Tensor block = matrix.to_dense().cpu().to(at::kDouble);
        dense.narrow(0, row_offset, matrix.n_rows_)
            .narrow(1, col_offset, matrix.n_cols_)
            .copy_(block);
        batch_meta.push_back(row_offset);
        batch_meta.push_back(matrix.n_rows_);
        batch_meta.push_back(col_offset);
        batch_meta.push_back(matrix.n_cols_);
        row_offset += matrix.n_rows_;
        col_offset += matrix.n_cols_;
    }

    BoundaryMatrix result = from_torch_sparse(dense.to_sparse(), Format::CSC);
    result.column_order_ = at::tensor(batch_meta, at::TensorOptions().dtype(at::kLong));
    return result;
}

BoundaryMatrix BoundaryMatrix::get_batch_item(int64_t idx) const
{
    TORCH_CHECK(is_batched(), "matrix is not batched");
    auto meta_cpu = column_order_.cpu();
    auto meta_acc = meta_cpu.accessor<int64_t, 1>();
    const int64_t batch_size = meta_acc[0];
    TORCH_CHECK(idx >= 0 && idx < batch_size, "batch index out of bounds");

    const int64_t base = 1 + idx * 4;
    const int64_t row_offset = meta_acc[base];
    const int64_t row_count = meta_acc[base + 1];
    const int64_t col_offset = meta_acc[base + 2];
    const int64_t col_count = meta_acc[base + 3];

    at::Tensor dense_cpu =
        detail::sparseToDenseCpu(format_, indices_, indptr_, data_, n_rows_, n_cols_);
    at::Tensor sub =
        dense_cpu.narrow(0, row_offset, row_count).narrow(1, col_offset, col_count).clone();
    BoundaryMatrix item = from_torch_sparse(sub.to_sparse(), Format::CSC);
    item.to(data_.device());
    return item;
}

bool BoundaryMatrix::is_batched() const
{
    if (!column_order_.defined() || column_order_.dim() != 1 || column_order_.numel() < 5)
    {
        return false;
    }
    const int64_t num = column_order_.numel();
    if ((num - 1) % 4 != 0)
    {
        return false;
    }
    const int64_t batch_size = column_order_.cpu()[0].item<int64_t>();
    return batch_size > 0 && (1 + 4 * batch_size) == num;
}

} // namespace nerve::torch
