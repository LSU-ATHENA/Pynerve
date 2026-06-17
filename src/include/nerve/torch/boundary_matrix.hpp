#pragma once
#if __has_include(<torch/torch.h>)

#include <torch/torch.h>

#include <cstdint>
#include <vector>

namespace nerve::torch
{

// Forward declaration
class SimplexTree;

/**
 * @brief Compressed sparse boundary matrix using ATen tensor storage.
 *
 * Stores boundary matrix in CSR (Compressed Sparse Row) format:
 * - data_[i]: value at position indices_[i] in row indptr_[row] <= i < indptr_[row+1]
 *
 * For simplicial complexes with Z_2 coefficients, values are 0 or 1.
 * For other coefficient rings, values can be arbitrary.
 *
 * Benefits:
 * - Efficient column/row operations for persistence
 * - GPU-accelerated matrix operations
 * - Zero-copy with PyTorch sparse tensors
 */
class BoundaryMatrix
{
public:
    enum class Format
    {
        CSR, // Compressed Sparse Row
        CSC, // Compressed Sparse Column
        COO  // Coordinate format
    };

private:
    // Storage format
    Format format_;

    // CSR/CSC storage
    at::Tensor indices_; // [nnz] int64 - row indices (CSC) or col indices (CSR)
    at::Tensor indptr_;  // [major_dim+1] int64 - index pointers
    at::Tensor data_;    // [nnz] double - values (typically +/-1 for mod2)

    // Dimensions
    int64_t n_rows_;
    int64_t n_cols_;
    int64_t nnz_;

    // Persistence state
    at::Tensor pivot_rows_; // [cols] int64 - pivot row for each column (-1 if none)
    at::Tensor is_reduced_; // [cols] bool - whether column is reduced

    // Filtration ordering for persistence
    at::Tensor column_order_; // [cols] int64 - sorted column indices

public:
    BoundaryMatrix();

    /// Build from simplex tree for given dimension
    explicit BoundaryMatrix(const SimplexTree &tree, int64_t dimension,
                            Format format = Format::CSR);

    /// Build from explicit data
    BoundaryMatrix(at::Tensor indices, at::Tensor indptr, at::Tensor data, int64_t n_rows,
                   int64_t n_cols, Format format = Format::CSR);

    // Copy/Move
    BoundaryMatrix(const BoundaryMatrix &other) = default;
    BoundaryMatrix(BoundaryMatrix &&other) noexcept = default;
    BoundaryMatrix &operator=(const BoundaryMatrix &other) = default;
    BoundaryMatrix &operator=(BoundaryMatrix &&other) noexcept = default;

    /// Matrix-vector multiplication: y = A @ x
    [[nodiscard]] at::Tensor matvec(const at::Tensor &x) const;

    /// Transpose matrix-vector: y = A^T @ x
    [[nodiscard]] at::Tensor matvec_transpose(const at::Tensor &x) const;

    /// Get column as dense vector
    [[nodiscard]] at::Tensor get_column(int64_t col_idx) const;

    /// Add column j to column i (for persistence reduction)
    void add_column(int64_t target_col, int64_t source_col);

    /// Reduce a single column (standard persistence algorithm)
    void reduce_column(int64_t col_idx);

    /// Reduce all columns
    void reduce_all();

    /// Get pivot row of a column (-1 if column is empty)
    [[nodiscard]] int64_t get_pivot(int64_t col_idx) const;

    /// Get pivot as tensor for all columns
    [[nodiscard]] at::Tensor get_pivots() const;

    /// Compute persistence pairs [birth_idx, death_idx]
    [[nodiscard]] std::pair<at::Tensor, at::Tensor> compute_persistence_pairs() const;

    /// Low-level persistence computation with custom coefficient ring
    template <typename CoefficientRing>
    [[nodiscard]] at::Tensor compute_persistence_templated() const;

    /// Find lowest 1 in a column (pivot)
    [[nodiscard]] int64_t find_pivot(const at::Tensor &column) const;

    /// Check if column is zero
    [[nodiscard]] bool is_zero_column(int64_t col_idx) const;

    /// Get dimension (boundary maps k-simplices to (k-1)-simplices)
    [[nodiscard]] int64_t dimension() const;

    void to(at::Device device);
    [[nodiscard]] at::Device device() const { return data_.device(); }
    [[nodiscard]] bool is_cuda() const { return data_.is_cuda(); }

    /// Convert to dense matrix
    [[nodiscard]] at::Tensor to_dense() const;

    /// Convert to CSR format (if not already)
    void convert_to_csr();

    /// Convert to CSC format
    void convert_to_csc();

    /// Convert to COO format
    void convert_to_coo();

    /// Convert to PyTorch sparse tensor
    [[nodiscard]] at::Tensor to_torch_sparse() const;

    /// Create from PyTorch sparse tensor
    [[nodiscard]] static BoundaryMatrix from_torch_sparse(const at::Tensor &sparse_tensor,
                                                          Format format = Format::CSR);

    [[nodiscard]] int64_t n_rows() const { return n_rows_; }
    [[nodiscard]] int64_t n_cols() const { return n_cols_; }
    [[nodiscard]] int64_t nnz() const { return nnz_; }
    [[nodiscard]] Format format() const { return format_; }

    [[nodiscard]] const at::Tensor &indices() const { return indices_; }
    [[nodiscard]] const at::Tensor &indptr() const { return indptr_; }
    [[nodiscard]] const at::Tensor &data() const { return data_; }

    // Batching Support

    /// Stack multiple boundary matrices (for batch processing)
    static BoundaryMatrix batch(const std::vector<BoundaryMatrix> &matrices);

    /// Extract a single matrix from a batch
    [[nodiscard]] BoundaryMatrix get_batch_item(int64_t idx) const;

    /// Check if this is a batched matrix
    [[nodiscard]] bool is_batched() const;
};

} // namespace nerve::torch

#endif
