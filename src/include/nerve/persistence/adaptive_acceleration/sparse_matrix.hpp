
#pragma once

#include "nerve/core_types.hpp"
#include "nerve/errors/errors.hpp"

#include <algorithm>
#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

namespace nerve::persistence::adaptive_acceleration
{

/**
 * @brief Sparse matrix representation for persistent homology
 *
 * This class provides an efficient sparse matrix representation suitable for
 * matrix multiplication algorithms and sparsification strategies.
 *
 * Key features:
 * - Compressed sparse storage using CSR format
 * - Fast row/column access patterns
 * - Memory-efficient storage for large sparse matrices
 * - Integration with Adaptive Acceleration algorithms
 */
class SparseMatrix
{
public:
    SparseMatrix()
        : n_rows_(0)
        , n_cols_(0)
        , nnz_(0)
    {
        row_starts_.push_back(0);
    }

    /**
     * @brief Create sparse matrix from boundary matrix
     * @param boundary_matrix Input boundary matrix
     * @return Sparse matrix representation
     */
    static errors::ErrorResult<SparseMatrix>
    fromBoundaryMatrix(const std::vector<std::vector<int>> &boundary_matrix, size_t n_rows,
                       size_t n_cols);

    /**
     * @brief Create sparse matrix from dense matrix
     * @param dense_matrix Dense matrix data
     * @param n_rows Number of rows
     * @param n_cols Number of columns
     * @return Sparse matrix representation
     */
    static errors::ErrorResult<SparseMatrix>
    fromDenseMatrix(const std::vector<double> &dense_matrix, size_t n_rows, size_t n_cols);

    /**
     * @brief Get number of rows
     * @return Number of rows in matrix
     */
    size_t numRows() const { return n_rows_; }

    /**
     * @brief Get number of columns
     * @return Number of columns in matrix
     */
    size_t numCols() const { return n_cols_; }

    /**
     * @brief Get number of non-zero elements
     * @return Count of non-zero elements
     */
    size_t nnz() const { return nnz_; }

    /**
     * @brief Get sparsity ratio
     * @return Sparsity ratio (0.0 = dense, 1.0 = completely sparse)
     */
    double sparsityRatio() const
    {
        const double total_entries = static_cast<double>(n_rows_) * static_cast<double>(n_cols_);
        return total_entries > 0.0 ? static_cast<double>(nnz_) / total_entries : 1.0;
    }

    /**
     * @brief Get row indices
     * @return Vector of row indices
     */
    const std::vector<int> &rowIndices() const { return row_indices_; }

    /**
     * @brief Get column indices
     * @return Vector of column indices
     */
    const std::vector<int> &colIndices() const { return col_indices_; }

    /**
     * @brief Get values
     * @return Vector of non-zero values
     */
    const std::vector<double> &values() const { return values_; }

    /**
     * @brief Get row starts
     * @return Vector of row start positions
     */
    const std::vector<int> &rowStarts() const { return row_starts_; }

    /**
     * @brief Access element at position
     * @param row Row index
     * @param col Column index
     * @return Value at position
     */
    double operator()(size_t row, size_t col) const;

    /**
     * @brief Get row slice
     * @param row Row index
     *return Row slice as vector
     */
    std::vector<double> getRow(size_t row) const;

    /**
     * @brief Get column slice
     * @param col Column index
     * @return Column slice as vector
     */
    std::vector<double> getColumn(size_t col) const;

    /**
     * @brief Transpose matrix
     @return Transposed sparse matrix
     */
    errors::ErrorResult<SparseMatrix> transpose() const;

    /**
     * @brief Extract submatrix
     * @param row_start Start row index
     * @param row_end End row index
     * @param col_start Start column index
     * @param col_end End column index
     *return Submatrix
     */
    errors::ErrorResult<SparseMatrix> submatrix(size_t row_start, size_t row_end, size_t col_start,
                                                size_t col_end) const;

    /**
     * @brief Validate matrix structure
     * @return True if matrix is valid
     */
    bool isValid() const;

    /**
     * @brief Get memory usage in bytes
     * @return Memory usage in bytes
     */
    size_t memoryUsage() const;

private:
    SparseMatrix(size_t n_rows, size_t n_cols);

    // CSR format storage
    std::vector<int> row_indices_;
    std::vector<int> col_indices_;
    std::vector<double> values_;
    std::vector<int> row_starts_;

    size_t n_rows_;
    size_t n_cols_;
    size_t nnz_;

    // Helper methods
    void initializeFromDense(const std::vector<double> &dense_matrix, size_t n_rows, size_t n_cols);

    void initializeFromBoundary(const std::vector<std::vector<int>> &boundary_matrix, size_t n_rows,
                                size_t n_cols);

    void compressStorage();
    void validateStructure() const;
};

/**
 * @brief Iterator for sparse matrix rows
 */
class SparseMatrixRowIterator
{
private:
    const SparseMatrix &matrix_;
    size_t current_row_;

public:
    SparseMatrixRowIterator(const SparseMatrix &matrix, size_t start_row = 0);

    SparseMatrixRowIterator &operator++();
    SparseMatrixRowIterator operator++(int);
    std::vector<double> operator*() const;
    bool operator!=(const SparseMatrixRowIterator &other) const;
    bool operator==(const SparseMatrixRowIterator &other) const;
};

/**
 * @brief Iterator for sparse matrix columns
 */
class SparseMatrixColumnIterator
{
private:
    const SparseMatrix &matrix_;
    size_t current_col_;

public:
    SparseMatrixColumnIterator(const SparseMatrix &matrix, size_t start_col = 0);

    SparseMatrixColumnIterator &operator++();
    SparseMatrixColumnIterator operator++(int);
    std::vector<double> operator*() const;
    bool operator!=(const SparseMatrixColumnIterator &other) const;
    bool operator==(const SparseMatrixColumnIterator &other) const;
};

/**
 * @brief Problem characteristics for algorithm selection
 */
#ifndef NERVE_ADAPTIVE_ACCELERATION_PROBLEM_CHARACTERISTICS_DEFINED
#define NERVE_ADAPTIVE_ACCELERATION_PROBLEM_CHARACTERISTICS_DEFINED
struct ProblemCharacteristics
{
    size_t num_points = 0;
    size_t point_dim = 0;
    size_t max_simplex_size = 0;
    size_t estimated_columns = 0;
    double density = 0.0;
    double apparent_pair_ratio = 0.0;
    double sparsity_ratio = 0.0;
    bool is_dense = false;
    bool is_sparse = false;
    bool isHighDimensional = false;
    bool hasRegularStructure = false;

    // Complexity estimation
    double estimated_complexity = 0.0;
    double memory_requirement_mb = 0.0;

    // Algorithm suitability metrics
    bool suitable_for_matrix_multiplication = false;
    bool suitable_for_sparsification = false;
    bool suitable_for_lockfree_multicore = false;
    bool suitable_for_gpu = false;
    bool suitable_for_streaming = false;
};
#endif

} // namespace nerve::persistence::adaptive_acceleration
