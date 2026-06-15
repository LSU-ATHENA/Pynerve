
#pragma once
#include "math/field_matrix.hpp"
#include "nerve/error/error_registry.hpp"

#include <algorithm>
#include <chrono>
#include <optional>
#include <vector>

namespace nerve::math
{

template <typename Field>
class MatrixReduction
{
public:
    struct ReductionResult
    {
        std::vector<size_t> pivots;        // Pivot columns
        std::vector<size_t> pivot_rows;    // Pivot rows
        FieldMatrix<Field> reduced_matrix; // Reduced matrix
        size_t operations_count;           // Number of field operations
        double computation_time;           // Time in seconds
        size_t rank;                       // Matrix rank
        std::vector<bool> zero_rows;       // Zero row indicators
        std::vector<bool> zero_columns;    // Zero column indicators
    };

    enum class Strategy
    {
        GAUSSIAN_DENSE,     // Standard Gaussian elimination
        SPARSE_ACCELERATED, // Accelerated for sparse matrices
        COHOMOLOGY,         // fast cohomology reduction
        HYBRID_APPROACH     // Adaptive hybrid method
    };

    static error::Result<ReductionResult> gaussianElimination(FieldMatrix<Field> &matrix)
    {
        auto start = std::chrono::high_resolution_clock::now();

        ReductionResult result;
        result.operations_count = 0;
        result.zero_rows.resize(matrix.rows(), false);
        result.zero_columns.resize(matrix.cols(), false);

        size_t current_row = 0;

        for (size_t col = 0; col < matrix.cols() && current_row < matrix.rows(); ++col)
        {
            auto pivot_row = matrix.findPivot(col, current_row);

            if (!pivot_row)
            {
                result.zero_columns[col] = true;
                continue;
            }

            if (*pivot_row != current_row)
            {
                matrix.swapRows(current_row, *pivot_row);
                result.operations_count++;
            }

            Field pivot_value = matrix(current_row, col);
            if (pivot_value != Field::one())
            {
                matrix.scaleRow(current_row, Field::one() / pivot_value);
                result.operations_count++;
            }

            for (size_t row = 0; row < matrix.rows(); ++row)
            {
                if (row != current_row && matrix(row, col) != Field::zero())
                {
                    Field factor = matrix(row, col);
                    matrix.addScaledRow(row, current_row, -factor);
                    result.operations_count++;
                }
            }

            result.pivots.push_back(col);
            result.pivot_rows.push_back(current_row);
            current_row++;
        }

        for (size_t row = current_row; row < matrix.rows(); ++row)
        {
            result.zero_rows[row] = matrix.isZeroRow(row);
        }

        result.reduced_matrix = matrix;
        result.rank = result.pivots.size();

        auto end = std::chrono::high_resolution_clock::now();
        result.computation_time = std::chrono::duration<double>(end - start).count();

        return error::Result<ReductionResult>::ok(result);
    }

    static error::Result<ReductionResult> sparseReduction(FieldMatrix<Field> &matrix)
    {
        auto start = std::chrono::high_resolution_clock::now();

        double sparsity = computeSparsity(matrix);

        if (sparsity > 0.5)
        {
            return gaussianElimination(matrix);
        }

        ReductionResult result;
        result.operations_count = 0;
        result.zero_rows.resize(matrix.rows(), false);
        result.zero_columns.resize(matrix.cols(), false);

        size_t current_row = 0;

        for (size_t col = 0; col < matrix.cols() && current_row < matrix.rows(); ++col)
        {
            auto pivot_row = findSparsePivot(matrix, col, current_row);

            if (!pivot_row)
            {
                result.zero_columns[col] = true;
                continue;
            }

            if (*pivot_row != current_row)
            {
                matrix.swapRows(current_row, *pivot_row);
                result.operations_count++;
            }

            Field pivot_value = matrix(current_row, col);
            if (pivot_value != Field::one())
            {
                matrix.scaleRow(current_row, Field::one() / pivot_value);
                result.operations_count++;
            }

            for (size_t row = 0; row < matrix.rows(); ++row)
            {
                if (row != current_row && matrix(row, col) != Field::zero())
                {
                    Field factor = matrix(row, col);
                    matrix.addScaledRow(row, current_row, -factor);
                    result.operations_count++;
                }
            }

            result.pivots.push_back(col);
            result.pivot_rows.push_back(current_row);
            current_row++;
        }

        for (size_t row = current_row; row < matrix.rows(); ++row)
        {
            result.zero_rows[row] = matrix.isZeroRow(row);
        }

        result.reduced_matrix = matrix;
        result.rank = result.pivots.size();

        auto end = std::chrono::high_resolution_clock::now();
        result.computation_time = std::chrono::duration<double>(end - start).count();

        return error::Result<ReductionResult>::ok(result);
    }

    static error::Result<ReductionResult> cohomologyReduction(FieldMatrix<Field> &matrix)
    {
        auto start = std::chrono::high_resolution_clock::now();

        FieldMatrix<Field> coboundary_matrix = matrix.transpose();

        ReductionResult result;
        result.operations_count = 0;
        result.zero_rows.resize(coboundary_matrix.rows(), false);
        result.zero_columns.resize(coboundary_matrix.cols(), false);

        std::vector<int> columnPivots(coboundary_matrix.cols(), -1);
        std::vector<bool> columnProcessed(coboundary_matrix.cols(), false);

        for (size_t col = 0; col < coboundary_matrix.cols(); ++col)
        {
            if (columnProcessed[col])
                continue;

            auto pivot_row = coboundary_matrix.findPivot(col);

            if (!pivot_row)
            {
                columnProcessed[col] = true;
                columnPivots[col] = -1;
                continue;
            }

            int matching_col = -1;
            for (size_t j = 0; j < columnPivots.size(); ++j)
            {
                if (columnPivots[j] == static_cast<int>(*pivot_row) &&
                    static_cast<int>(j) != static_cast<int>(col) && !columnProcessed[j])
                {
                    matching_col = static_cast<int>(j);
                    break;
                }
            }

            if (matching_col != -1)
            {
                addCocycles(coboundary_matrix, col, matching_col);
                result.operations_count++;

                col--; // Re-process this column
                continue;
            }
            else
            {
                columnProcessed[col] = true;
                columnPivots[col] = static_cast<int>(*pivot_row);
            }
        }

        for (size_t i = 0; i < columnPivots.size(); ++i)
        {
            if (columnPivots[i] != -1)
            {
                result.pivots.push_back(i);
                result.pivot_rows.push_back(static_cast<size_t>(columnPivots[i]));
            }
        }

        result.reduced_matrix = coboundary_matrix.transpose();
        result.rank = result.pivots.size();

        auto end = std::chrono::high_resolution_clock::now();
        result.computation_time = std::chrono::duration<double>(end - start).count();

        return error::Result<ReductionResult>::ok(result);
    }

    static error::Result<ReductionResult> adaptiveReduction(FieldMatrix<Field> &matrix)
    {
        Strategy strategy = selectOptimalStrategy(matrix);

        switch (strategy)
        {
            case Strategy::GAUSSIAN_DENSE:
                return gaussianElimination(matrix);
            case Strategy::SPARSE_ACCELERATED:
                return sparseReduction(matrix);
            case Strategy::COHOMOLOGY:
                return cohomologyReduction(matrix);
            case Strategy::HYBRID_APPROACH:
            default:
                return hybridReduction(matrix);
        }
    }

private:
    static double computeSparsity(const FieldMatrix<Field> &matrix)
    {
        if (matrix.empty())
            return 0.0;

        size_t non_zero_count = 0;
        for (size_t i = 0; i < matrix.rows(); ++i)
        {
            for (size_t j = 0; j < matrix.cols(); ++j)
            {
                if (matrix(i, j) != Field::zero())
                {
                    non_zero_count++;
                }
            }
        }

        return static_cast<double>(non_zero_count) / matrix.size();
    }

    static std::optional<size_t> findSparsePivot(const FieldMatrix<Field> &matrix, size_t col,
                                                 size_t start_row)
    {
        for (size_t row = start_row; row < matrix.rows(); ++row)
        {
            if (matrix(row, col) != Field::zero())
            {
                return row;
            }
        }
        return std::nullopt;
    }

    static void addCocycles(FieldMatrix<Field> &matrix, size_t target, size_t source)
    {
        for (size_t row = 0; row < matrix.rows(); ++row)
        {
            if (matrix(row, source) != Field::zero())
            {
                matrix(row, target) = matrix(row, target) + matrix(row, source);
            }
        }
    }

    static Strategy selectOptimalStrategy(const FieldMatrix<Field> &matrix)
    {
        size_t size = std::max(matrix.rows(), matrix.cols());
        double sparsity = computeSparsity(matrix);

        if (size <= 1000)
        {
            return Strategy::GAUSSIAN_DENSE;
        }
        else if (sparsity < 0.1)
        {
            return Strategy::SPARSE_ACCELERATED;
        }
        else if (size <= 50000 && matrix.isSquare())
        {
            return Strategy::COHOMOLOGY;
        }
        else
        {
            return Strategy::HYBRID_APPROACH;
        }
    }

    static error::Result<ReductionResult> hybridReduction(FieldMatrix<Field> &matrix)
    {
        const double sparsity = computeSparsity(matrix);
        const size_t max_extent = std::max(matrix.rows(), matrix.cols());

        if (sparsity < 0.08)
        {
            return sparseReduction(matrix);
        }

        if (matrix.isSquare() && max_extent >= 2048 && sparsity < 0.35)
        {
            FieldMatrix<Field> cohomology_matrix = matrix;
            auto cohomology_result = cohomologyReduction(cohomology_matrix);
            if (cohomology_result.isOk())
            {
                matrix = cohomology_result.value().reduced_matrix;
                return cohomology_result;
            }
        }

        FieldMatrix<Field> sparse_stage_matrix = matrix;
        auto sparse_stage = sparseReduction(sparse_stage_matrix);
        if (sparse_stage.isOk())
        {
            matrix = sparse_stage.value().reduced_matrix;
            auto dense_stage = gaussianElimination(matrix);
            if (dense_stage.isOk())
            {
                auto merged = dense_stage.value();
                merged.operations_count += sparse_stage.value().operations_count;
                merged.computation_time += sparse_stage.value().computation_time;
                return error::Result<ReductionResult>::ok(std::move(merged));
            }
        }

        return gaussianElimination(matrix);
    }
};

class AdaptiveReductionSelector
{
public:
    enum class MatrixType
    {
        DENSE_SMALL,  // Small dense matrices
        DENSE_LARGE,  // Large dense matrices
        SPARSE_SMALL, // Small sparse matrices
        SPARSE_LARGE, // Large sparse matrices
        SQUARE_MEDIUM // Medium square matrices
    };

    static MatrixType classifyMatrix(size_t rows, size_t cols, double sparsity)
    {
        size_t size = std::max(rows, cols);
        bool isSquare = (rows == cols);
        bool is_small = (size <= 1000);
        bool is_sparse = (sparsity < 0.1);

        if (is_small)
        {
            return is_sparse ? MatrixType::SPARSE_SMALL : MatrixType::DENSE_SMALL;
        }
        else if (isSquare && size <= 50000)
        {
            return MatrixType::SQUARE_MEDIUM;
        }
        else
        {
            return is_sparse ? MatrixType::SPARSE_LARGE : MatrixType::DENSE_LARGE;
        }
    }

    template <typename Field>
    static typename MatrixReduction<Field>::Strategy
    selectStrategy(const FieldMatrix<Field> &matrix)
    {
        MatrixType type = classifyMatrix(matrix.rows(), matrix.cols(),
                                         MatrixReduction<Field>::computeSparsity(matrix));

        switch (type)
        {
            case MatrixType::DENSE_SMALL:
                return MatrixReduction<Field>::Strategy::GAUSSIAN_DENSE;
            case MatrixType::SPARSE_SMALL:
                return MatrixReduction<Field>::Strategy::SPARSE_ACCELERATED;
            case MatrixType::SQUARE_MEDIUM:
                return MatrixReduction<Field>::Strategy::COHOMOLOGY;
            case MatrixType::SPARSE_LARGE:
                return MatrixReduction<Field>::Strategy::SPARSE_ACCELERATED;
            case MatrixType::DENSE_LARGE:
            default:
                return MatrixReduction<Field>::Strategy::HYBRID_APPROACH;
        }
    }
};

template <typename Field>
error::Result<typename MatrixReduction<Field>::ReductionResult>
performGaussianElimination(FieldMatrix<Field> &matrix)
{
    return MatrixReduction<Field>::gaussianElimination(matrix);
}

template <typename Field>
error::Result<typename MatrixReduction<Field>::ReductionResult>
performSparseReduction(FieldMatrix<Field> &matrix)
{
    return MatrixReduction<Field>::sparseReduction(matrix);
}

template <typename Field>
error::Result<typename MatrixReduction<Field>::ReductionResult>
performCohomologyReduction(FieldMatrix<Field> &matrix)
{
    return MatrixReduction<Field>::cohomologyReduction(matrix);
}

template <typename Field>
error::Result<typename MatrixReduction<Field>::ReductionResult>
performAdaptiveReduction(FieldMatrix<Field> &matrix)
{
    return MatrixReduction<Field>::adaptiveReduction(matrix);
}

inline error::Result<MatrixReduction<Z2>::ReductionResult> performZ2Reduction(Z2Matrix &matrix)
{
    return MatrixReduction<Z2>::adaptiveReduction(matrix);
}

inline error::Result<MatrixReduction<Z3>::ReductionResult> performZ3Reduction(Z3Matrix &matrix)
{
    return MatrixReduction<Z3>::adaptiveReduction(matrix);
}

inline error::Result<MatrixReduction<Z5>::ReductionResult> performZ5Reduction(Z5Matrix &matrix)
{
    return MatrixReduction<Z5>::adaptiveReduction(matrix);
}

} // namespace nerve::math
