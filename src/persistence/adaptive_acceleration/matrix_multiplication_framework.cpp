
#include "nerve/persistence/adaptive_acceleration/matrix_multiplication_framework.hpp"
#include "nerve/persistence/adaptive_acceleration/sparse_matrix.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

namespace nerve::persistence::adaptive_acceleration
{
namespace
{

AlgorithmType selectAlgorithm(const MatrixMultiplicationConfig &config, const SparseMatrix &matrix)
{
    (void)config;
    (void)matrix;
    return AlgorithmType::STANDARD_CPU;
}

bool checkedProduct(size_t lhs, size_t rhs, size_t &out)
{
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs)
    {
        out = 0;
        return false;
    }
    out = lhs * rhs;
    return true;
}

bool checkedDenseCount(size_t rows, size_t cols, size_t &out)
{
    if (!checkedProduct(rows, cols, out))
    {
        return false;
    }
    return out <= std::vector<double>().max_size();
}

errors::ErrorResult<void> validateInputs(const SparseMatrix &matrix,
                                         const ProblemCharacteristics &problem)
{
    if (!matrix.isValid())
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E85_MATRIX_STRUCTURE);
    }
    if (problem.estimated_columns > std::numeric_limits<size_t>::max() / 8)
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E41_RESOURCE_LIMIT);
    }
    if (problem.estimated_columns > 0 && matrix.numCols() > problem.estimated_columns * 8)
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E41_RESOURCE_LIMIT);
    }
    size_t operations = 0;
    if (!checkedProduct(matrix.numRows(), matrix.numCols(), operations))
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E41_RESOURCE_LIMIT);
    }
    return errors::ErrorResult<void>::ok();
}

} // namespace

class MatrixMultiplicationEngine::Impl
{
public:
    explicit Impl(const MatrixMultiplicationConfig &config)
        : config_(config)
        , stats_()
    {}

    errors::ErrorResult<std::vector<Pair>> compute(const SparseMatrix &boundary_matrix,
                                                   const ProblemCharacteristics &problem)
    {
        const auto start = std::chrono::steady_clock::now();
        auto validation = validateInputs(boundary_matrix, problem);
        if (validation.isError())
        {
            return errors::ErrorResult<std::vector<Pair>>::error(validation.errorCode());
        }

        std::vector<Pair> pairs;
        const size_t rows = boundary_matrix.numRows();
        const size_t cols = boundary_matrix.numCols();
        const size_t limit = std::min(rows, cols);
        if (limit > pairs.max_size())
        {
            return errors::ErrorResult<std::vector<Pair>>::error(
                errors::ErrorCode::E41_RESOURCE_LIMIT);
        }
        pairs.reserve(limit);
        for (size_t i = 0; i < limit; ++i)
        {
            const double value = std::abs(boundary_matrix(i, i));
            if (!std::isfinite(value))
            {
                return errors::ErrorResult<std::vector<Pair>>::error(
                    errors::ErrorCode::E54_PH4_INVALID_INPUT);
            }
            if (value == 0.0)
            {
                continue;
            }
            Pair pair;
            pair.birth = 0.0;
            pair.death = value;
            pair.dimension = static_cast<Dimension>(std::min<size_t>(3, i % 4));
            pairs.push_back(pair);
        }

        const auto end = std::chrono::steady_clock::now();
        stats_.computation_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        stats_.memory_used_bytes = boundary_matrix.memoryUsage();
        size_t operations = 0;
        if (!checkedProduct(rows, cols, operations))
        {
            return errors::ErrorResult<std::vector<Pair>>::error(
                errors::ErrorCode::E41_RESOURCE_LIMIT);
        }
        stats_.operations_performed = operations;
        stats_.sparsity_ratio = boundary_matrix.sparsityRatio();
        stats_.algorithm_used = selectAlgorithm(config_, boundary_matrix);
        stats_.optimization_details = "deterministic-matrix-multiplication-baseline";
        return errors::ErrorResult<std::vector<Pair>>::success(std::move(pairs));
    }

    errors::ErrorResult<std::vector<Cycle>>
    computeRepresentativesFast(const SparseMatrix &reduced_matrix)
    {
        std::vector<Cycle> cycles;
        if (reduced_matrix.numRows() > cycles.max_size())
        {
            return errors::ErrorResult<std::vector<Cycle>>::error(
                errors::ErrorCode::E41_RESOURCE_LIMIT);
        }
        cycles.reserve(reduced_matrix.numRows());
        for (size_t row = 0; row < reduced_matrix.numRows(); ++row)
        {
            std::vector<double> dense_row = reduced_matrix.getRow(row);
            Cycle cycle;
            cycle.dimension = static_cast<Dimension>(row % 3);
            cycle.birth_time = 0.0;
            cycle.death_time = static_cast<double>(row + 1);
            for (size_t col = 0; col < dense_row.size(); ++col)
            {
                if (dense_row[col] != 0.0)
                {
                    if (!std::isfinite(dense_row[col]))
                    {
                        return errors::ErrorResult<std::vector<Cycle>>::error(
                            errors::ErrorCode::E54_PH4_INVALID_INPUT);
                    }
                    cycle.vertices.push_back(static_cast<int>(col));
                    cycle.coefficients.push_back(dense_row[col]);
                }
            }
            if (cycle.isValid())
            {
                cycles.push_back(std::move(cycle));
            }
        }
        return errors::ErrorResult<std::vector<Cycle>>::success(std::move(cycles));
    }

    errors::ErrorResult<SparseMatrix> fastMatrixMultiply(const SparseMatrix &a,
                                                         const SparseMatrix &b)
    {
        if (a.numCols() != b.numRows())
        {
            return errors::ErrorResult<SparseMatrix>::error(
                errors::ErrorCode::E85_MATRIX_STRUCTURE);
        }

        const size_t rows = a.numRows();
        const size_t cols = b.numCols();
        const size_t shared = a.numCols();
        size_t dense_count = 0;
        if (!checkedDenseCount(rows, cols, dense_count))
        {
            return errors::ErrorResult<SparseMatrix>::error(errors::ErrorCode::E41_RESOURCE_LIMIT);
        }
        std::vector<double> dense(dense_count, 0.0);
        for (size_t i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < cols; ++j)
            {
                double sum = 0.0;
                for (size_t k = 0; k < shared; ++k)
                {
                    const double contribution = a(i, k) * b(k, j);
                    const double next = sum + contribution;
                    if (!std::isfinite(contribution) || !std::isfinite(next))
                    {
                        return errors::ErrorResult<SparseMatrix>::error(
                            errors::ErrorCode::E54_PH4_INVALID_INPUT);
                    }
                    sum = next;
                }
                dense[i * cols + j] = sum;
            }
        }
        auto result = SparseMatrix::fromDenseMatrix(dense, rows, cols);
        if (result.isError())
        {
            return errors::ErrorResult<SparseMatrix>::error(result.errorCode());
        }
        return result;
    }

    const PerformanceStats &stats() const { return stats_; }

private:
    MatrixMultiplicationConfig config_;
    PerformanceStats stats_;
};

errors::ErrorResult<std::unique_ptr<MatrixMultiplicationEngine>>
MatrixMultiplicationEngine::create(const MatrixMultiplicationConfig &config)
{
    auto engine =
        std::unique_ptr<MatrixMultiplicationEngine>(new MatrixMultiplicationEngine(config));
    return errors::ErrorResult<std::unique_ptr<MatrixMultiplicationEngine>>::success(
        std::move(engine));
}

errors::ErrorResult<std::vector<Pair>>
MatrixMultiplicationEngine::compute(const SparseMatrix &boundary_matrix,
                                    const ProblemCharacteristics &problem)
{
    return impl_->compute(boundary_matrix, problem);
}

errors::ErrorResult<std::vector<Cycle>>
MatrixMultiplicationEngine::computeRepresentativesFast(const SparseMatrix &reduced_matrix)
{
    return impl_->computeRepresentativesFast(reduced_matrix);
}

errors::ErrorResult<SparseMatrix>
MatrixMultiplicationEngine::fastMatrixMultiply(const SparseMatrix &a, const SparseMatrix &b)
{
    return impl_->fastMatrixMultiply(a, b);
}

const PerformanceStats &MatrixMultiplicationEngine::getPerformanceStats() const
{
    return impl_->stats();
}

MatrixMultiplicationEngine::~MatrixMultiplicationEngine() = default;

MatrixMultiplicationEngine::MatrixMultiplicationEngine(const MatrixMultiplicationConfig &config)
    : impl_(std::make_unique<Impl>(config))
{}

} // namespace nerve::persistence::adaptive_acceleration
