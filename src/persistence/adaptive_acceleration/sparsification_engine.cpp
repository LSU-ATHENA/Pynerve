
#include "nerve/persistence/adaptive_acceleration/sparse_matrix.hpp"
#include "nerve/persistence/adaptive_acceleration/sparsification_engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <ranges>
#include <utility>

namespace nerve::persistence::adaptive_acceleration
{
namespace
{

constexpr std::size_t kMaxDenseEntries = static_cast<std::size_t>(std::numeric_limits<int>::max());

bool checkedDenseCount(std::size_t rows, std::size_t cols, std::size_t &out)
{
    if (cols != 0 && rows > std::numeric_limits<std::size_t>::max() / cols)
    {
        out = 0;
        return false;
    }
    out = rows * cols;
    return out <= kMaxDenseEntries && out <= std::vector<double>().max_size();
}

errors::ErrorResult<void> validateFiniteValues(const SparseMatrix &matrix)
{
    for (const double value : matrix.values())
    {
        if (!std::isfinite(value))
        {
            return errors::ErrorResult<void>::error(errors::ErrorCode::E54_PH4_INVALID_INPUT);
        }
    }
    return errors::ErrorResult<void>::ok();
}

errors::ErrorResult<void> validateSparsificationInputs(const SparseMatrix &matrix,
                                                       double target_sparsity)
{
    if (!matrix.isValid())
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E85_MATRIX_STRUCTURE);
    }
    if (!std::isfinite(target_sparsity) || target_sparsity < 0.0 || target_sparsity > 1.0)
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E54_PH4_INVALID_INPUT);
    }
    return validateFiniteValues(matrix);
}

errors::ErrorResult<std::vector<double>> toDense(const SparseMatrix &matrix)
{
    std::size_t dense_count = 0;
    if (!checkedDenseCount(matrix.numRows(), matrix.numCols(), dense_count))
    {
        return errors::ErrorResult<std::vector<double>>::error(
            errors::ErrorCode::E41_RESOURCE_LIMIT);
    }
    auto finite_values = validateFiniteValues(matrix);
    if (finite_values.isError())
    {
        return errors::ErrorResult<std::vector<double>>::error(finite_values.errorCode());
    }

    std::vector<double> dense(dense_count, 0.0);
    const auto &row_indices = matrix.rowIndices();
    const auto &col_indices = matrix.colIndices();
    const auto &values = matrix.values();
    for (std::size_t entry = 0; entry < values.size(); ++entry)
    {
        const auto row = static_cast<std::size_t>(row_indices[entry]);
        const auto col = static_cast<std::size_t>(col_indices[entry]);
        dense[row * matrix.numCols() + col] = values[entry];
    }
    return errors::ErrorResult<std::vector<double>>::success(std::move(dense));
}

errors::ErrorResult<SparseMatrix> buildSparse(const std::vector<double> &dense, std::size_t rows,
                                              std::size_t cols)
{
    auto result = SparseMatrix::fromDenseMatrix(dense, rows, cols);
    if (result.isError())
    {
        return errors::ErrorResult<SparseMatrix>::error(result.errorCode());
    }
    return result;
}

errors::ErrorResult<double> percentileThreshold(const std::vector<double> &magnitudes,
                                                double keep_fraction)
{
    if (!std::isfinite(keep_fraction))
    {
        return errors::ErrorResult<double>::error(errors::ErrorCode::E54_PH4_INVALID_INPUT);
    }
    if (magnitudes.empty())
    {
        return errors::ErrorResult<double>::success(0.0);
    }
    const double clamped_keep = std::clamp(keep_fraction, 0.0, 1.0);
    const std::size_t keep_count =
        static_cast<std::size_t>(std::ceil(clamped_keep * static_cast<double>(magnitudes.size())));
    if (keep_count == 0)
    {
        return errors::ErrorResult<double>::success(std::numeric_limits<double>::max());
    }
    if (keep_count >= magnitudes.size())
    {
        return errors::ErrorResult<double>::success(0.0);
    }

    std::vector<double> sorted = magnitudes;
    std::sort(sorted.begin(), sorted.end(), std::greater<double>());
    return errors::ErrorResult<double>::success(double{sorted[keep_count - 1]});
}

errors::ErrorResult<std::vector<double>>
thresholdDense(const SparseMatrix &matrix, double keep_fraction, bool preserve_diagonal)
{
    auto dense_result = toDense(matrix);
    if (dense_result.isError())
    {
        return errors::ErrorResult<std::vector<double>>::error(dense_result.errorCode());
    }
    std::vector<double> dense = dense_result.moveValue();
    std::vector<double> magnitudes;
    magnitudes.reserve(dense.size());
    for (const double value : dense)
    {
        if (!std::isfinite(value))
        {
            return errors::ErrorResult<std::vector<double>>::error(
                errors::ErrorCode::E54_PH4_INVALID_INPUT);
        }
        if (value != 0.0)
        {
            magnitudes.push_back(std::abs(value));
        }
    }

    auto threshold_result = percentileThreshold(magnitudes, keep_fraction);
    if (threshold_result.isError())
    {
        return errors::ErrorResult<std::vector<double>>::error(threshold_result.errorCode());
    }
    const double threshold = threshold_result.value();
    const bool drop_all = keep_fraction <= 0.0;
    for (std::size_t row = 0; row < matrix.numRows(); ++row)
    {
        for (std::size_t col = 0; col < matrix.numCols(); ++col)
        {
            if (preserve_diagonal && row == col)
            {
                continue;
            }
            const std::size_t index = row * matrix.numCols() + col;
            if (drop_all || std::abs(dense[index]) < threshold)
            {
                dense[index] = 0.0;
            }
        }
    }
    return errors::ErrorResult<std::vector<double>>::success(std::move(dense));
}

} // namespace

class SparsificationEngine::Impl
{
public:
    explicit Impl(const SparsificationConfig &config)
        : config(config)
    {}

    SparsificationConfig config;
};

errors::ErrorResult<std::unique_ptr<SparsificationEngine>>
SparsificationEngine::create(const SparsificationConfig &config)
{
    if (!std::isfinite(config.target_sparsity) || config.target_sparsity < 0.0 ||
        config.target_sparsity > 1.0 || !std::isfinite(config.max_sparsity_loss) ||
        config.max_sparsity_loss < 0.0)
    {
        return errors::ErrorResult<std::unique_ptr<SparsificationEngine>>::error(
            errors::ErrorCode::E54_PH4_INVALID_INPUT);
    }
    std::unique_ptr<SparsificationEngine> engine(new SparsificationEngine(config));
    return errors::ErrorResult<std::unique_ptr<SparsificationEngine>>::success(std::move(engine));
}

errors::ErrorResult<SparseMatrix> SparsificationEngine::sparsify(const SparseMatrix &matrix,
                                                                 SparsificationStrategy strategy,
                                                                 double target_sparsity)
{
    auto validation = validateSparsificationInputs(matrix, target_sparsity);
    if (validation.isError())
    {
        return errors::ErrorResult<SparseMatrix>::error(validation.errorCode());
    }

    const auto start = std::chrono::steady_clock::now();
    errors::ErrorResult<SparseMatrix> result =
        errors::ErrorResult<SparseMatrix>::error(errors::ErrorCode::E85_MATRIX_STRUCTURE);
    switch (strategy)
    {
        case SparsificationStrategy::SWAP_REDUCTION:
            result = swapReduction(matrix);
            break;
        case SparsificationStrategy::EXHAUSTIVE_REDUCTION:
            result = exhaustiveReduction(matrix);
            break;
        case SparsificationStrategy::RETROSPECTIVE_REDUCTION:
            result = retrospectiveReduction(matrix);
            break;
        case SparsificationStrategy::ADAPTIVE_HYBRID:
        {
            ProblemCharacteristics problem;
            problem.sparsity_ratio = matrix.sparsityRatio();
            problem.estimated_columns = matrix.numCols();
            result = adaptiveHybrid(matrix, problem);
            break;
        }
    }

    if (result.isError())
    {
        return errors::ErrorResult<SparseMatrix>::error(result.errorCode());
    }

    const bool preserve_diagonal = impl_->config.preserve_diagonal;
    auto dense_result = thresholdDense(result.value(), target_sparsity, preserve_diagonal);
    if (dense_result.isError())
    {
        return errors::ErrorResult<SparseMatrix>::error(dense_result.errorCode());
    }
    std::vector<double> dense = dense_result.moveValue();
    auto thresholded = buildSparse(dense, matrix.numRows(), matrix.numCols());
    if (thresholded.isError())
    {
        return errors::ErrorResult<SparseMatrix>::error(thresholded.errorCode());
    }

    const auto end = std::chrono::steady_clock::now();
    performance_stats_.computation_time_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    performance_stats_.original_nonzero_count = matrix.nnz();
    performance_stats_.sparsified_nonzero_count = thresholded.value().nnz();
    performance_stats_.sparsity_ratio =
        matrix.nnz() == 0
            ? 1.0
            : static_cast<double>(thresholded.value().nnz()) / static_cast<double>(matrix.nnz());
    performance_stats_.strategy_used = strategy;
    performance_stats_.speedup_factor = estimateSparsificationBenefit(matrix, strategy);
    performance_stats_.optimization_details = "deterministic sparsification";

    return thresholded;
}

SparsificationStrategy
SparsificationEngine::selectOptimalStrategy(const ProblemCharacteristics &problem)
{
    if (problem.sparsity_ratio < 0.05)
    {
        return SparsificationStrategy::EXHAUSTIVE_REDUCTION;
    }
    if (problem.sparsity_ratio < 0.20)
    {
        return SparsificationStrategy::RETROSPECTIVE_REDUCTION;
    }
    if (problem.estimated_columns > 4096)
    {
        return SparsificationStrategy::ADAPTIVE_HYBRID;
    }
    return SparsificationStrategy::SWAP_REDUCTION;
}

double SparsificationEngine::estimateSparsificationBenefit(const SparseMatrix &matrix,
                                                           SparsificationStrategy strategy)
{
    const double density = matrix.sparsityRatio();
    if (!std::isfinite(density))
    {
        return 1.0;
    }
    const double bounded_density = std::clamp(density, 0.0, 1.0);
    const double base = std::max(1.0, 1.0 + (1.0 - bounded_density) * 2.0);
    double benefit = base;
    switch (strategy)
    {
        case SparsificationStrategy::SWAP_REDUCTION:
            break;
        case SparsificationStrategy::RETROSPECTIVE_REDUCTION:
            benefit = base * 1.1;
            break;
        case SparsificationStrategy::EXHAUSTIVE_REDUCTION:
            benefit = base * 1.2;
            break;
        case SparsificationStrategy::ADAPTIVE_HYBRID:
            benefit = base * 1.15;
            break;
    }
    return std::isfinite(benefit) ? benefit : 1.0;
}

errors::ErrorResult<SparseMatrix> SparsificationEngine::swapReduction(const SparseMatrix &matrix)
{
    auto dense_result = toDense(matrix);
    if (dense_result.isError())
    {
        return errors::ErrorResult<SparseMatrix>::error(dense_result.errorCode());
    }
    std::vector<double> dense = dense_result.moveValue();
    const std::size_t paired_extent = std::min(matrix.numRows(), matrix.numCols());
    for (std::size_t row = 0; row < paired_extent; ++row)
    {
        for (std::size_t col = row + 1; col < paired_extent; ++col)
        {
            const std::size_t upper = row * matrix.numCols() + col;
            const std::size_t lower = col * matrix.numCols() + row;
            if (std::abs(dense[upper]) > std::abs(dense[lower]))
            {
                std::swap(dense[upper], dense[lower]);
            }
        }
    }
    return buildSparse(dense, matrix.numRows(), matrix.numCols());
}

errors::ErrorResult<SparseMatrix>
SparsificationEngine::exhaustiveReduction(const SparseMatrix &matrix)
{
    auto dense_result = thresholdDense(matrix, 0.25, true);
    if (dense_result.isError())
    {
        return errors::ErrorResult<SparseMatrix>::error(dense_result.errorCode());
    }
    std::vector<double> dense = dense_result.moveValue();
    return buildSparse(dense, matrix.numRows(), matrix.numCols());
}

errors::ErrorResult<SparseMatrix>
SparsificationEngine::retrospectiveReduction(const SparseMatrix &matrix)
{
    auto dense_result = thresholdDense(matrix, 0.50, true);
    if (dense_result.isError())
    {
        return errors::ErrorResult<SparseMatrix>::error(dense_result.errorCode());
    }
    std::vector<double> dense = dense_result.moveValue();
    return buildSparse(dense, matrix.numRows(), matrix.numCols());
}

errors::ErrorResult<SparseMatrix>
SparsificationEngine::adaptiveHybrid(const SparseMatrix &matrix,
                                     const ProblemCharacteristics &problem)
{
    const SparsificationStrategy selected = selectOptimalStrategy(problem);
    switch (selected)
    {
        case SparsificationStrategy::SWAP_REDUCTION:
            return swapReduction(matrix);
        case SparsificationStrategy::EXHAUSTIVE_REDUCTION:
            return exhaustiveReduction(matrix);
        case SparsificationStrategy::RETROSPECTIVE_REDUCTION:
        case SparsificationStrategy::ADAPTIVE_HYBRID:
            return retrospectiveReduction(matrix);
    }
    return retrospectiveReduction(matrix);
}

bool SparsificationEngine::shouldPreserveElement(const SparseMatrix &matrix, std::size_t row,
                                                 std::size_t col, SparsificationStrategy strategy)
{
    if (row == col)
    {
        return true;
    }
    const double magnitude = std::abs(matrix(row, col));
    if (!std::isfinite(magnitude))
    {
        return false;
    }
    if (strategy == SparsificationStrategy::EXHAUSTIVE_REDUCTION)
    {
        return magnitude > 0.0;
    }
    return magnitude > 1e-12;
}

double SparsificationEngine::computeSparsificationLoss(const SparseMatrix &original,
                                                       const SparseMatrix &sparsified)
{
    double loss = 0.0;
    for (std::size_t row = 0; row < original.numRows(); ++row)
    {
        for (std::size_t col = 0; col < original.numCols(); ++col)
        {
            const double delta = std::abs(original(row, col) - sparsified(row, col));
            const double next = loss + delta;
            if (!std::isfinite(delta) || !std::isfinite(next))
            {
                return std::numeric_limits<double>::max();
            }
            loss = next;
        }
    }
    return loss;
}

SparsificationEngine::SparsificationEngine(const SparsificationConfig &config)
    : impl_(std::make_unique<Impl>(config))
    , performance_stats_()
{}

SparsificationEngine::~SparsificationEngine() = default;

} // namespace nerve::persistence::adaptive_acceleration
