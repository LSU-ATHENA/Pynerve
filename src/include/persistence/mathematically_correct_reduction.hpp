
#pragma once
#include "math/field_matrix.hpp"
#include "math/finite_field.hpp"
#include "math/matrix_reduction.hpp"
#include "math/precision_manager.hpp"
#include "nerve/error/error_registry.hpp"
#include "nerve/persistence/reduction/reducer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>
namespace nerve::persistence
{

using ::nerve::math::FieldMatrix;
using ::nerve::math::FiniteField;
using ::nerve::math::makeFieldMatrix;
using ::nerve::math::MatrixReduction;
using ::nerve::math::performAdaptiveReduction;
using ::nerve::math::performCohomologyReduction;
using ::nerve::math::performSparseReduction;
using ::nerve::math::PrecisionManager;

class MathematicallyCorrectReduction
{
private:
    int field_characteristic_ = 2;
    const algebra::BoundaryMatrix *boundary_matrix_ = nullptr;
    std::vector<Pair> persistence_pairs_;
    std::vector<Size> betti_numbers_;
    std::vector<Index> essential_cycles_;
    double computation_time_ = 0.0;
    size_t operations_count_ = 0;
    template <typename Target, typename Source>
    static error::Result<Target> forwardError(const error::Result<Source> &result)
    {
        return error::Result<Target>::err(static_cast<error::TDAErrorCode>(result.error().value()),
                                          std::string(result.detail()), result.where());
    }

    template <typename Source>
    static error::Result<void> forwardVoidError(const error::Result<Source> &result)
    {
        return error::Result<void>::err(static_cast<error::TDAErrorCode>(result.error().value()),
                                        std::string(result.detail()), result.where());
    }

    template <typename Field>
    error::Result<std::vector<Pair>> computeReductionWithField()
    {
        auto matrix_result = convertToFieldMatrix<Field>();
        if (matrix_result.isErr())
        {
            return forwardError<std::vector<Pair>>(matrix_result);
        }
        auto &fieldMatrix = matrix_result.value();
        auto reduction_result = performAdaptiveReduction(fieldMatrix);
        if (reduction_result.isErr())
        {
            return forwardError<std::vector<Pair>>(reduction_result);
        }
        auto &result = reduction_result.value();
        operations_count_ = result.operations_count;
        computation_time_ = result.computation_time;
        return convertToPersistencePairs<Field>(result);
    }
    template <typename Field>
    error::Result<FieldMatrix<Field>> convertToFieldMatrix() const
    {
        try
        {
            if (!boundary_matrix_)
            {
                return error::Result<FieldMatrix<Field>>::err(error::TDAErrorCode::InvalidDimension,
                                                              "No boundary matrix provided");
            }
            size_t rows = boundary_matrix_->rows();
            size_t cols = boundary_matrix_->cols();
            auto matrix_result = makeFieldMatrix<Field>(rows, cols);
            if (matrix_result.isErr())
            {
                return forwardError<FieldMatrix<Field>>(matrix_result);
            }
            auto &fieldMatrix = matrix_result.value();
            auto precision = PrecisionManager::determinePrecision(
                PrecisionManager::OperationType::MATRIX_REDUCTION);
            for (size_t i = 0; i < rows; ++i)
            {
                for (size_t j = 0; j < cols; ++j)
                {
                    double value = boundary_matrix_->getMatrixEntry(i, j);
                    if (PrecisionManager::isZero(value, precision))
                    {
                        fieldMatrix(i, j) = Field::zero();
                    }
                    else
                    {
                        int field_value = static_cast<int>(std::round(value));
                        fieldMatrix(i, j) = Field(field_value);
                    }
                }
            }
            return error::Result<FieldMatrix<Field>>::ok(fieldMatrix);
        }
        catch (const std::exception &e)
        {
            return error::Result<FieldMatrix<Field>>::err(
                error::TDAErrorCode::InvalidFieldOperation,
                std::string("Failed to convert to field matrix: ") + e.what());
        }
    }
    template <typename Field>
    error::Result<std::vector<Pair>>
    convertToPersistencePairs(const MatrixReduction<Field>::ReductionResult &result) const
    {
        try
        {
            std::vector<Pair> pairs;
            for (size_t i = 0; i < result.pivots.size(); ++i)
            {
                size_t pivot_col = result.pivots[i];
                size_t pivot_row = result.pivot_rows[i];
                double birth_time = getFiltrationValue(pivot_col);
                double death_time = getFiltrationValue(pivot_row);
                size_t dimension = getSimplexDimension(pivot_col);
                Pair pair{birth_time, death_time, static_cast<Dimension>(dimension),
                          static_cast<Index>(pivot_col), static_cast<Index>(pivot_row)};
                if (death_time < std::numeric_limits<double>::infinity())
                {
                    pairs.push_back(pair);
                }
                else
                {
                    pair.death = std::numeric_limits<double>::infinity();
                    pair.death_index = -1;
                    pairs.push_back(pair);
                }
            }
            for (size_t col = 0; col < boundary_matrix_->cols(); ++col)
            {
                if (std::find(result.pivots.begin(), result.pivots.end(), col) ==
                    result.pivots.end())
                {
                    double birth_time = getFiltrationValue(col);
                    size_t dimension = getSimplexDimension(col);
                    pairs.push_back(Pair{birth_time, std::numeric_limits<double>::infinity(),
                                         static_cast<Dimension>(dimension), static_cast<Index>(col),
                                         -1});
                }
            }
            return error::Result<std::vector<Pair>>::ok(pairs);
        }
        catch (const std::exception &e)
        {
            return error::Result<std::vector<Pair>>::err(
                error::TDAErrorCode::InvalidFieldOperation,
                std::string("Failed to convert to persistence pairs: ") + e.what());
        }
    }
    double getFiltrationValue(size_t column) const
    {
        if (!boundary_matrix_)
            return 0.0;
        return boundary_matrix_->getFiltrationValue(column);
    }
    size_t getSimplexDimension(size_t column) const
    {
        if (!boundary_matrix_)
            return 0;
        return static_cast<size_t>(boundary_matrix_->getColSimplexDimension(column));
    }
    template <int P>
    error::Result<std::vector<Pair>> computeReductionWithArbitraryField()
    {
        static_assert(isPrime(P), "Characteristic must be prime");
        auto matrix_result = convertToFieldMatrix<FiniteField<P>>();
        if (matrix_result.isErr())
        {
            return forwardError<std::vector<Pair>>(matrix_result);
        }
        auto &matrix = matrix_result.value();
        auto reduction_result = performAdaptiveReduction(matrix);
        if (reduction_result.isErr())
        {
            return forwardError<std::vector<Pair>>(reduction_result);
        }
        return convertToPersistencePairs<FiniteField<P>>(reduction_result.value());
    }
    static constexpr bool isPrime(int n)
    {
        if (n <= 1)
            return false;
        if (n <= 3)
            return true;
        if (n % 2 == 0 || n % 3 == 0)
            return false;
        for (int i = 5; i * i <= n; i += 6)
        {
            if (n % i == 0 || n % (i + 2) == 0)
            {
                return false;
            }
        }
        return true;
    }

public:
    MathematicallyCorrectReduction() = default;
    explicit MathematicallyCorrectReduction(const algebra::BoundaryMatrix &boundary_matrix)
        : boundary_matrix_(&boundary_matrix)
    {}
    error::Result<void> setFieldCharacteristic(int characteristic)
    {
        if (characteristic < 2)
        {
            return error::Result<void>::err(error::TDAErrorCode::InvalidFieldOperation,
                                            "Field characteristic must be at least 2");
        }
        if (!isPrime(characteristic))
        {
            return error::Result<void>::err(error::TDAErrorCode::InvalidFieldOperation,
                                            "Field characteristic must be prime");
        }
        field_characteristic_ = characteristic;
        return error::Result<void>::ok();
    }
    error::Result<void> compute()
    {
        auto start = std::chrono::high_resolution_clock::now();
        auto result = error::Result<std::vector<Pair>>::err(
            error::TDAErrorCode::InvalidFieldOperation, "No field reduction selected");
        switch (field_characteristic_)
        {
            case 2:
                result = computeReductionWithField<FiniteField<2>>();
                break;
            case 3:
                result = computeReductionWithField<FiniteField<3>>();
                break;
            case 5:
                result = computeReductionWithField<FiniteField<5>>();
                break;
            case 7:
                result = computeReductionWithField<FiniteField<7>>();
                break;
            default:
                switch (field_characteristic_)
                {
                    case 11:
                        result = computeReductionWithArbitraryField<11>();
                        break;
                    case 13:
                        result = computeReductionWithArbitraryField<13>();
                        break;
                    case 17:
                        result = computeReductionWithArbitraryField<17>();
                        break;
                    case 19:
                        result = computeReductionWithArbitraryField<19>();
                        break;
                    default:
                        return error::Result<void>::err(
                            error::TDAErrorCode::InvalidFieldOperation,
                            "Unsupported field characteristic in production reducer");
                }
                break;
        }
        if (result.isErr())
        {
            return forwardVoidError(result);
        }
        persistence_pairs_ = result.value();
        computeBettiNumbersFromPivots();
        classifyEssentialCycles();
        auto end = std::chrono::high_resolution_clock::now();
        computation_time_ = std::chrono::duration<double>(end - start).count();
        return error::Result<void>::ok();
    }
    error::Result<void> standardReduction()
    {
        auto result = computeReductionWithField<FiniteField<2>>();
        if (result.isErr())
        {
            return forwardVoidError(result);
        }
        persistence_pairs_ = result.value();
        return error::Result<void>::ok();
    }
    error::Result<void> fastPhase1Reduction()
    {
        auto matrix_result = convertToFieldMatrix<FiniteField<2>>();
        if (matrix_result.isErr())
        {
            return forwardVoidError(matrix_result);
        }
        auto &matrix = matrix_result.value();
        auto reduction_result = performSparseReduction(matrix);
        if (reduction_result.isErr())
        {
            return forwardVoidError(reduction_result);
        }
        auto pairs_result = convertToPersistencePairs<FiniteField<2>>(reduction_result.value());
        if (pairs_result.isErr())
        {
            return forwardVoidError(pairs_result);
        }
        persistence_pairs_ = pairs_result.value();
        return error::Result<void>::ok();
    }
    error::Result<void> acceleratedPhase2Reduction()
    {
        auto matrix_result = convertToFieldMatrix<FiniteField<2>>();
        if (matrix_result.isErr())
        {
            return forwardVoidError(matrix_result);
        }
        auto &matrix = matrix_result.value();
        auto reduction_result = performCohomologyReduction(matrix);
        if (reduction_result.isErr())
        {
            return forwardVoidError(reduction_result);
        }
        auto pairs_result = convertToPersistencePairs<FiniteField<2>>(reduction_result.value());
        if (pairs_result.isErr())
        {
            return forwardVoidError(pairs_result);
        }
        persistence_pairs_ = pairs_result.value();
        return error::Result<void>::ok();
    }
    error::Result<void> CohomologyReduction()
    {
        auto matrix_result = convertToFieldMatrix<FiniteField<2>>();
        if (matrix_result.isErr())
        {
            return forwardVoidError(matrix_result);
        }
        auto &matrix = matrix_result.value();
        auto reduction_result = performCohomologyReduction(matrix);
        if (reduction_result.isErr())
        {
            return forwardVoidError(reduction_result);
        }
        auto pairs_result = convertToPersistencePairs<FiniteField<2>>(reduction_result.value());
        if (pairs_result.isErr())
        {
            return forwardVoidError(pairs_result);
        }
        persistence_pairs_ = pairs_result.value();
        return error::Result<void>::ok();
    }
    const std::vector<Pair> &getPersistencePairs() const { return persistence_pairs_; }
    const std::vector<Size> &getBettiNumbers() const { return betti_numbers_; }
    const std::vector<Index> &getEssentialCycles() const { return essential_cycles_; }
    double getComputationTime() const { return computation_time_; }
    size_t getOperationsCount() const { return operations_count_; }
    int getFieldCharacteristic() const { return field_characteristic_; }

private:
    void computeBettiNumbersFromPivots()
    {
        betti_numbers_.clear();
        std::map<Size, Size> essential_by_dim;
        for (const auto &pair : persistence_pairs_)
        {
            if (pair.isInfinite())
            {
                essential_by_dim[static_cast<Size>(pair.dimension)]++;
            }
        }
        for (Size dim = 0; dim <= 5; ++dim)
        {
            Size betti = essential_by_dim[dim];
            betti_numbers_.push_back(betti);
        }
    }
    void classifyEssentialCycles()
    {
        essential_cycles_.clear();
        for (const auto &pair : persistence_pairs_)
        {
            if (pair.isInfinite())
            {
                essential_cycles_.push_back(pair.birth_index);
            }
        }
    }
};
inline error::Result<std::unique_ptr<MathematicallyCorrectReduction>>
makeMathematicallyCorrectReduction()
{
    try
    {
        auto reduction = std::make_unique<MathematicallyCorrectReduction>();
        return error::Result<std::unique_ptr<MathematicallyCorrectReduction>>::ok(
            std::move(reduction));
    }
    catch (const std::exception &e)
    {
        return error::Result<std::unique_ptr<MathematicallyCorrectReduction>>::err(
            error::TDAErrorCode::AllocationFailed,
            std::string("Failed to create reduction: ") + e.what());
    }
}
inline error::Result<std::unique_ptr<MathematicallyCorrectReduction>>
makeMathematicallyCorrectReduction(const algebra::BoundaryMatrix &boundary_matrix)
{
    try
    {
        auto reduction = std::make_unique<MathematicallyCorrectReduction>(boundary_matrix);
        return error::Result<std::unique_ptr<MathematicallyCorrectReduction>>::ok(
            std::move(reduction));
    }
    catch (const std::exception &e)
    {
        return error::Result<std::unique_ptr<MathematicallyCorrectReduction>>::err(
            error::TDAErrorCode::AllocationFailed,
            std::string("Failed to create reduction with boundary matrix: ") + e.what());
    }
}
} // namespace nerve::persistence
