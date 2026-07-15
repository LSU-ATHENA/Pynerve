#pragma once

#include "nerve/core_types.hpp"
#include "nerve/persistence/core/core_types.hpp"
#include "nerve/types.hpp"

#include <cstddef>
#include <initializer_list>
#include <unordered_map>
#include <utility>
#include <vector>

namespace nerve::persistence
{

enum class PH4Algorithm
{
    EXACT_COMPUTE,
    BUDGETED_COMPUTE,
    SPARSE_MATRIX,
    WITNESS_SAMPLING
};

enum class ComputeMode
{
    EXACT,
    APPROXIMATE,
    BUDGETED
};

enum class WitnessStrategy
{
    RANDOM_SAMPLING,
    LANDMARK_SAMPLING,
    MAX_PERSISTENCE,
    COFACTOR_PERSISTENCE
};

template <typename Scalar = int>
class SparseBoundaryMatrix
{
public:
    SparseBoundaryMatrix(int rows, int cols)
        : rows_(rows)
        , cols_(cols)
    {}

    int numRows() const { return rows_; }
    int numCols() const { return cols_; }
    int numNonzero() const { return static_cast<int>(entries_.size()); }

    void addColumn(int col, std::initializer_list<std::pair<int, Scalar>> entries)
    {
        for (const auto &[row, val] : entries)
        {
            set(row, col, val);
        }
    }

    void addRow(int row, std::initializer_list<std::pair<int, Scalar>> entries)
    {
        for (const auto &[col, val] : entries)
        {
            set(row, col, val);
        }
    }

    void set(int row, int col, Scalar value)
    {
        if (value != Scalar{})
        {
            entries_[{row, col}] = value;
        }
        else
        {
            entries_.erase({row, col});
        }
    }

    Scalar get(int row, int col) const
    {
        auto it = entries_.find({row, col});
        return it != entries_.end() ? it->second : Scalar{};
    }

    bool isNonzero(int row, int col) const { return entries_.find({row, col}) != entries_.end(); }

private:
    int rows_ = 0;
    int cols_ = 0;

    struct PairHash
    {
        std::size_t operator()(const std::pair<int, int> &p) const noexcept
        {
            return static_cast<std::size_t>(p.first) * 7919 + static_cast<std::size_t>(p.second);
        }
    };

    std::unordered_map<std::pair<int, int>, Scalar, PairHash> entries_;
};

class StabilityCertificate
{
public:
    StabilityCertificate() = default;
    StabilityCertificate(double stability_constant, double numerical_residual, bool is_exact);

    [[nodiscard]] double stabilityConstant() const noexcept { return stability_constant_; }
    [[nodiscard]] double numericalResidual() const noexcept { return numerical_residual_; }
    [[nodiscard]] bool isExact() const noexcept { return is_exact_; }
    [[nodiscard]] bool isValid() const noexcept { return is_valid_; }
    void validateCertificate() const;

private:
    double stability_constant_ = 0.0;
    double numerical_residual_ = 0.0;
    bool is_exact_ = false;
    bool is_valid_ = true;
};

class CompactSummary
{
public:
    CompactSummary();

    void addPair(const Pair &pair);
    [[nodiscard]] std::vector<Pair> getTopPairs(std::size_t k) const;
    [[nodiscard]] double getCompressionRatio() const;
    [[nodiscard]] std::size_t getTotalPairs() const { return total_pairs_; }
    [[nodiscard]] std::size_t getMemorySavedBytes() const { return memory_saved_bytes_; }
    [[nodiscard]] double getTotalPersistence() const { return total_persistence_; }

private:
    void compressPairs();

    std::vector<Pair> top_pairs_;
    std::size_t total_pairs_;
    double total_persistence_;
    std::size_t memory_saved_bytes_;
};

class PH4
{
public:
    PH4();
    PH4(PH4Algorithm algorithm, ComputeMode mode, std::size_t max_memory_mb);

    Diagram compute(const std::vector<Simplex> &complex,
                    const ::nerve::core::DeterminismContract &contract);

    Diagram computePersistenceApproximate(const std::vector<Simplex> &complex,
                                          const ::nerve::core::DeterminismContract &contract);

    [[nodiscard]] std::pair<Diagram, StabilityCertificate>
    computePersistenceWithCertificate(const std::vector<Simplex> &complex,
                                      const ::nerve::core::DeterminismContract &contract);

    [[nodiscard]] std::pair<CompactSummary, StabilityCertificate>
    computePersistenceBudgeted(const std::vector<Simplex> &complex, std::size_t max_memory_mb,
                               const ::nerve::core::DeterminismContract &contract);

    [[nodiscard]] std::pair<std::vector<Index>, StabilityCertificate>
    sampleWitnessesWithCertificate(const std::vector<Simplex> &complex, std::size_t num_witnesses,
                                   WitnessStrategy strategy,
                                   const ::nerve::core::DeterminismContract &contract);

    [[nodiscard]] std::vector<Index>
    sampleWitnesses(const std::vector<Simplex> &complex, std::size_t num_witnesses,
                    WitnessStrategy strategy, const ::nerve::core::DeterminismContract &contract);

    void setAlgorithm(PH4Algorithm algorithm);
    void setComputeMode(ComputeMode mode);
    void setMaxMemory(std::size_t max_memory_mb);
    void setWitnessStrategy(WitnessStrategy strategy);

    [[nodiscard]] std::size_t getComputationTimeMs() const;
    [[nodiscard]] std::size_t getMemoryUsedMb() const;
    [[nodiscard]] bool wasBudgetExceeded() const;

private:
    PH4Algorithm algorithm_ = PH4Algorithm::EXACT_COMPUTE;
    ComputeMode compute_mode_ = ComputeMode::EXACT;
    std::size_t max_memory_mb_ = 256;
    WitnessStrategy witness_strategy_ = WitnessStrategy::RANDOM_SAMPLING;
    std::size_t computation_time_ms_ = 0;
    std::size_t memory_used_mb_ = 0;
    bool budget_exceeded_ = false;

    Diagram computeExactPersistence(const std::vector<Simplex> &complex,
                                    const ::nerve::core::DeterminismContract &contract);
    Diagram computeSparsePersistence(const std::vector<Simplex> &complex,
                                     const ::nerve::core::DeterminismContract &contract);
    Diagram computeApproximatePersistence(const std::vector<Simplex> &complex,
                                          const ::nerve::core::DeterminismContract &contract);

    std::vector<Index> sampleLandmarkWitnesses(const std::vector<Simplex> &complex,
                                               std::size_t num_witnesses,
                                               const ::nerve::core::DeterminismContract &contract);
    std::vector<Index>
    sampleMaxPersistenceWitnesses(const std::vector<Simplex> &complex, std::size_t num_witnesses,
                                  const ::nerve::core::DeterminismContract &contract);

    void checkMemoryBudget(std::size_t required_memory_mb);
    [[nodiscard]] StabilityCertificate computeStabilityCertificate(const Diagram &diagram);
    [[nodiscard]] CompactSummary computeCompactSummary(const Diagram &diagram,
                                                       std::size_t max_memory_mb);
};

} // namespace nerve::persistence
