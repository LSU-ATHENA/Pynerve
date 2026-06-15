
#pragma once
#include "nerve/config.hpp"
#include "nerve/core/rng/determinism_contract.hpp"
#include "nerve/errors/errors.hpp"
#include "nerve/persistence/core/core_types.hpp"
#include "nerve/persistence/streaming/streaming_reducer.hpp"

#include <algorithm>
#include <chrono>
#include <memory>
#include <random>
#include <unordered_map>
#include <vector>
namespace nerve::persistence
{
enum class PH4Algorithm
{
    SPARSE_MATRIX = 0,
    WITNESS_SAMPLING,
    BUDGETED_COMPUTE,
    EXACT_COMPUTE
};
enum class WitnessStrategy
{
    RANDOM_SAMPLING,
    LANDMARK_SAMPLING,
    MAX_PERSISTENCE,
    COFACTOR_PERSISTENCE
};
enum class ComputeMode
{
    EXACT,
    APPROXIMATE,
    BUDGETED
};
class StabilityCertificate
{
public:
    StabilityCertificate(double stability_constant = 0.0, double numerical_residual = 0.0,
                         bool isExact = false);
    double getStabilityConstant() const { return stability_constant_; }
    double getNumericalResidual() const { return numerical_residual_; }
    bool isExact() const { return is_exact_; }
    bool isValid() const { return is_valid_; }
    void validateCertificate() const;

private:
    double stability_constant_;
    double numerical_residual_;
    bool is_exact_;
    bool is_valid_;
};
class CompactSummary
{
public:
    CompactSummary();
    void addPair(const Pair &pair);
    std::vector<Pair> getTopPairs(size_t k) const;
    size_t getTotalPairs() const { return total_pairs_; }
    double getTotalPersistence() const { return total_persistence_; }
    size_t getMemorySavedBytes() const { return memory_saved_bytes_; }
    double getCompressionRatio() const;

private:
    std::vector<Pair> top_pairs_;
    size_t total_pairs_;
    double total_persistence_;
    size_t memory_saved_bytes_;
    void compressPairs();
};
template <typename Index = ::nerve::Index>
class SparseBoundaryMatrix
{
public:
    using value_type = Index;
    using size_type = std::size_t;
    SparseBoundaryMatrix() = default;
    explicit SparseBoundaryMatrix(std::size_t numRows, std::size_t numCols);
    void addColumn(Index col, const std::vector<std::pair<Index, value_type>> &entries);
    void addRow(Index row, const std::vector<std::pair<Index, value_type>> &entries);
    void set(Index row, Index col, value_type value);
    value_type get(Index row, Index col) const;
    bool isNonzero(Index row, Index col) const;
    size_type numRows() const { return num_rows_; }
    size_type numCols() const { return num_cols_; }
    size_type numNonzero() const;
    auto beginRow(Index row) const;
    auto endRow(Index row) const;

private:
    std::vector<std::unordered_map<Index, value_type>> data_;
    std::size_t num_rows_;
    std::size_t num_cols_;
    auto findColumn(Index col) const -> const std::unordered_map<Index, value_type> &;
};
class PH4
{
public:
    PH4();
    explicit PH4(PH4Algorithm algorithm = PH4Algorithm::SPARSE_MATRIX,
                 ComputeMode mode = ComputeMode::EXACT, std::size_t max_memory_mb = 1024);
    std::pair<Diagram, StabilityCertificate>
    computePersistenceWithCertificate(const std::vector<Simplex> &complex,
                                      const ::nerve::core::DeterminismContract &contract = {});
    std::pair<CompactSummary, StabilityCertificate>
    computePersistenceBudgeted(const std::vector<Simplex> &complex,
                               std::size_t max_memory_mb = 1024,
                               const ::nerve::core::DeterminismContract &contract = {});
    Diagram compute(const std::vector<Simplex> &complex,
                    const ::nerve::core::DeterminismContract &contract = {});
    Diagram computePersistenceApproximate(const std::vector<Simplex> &complex,
                                          const ::nerve::core::DeterminismContract &contract = {});
    std::pair<std::vector<Index>, StabilityCertificate>
    sampleWitnessesWithCertificate(const std::vector<Simplex> &complex, std::size_t num_witnesses,
                                   WitnessStrategy strategy = WitnessStrategy::RANDOM_SAMPLING,
                                   const ::nerve::core::DeterminismContract &contract = {});
    std::vector<Index> sampleWitnesses(const std::vector<Simplex> &complex,
                                       std::size_t num_witnesses,
                                       WitnessStrategy strategy = WitnessStrategy::RANDOM_SAMPLING,
                                       const ::nerve::core::DeterminismContract &contract = {});
    void setAlgorithm(PH4Algorithm algorithm);
    void setComputeMode(ComputeMode mode);
    void setMaxMemory(std::size_t max_memory_mb);
    void setWitnessStrategy(WitnessStrategy strategy);
    std::size_t getComputationTimeMs() const;
    std::size_t getMemoryUsedMb() const;
    bool wasBudgetExceeded() const;

private:
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
    StabilityCertificate computeStabilityCertificate(const Diagram &diagram);
    CompactSummary computeCompactSummary(const Diagram &diagram, std::size_t max_memory_mb);
    PH4Algorithm algorithm_ = PH4Algorithm::SPARSE_MATRIX;
    ComputeMode compute_mode_ = ComputeMode::EXACT;
    std::size_t max_memory_mb_ = 1024;
    WitnessStrategy witness_strategy_ = WitnessStrategy::RANDOM_SAMPLING;
    std::size_t computation_time_ms_ = 0;
    std::size_t memory_used_mb_ = 0;
    bool budget_exceeded_ = false;
};
} // namespace nerve::persistence
