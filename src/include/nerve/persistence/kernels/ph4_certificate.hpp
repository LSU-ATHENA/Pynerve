#pragma once

#include "nerve/persistence/core/core_types.hpp"

#include <cstddef>
#include <vector>

namespace nerve::persistence
{

/**
 * @brief Lightweight stability certificate for PH4/PH5/PH6 computations.
 *
 * Tracks stability constant, numerical residual, and exact flag
 * to certify that persistence results meet numerical guarantees.
 */
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

/**
 * @brief Compact summary of persistence pairs.
 *
 * Stores top persistence pairs by lifetime for memory-efficient
 * representation of large persistence diagrams.
 */
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

} // namespace nerve::persistence
