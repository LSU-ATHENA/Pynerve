
#pragma once

// Internal structures (Container, RoaringBitmapImpl) - include first
#include "nerve/core.hpp"
#include "roaring_bitmap_internal.hpp"

#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

namespace nerve::persistence::roaring
{

/**
 * @brief Compressed sparse bitmap using roaring format
 *
 * Roaring bitmaps compress sparse bit-vectors while maintaining fast operations.
 * They use a hybrid approach: array containers for sparse chunks, bitset containers
 * for dense chunks, and run containers for sequential data.
 *
 * For persistent homology with very sparse columns (H5-H6), this provides:
 * - Lower memory footprint than dense bit-vectors on sparse workloads
 * - Better locality from compressed containers
 * - Fast AND, OR, XOR operations
 */

// Forward declaration
struct RoaringBitmapImpl;

/**
 * @brief Roaring bitmap column for sparse PH
 */
class RoaringColumn
{
public:
    RoaringColumn();
    ~RoaringColumn();

    // Disable copy, enable move
    RoaringColumn(const RoaringColumn &) = delete;
    RoaringColumn &operator=(const RoaringColumn &) = delete;
    RoaringColumn(RoaringColumn &&) noexcept;
    RoaringColumn &operator=(RoaringColumn &&) noexcept;

    // Construction from sparse indices
    static RoaringColumn fromSparseIndices(const std::vector<int> &indices, int max_row);
    static RoaringColumn fromBitVector(const std::vector<uint64_t> &words);

    // Core operations (Z2 arithmetic)
    void xorInPlace(const RoaringColumn &other);
    RoaringColumn xorCopy(const RoaringColumn &other) const;

    // Set operations
    void add(int index);
    void remove(int index);
    bool contains(int index) const;

    // Pivot (highest set bit)
    int computePivot() const;
    int getPivot() const { return pivot_; }

    // Sparsity info
    size_t cardinality() const; // Number of set bits
    double sparsity() const;    // 1.0 = empty, 0.0 = full
    size_t memoryUsage() const; // Bytes used

    // Conversion
    std::vector<int> toSparseIndices() const;
    std::vector<uint64_t> toBitVector(int num_words) const;

    // Iteration
    void forEach(std::function<void(int)> callback) const;

    // Comparison
    bool operator==(const RoaringColumn &other) const;
    bool operator!=(const RoaringColumn &other) const;

    // Check if empty
    bool isEmpty() const;
    void clear();

private:
    std::unique_ptr<RoaringBitmapImpl> impl_;
    mutable int pivot_ = -1;
    mutable bool pivot_valid_ = false;

    void invalidatePivot();
};

/**
 * @brief Threshold for using roaring vs bit-vector
 */
inline bool shouldUseRoaring(size_t cardinality, size_t max_rows)
{
    // Use roaring if sparsity > 90%
    if (max_rows == 0)
        return false;
    double sparsity = 1.0 - (static_cast<double>(cardinality) / static_cast<double>(max_rows));
    return sparsity > 0.9;
}

inline bool shouldUseRoaring(double sparsity)
{
    if (!std::isfinite(sparsity) || sparsity < 0.0 || sparsity > 1.0)
    {
        throw std::invalid_argument("Roaring sparsity must be finite and in [0, 1]");
    }
    return sparsity > 0.9;
}

/**
 * @brief Hybrid column that switches between dense and sparse
 */
class HybridColumn
{
public:
    enum class Type
    {
        DENSE,
        SPARSE
    };

    HybridColumn(int max_rows);

    // Add index
    void add(int index);

    // XOR operation
    void xorInPlace(const HybridColumn &other);

    // Pivot
    int computePivot() const;

    // Auto-convert based on density
    void optimizeStorage();

    Type getType() const { return type_; }

    // Check if column is empty
    bool isEmpty() const;

    // Get sparsity ratio (0.0 = dense, 1.0 = completely sparse)
    double sparsity() const;

private:
    int max_rows_;
    Type type_ = Type::SPARSE;

    std::vector<uint64_t> dense_data_;
    RoaringColumn sparse_data_;

    static constexpr double SPARSE_THRESHOLD = 0.1; // 10% density
};

/**
 * @brief Roaring-optimized matrix reduction result
 */
struct RoaringReductionResult
{
    std::vector<std::pair<int, int>> pairs; // birth, death
    size_t peak_memory_bytes;
    double reduction_time_ms;
    size_t num_columns_processed;
};

/**
 * @brief Reduce matrix using roaring bitmaps
 *
 * Uses roaring columns for sparse sections, bit-vectors for dense.
 */
RoaringReductionResult reduceMatrixRoaring(std::vector<HybridColumn> &columns,
                                           const std::vector<double> &filtration_values);

/**
 * @brief Benchmark roaring vs bit-vector
 */
struct RoaringBenchmark
{
    double bitvector_time_ms;
    double roaring_time_ms;
    size_t bitvector_memory_bytes;
    size_t roaring_memory_bytes;
    double speedup;
    double memory_reduction;
};

RoaringBenchmark benchmarkRoaring(size_t num_columns, size_t sparsity_percent);

/**
 * @brief Configuration for roaring optimization
 */
struct RoaringConfig
{
    double sparsity_threshold = 0.9; // Use roaring above this
    bool auto_convert = true;        // Auto dense<->sparse
    bool parallel_build = true;      // Parallel bitmap construction
};

/**
 * @brief Get optimal roaring config
 */
RoaringConfig getOptimalRoaringConfig(size_t num_columns, double avg_sparsity);

/**
 * @brief Estimate roaring benefit
 */
struct RoaringEstimate
{
    double memory_reduction; // Factor (10 = 10x less memory)
    double speedup;          // Performance gain
    bool recommended;        // Should use roaring?
};

RoaringEstimate estimateRoaringBenefit(size_t num_rows, size_t expected_cardinality);

} // namespace nerve::persistence::roaring
