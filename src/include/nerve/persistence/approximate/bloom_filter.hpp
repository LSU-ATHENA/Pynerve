
#pragma once

#include "nerve/core.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace nerve::persistence::bloom
{

/**
 * @brief Bloom Filter for Probabilistic Pivot Lookup
 *
 * **O(1) PIVOT EXISTENCE CHECK WITH 1% FALSE POSITIVE RATE**
 *
 * Use bloom filter before hash map lookup:
 * - If bloom says "definitely not": O(1) negative
 * - If bloom says "maybe": do full hash lookup
 *
 * Speedup: 1.2-1.5x for large matrices (most pivots don't exist)
 * Memory: ~1 bit per element
 */

class BloomFilter
{
public:
    /**
     * @brief Create bloom filter for expected items with target FPP
     *
     * @param expected_items Expected number of items to insert
     * @param target_fpp Target false positive probability (default 1%)
     */
    BloomFilter(size_t expected_items, double target_fpp = 0.01);

    // Insert a key
    void add(int key);

    // Check if key might be present
    // Returns: true = "maybe" (could be false positive)
    //          false = "definitely not" (100% accurate)
    bool mightContain(int key) const;

    // Definitely not contains (100% accurate)
    bool definitelyNotContains(int key) const { return !mightContain(key); }

    // Get stats
    size_t memoryUsage() const { return bits_.size() * sizeof(uint64_t); }
    double currentFPP() const; // Estimated current false positive rate
    int numHashFunctions() const { return k_; }

    // Clear
    void clear();

private:
    std::vector<uint64_t> bits_;
    int k_;        // Number of hash functions
    size_t m_;     // Number of bits
    size_t n_ = 0; // Number of inserted items

    // Double hashing for multiple hash functions
    std::pair<size_t, size_t> hash(int key) const;
    size_t nthHash(size_t h1, size_t h2, int n) const;

    void setBit(size_t index);
    bool getBit(size_t index) const;
};

// Bloom-backed pivot lookup
class BloomPivotLookup
{
public:
    BloomPivotLookup(size_t expected_pivots);

    // Register a pivot
    void registerPivot(int pivot, int column_idx);

    // Lookup with bloom optimization
    // Returns: column_idx if found, -1 if not found
    int find(int pivot) const;

    // Quick check: is pivot definitely not present?
    bool definitelyNotPresent(int pivot) const;

private:
    BloomFilter bloom_;
    // Full hash map for verification (only checked if bloom says "maybe")
    std::unordered_map<int, int> full_map_;
};

// Benchmark
struct BloomBenchmark
{
    double full_lookup_time_ms = 0.0;
    double bloom_lookup_time_ms = 0.0;
    double speedup = 1.0;
    size_t false_positives = 0;
    double actual_fpp = 0.0;
};

BloomBenchmark benchmarkBloom(size_t num_pivots, size_t num_lookups);

} // namespace nerve::persistence::bloom
