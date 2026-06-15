
#pragma once

#include "nerve/core.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace nerve::persistence::perfect
{

/**
 * @brief Compact exact static map for pivot to column lookup.
 *
 * The implementation uses direct addressing for dense integer ranges and a
 * bucketed displacement table for sparse ranges. Every occupied slot stores
 * the original key, so absent lookups return a miss instead of a colliding
 * value.
 */

/**
 * @brief Collision-free static hash table for pivot to column mapping.
 */
class PerfectPivotMap
{
public:
    PerfectPivotMap();

    // Build from key-value pairs
    // O(n) expected time
    bool build(const std::vector<int> &keys, const std::vector<int> &values);

    // Lookup value for key.
    int lookup(int key) const;

    // Check if key exists (for non-minimal variant)
    bool contains(int key) const;

    // Get stats
    size_t memoryUsage() const;
    double bitsPerKey() const;

    // Serialization
    void save(const std::string &filename) const;
    bool load(const std::string &filename);

private:
    // Seeds for hash functions
    uint64_t seed1_, seed2_;

    // Pilot values for displacement
    std::vector<uint8_t> pilots_;

    // Stored values (columns), indexed either by direct offset or hash slot.
    std::vector<int> values_;

    // Original keys stored in each value slot for exact lookup verification.
    std::vector<int> slot_keys_;

    // Occupancy bits for slot_keys_/values_.
    std::vector<uint8_t> slot_occupied_;

    // Sorted original keys retained for persistence and diagnostics.
    std::vector<int> keys_;

    // Offset metadata for direct addressing mode.
    int min_key_;
    bool direct_mode_;

    // Number of keys
    size_t num_keys_;

    // Internal bucket structure
    struct Bucket
    {
        std::vector<size_t> keys;
    };

    // Hash functions
    uint64_t hash1(int key) const;
    uint64_t hash2(int key) const;
    uint64_t hashCombined(int key, uint8_t pilot) const;
};

/**
 * @brief Fast, compact hash for static sets
 *
 * Uses a simpler approach than full MPHF:
 * - Two-level hashing
 * - First level: bucket index
 * - Second level: offset within bucket
 */
class FastStaticMap
{
public:
    // Build from sorted keys
    bool build(const std::vector<int> &keys, const std::vector<int> &values);

    // Lookup
    std::optional<int> find(int key) const;

    // Stats
    size_t size() const { return keys_.size(); }
    size_t memoryUsage() const;

private:
    std::vector<int> keys_;
    std::vector<int> values_;
    std::vector<uint32_t> bucket_offsets_;

    size_t num_buckets_ = 0;
    size_t bucket_size_ = 0;
};

/**
 * @brief Benchmark perfect hashing vs alternatives
 */
struct PerfectHashBenchmark
{
    double std_unordered_map_time_ms = 0.0;
    double robin_hood_time_ms = 0.0;
    double perfect_hash_time_ms = 0.0;

    size_t std_memory_bytes = 0;
    size_t robin_hood_memory_bytes = 0;
    size_t perfect_hash_memory_bytes = 0;

    double speedup_vs_std = 1.0;
    double speedup_vs_robin_hood = 1.0;
    double memory_reduction_vs_std = 1.0;
    double memory_reduction_vs_robin_hood = 1.0;
};

PerfectHashBenchmark benchmarkPerfectHash(const std::vector<int> &keys,
                                          int lookup_iterations = 1000000);

/**
 * @brief Configuration
 */
struct PerfectHashConfig
{
    size_t num_buckets_factor = 2; // num_buckets = num_keys * factor
    uint8_t max_pilot_value = 255; // Max pilot value to try
    bool verbose = false;
};

/**
 * @brief Get optimal config
 */
PerfectHashConfig getOptimalPerfectHashConfig(size_t num_keys);

/**
 * @brief Estimate performance
 */
struct PerfectHashEstimate
{
    double query_speedup = 1.0;    // vs std::unordered_map
    double memory_reduction = 1.0; // vs std::unordered_map
    double build_time_ms = 0.0;    // Estimated build time
    bool recommended = false;      // Should use perfect hashing?
};

PerfectHashEstimate estimatePerfectHashBenefit(size_t num_keys, int num_lookups_expected);

} // namespace nerve::persistence::perfect
