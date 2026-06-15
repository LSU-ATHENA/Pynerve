// Bloom Filter Implementation for Persistent Homology
// Probabilistic O(1) pivot existence checking

#include "nerve/persistence/approximate/bloom_filter.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>

namespace nerve::persistence::bloom
{

// Bloom filter constants
constexpr size_t BITS_PER_WORD = 64;
constexpr size_t WORD_SIZE_MASK = BITS_PER_WORD - 1; // 63
constexpr int MIN_HASH_FUNCTIONS = 1;
constexpr int MAX_HASH_FUNCTIONS = 30;
constexpr int BYTES_PER_INT = 4;
constexpr int MURMUR_SHIFT_CONSTANT = 33;

// BloomPivotLookup constants
constexpr double PIVOT_FPP_TARGET = 0.01;

double finiteBenchmarkSpeedup(double baseline_ms, double accelerated_ms)
{
    if (!std::isfinite(baseline_ms) || baseline_ms < 0.0 || !std::isfinite(accelerated_ms) ||
        accelerated_ms <= 0.0)
    {
        return 1.0;
    }
    const double speedup = baseline_ms / accelerated_ms;
    return std::isfinite(speedup) && speedup >= 0.0 ? speedup : 1.0;
}

double sanitizeTargetFpp(double target_fpp)
{
    if (!std::isfinite(target_fpp))
    {
        throw std::invalid_argument("target_fpp must be finite");
    }
    return std::clamp(target_fpp, std::numeric_limits<double>::epsilon(), 0.5);
}

size_t checkedBloomBitCount(size_t expected_items, double target_fpp)
{
    const long double n = static_cast<long double>(std::max<size_t>(expected_items, 1));
    const long double p = static_cast<long double>(target_fpp);
    const long double log2 = std::log(2.0L);
    const long double raw_bits = -n * std::log(p) / (log2 * log2);
    if (!std::isfinite(raw_bits) || raw_bits <= 0.0L ||
        raw_bits > static_cast<long double>(std::numeric_limits<size_t>::max() - WORD_SIZE_MASK))
    {
        throw std::length_error("Bloom filter bit count exceeds size_t limits");
    }
    size_t bits = static_cast<size_t>(std::ceil(raw_bits));
    bits = std::max(bits, size_t(BITS_PER_WORD));
    if (bits > std::numeric_limits<size_t>::max() - WORD_SIZE_MASK)
    {
        throw std::length_error("Bloom filter bit count exceeds size_t limits");
    }
    return (bits + WORD_SIZE_MASK) & ~WORD_SIZE_MASK;
}

BloomFilter::BloomFilter(size_t expected_items, double target_fpp)
{
    expected_items = std::max<size_t>(expected_items, 1);
    target_fpp = sanitizeTargetFpp(target_fpp);

    const double n = static_cast<double>(expected_items);
    m_ = checkedBloomBitCount(expected_items, target_fpp);

    k_ = static_cast<int>(std::round((static_cast<double>(m_) / n) * std::log(2)));
    k_ = std::clamp(k_, MIN_HASH_FUNCTIONS, MAX_HASH_FUNCTIONS); // Clamp hash functions

    bits_.resize(m_ / BITS_PER_WORD, 0);
}

void BloomFilter::add(int key)
{
    auto [h1, h2] = hash(key);

    for (int i = 0; i < k_; ++i)
    {
        size_t idx = nthHash(h1, h2, i) % m_;
        setBit(idx);
    }

    ++n_;
}

bool BloomFilter::mightContain(int key) const
{
    auto [h1, h2] = hash(key);

    for (int i = 0; i < k_; ++i)
    {
        size_t idx = nthHash(h1, h2, i) % m_;
        if (!getBit(idx))
        {
            return false;
        }
    }

    return true; // Maybe present (could be false positive)
}

double BloomFilter::currentFPP() const
{
    // (1 - e^(-kn/m))^k
    if (m_ == 0 || n_ == 0)
        return 0.0;

    double exponent =
        -(static_cast<double>(k_) * static_cast<double>(n_)) / static_cast<double>(m_);
    double base = 1.0 - std::exp(exponent);
    return std::pow(base, k_);
}

void BloomFilter::clear()
{
    std::fill(bits_.begin(), bits_.end(), 0);
    n_ = 0;
}

std::pair<size_t, size_t> BloomFilter::hash(int key) const
{
    // FNV-1a inspired hash for h1
    const uint32_t unsigned_key = static_cast<uint32_t>(key);
    size_t h1 = 14695981039346656037ULL;
    for (int i = 0; i < BYTES_PER_INT; ++i)
    {
        h1 ^= static_cast<uint8_t>(unsigned_key >> (i * 8));
        h1 *= 1099511628211ULL;
    }

    // Murmur-inspired hash for h2
    size_t h2 = unsigned_key;
    h2 ^= h2 >> MURMUR_SHIFT_CONSTANT;
    h2 *= 0xff51afd7ed558ccdULL;
    h2 ^= h2 >> MURMUR_SHIFT_CONSTANT;
    h2 *= 0xc4ceb9fe1a85ec53ULL;
    h2 ^= h2 >> MURMUR_SHIFT_CONSTANT;

    return {h1, h2};
}

size_t BloomFilter::nthHash(size_t h1, size_t h2, int n) const
{
    // Double hashing: h_i = (h1 + i * h2) % m
    return h1 + n * h2;
}

void BloomFilter::setBit(size_t index)
{
    size_t word_idx = index / BITS_PER_WORD;
    size_t bit_idx = index % BITS_PER_WORD;
    bits_[word_idx] |= (1ULL << bit_idx);
}

bool BloomFilter::getBit(size_t index) const
{
    size_t word_idx = index / BITS_PER_WORD;
    size_t bit_idx = index % BITS_PER_WORD;
    return (bits_[word_idx] >> bit_idx) & 1ULL;
}

// BloomPivotLookup implementation
BloomPivotLookup::BloomPivotLookup(size_t expected_pivots)
    : bloom_(expected_pivots > std::numeric_limits<size_t>::max() / 2
                 ? std::numeric_limits<size_t>::max()
                 : expected_pivots * 2,
             PIVOT_FPP_TARGET)
{ // 2x for safety, 1% FPP
}

void BloomPivotLookup::registerPivot(int pivot, int column_idx)
{
    bloom_.add(pivot);
    full_map_[pivot] = column_idx;
}

int BloomPivotLookup::find(int pivot) const
{
    // Fast negative check
    if (bloom_.definitelyNotContains(pivot))
    {
        return -1;
    }

    // Might be present - check full map
    auto it = full_map_.find(pivot);
    if (it != full_map_.end())
    {
        return it->second;
    }

    // False positive
    return -1;
}

bool BloomPivotLookup::definitelyNotPresent(int pivot) const
{
    return bloom_.definitelyNotContains(pivot);
}

// Benchmark
BloomBenchmark benchmarkBloom(size_t num_pivots, size_t num_lookups)
{
    BloomBenchmark bench{};
    if (num_pivots == 0 || num_lookups == 0)
    {
        return bench;
    }

    std::vector<int> pivots;
    pivots.reserve(num_pivots);
    for (size_t i = 0; i < num_pivots; ++i)
    {
        pivots.push_back(static_cast<int>(i * 2));
    }

    std::vector<int> lookups;
    lookups.reserve(num_lookups);
    for (size_t i = 0; i < num_lookups; ++i)
    {
        const bool present = (i % 4) == 0;
        lookups.push_back(present ? pivots[i % pivots.size()]
                                  : static_cast<int>(num_pivots * 2 + i * 2 + 1));
    }

    std::unordered_map<int, int> std_map;
    std_map.reserve(num_pivots);
    for (int p : pivots)
    {
        std_map[p] = 1;
    }

    auto start_std = std::chrono::high_resolution_clock::now();
    size_t std_found = 0;
    for (int key : lookups)
    {
        if (std_map.find(key) != std_map.end())
        {
            ++std_found;
        }
    }
    auto end_std = std::chrono::high_resolution_clock::now();
    bench.full_lookup_time_ms =
        std::chrono::duration<double, std::milli>(end_std - start_std).count();

    BloomFilter bloom(num_pivots, 0.01);
    for (int p : pivots)
    {
        bloom.add(p);
    }

    auto start_bloom = std::chrono::high_resolution_clock::now();
    size_t bloom_found = 0;
    size_t false_positives = 0;
    for (int key : lookups)
    {
        if (bloom.mightContain(key))
        {
            // Verify
            if (std_map.find(key) != std_map.end())
            {
                ++bloom_found;
            }
            else
            {
                ++false_positives;
            }
        }
    }
    auto end_bloom = std::chrono::high_resolution_clock::now();
    bench.bloom_lookup_time_ms =
        std::chrono::duration<double, std::milli>(end_bloom - start_bloom).count();

    if (bloom_found != std_found)
    {
        throw std::runtime_error("Bloom benchmark produced false negatives");
    }

    bench.speedup = finiteBenchmarkSpeedup(bench.full_lookup_time_ms, bench.bloom_lookup_time_ms);
    bench.false_positives = false_positives;
    const size_t absent_lookups = num_lookups - std_found;
    bench.actual_fpp = absent_lookups > 0 ? static_cast<double>(false_positives) /
                                                static_cast<double>(absent_lookups)
                                          : 0.0;

    return bench;
}

} // namespace nerve::persistence::bloom
