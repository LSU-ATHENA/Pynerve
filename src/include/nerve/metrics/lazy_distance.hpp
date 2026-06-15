#pragma once

/// @file lazy_distance.hpp

#include <chrono>
#include <mutex>
#include <shared_mutex>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace nerve::metrics::lazy
{

// Cache constants
constinit const size_t DEFAULT_CACHE_SIZE = 1000000;
constinit const double CACHE_EVICTION_THRESHOLD = 0.9;
constinit const size_t PAIR_HASH_PRIME = 1000003ULL;

// LazyDistanceMatrix

class LazyDistanceMatrix
{
public:
    LazyDistanceMatrix(std::span<const double> points, size_t n_points, size_t point_dim,
                       const std::string &metric = "euclidean",
                       size_t max_cache_size = DEFAULT_CACHE_SIZE);

    [[nodiscard]] double getDistance(size_t i, size_t j);
    [[nodiscard]] double getDistanceNoCache(size_t i, size_t j) const;
    [[nodiscard]] bool isWithinRadius(size_t i, size_t j, double radius);
    [[nodiscard]] double getMaxDistanceInSubset(std::span<const size_t> indices);
    [[nodiscard]] std::vector<std::pair<size_t, double>>
    getKNearestNeighbors(size_t point_idx, size_t k,
                         double max_radius = std::numeric_limits<double>::infinity());

    // Cache management
    [[nodiscard]] size_t getCacheSize() const;
    [[nodiscard]] size_t getMaxCacheSize() const;
    void clearCache();
    [[nodiscard]] double getCacheHitRate() const;
    [[nodiscard]] size_t memoryBytes() const;
    [[nodiscard]] double memoryReduction() const;

private:
    std::span<const double> points_;
    size_t n_points_;
    size_t point_dim_;
    std::string metric_;

    struct PairHash
    {
        size_t operator()(const std::pair<size_t, size_t> &p) const
        {
            return p.first * PAIR_HASH_PRIME + p.second;
        }
    };

    std::unordered_map<std::pair<size_t, size_t>, double, PairHash> cache_;
    std::unordered_map<std::pair<size_t, size_t>, std::chrono::steady_clock::time_point, PairHash>
        access_times_;
    mutable std::shared_mutex cache_mutex_;
    size_t max_cache_size_;
    mutable size_t cache_hits_ = 0;
    mutable size_t total_lookups_ = 0;

    [[nodiscard]] double computeDistance(size_t i, size_t j) const;
    void evictOldest();
    void recordAccess(const std::pair<size_t, size_t> &key);
};

// SparseDistanceMatrix

class SparseDistanceMatrix
{
public:
    SparseDistanceMatrix(std::span<const double> points, size_t n_points, size_t point_dim,
                         double threshold, const std::string &metric = "euclidean");

    [[nodiscard]] double getDistance(size_t i, size_t j) const;
    [[nodiscard]] bool isEdge(size_t i, size_t j) const;
    [[nodiscard]] size_t memoryBytes() const;
    [[nodiscard]] double getSparsity() const;

private:
    size_t n_points_;

    struct PairHash
    {
        size_t operator()(const std::pair<size_t, size_t> &p) const
        {
            return p.first * 1000003 + p.second;
        }
    };

    std::unordered_map<std::pair<size_t, size_t>, double, PairHash> edges_;

    [[nodiscard]] static double computeDistance(std::span<const double> points, size_t i, size_t j,
                                                size_t point_dim, const std::string &metric);
};

// High-level API

[[nodiscard]] std::vector<std::vector<int>> buildVRLazy(std::span<const double> points,
                                                        size_t n_points, size_t point_dim,
                                                        double max_distance, int max_dim);

void expandCliquesRecursive(const std::vector<std::vector<int>> &adjacency,
                            const std::vector<int> &current_clique,
                            const std::vector<int> &candidates, int current_dim, int max_dim,
                            std::vector<std::vector<int>> &simplices, size_t max_simplices);

} // namespace nerve::metrics::lazy
