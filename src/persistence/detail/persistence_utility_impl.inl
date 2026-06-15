#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "nerve/core_types.hpp"

namespace nerve::persistence::detail {

template <typename T>
std::vector<Pair> filterFinitePairs(const std::vector<Pair>& pairs) {
    std::vector<Pair> result;
    result.reserve(pairs.size());
    for (const auto& p : pairs) {
        if (!std::isinf(p.death)) {
            result.push_back(p);
        }
    }
    return result;
}

template <typename T>
std::vector<Pair> filterInfinitePairs(const std::vector<Pair>& pairs) {
    std::vector<Pair> result;
    result.reserve(pairs.size());
    for (const auto& p : pairs) {
        if (std::isinf(p.death)) {
            result.push_back(p);
        }
    }
    return result;
}

template <typename T>
std::vector<Pair> filterDimension(const std::vector<Pair>& pairs, Index dim) {
    std::vector<Pair> result;
    result.reserve(pairs.size());
    for (const auto& p : pairs) {
        if (p.dimension == dim) {
            result.push_back(p);
        }
    }
    return result;
}

template <typename T>
double computePairPersistence(const Pair& p) {
    if (std::isinf(p.death)) return std::numeric_limits<double>::infinity();
    return p.death - p.birth;
}

template <typename T>
std::vector<double> computePersistenceValues(const std::vector<Pair>& pairs) {
    std::vector<double> values;
    values.reserve(pairs.size());
    for (const auto& p : pairs) {
        values.push_back(computePairPersistence<T>(p));
    }
    return values;
}

template <typename T>
double computeTotalPersistence(const std::vector<Pair>& pairs) {
    double total = 0.0;
    for (const auto& p : pairs) {
        if (!std::isinf(p.death)) {
            total += (p.death - p.birth);
        }
    }
    return total;
}

template <typename T>
double computeMaxPersistence(const std::vector<Pair>& pairs) {
    double max_pers = 0.0;
    for (const auto& p : pairs) {
        if (!std::isinf(p.death)) {
            max_pers = std::max(max_pers, p.death - p.birth);
        }
    }
    return max_pers;
}

template <typename T>
Size countPairsInRange(const std::vector<Pair>& pairs,
                        double min_persistence, double max_persistence) {
    Size count = 0;
    for (const auto& p : pairs) {
        if (std::isinf(p.death)) continue;
        double pers = p.death - p.birth;
        if (pers >= min_persistence && pers <= max_persistence) {
            ++count;
        }
    }
    return count;
}

template <typename T>
std::vector<Pair> sortByPersistence(std::vector<Pair>& pairs, bool descending) {
    std::sort(pairs.begin(), pairs.end(),
              [descending](const Pair& a, const Pair& b) {
                  double pa = computePairPersistence<T>(a);
                  double pb = computePairPersistence<T>(b);
                  if (std::isinf(pa)) pa = std::numeric_limits<double>::max();
                  if (std::isinf(pb)) pb = std::numeric_limits<double>::max();
                  return descending ? pa > pb : pa < pb;
              });
    return pairs;
}

template <typename T>
std::vector<Pair> sortByBirthTime(std::vector<Pair>& pairs) {
    std::sort(pairs.begin(), pairs.end(),
              [](const Pair& a, const Pair& b) {
                  return a.birth < b.birth;
              });
    return pairs;
}

template <typename T>
double computeBirthTimeRange(const std::vector<Pair>& pairs) {
    if (pairs.empty()) return 0.0;
    double min_birth = std::numeric_limits<double>::infinity();
    double max_birth = -std::numeric_limits<double>::infinity();
    for (const auto& p : pairs) {
        min_birth = std::min(min_birth, p.birth);
        max_birth = std::max(max_birth, p.birth);
    }
    return max_birth - min_birth;
}

}  // namespace nerve::persistence::detail
