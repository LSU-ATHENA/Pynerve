
#pragma once

#include "nerve/core.hpp"

#include <algorithm>
#include <memory>
#include <queue>
#include <utility>
#include <vector>

namespace nerve::dmt
{

namespace parallel
{

class SimplexPairOps
{
public:
    static bool canFormGradientPair(const std::vector<int> &simplex_a,
                                    const std::vector<int> &simplex_b);

private:
    static bool isSubsetSIMD(const std::vector<int> &subset, const std::vector<int> &superset);
};
using SIMDSimplexOps = SimplexPairOps;

class ParallelMorsePairFinder
{
public:
    struct Config
    {
        int num_threads = 0;
        int batch_size = 1024;
        bool use_simd = true;
    };

    explicit ParallelMorsePairFinder(const Config &config);
    ~ParallelMorsePairFinder();

    std::vector<std::pair<int, int>> findMorsePairs(const std::vector<std::vector<int>> &simplices,
                                                    const std::vector<float> &filtration_values);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

class CacheOptimizedMorseTraversal
{
public:
    template <typename Callback>
    void traversePaths(const std::vector<std::pair<int, int>> &gradient_pairs,
                       const std::vector<std::vector<int>> &simplex_boundaries, int block_size,
                       Callback &&callback)
    {
        const int n = static_cast<int>(gradient_pairs.size());
        const int step = std::max(1, block_size);
        for (int block_start = 0; block_start < n; block_start += step)
        {
            const int block_end = std::min(block_start + step, n);
            for (int i = block_start; i < block_end; ++i)
            {
                traverseFromSimplex(gradient_pairs[i].first, simplex_boundaries, callback);
            }
        }
    }

private:
    template <typename Callback>
    void traverseFromSimplex(int start, const std::vector<std::vector<int>> &boundaries,
                             Callback &&callback)
    {
        if (start < 0 || static_cast<std::size_t>(start) >= boundaries.size())
        {
            return;
        }
        std::queue<int> queue;
        std::vector<char> visited(boundaries.size(), 0);
        queue.push(start);
        visited[static_cast<std::size_t>(start)] = 1;
        while (!queue.empty())
        {
            const int current = queue.front();
            queue.pop();
            callback(current);
            for (int neighbor : boundaries[static_cast<std::size_t>(current)])
            {
                if (neighbor < 0 || static_cast<std::size_t>(neighbor) >= boundaries.size())
                {
                    continue;
                }
                if (!visited[static_cast<std::size_t>(neighbor)])
                {
                    visited[static_cast<std::size_t>(neighbor)] = 1;
                    queue.push(neighbor);
                }
            }
        }
    }
};

class MorseMemoryPool
{
public:
    explicit MorseMemoryPool(size_t initial_capacity = 10000);
    ~MorseMemoryPool();

    struct MorseCell
    {
        int critical_index;
        std::vector<int> boundary;
        float filtration;
    };

    MorseCell *allocateCell();
    void reset();
    size_t size() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

class StreamingMorseBuilder
{
public:
    struct Config
    {
        int chunk_size = 1000;
        int max_dimension = 2;
        float max_filtration = 1.0f;
    };

    explicit StreamingMorseBuilder(const Config &config);

    template <typename SimplexSource, typename Callback>
    void buildStreaming(SimplexSource &source, Callback &&callback)
    {
        std::vector<std::vector<int>> chunk;
        std::vector<float> filtration_values;
        chunk.reserve(static_cast<std::size_t>(std::max(1, config_.chunk_size)));
        filtration_values.reserve(chunk.capacity());
        while (source.hasNext())
        {
            auto [vertices, filtration] = source.next();
            chunk.push_back(std::move(vertices));
            filtration_values.push_back(filtration);
            if (static_cast<int>(chunk.size()) >= config_.chunk_size)
            {
                processChunk(chunk, filtration_values, callback);
                chunk.clear();
                filtration_values.clear();
            }
        }
        if (!chunk.empty())
        {
            processChunk(chunk, filtration_values, callback);
        }
    }

private:
    Config config_;

    template <typename Callback>
    void processChunk(const std::vector<std::vector<int>> &chunk,
                      const std::vector<float> &filtration_values, Callback &&callback)
    {
        ParallelMorsePairFinder::Config finder_config;
        finder_config.batch_size = std::max(1, config_.chunk_size);
        ParallelMorsePairFinder finder(finder_config);
        callback(chunk, finder.findMorsePairs(chunk, filtration_values));
    }
};

struct ParallelDMTBenchmark
{
    double sequential_time_ms;
    double parallel_time_ms;
    double simd_time_ms;
    double speedup_parallel;
    double speedup_simd;
    int num_simplices;
    int num_pairs;
};

ParallelDMTBenchmark benchmarkParallelDMT(int num_simplices);

} // namespace parallel

struct DMTConfig
{
    int max_dimension = 2;
    bool use_parallel = true;
    bool use_simd = true;
    int num_threads = 0;
    bool use_priority_queue = true;
};

struct MorseResult
{
    std::vector<int> critical_simplices;
    std::vector<std::pair<int, int>> gradient_pairs;
    double computation_time_ms = 0.0;
};

class DMTEngine
{
public:
    explicit DMTEngine(const DMTConfig &config = {});
    ~DMTEngine();

    MorseResult computeMorseComplex(const std::vector<std::vector<int>> &simplices,
                                    const std::vector<float> &filtration);

    std::vector<int> findCriticalPoints() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace nerve::dmt
