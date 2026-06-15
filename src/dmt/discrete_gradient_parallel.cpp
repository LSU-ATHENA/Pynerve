#include "nerve/dmt/gpu_dmt.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <deque>
#include <limits>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace nerve::dmt::parallel
{
namespace
{

bool hasCofacetCardinality(const std::vector<int> &a, const std::vector<int> &b)
{
    return a.size() + 1 == b.size() || b.size() + 1 == a.size();
}

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

bool containsAsSortedSubset(std::vector<int> subset, std::vector<int> superset)
{
    if (subset.size() > superset.size())
    {
        return false;
    }
    std::ranges::sort(subset);
    std::ranges::sort(superset);
    return std::ranges::includes(superset, subset);
}

bool canFormPairScalar(const std::vector<int> &a, const std::vector<int> &b)
{
    if (!hasCofacetCardinality(a, b))
    {
        return false;
    }
    return a.size() < b.size() ? containsAsSortedSubset(a, b) : containsAsSortedSubset(b, a);
}

struct CandidatePair
{
    int lower = -1;
    int upper = -1;
    float filtration = std::numeric_limits<float>::infinity();
};

std::vector<std::vector<int>> makeBenchmarkSimplices(int num_simplices)
{
    const int safe_count = std::max(0, num_simplices);
    const int vertex_count = std::max(2, safe_count / 2 + 1);
    std::vector<std::vector<int>> simplices;
    simplices.reserve(static_cast<std::size_t>(safe_count));
    for (int i = 0; i < safe_count; ++i)
    {
        const int vertex = i / 2;
        if ((i & 1) == 0)
        {
            simplices.push_back({vertex});
        }
        else
        {
            simplices.push_back({vertex, (vertex + 1) % vertex_count});
        }
    }
    return simplices;
}

std::vector<float> makeBenchmarkFiltration(int num_simplices)
{
    std::vector<float> filtration(static_cast<std::size_t>(std::max(0, num_simplices)));
    for (int i = 0; i < num_simplices; ++i)
    {
        filtration[static_cast<std::size_t>(i)] = static_cast<float>(i) * 0.01f;
    }
    return filtration;
}

double elapsedMs(std::chrono::steady_clock::time_point start,
                 std::chrono::steady_clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

} // namespace

bool SimplexPairOps::canFormGradientPair(const std::vector<int> &simplex_a,
                                         const std::vector<int> &simplex_b)
{
    return canFormPairScalar(simplex_a, simplex_b);
}

bool SimplexPairOps::isSubsetSIMD(const std::vector<int> &subset, const std::vector<int> &superset)
{
    return containsAsSortedSubset(subset, superset);
}

class ParallelMorsePairFinder::Impl
{
public:
    explicit Impl(Config config)
        : config_(config)
    {
        if (config_.num_threads < 0 || config_.batch_size <= 0)
        {
            throw std::invalid_argument("ParallelMorsePairFinder config contains invalid values");
        }
    }

    std::vector<std::pair<int, int>>
    findMorsePairs(const std::vector<std::vector<int>> &simplices,
                   const std::vector<float> &filtration_values) const
    {
        const int n = static_cast<int>(simplices.size());
        if (filtration_values.size() != simplices.size())
        {
            throw std::invalid_argument("filtration_values size must match simplices size");
        }
        if (n == 0)
        {
            return {};
        }

        std::vector<CandidatePair> candidates(static_cast<std::size_t>(n));
        int thread_count = 1;
#ifdef _OPENMP
        thread_count = config_.num_threads > 0 ? config_.num_threads : omp_get_max_threads();
#endif
        const int chunk_size = std::max(1, config_.batch_size);
#pragma omp parallel for schedule(dynamic, chunk_size) num_threads(thread_count)
        for (int i = 0; i < n; ++i)
        {
            CandidatePair best;
            best.lower = i;
            for (int j = i + 1; j < n; ++j)
            {
                const bool can_pair =
                    config_.use_simd
                        ? SimplexPairOps::canFormGradientPair(simplices[i], simplices[j])
                        : canFormPairScalar(simplices[i], simplices[j]);
                if (!can_pair)
                {
                    continue;
                }
                const float filtration = filtration_values[static_cast<std::size_t>(j)];
                if (filtration < best.filtration ||
                    (filtration == best.filtration && j < best.upper))
                {
                    best.upper = j;
                    best.filtration = filtration;
                }
            }
            candidates[static_cast<std::size_t>(i)] = best;
        }

        std::erase_if(candidates,
                      [](const CandidatePair &pair) { return pair.lower < 0 || pair.upper < 0; });
        std::ranges::sort(candidates, [](const CandidatePair &lhs, const CandidatePair &rhs) {
            if (lhs.filtration != rhs.filtration)
            {
                return lhs.filtration < rhs.filtration;
            }
            if (lhs.lower != rhs.lower)
            {
                return lhs.lower < rhs.lower;
            }
            return lhs.upper < rhs.upper;
        });

        std::vector<char> paired(static_cast<std::size_t>(n), 0);
        std::vector<std::pair<int, int>> pairs;
        pairs.reserve(candidates.size());
        for (const CandidatePair &candidate : candidates)
        {
            const auto lower = static_cast<std::size_t>(candidate.lower);
            const auto upper = static_cast<std::size_t>(candidate.upper);
            if (!paired[lower] && !paired[upper])
            {
                paired[lower] = 1;
                paired[upper] = 1;
                pairs.emplace_back(candidate.lower, candidate.upper);
            }
        }
        return pairs;
    }

private:
    Config config_;
};

ParallelMorsePairFinder::ParallelMorsePairFinder(const Config &config)
    : impl_(std::make_unique<Impl>(config))
{}

ParallelMorsePairFinder::~ParallelMorsePairFinder() = default;

std::vector<std::pair<int, int>>
ParallelMorsePairFinder::findMorsePairs(const std::vector<std::vector<int>> &simplices,
                                        const std::vector<float> &filtration_values)
{
    return impl_->findMorsePairs(simplices, filtration_values);
}

class MorseMemoryPool::Impl
{
public:
    explicit Impl(std::size_t initial_capacity)
    {
        cells_.resize(std::max<std::size_t>(1, initial_capacity));
    }

    MorseCell *allocateCell()
    {
        const std::size_t idx = offset_.fetch_add(1, std::memory_order_relaxed);
        if (idx >= cells_.size())
        {
            std::lock_guard<std::mutex> lock(grow_mutex_);
            while (idx >= cells_.size())
            {
                cells_.resize(cells_.size() * 2);
            }
        }
        return &cells_[idx];
    }

    void reset() { offset_.store(0, std::memory_order_release); }
    std::size_t size() const { return offset_.load(std::memory_order_acquire); }

private:
    std::deque<MorseCell> cells_;
    std::atomic<std::size_t> offset_{0};
    std::mutex grow_mutex_;
};

MorseMemoryPool::MorseMemoryPool(size_t initial_capacity)
    : impl_(std::make_unique<Impl>(initial_capacity))
{}

MorseMemoryPool::~MorseMemoryPool() = default;

MorseMemoryPool::MorseCell *MorseMemoryPool::allocateCell()
{
    return impl_->allocateCell();
}

void MorseMemoryPool::reset()
{
    impl_->reset();
}

size_t MorseMemoryPool::size() const
{
    return impl_->size();
}

StreamingMorseBuilder::StreamingMorseBuilder(const Config &config)
    : config_(config)
{}

ParallelDMTBenchmark benchmarkParallelDMT(int num_simplices)
{
    ParallelDMTBenchmark bench{};
    bench.num_simplices = std::max(0, num_simplices);

    const auto simplices = makeBenchmarkSimplices(bench.num_simplices);
    const auto filtration = makeBenchmarkFiltration(bench.num_simplices);

    ParallelMorsePairFinder::Config sequential_config;
    sequential_config.num_threads = 1;
    sequential_config.use_simd = false;
    ParallelMorsePairFinder sequential(sequential_config);
    const auto start_seq = std::chrono::steady_clock::now();
    auto seq_pairs = sequential.findMorsePairs(simplices, filtration);
    const auto end_seq = std::chrono::steady_clock::now();
    bench.sequential_time_ms = elapsedMs(start_seq, end_seq);

    ParallelMorsePairFinder::Config parallel_config;
    parallel_config.use_simd = false;
    ParallelMorsePairFinder parallel(parallel_config);
    const auto start_parallel = std::chrono::steady_clock::now();
    auto parallel_pairs = parallel.findMorsePairs(simplices, filtration);
    const auto end_parallel = std::chrono::steady_clock::now();
    bench.parallel_time_ms = elapsedMs(start_parallel, end_parallel);

    ParallelMorsePairFinder::Config fast_config;
    fast_config.use_simd = true;
    ParallelMorsePairFinder fast(fast_config);
    const auto start_fast = std::chrono::steady_clock::now();
    auto fast_pairs = fast.findMorsePairs(simplices, filtration);
    const auto end_fast = std::chrono::steady_clock::now();
    bench.simd_time_ms = elapsedMs(start_fast, end_fast);

    bench.num_pairs = static_cast<int>(fast_pairs.size());
    bench.speedup_parallel =
        finiteBenchmarkSpeedup(bench.sequential_time_ms, bench.parallel_time_ms);
    bench.speedup_simd = finiteBenchmarkSpeedup(bench.sequential_time_ms, bench.simd_time_ms);
    if (parallel_pairs.size() != seq_pairs.size())
    {
        throw std::runtime_error("parallel Morse benchmark produced inconsistent pair count");
    }
    return bench;
}

} // namespace nerve::dmt::parallel

namespace nerve::dmt
{

class DMTEngine::Impl
{
public:
    explicit Impl(DMTConfig config)
        : config_(config)
    {
        if (config_.max_dimension < 0 || config_.num_threads < 0)
        {
            throw std::invalid_argument("DMTConfig contains invalid values");
        }
    }

    MorseResult computeMorseComplex(const std::vector<std::vector<int>> &simplices,
                                    const std::vector<float> &filtration)
    {
        const auto start = std::chrono::steady_clock::now();
        parallel::ParallelMorsePairFinder::Config finder_config;
        finder_config.num_threads = config_.num_threads;
        finder_config.use_simd = config_.use_simd;
        parallel::ParallelMorsePairFinder finder(finder_config);
        last_pairs_ = finder.findMorsePairs(simplices, filtration);

        std::vector<char> paired(simplices.size(), 0);
        for (const auto &[lower, upper] : last_pairs_)
        {
            if (lower >= 0 && static_cast<std::size_t>(lower) < paired.size())
            {
                paired[static_cast<std::size_t>(lower)] = 1;
            }
            if (upper >= 0 && static_cast<std::size_t>(upper) < paired.size())
            {
                paired[static_cast<std::size_t>(upper)] = 1;
            }
        }

        last_critical_.clear();
        for (std::size_t i = 0; i < paired.size(); ++i)
        {
            if (!paired[i])
            {
                last_critical_.push_back(static_cast<int>(i));
            }
        }

        const auto end = std::chrono::steady_clock::now();
        return MorseResult{last_critical_, last_pairs_,
                           std::chrono::duration<double, std::milli>(end - start).count()};
    }

    const std::vector<int> &findCriticalPoints() const { return last_critical_; }

private:
    DMTConfig config_;
    std::vector<int> last_critical_;
    std::vector<std::pair<int, int>> last_pairs_;
};

DMTEngine::DMTEngine(const DMTConfig &config)
    : impl_(std::make_unique<Impl>(config))
{}
DMTEngine::~DMTEngine() = default;

MorseResult DMTEngine::computeMorseComplex(const std::vector<std::vector<int>> &simplices,
                                           const std::vector<float> &filtration)
{
    return impl_->computeMorseComplex(simplices, filtration);
}

std::vector<int> DMTEngine::findCriticalPoints() const
{
    return impl_->findCriticalPoints();
}

} // namespace nerve::dmt
