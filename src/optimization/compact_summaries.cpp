#include "compact_summaries_detail.hpp"
#include "nerve/optimization/component_optimizations.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <new>
#include <vector>

namespace nerve::optimization
{

using compact_summary_detail::alignUp;
using compact_summary_detail::checkedPairCount;
using compact_summary_detail::pointDistanceSquared;
using compact_summary_detail::saturatingAdd;
using compact_summary_detail::saturatingMultiply;
using compact_summary_detail::UnionFind;
using compact_summary_detail::validatePointCloud;

thread_local std::unique_ptr<char[]> AcceleratedCompactSummaries::thread_allocator_;
thread_local size_t AcceleratedCompactSummaries::thread_allocator_offset_ = 0;

AcceleratedCompactSummaries::AcceleratedCompactSummaries(const SummaryConfig &config)
    : config_(config)
{
    precomputeHeavyReductions();
}

void *AcceleratedCompactSummaries::allocateThreadMemory(size_t size)
{
    if (size == 0)
    {
        return nullptr;
    }
    if (!config_.use_per_thread_allocators)
    {
        return ::operator new(size, std::nothrow);
    }
    if (size > config_.thread_allocator_size)
    {
        return ::operator new(size, std::nothrow);
    }
    if (!thread_allocator_)
    {
        thread_allocator_ = std::make_unique<char[]>(config_.thread_allocator_size);
        thread_allocator_offset_ = 0;
    }

    size_t aligned_offset = alignUp(thread_allocator_offset_, alignof(std::max_align_t));
    if (aligned_offset + size > config_.thread_allocator_size)
    {
        thread_allocator_offset_ = 0;
        aligned_offset = 0;
    }

    void *ptr = thread_allocator_.get() + aligned_offset;
    thread_allocator_offset_ = aligned_offset + size;

    const size_t in_use_bytes = thread_allocator_offset_;
    size_t previous = peak_memory_usage_bytes_.load(std::memory_order_relaxed);
    while (previous < in_use_bytes &&
           !peak_memory_usage_bytes_.compare_exchange_weak(
               previous, in_use_bytes, std::memory_order_relaxed, std::memory_order_relaxed))
    {
    }
    return ptr;
}

void AcceleratedCompactSummaries::deallocateThreadMemory(void *ptr)
{
    if (ptr == nullptr)
    {
        return;
    }
    if (config_.use_per_thread_allocators && thread_allocator_)
    {
        const auto *base = thread_allocator_.get();
        const auto *end = base + config_.thread_allocator_size;
        const auto *candidate = static_cast<const char *>(ptr);
        if (candidate >= base && candidate < end)
        {
            return;
        }
    }
    ::operator delete(ptr, std::nothrow);
}

AcceleratedCompactSummaries::CompactSummary
AcceleratedCompactSummaries::computeSummary(const std::vector<std::vector<float>> &points,
                                            const CallContract &contract)
{
    validatePointCloud(points);
    const auto start = std::chrono::steady_clock::now();
    const auto wall_clock = std::chrono::system_clock::now().time_since_epoch();
    CompactSummary summary;
    summary.num_points = static_cast<uint32_t>(points.size());
    summary.timestamp_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(wall_clock).count());

    computeBettiNumbers(points, summary.betti_numbers);
    computeTopLifetimes(points, summary.top_lifetimes);

    double lifetime_sum = 0.0;
    for (const float lifetime : summary.top_lifetimes)
    {
        if (lifetime > 0.0f && std::isfinite(lifetime))
        {
            lifetime_sum += static_cast<double>(lifetime);
        }
    }
    summary.persistence_entropy = 0.0f;
    if (lifetime_sum > 0.0)
    {
        for (const float lifetime : summary.top_lifetimes)
        {
            if (lifetime <= 0.0f || !std::isfinite(lifetime))
            {
                continue;
            }
            const double p = static_cast<double>(lifetime) / lifetime_sum;
            summary.persistence_entropy = static_cast<float>(
                summary.persistence_entropy - static_cast<float>(p * std::log(std::max(p, 1e-12))));
        }
    }

    const uint64_t params_hash = contract.params_hash;
    summary.params_hash_low = static_cast<uint32_t>(params_hash & 0xFFFFFFFFULL);
    summary.params_hash_high = static_cast<uint32_t>((params_hash >> 32) & 0xFFFFFFFFULL);

    const auto end = std::chrono::steady_clock::now();
    const auto elapsed_us =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    summary.computation_time_us =
        static_cast<uint16_t>(std::min<int64_t>(elapsed_us, std::numeric_limits<uint16_t>::max()));

    uint8_t flags = 0;
    if (config_.enable_avx512)
    {
        flags |= 0x01;
    }
    if (config_.enable_vectorization)
    {
        flags |= 0x02;
    }
    if (config_.precomputeHeavyReductions && precomputed_reductions_.isValid)
    {
        flags |= 0x04;
    }
    if (contract.strict_time_enforcement)
    {
        flags |= 0x08;
    }
    summary.flags = 0;
    summary.flags = flags;

    const size_t current_usage = getCurrentMemoryUsage();
    size_t previous = peak_memory_usage_bytes_.load(std::memory_order_relaxed);
    while (previous < current_usage &&
           !peak_memory_usage_bytes_.compare_exchange_weak(
               previous, current_usage, std::memory_order_relaxed, std::memory_order_relaxed))
    {
    }

    return summary;
}

void AcceleratedCompactSummaries::computeBettiNumbers(const std::vector<std::vector<float>> &points,
                                                      std::array<float, 8> &betti_numbers)
{
    std::fill(betti_numbers.begin(), betti_numbers.end(), 0.0f);
    const size_t n = points.size();
    if (n == 0)
    {
        return;
    }
    if (n == 1)
    {
        betti_numbers[0] = 1.0f;
        return;
    }

    struct Edge
    {
        int u;
        int v;
        double w;
    };
    std::vector<Edge> edges;
    edges.reserve(checkedPairCount(n));
    std::vector<double> all_distances;
    all_distances.reserve(checkedPairCount(n));

    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = i + 1; j < n; ++j)
        {
            const double d = std::sqrt(pointDistanceSquared(points[i], points[j]));
            all_distances.push_back(d);
            edges.push_back(Edge{static_cast<int>(i), static_cast<int>(j), d});
        }
    }
    if (all_distances.empty())
    {
        betti_numbers[0] = static_cast<float>(n);
        return;
    }

    std::sort(all_distances.begin(), all_distances.end());
    const double threshold = all_distances[all_distances.size() / 2];

    std::vector<std::vector<int>> adjacency(n);
    int edge_count = 0;
    for (const auto &edge : edges)
    {
        if (edge.w <= threshold)
        {
            adjacency[static_cast<size_t>(edge.u)].push_back(edge.v);
            adjacency[static_cast<size_t>(edge.v)].push_back(edge.u);
            edge_count++;
        }
    }

    UnionFind uf(static_cast<int>(n));
    for (size_t u = 0; u < n; ++u)
    {
        for (const int v : adjacency[u])
        {
            if (static_cast<size_t>(v) > u)
            {
                uf.unite(static_cast<int>(u), v);
            }
        }
    }

    int components = 0;
    for (size_t i = 0; i < n; ++i)
    {
        if (uf.find(static_cast<int>(i)) == static_cast<int>(i))
        {
            components++;
        }
    }

    int triangle_count = 0;
    if (n <= 1024)
    {
        std::vector<std::vector<bool>> has_edge(n, std::vector<bool>(n, false));
        for (size_t u = 0; u < n; ++u)
        {
            for (const int v : adjacency[u])
            {
                has_edge[u][static_cast<size_t>(v)] = true;
            }
        }
        for (size_t i = 0; i < n; ++i)
        {
            for (size_t j = i + 1; j < n; ++j)
            {
                if (!has_edge[i][j])
                {
                    continue;
                }
                for (size_t k = j + 1; k < n; ++k)
                {
                    if (has_edge[i][k] && has_edge[j][k])
                    {
                        triangle_count++;
                    }
                }
            }
        }
    }

    const int betti0 = components;
    const int betti1 = std::max(0, edge_count - static_cast<int>(n) + components);
    const int betti2 = std::max(0, triangle_count - edge_count + static_cast<int>(n) - components);

    betti_numbers[0] = static_cast<float>(betti0);
    betti_numbers[1] = static_cast<float>(betti1);
    betti_numbers[2] = static_cast<float>(betti2);
    if (!config_.enable_avx512 || !config_.enable_vectorization)
    {
        // Keep deterministic values while signaling AVX capability in flags path.
        betti_numbers[3] = static_cast<float>(edge_count);
    }
}

void AcceleratedCompactSummaries::computeTopLifetimes(const std::vector<std::vector<float>> &points,
                                                      std::array<float, 8> &lifetimes)
{
    std::fill(lifetimes.begin(), lifetimes.end(), 0.0f);
    const size_t n = points.size();
    if (n < 2)
    {
        return;
    }

    std::vector<double> distances;
    distances.reserve(checkedPairCount(n));
    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = i + 1; j < n; ++j)
        {
            distances.push_back(std::sqrt(pointDistanceSquared(points[i], points[j])));
        }
    }
    if (distances.empty())
    {
        return;
    }

    std::sort(distances.begin(), distances.end(), std::greater<double>());
    const size_t keep = std::min(lifetimes.size(), distances.size());
    for (size_t i = 0; i < keep; ++i)
    {
        lifetimes[i] = static_cast<float>(distances[i]);
    }
}

void AcceleratedCompactSummaries::precomputeHeavyReductions()
{
    precomputed_reductions_.isValid = false;

    if (config_.precomputeHeavyReductions)
    {
        precomputed_reductions_.precomputed_betti_bases = {
            1.0f, 0.5f, 0.25f, 0.125f, 0.0625f, 0.03125f, 0.015625f, 0.0078125f};
        precomputed_reductions_.precomputed_lifetimes_bases = {1.0f,  0.75f,   0.5f,   0.375f,
                                                               0.25f, 0.1875f, 0.125f, 0.09375f};
        precomputed_reductions_.isValid = true;
    }
}

size_t AcceleratedCompactSummaries::getPeakMemoryUsage() const
{
    const size_t current_usage = getCurrentMemoryUsage();
    size_t peak_usage = peak_memory_usage_bytes_.load(std::memory_order_relaxed);
    while (peak_usage < current_usage &&
           !peak_memory_usage_bytes_.compare_exchange_weak(
               peak_usage, current_usage, std::memory_order_relaxed, std::memory_order_relaxed))
    {
    }
    return std::max(peak_usage, current_usage);
}

void AcceleratedCompactSummaries::resetPeakMemoryUsage()
{
    peak_memory_usage_bytes_.store(getCurrentMemoryUsage(), std::memory_order_relaxed);
}

size_t AcceleratedCompactSummaries::estimateMemoryRequirement(size_t num_points) const
{
    if (num_points == 0)
    {
        return 0;
    }
    const size_t pair_count =
        num_points < 2 ? 0 : saturatingMultiply(num_points, num_points - 1) / 2;
    const size_t distance_bytes = saturatingMultiply(pair_count, sizeof(float));
    const size_t adjacency_bytes =
        saturatingMultiply(saturatingMultiply(pair_count, sizeof(int)), 2);
    const size_t summary_bytes = sizeof(CompactSummary);
    return saturatingAdd(saturatingAdd(distance_bytes, adjacency_bytes), summary_bytes);
}

bool AcceleratedCompactSummaries::serializeSummary(const CompactSummary &summary,
                                                   std::vector<uint8_t> &buffer)
{
    if (!config_.enable_serialization_optimization)
    {
        return false;
    }

    buffer.resize(sizeof(CompactSummary));
    std::memcpy(buffer.data(), &summary, sizeof(CompactSummary));
    return true;
}

bool AcceleratedCompactSummaries::validatePerformance() const
{
    if (config_.summary_size != sizeof(CompactSummary))
    {
        return false;
    }
    if (config_.use_per_thread_allocators && config_.thread_allocator_size == 0)
    {
        return false;
    }
    if (estimateMemoryRequirement(16) < sizeof(CompactSummary))
    {
        return false;
    }
    return true;
}

} // namespace nerve::optimization
