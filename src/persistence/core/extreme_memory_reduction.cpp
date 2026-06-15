#include "nerve/persistence/core/core_types.hpp"
#include "nerve/persistence/streaming/streaming_reducer.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <span>

namespace nerve::persistence::extreme
{

constexpr size_t BYTES_PER_GB = 1024ULL * 1024ULL * 1024ULL;

constinit const size_t DEFAULT_EXTREME_MEMORY_LIMIT = 4ULL * BYTES_PER_GB; // 4GB default

enum class MemoryTier
{
    IN_MEMORY,
    STREAMING
};

class AutomaticMemoryOptimizer
{
public:
    explicit AutomaticMemoryOptimizer(size_t max_ram_bytes);

    /**
     * @brief Select optimal memory tier
     */
    [[nodiscard]] MemoryTier selectTier(size_t n_points, size_t point_dim,
                                        double max_distance) const;

    /**
     * @brief Compute persistence with automatic optimization
     *
     * One-call function that automatically handles all memory decisions.
     */
    [[nodiscard]] PersistenceDiagram compute(std::span<const double> points, size_t n_points,
                                             size_t point_dim, double max_distance, int max_dim);

private:
    size_t max_ram_ = 0;

    [[nodiscard]] size_t estimateMemoryBytes(size_t n_points, size_t point_dim,
                                             double max_distance) const;
};

#include "detail/extreme_memory_reduction_impl.inl"

} // namespace nerve::persistence::extreme
