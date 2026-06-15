# PH6 Internals

## Experimental Feature Registry

PH6 maintains a registry of experimental features with their status and configuration:

```cpp
class ExperimentalFeatureRegistry {
    struct Feature {
        std::string name;
        bool active;
        bool requires_verification;
        std::function<bool(const PH5PH6Config&)> activator;
        std::function<bool(const PH5PH6Config&)> validator;
    };

    std::vector<Feature> features;

public:
    ExperimentalFeatureRegistry() {
        registerFeature("adaptive_ordering", {
            .active = false,
            .requires_verification = true,
            .activator = [](const PH5PH6Config& c) {
                return c.experimental_reduction_ordering == "adaptive";
            },
            .validator = [](const PH5PH6Config& c) {
                return c.experimental_reduction_ordering == "adaptive"
                    || c.experimental_reduction_ordering == "default";
            }
        });

        registerFeature("approximate_clearing", {
            .active = false,
            .requires_verification = true,
            .activator = [](const PH5PH6Config& c) {
                return c.experimental_clearing == "approximate";
            },
            .validator = [](const PH5PH6Config& c) {
                return c.experimental_clearing == "exact"
                    || c.experimental_clearing == "approximate";
            }
        });

        registerFeature("speculative_reduction", {
            .active = false,
            .requires_verification = true,
            .activator = [](const PH5PH6Config& c) {
                return c.experimental_speculative_threads > 1;
            },
            .validator = [](const PH5PH6Config& c) {
                return c.experimental_speculative_threads >= 1
                    && c.experimental_speculative_threads <= 64;
            }
        });
    }

    std::vector<std::string> getActiveFeatures(const PH5PH6Config& config) const {
        std::vector<std::string> active;
        for (const auto& f : features) {
            if (f.activator(config)) {
                active.push_back(f.name);
            }
        }
        return active;
    }

    bool validateConfiguration(const PH5PH6Config& config) const {
        for (const auto& f : features) {
            if (f.activator(config) && !f.validator(config)) {
                return false;  // active feature has invalid configuration
            }
        }
        return true;
    }
};
```

## Adaptive Ordering Implementation

The adaptive ordering strategy dynamically reorders columns based on their estimated reduction cost:

```cpp
void reorderColumnsAdaptively(std::vector<uint32_t>& order,
                              const std::vector<Column>& columns,
                              const std::vector<bool>& cleared) {
    // Score each remaining column by estimated reduction difficulty
    struct ScoredColumn {
        uint32_t index;
        float score;  // lower = easier to reduce
    };

    std::vector<ScoredColumn> scored;
    for (uint32_t j = 0; j < columns.size(); ++j) {
        if (cleared[j]) continue;
        float density = static_cast<float>(columns[j].nonZeroCount())
                      / columns[j].size();
        float pivot_distance = columns[j].hasPivot()
            ? static_cast<float>(columns[j].size() - columns[j].pivot())
            : 0.0f;
        // Heuristic: process dense columns with close-to-bottom pivots first
        float score = density * 0.6f + (pivot_distance / columns[j].size()) * 0.4f;
        scored.push_back({j, score});
    }

    // Sort by score (ascending)
    std::sort(scored.begin(), scored.end(),
              [](const ScoredColumn& a, const ScoredColumn& b) {
                  return a.score < b.score;
              });

    // Extract the new order
    order.clear();
    for (const auto& sc : scored) {
        order.push_back(sc.index);
    }
}
```

The reordering is triggered every `N_columns / 10` steps or when the pivot conflict rate exceeds a threshold.

## Approximate Clearing Implementation

```cpp
void approximateClearing(std::vector<Column>& columns, float threshold) {
    for (uint32_t j = 0; j < columns.size(); ++j) {
        if (columns[j].cleared()) continue;

        float density = columns[j].density();
        if (density > threshold) {
            // Heuristically clear this column
            columns[j].clear();
            cleared_flags[j] = true;

            // Track for verification pass
            approximate_clears.push_back(j);
        }
    }
}

void verifyApproximateClears(std::vector<Column>& columns) {
    // Verify each heuristically cleared column
    for (uint32_t idx : approximate_clears) {
        // Re-process the column exactly
        Column original = reconstructColumn(idx);
        reduceExact(original, idx);

        if (original.hasPivot() != columns[idx].hasPivot()
            || original.pivot() != columns[idx].pivot()) {
            // False clear detected! Revert and re-reduce.
            false_clear_count++;
            columns[idx] = std::move(original);
        }
    }
}
```

## Block-Sparse Memory Layout

The block-sparse reduction organizes column storage into cache-sized blocks:

```cpp
struct Block {
    static constexpr size_t BLOCK_SIZE = 4096;  // columns per block
    static constexpr size_t CACHE_LINE = 64;     // bytes

    // Bit-packed column data within this block
    alignas(CACHE_LINE) uint64_t data[BLOCK_SIZE][8]; // 4096 cols x 512 rows each
    uint32_t pivot_map[BLOCK_SIZE];
    uint8_t dimensions[BLOCK_SIZE];

    // Block fits in:
    // data:      4096 * 8 * 8 = 262 KB
    // pivot_map: 4096 * 4     = 16 KB
    // dimensions: 4096 * 1    = 4 KB
    // Total: ~282 KB (fits in L2 cache on most CPUs)
};
```

Columns wider than 512 rows are split across multiple blocks. Block-local operations avoid main memory traffic:

```cpp
void reduceBlockColumn(Block& block, uint32_t local_j) {
    // Column XOR within a block: all data is in L2 cache
    while (true) {
        uint32_t p = block.pivot_map[local_j];
        if (p == UINT32_MAX || pivot_global[p] == UINT32_MAX) break;

        uint32_t k = pivot_global[p];
        // XOR columns within the same block
        for (size_t w = 0; w < 8; ++w) {
            block.data[local_j][w] ^= block.data[k][w];
        }
        block.pivot_map[local_j] = recomputePivot(block.data[local_j]);
    }
}
```

## Experimental Feature Interaction Matrix

Some experimental features interact in non-trivial ways. Adaptive ordering and speculative reduction are compatible because each speculative thread uses its own ordering. Adaptive ordering and approximate clearing are compatible because ordering runs on remaining (non-cleared) columns. Adaptive ordering and block-sparse are compatible because ordering operates at block granularity. Approximate clearing combined with speculative reduction requires caution because false clears may differ across threads, though majority voting helps. Approximate clearing combined with block-sparse also requires caution because clearing breaks block alignment. Speculative reduction and block-sparse are compatible because each thread gets independent block partitions. Adaptive pivoting is orthogonal and compatible with any feature.


[Back to PH6 Index](index.md)
