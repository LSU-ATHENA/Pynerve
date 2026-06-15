# API

## Python API

```python
from pynerve.core import CompactSummaryPipeline

def summarize(points, max_size=1024, max_persistence_dim=2,
              max_filtration_radius=1.0, seed=42) -> CompactSummary: ...

class CompactSummary:
    top_lifetimes: list[Lifetime]
    betti_counts: list[int]
    top_eigenvalues: list[Eigenvalue]
    persistence_entropy: float
    betti_entropy: float
    spectral_entropy: float
    timestamp_ns: int
    symbol_id: int
    computation_time_us: int
    data_points_count: int
    noise_level: float
    def serialize(self) -> bytes: ...
    def deserialize(self, data: bytes) -> bool: ...
    def is_valid(self) -> bool: ...
```


### C++ API

```cpp
#include <nerve/summary/compact_summary.hpp>

namespace nerve::summary {

struct CompactSummary {
    static constexpr size_t MAX_LIFETIMES = 10;
    static constexpr size_t MAX_BETTI_DIM = 5;
    static constexpr size_t MAX_EIGENVALUES = 10;
    static constexpr size_t TARGET_SIZE_BYTES = 1024;

    struct Lifetime {
        float birth, death;
        uint8_t dimension;
        float persistence;
    };

    struct Eigenvalue {
        float value;
        uint16_t multiplicity;
    };

    std::array<Lifetime, MAX_LIFETIMES> top_lifetimes;
    uint8_t lifetime_count;
    std::array<uint16_t, MAX_BETTI_DIM> betti_counts;
    uint8_t betti_dimension_count;
    std::array<Eigenvalue, MAX_EIGENVALUES> top_eigenvalues;
    uint8_t eigenvalue_count;

    float persistence_entropy;
    float betti_entropy;
    float spectral_entropy;

    int64_t timestamp_ns;
    int64_t symbol_id;
    uint32_t computation_time_us;
    uint16_t data_points_count;
    float noise_level;

    // High-dimensional extension
    struct HighDimExtension {
        std::array<uint16_t, 8> highdim_betti_top8;
        std::array<LifetimeStats, 8> highdim_lifetime_stats;
        std::array<float, 8> dimension_complexity;
        std::array<uint32_t, 8> simplex_counts;
        bool truncated_by_budget;
        float compression_ratio;
    };

    std::vector<uint8_t> serialize() const;
    bool deserialize(const std::vector<uint8_t>& data);
    size_t sizeBytes() const;
    bool isValid() const;
    bool isUnderSizeLimit() const;
    bool hasHighdimData() const;
    const HighDimExtension& getHighdimExtension() const;
    void setHighdimExtension(const HighDimExtension&);
    uint16_t getHighdimBetti(uint8_t dimension) const;
    float getDimensionComplexity(uint8_t dimension) const;
};

}
```


[Back to index](index.md)
