## Compact summaries

`AcceleratedCompactSummaries` reduces a point cloud into a fixed-size 128-byte
`CompactSummary` that preserves topological features.

```cpp
class AcceleratedCompactSummaries {
public:
    struct SummaryConfig {
        bool enable_avx512 = true;
        bool use_per_thread_allocators = true;
        size_t thread_allocator_size = 1024ULL * 1024ULL;
        bool precomputeHeavyReductions = true;
        bool enable_vectorization = true;
        size_t summary_size = 128;
        bool enable_serialization_optimization = true;
    };

    explicit AcceleratedCompactSummaries(const SummaryConfig& config);

    struct CompactSummary {
        std::array<float, 8> betti_numbers;       // Betti 0-7
        std::array<float, 8> top_lifetimes;       // longest-lived pairs
        float persistence_entropy;                 // Shannon entropy
        std::array<float, 4> laplacian_top4;       // top Laplacian eigenvalues
        uint32_t num_points;
        uint64_t timestamp_ns;
        uint32_t params_hash_low;
        uint32_t params_hash_high;
        uint16_t computation_time_us;
        uint8_t flags;
        uint8_t reserved[15];
    };
    static_assert(sizeof(CompactSummary) == 128);

    CompactSummary computeSummary(
        const std::vector<std::vector<float>>& points,
        const CallContract& contract);

    void* allocateThreadMemory(size_t size);
    void deallocateThreadMemory(void* ptr);
    bool serializeSummary(const CompactSummary& summary,
                          std::vector<uint8_t>& buffer);
    bool validatePerformance() const;
    size_t estimateMemoryRequirement(size_t num_points) const;
    size_t getCurrentMemoryUsage() const;
    size_t getPeakMemoryUsage() const;
    void resetPeakMemoryUsage();

private:
    SummaryConfig config_;
    thread_local static std::unique_ptr<char[]> thread_allocator_;
};
```

The summary packs topological features into a fixed 128-byte structure
suitable for serialization, MPI broadcast, or real-time dashboards.
Precomputes heavy reductions (PCA-like bases for Betti numbers and lifetimes)
for amortized O(1) per-window latency.

```python
from pynerve.optimization import AcceleratedCompactSummaries

summarizer = AcceleratedCompactSummaries(
    summary_size=128,
    enable_avx512=True,
)

summary = summarizer.compute_summary(points)
print(summary.betti_numbers)         # [1, 1, 0, ...]
print(summary.persistence_entropy)   # 0.89
print(summary.top_lifetimes)         # [2.3, 1.8, 0.9, ...]
```

### When to use compact summaries

Compact summaries are suitable for real-time monitoring with a small fixed size per snapshot and O(1) bandwidth, sliding windows where summaries are computed per window and compared, distributed aggregation with MPI broadcast of a small fixed size per node, and historical storage as a fixed-size record that is easy to index.


## Summary fields in detail

### Betti numbers (8 floats)

Stores Betti numbers for dimensions 0-7 as 32-bit floats. For typical 3D data, only dimensions 0-2 are non-zero. Dimensions 3-7 are zero for most real-world data.

### Top lifetimes (8 floats)

The 8 longest-lived persistence pairs. Stored as (birth, death) interleaved. Sorted by persistence descending.

### Persistence entropy (1 float)

```
H = -sum_i (L_i / L_total) * log(L_i / L_total)
```

where L_i is the persistence of the i-th pair and L_total = sum L_i.

### Laplacian top 4 (4 floats)

The 4 largest eigenvalues of the graph Laplacian of the 1-skeleton. Indicate spectral properties: number of connected components (zero eigenvalues) and graph expansion (spectral gap).


## Memory usage breakdown

The CompactSummary struct consists of: Betti numbers (32 bytes, 8 x float32), Top lifetimes (64 bytes, 8 x birth + death pairs), Persistence entropy (4 bytes, float32), Laplacian top 4 (16 bytes, 4 x float32), Num points (4 bytes, uint32), Timestamp (8 bytes, uint64), Params hash (8 bytes, 2 x uint32), Computation time (2 bytes, uint16 in microseconds), Flags (1 byte, uint8), and Reserved (15 bytes for future use).

### Memory estimation for batch summaries

```python
num_windows = 10000
summary_size = 128  # bytes
total_bytes = num_windows * summary_size
print(f"Batch memory: {total_bytes / 1024:.1f} KB")

# 128-byte summaries fit in L1 cache lines
# 10k summaries fit in around a megabyte (L2 cache on modern CPUs)
```

### Performance monitoring

```python
summarizer = AcceleratedCompactSummaries(cfg)
summary = summarizer.computeSummary(points, contract)

# Check performance
assert summarizer.validatePerformance()
print(f"Memory: {summarizer.getCurrentMemoryUsage()} bytes")
print(f"Peak: {summarizer.getPeakMemoryUsage()} bytes")

# Memory tracking
summarizer.resetPeakMemoryUsage()
```


## FAQ

**Q: Why 8 Betti numbers when most data has <4 relevant dimensions?**
A: 8 fields ensure the structure is future-proof for high-dimensional data (e.g., word embeddings, multi-sensor fusion). They are packed as floats so unused dimensions can store other features without schema changes.

**Q: How are top lifetimes selected?**
A: All pairs from the persistence computation are sorted by persistence. The top 8 (or fewer, if the diagram has <8 pairs) are stored. If there are more than 8, only the 8 most persistent are kept.

**Q: Can I extract more than 8 lifetimes?**
A: For full analysis, use the full persistence diagram. The compact summary is designed for bandwidth-constrained scenarios (MPI, dashboards). For detailed analysis, access the `pynerve.persistence` module directly.

**Q: Is the summary thread-safe?**
A: Yes. Each thread has its own allocator (`thread_local static`). The `computeSummary` method is fully thread-safe and can be called from multiple threads concurrently.


### Cross-references

- `pynerve.optimization`: Optimization module overview
- `pynerve.optimization.streaming_ph`: Streaming persistence for summaries
- `pynerve.summary`: Summary data structure
- `pynerve.serialization`: Serialization of summaries
- `pynerve.spectral.laplacian`: Laplacian eigenvalue computation
