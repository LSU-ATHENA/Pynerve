## Parallel gradient computation

```cpp
namespace nerve::dmt::parallel {

struct ParallelMorsePairFinder::Config {
    int num_threads = 0;
    int batch_size = 1024;
    bool use_simd = true;
};

class ParallelMorsePairFinder {
public:
    explicit ParallelMorsePairFinder(const Config& config);

    std::vector<std::pair<int, int>> findMorsePairs(
        const std::vector<std::vector<int>>& simplices,
        const std::vector<float>& filtration_values);
};

}
```

Uses OpenMP dynamic scheduling. Each thread processes a batch of simplices,
finding and matching candidate gradient pairs. The `use_simd` flag enables
AVX-512/AVX2 subset checks (`SimplexPairOps::isSubsetSIMD`) for faster face
comparisons.

**SIMD subset check** (`SimplexPairOps::canFormGradientPair`):
- Two simplices a, b form a gradient pair iff:
  - dim(b) = dim(a) + 1
  - a is a subset of b (all vertices of a appear in b)
  - The pair is acyclic (no directed cycles in gradient field)

`isSubsetSIMD` compares 8 vertices at a time with packed comparisons on
AVX-512, reducing O(k*m) face checks to O(k*m/SIMD_width).

```cpp
// Example: parallel Morse pair finding
ParallelMorsePairFinder::Config config;
config.num_threads = 8;
config.use_simd = true;

ParallelMorsePairFinder finder(config);
auto pairs = finder.findMorsePairs(simplices, filtration_values);
```

### Cache-optimized traversal

```cpp
class CacheOptimizedMorseTraversal {
public:
    template <typename Callback>
    void traversePaths(
        const std::vector<std::pair<int, int>>& gradient_pairs,
        const std::vector<std::vector<int>>& simplex_boundaries,
        int block_size,
        Callback&& callback);
};
```

Blocked BFS traversal through gradient pairs. Processes gradient pairs in
blocks of `block_size` to improve cache locality. Each block traverses the
boundary graph starting from each pair's face index.

### Streaming construction

```cpp
class StreamingMorseBuilder {
public:
    struct Config {
        int chunk_size = 1000;
        int max_dimension = 2;
        float max_filtration = 1.0f;
    };

    explicit StreamingMorseBuilder(const Config& config);

    template <typename SimplexSource, typename Callback>
    void buildStreaming(SimplexSource& source, Callback&& callback);
};
```

Processes simplices in chunks to bound memory usage. The `SimplexSource` must
provide `hasNext()` and `next()` returning `(vertices, filtration)`. Each
chunk is reduced independently; critical cells propagate to the next chunk.

Use for complexes too large to fit in memory (e.g., high-dimensional VR on
100k+ points).

### Memory pool

```cpp
class MorseMemoryPool {
public:
    explicit MorseMemoryPool(size_t initial_capacity = 10000);
    ~MorseMemoryPool();

    struct MorseCell {
        int critical_index;
        std::vector<int> boundary;
        float filtration;
    };

    MorseCell* allocateCell();
    void reset();
    size_t size() const;
};
```

Thread-safe pool of `MorseCell` objects. Uses a deque + atomic offset + mutex
for concurrent allocation during parallel gradient computation.

### Benchmark utility

```cpp
struct ParallelDMTBenchmark {
    double sequential_time_ms;
    double parallel_time_ms;
    double simd_time_ms;
    double speedup_parallel;
    double speedup_simd;
    int num_simplices;
    int num_pairs;
};

ParallelDMTBenchmark benchmarkParallelDMT(int num_simplices);
```


## OpenMP schedule strategy

```cpp
// Dynamic scheduling with batch_size=1024
#pragma omp parallel for schedule(dynamic, 1024)
for (int i = 0; i < num_simplices; i++) {
    process_simplex(i);
}
```

The batch size balances load distribution vs scheduling overhead:
- Too small (<128): high OpenMP overhead from frequent scheduling
- Too large (>4096): poor load balance (some threads idle)

## SIMD subset check implementation

```cpp
// SimplexPairOps::isSubsetSIMD with AVX-512
bool isSubsetSIMD(const int* a, int na, const int* b, int nb) {
    // Check if all vertices of a appear in b
    // b has exactly one more vertex than a (codim-1 coface)

    __m512i va, vb;
    int mask;

    for (int i = 0; i < na; i += 8) {
        va = _mm512_loadu_si512(a + i);
        // Broadcast each vertex of a and compare with b
        for (int j = 0; j < 8 && i + j < na; j++) {
            int v = a[i + j];
            __m512i v_bcast = _mm512_set1_epi32(v);
            vb = _mm512_loadu_si512(b);
            __mmask16 eq = _mm512_cmpeq_epi32_mask(v_bcast, vb);
            if (eq == 0) return false;  // vertex not found
        }
    }
    return true;
}
```

## Performance benchmarks

```python
from pynerve.dmt import benchmarkParallelDMT

for n in [1000, 10000, 100000]:
    bm = benchmarkParallelDMT(num_simplices=n)
    print(f"n={n}:")
    print(f"  Sequential: {bm.sequential_time_ms:.1f}ms")
    print(f"  Parallel:   {bm.parallel_time_ms:.1f}ms")
    print(f"  SIMD:       {bm.simd_time_ms:.1f}ms")
    print(f"  Speedup:    {bm.speedup_parallel:.1f}x (par) / "
          f"{bm.speedup_simd:.1f}x (simd)")
```

Expected performance on 96-core AMD EPYC:
- Sequential: baseline
- Parallel (96 threads): 20-60x speedup
- Parallel + SIMD: 25-80x speedup

## Streaming DMT configuration

```cpp
StreamingMorseBuilder::Config cfg;
cfg.chunk_size = 5000;        // process 5000 simplices per chunk
cfg.max_dimension = 3;
cfg.max_filtration = 5.0f;

StreamingMorseBuilder builder(cfg);

// Stream simplices from a large dataset
// (e.g., high-dimensional VR with 100k+ points)
for (auto& chunk : chunked_simplices) {
    builder.buildStreaming(chunk, [](int idx, float filtration) {
        // Callback for each critical cell
        // Critical cells propagate between chunks
    });
}
```


## Memory pool tuning

```cpp
// Tune memory pool for expected load
MorseMemoryPool pool(100000);  // initial capacity

// Parallel allocation is thread-safe
#pragma omp parallel for
for (int i = 0; i < num_cells; i++) {
    MorseCell* cell = pool.allocateCell();
    cell->critical_index = i;
    // ... populate cell ...
}
```

Pool growth doubles capacity when exhausted, bounded by the total number of cells.


## FAQ

**Q: How many threads should I use for parallel DMT?**
A: Start with `num_threads` equal to your available cores. On a 96-core AMD EPYC, parallel speedup reaches 20-60x. Adding SIMD (AVX-512) improves this to 25-80x. Avoid over-subscription (more threads than hardware cores) as OpenMP overhead grows.

**Q: When should I use streaming mode instead of in-memory?**
A: Use `StreamingMorseBuilder` when the complex is too large to fit in memory -- for example, high-dimensional VR on 100k+ points. Streaming processes simplices in chunks (default 5000 per chunk) and propagates critical cells between chunks.

**Q: What batch size should I choose for OpenMP scheduling?**
A: A `batch_size` of 1024 balances load distribution against scheduling overhead. Below 128, OpenMP scheduling overhead is significant. Above 4096, load imbalance increases (some threads sit idle while others still have work).


### Cross-references

- `pynerve.dmt`: DMT module overview
- `pynerve.dmt.gpu`: GPU DMT
- `pynerve.core.thread_pool`: Thread pool
- `pynerve.memory`: Memory pools
- `pynerve.core.simd_ops`: SIMD comparison operations
