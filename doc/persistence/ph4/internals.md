# PH4 Engine Internals

### Core Data Structures

```cpp
// Internal: the PH4 reduction engine state
struct PH4EngineState {
    // Column storage
    std::vector<Column> columns;            // boundary/coboundary columns
    std::vector<uint32_t> pivot_map;        // row -> column mapping (SIZE_MAX = empty)
    std::vector<int8_t> column_dimensions;  // cached dimension per column
    std::vector<uint32_t> filtration_order; // simplex order by filtration value

    // Reduction state
    std::vector<bool> cleared;              // per-column cleared flag
    std::vector<std::pair<uint32_t, uint32_t>> pairs;
    std::vector<uint32_t> betti_numbers;

    // Parallelism
    int num_threads;
    std::vector<std::vector<Column>> thread_local_columns; // per-thread working buffers

    // Algorithm selection
    ReductionStrategy strategy;  // STANDARD, COHOMOLOGY, or AUTO
};
```

### Column Representation Selection

PH4 selects column representation at construction time based on estimated density:

```cpp
ColumnRepresentation selectRepresentation(uint32_t n_rows, float estimated_density) {
    // For small columns: always use inline bitset (single uint64_t)
    if (n_rows <= 64) {
        return ColumnRepresentation::BITSET_INLINE;
    }

    // For dense columns: use heap-allocated bitset
    if (estimated_density > 0.1) {
        return ColumnRepresentation::BITSET_EXTERNAL;
    }

    // For sparse columns: use sorted index list
    return ColumnRepresentation::SPARSE_SORTED;
}
```

The density estimate is computed from the boundary/coboundary statistics:

```cpp
float estimateColumnDensity(const Simplex& sigma, const Filtration& filtration) {
    int d = sigma.dimension();
    if (d == 0) return 0.0f;  // vertices have empty boundary

    if (strategy == STANDARD) {
        // Boundary of a d-simplex has exactly d+1 entries
        return static_cast<float>(d + 1) / filtration.size();
    } else {
        // Coboundary: estimate from coface count
        uint32_t cofaces = estimateCofaceCount(sigma, filtration);
        return static_cast<float>(cofaces) / filtration.size();
    }
}
```

### OpenMP Parallelism

PH4 uses OpenMP for thread-level parallelism:

```cpp
void reduceColumns() {
    #pragma omp parallel for schedule(static) num_threads(num_threads)
    for (uint32_t j = 0; j < num_columns; ++j) {
        if (cleared[j]) continue;

        // Each thread reduces its assigned column
        reduceColumn(j);
    }
}
```

The `schedule(static)` clause ensures each thread gets a fixed set of columns, enabling bitwise reproducibility. The chunk size is computed as `num_columns / num_threads`.

### Deterministic Seed Propagation

```cpp
class DeterminismManager {
    uint64_t global_seed;

public:
    DeterminismManager() {
        // Seed from high-resolution entropy source
        std::random_device rd;
        global_seed = rd();
    }

    uint64_t getThreadSeed(int thread_id) {
        // Deterministic: same thread_id always gets same seed
        // Uses splitmix64 to avoid correlation between consecutive seeds
        uint64_t z = global_seed + 0x9e3779b97f4a7c15ULL * thread_id;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
};
```

### Error Handling

PH4 categorizes errors and warnings:

```cpp
enum class PH4Error {
    // Fatal errors (computation stops)
    OUT_OF_MEMORY,
    INVALID_POINTS,
    INVALID_PARAMETER,

    // Non-fatal warnings (computation continues)
    MEMORY_NEAR_LIMIT,
    UNUSUALLY_SLOW_COLUMN,
    HIGH_PIVOT_CONFLICT_RATE,
    DEGENERATE_FILTRATION,
};

// Errors are reported through a callback or stored in the result metadata
struct PH4Warning {
    PH4Error code;
    std::string message;
    std::optional<uint32_t> column;
    double timestamp_ms;
};
```

### Thread-Local Storage Layout

Each thread has a dedicated working buffer for column XOR operations:

```
Thread-local buffer (per thread, per dimension):

  [ metadata | column_a_slice | column_b_slice | result_buffer ]

  metadata: 64 bytes (current column, pivot, status flags)
  column_a_slice: up to 8 KB (cached portion of column being reduced)
  column_b_slice: up to 8 KB (cached portion of additive column)
  result_buffer: up to 8 KB (temporary space for XOR result)

Total per thread: ~24 KB per dimension, ~72 KB for max_dim=3
```

This per-thread allocation avoids malloc/free in the inner loop, which would cause contention and degrade performance.

### Memory Allocation Strategy

PH4 uses a custom arena allocator for column storage:

```cpp
class ColumnArena {
    std::vector<std::unique_ptr<char[]>> blocks;
    size_t block_size;
    size_t current_offset;

public:
    void* allocate(size_t bytes) {
        if (current_offset + bytes > block_size) {
            // Allocate a new block (megabytes default)
            blocks.push_back(std::make_unique<char[]>(block_size));
            current_offset = 0;
        }
        void* ptr = blocks.back().get() + current_offset;
        current_offset += alignUp(bytes, 64);  // align to cache line
        return ptr;
    }

    void reset() {
        // Reuse existing blocks; just reset the offset
        current_offset = 0;
    }
};
```

The arena allocator provides:
- O(1) allocation (pointer bump)
- Zero fragmentation
- Cache-line alignment for all allocations
- Bulk deallocation (reset the arena)

For n = 10^6 columns with average size 16 bytes, the arena uses a few tens of megabytes, fitting in L3 cache.

Back to [PH4 Engine Overview](index.md)
