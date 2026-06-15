# Discrete Morse Theory

## Quick start

```python
import pynerve.dmt as dmt

simplices = [[0], [1], [2], [0,1], [1,2], [0,2], [0,1,2]]
filtration = [0.0, 0.0, 0.0, 1.0, 1.5, 2.0, 3.0]

result = dmt.compute_gradient(simplices, filtration)
reduced = dmt.reduce_complex(simplices, filtration, result)
```

Discrete Morse Theory reduces a simplicial complex to a homotopy-equivalent
complex with far fewer cells (10-100x smaller). Run DMT before persistence
to drastically reduce computation time.


## Pages

- [gradient_field.md](gradient_field.md) -- computation, critical cells
- [gpu.md](gpu.md) -- CUDA DMT kernel
- [parallel.md](parallel.md) -- parallel Morse pair finding


## Core algorithm

The DMT gradient field is a matching between simplices of adjacent dimensions.
A pair (s, t) with dim(s) = k and dim(t) = k+1 is a gradient pair if s is a
face of t (s < t) and the pairing is acyclic.

**Algorithm:**
1. For each simplex, find candidate cofaces that differ by one vertex
2. Sort candidates by filtration value (ascending)
3. Greedy: pair an unmatched simplex with its best unmatched coface
4. Unpaired simplices are critical cells

The reduced complex is built from the critical cells plus gradient path
information, which preserves homotopy type.


## DMTEngine

```cpp
namespace nerve::dmt {

struct DMTConfig {
    int max_dimension = 2;
    bool use_gpu = false;
    bool use_parallel = true;
    bool use_simd = true;
    int num_threads = 0;
    bool use_priority_queue = true;
};

struct MorseResult {
    std::vector<int> critical_simplices;
    std::vector<std::pair<int, int>> gradient_pairs;
    double computation_time_ms = 0.0;
};

class DMTEngine {
public:
    explicit DMTEngine(const DMTConfig& config = {});
    ~DMTEngine();

    MorseResult computeMorseComplex(
        const std::vector<std::vector<int>>& simplices,
        const std::vector<float>& filtration);

    std::vector<int> findCriticalPoints() const;
};

}
```

**Example:**
```cpp
DMTEngine engine(DMTConfig{.max_dimension = 2, .use_parallel = true});
MorseResult result = engine.computeMorseComplex(simplices, filtration);
```


## When to use DMT

**Scenario-based recommendations:** Dense 2D VR (500 pts) achieves 5-10x reduction, worth it for repeated computation. Dense 3D VR (1000 pts) achieves 10-50x reduction -- recommended. High-dim VR (100 pts, dim=5) achieves 20-100x reduction, strongly recommended. For small complexes (<100 simplices), reduction is only 1-2x and overhead dominates -- skip DMT. Streaming / out-of-core achieves 10-100x reduction, use StreamingMorseBuilder. On GPU with CUDA, expect 10-50x reduction for large complexes.


## Complexity notes

**Complexity costs (m = number of simplices):** Gradient pair finding (sequential) is O(m * f) where f = average coface count. Gradient pair finding (parallel) is O(m * f / p) with OpenMP. SIMD subset check is O(k / SIMD_width) per pair. BFS traversal is O(gradient_pairs * avg_degree). GPU gradient computation is O(m * f / CUDA_cores). Memory is O(m) for gradient array plus O(pairs) for result.

Typical m = number of simplices. For VR on n points with max_dim = 2,
m = O(n^3) worst case, but DMT typically reduces to O(n^2) or less.



## Practical guidance

### When DMT is most effective

**Effectiveness by complex type:** VR complex (dense 2D, 500 pts) with ~42k simplices gives 5-10x reduction -- moderate benefit. VR complex (dense 3D, 1000 pts) with ~167k simplices gives 10-50x -- high benefit. High-dim VR (100 pts, dim=5) with ~100M simplices gives 20-100x -- critical benefit. Alpha complex (3D, 10k pts) with ~200k simplices gives 3-5x -- moderate benefit. Cubical complex (2D image) with ~1M cells gives 10-100x -- high benefit. Random point cloud gives 1-2x -- low benefit (no structure).

### Common pitfalls

1. **DMT overhead for small complexes**: For complexes with <100 simplices, the gradient pair finding overhead exceeds the reduction benefit. Always check the reduction factor before using DMT in a pipeline.
2. **Acyclicity violation**: The greedy matching may produce cycles if the filtration has equal values for adjacent simplices. Add a small epsilon to break ties consistently.
3. **Memory in parallel mode**: Each thread maintains its own candidate queue. With `num_threads=48`, memory usage can spike. Set `use_priority_queue=false` to bound memory.
4. **GPU overhead for small data**: The PCIe transfer for complexes <50k simplices makes GPU slower than CPU. Use GPU only for large complexes (>50k simplices).

### Integration with persistence pipeline

```python
from pynerve.dmt import DMTEngine
from pynerve.persistence import compute_persistence

# DMT pre-processing
cfg = DMTConfig(max_dimension=2, use_parallel=True)
engine = DMTEngine(cfg)
result = engine.computeMorseComplex(simplices, filtration)

# Build reduced complex from critical cells
reduced_simplices = [simplices[i] for i in result.critical_simplices]
reduced_filtration = [filtration[i] for i in result.critical_simplices]

# Compute persistence on reduced complex (10-100x faster)
pairs = compute_persistence(reduced_simplices, reduced_filtration)
```

### Verification: DMT preserves homotopy

```python
from pynerve.dmt import verify_homotopy_equivalence

# Verify that the reduced complex is homotopy-equivalent
ok, error = verify_homotopy_equivalence(
    original_simplices, original_filtration,
    result.critical_simplices, result.gradient_pairs,
)

if ok:
    print("Homotopy equivalence verified")
else:
    print(f"Verification failed: {error}")
```


## Advanced topics

### Gradient path visualization

```python
from pynerve.dmt import visualize_gradient_field

# Visualize gradient pairs on a 2D complex
visualize_gradient_field(
    simplices, result.gradient_pairs,
    show_critical=True,  # highlight critical cells
    show_paths=True,     # draw gradient flow lines
)
```

### Custom pairing strategy

```cpp
class MyPairingStrategy : public PairingStrategy {
public:
    std::vector<std::pair<int, int>> findPairs(
        const std::vector<std::vector<int>>& simplices,
        const std::vector<float>& filtration) override {

        // Custom: pair by geometric distance instead of filtration
        std::vector<std::pair<int, int>> pairs;
        // ... custom logic ...
        return pairs;
    }
};

DMTEngine engine(cfg);
engine.setPairingStrategy(std::make_shared<MyPairingStrategy>());
```


## FAQ

**Q: Does DMT guarantee homotopy equivalence?**
A: Yes, when the gradient field is acyclic (no directed cycles). The greedy matching strategy used by DMTEngine is provably acyclic for generic filtrations (all distinct filtration values).

**Q: Can DMT handle weighted complexes?**
A: Yes. The gradient field is computed from filtration values. Edge weights propagate to higher-dimensional simplices via the maximum weight of constituent edges (standard VR rule).

**Q: What is the memory overhead of DMT?**
A: DMT stores one gradient pair per simplex (~8 bytes per simplex) plus the pair list. For a complex with 1M simplices, expect tens of megabytes of additional memory.

**Q: How does DMT compare to apparent pairs?**
A: Apparent pairs are a subset of DMT gradient pairs that can be detected directly (in O(n) time) without the full greedy matching. DMT finds more pairs overall (10-50% more reduction), but apparent pairs are faster. Use apparent pairs when speed matters more than reduction, DMT when reduction matters more.


### Cross-references

- `pynerve.algebra`: Simplicial complex used by DMT
- `pynerve.persistence`: Persistence on DMT-reduced complex
- `pynerve.core.thread_pool`: Parallel DMT uses thread pool
- `pynerve.cuda`: GPU DMT kernel
- `pynerve.validation.benchmarks`: DMT benchmark suite
