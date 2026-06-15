## Sheaf Morphisms

Maps between sheaves -- chain maps commuting with restriction.

```python
from pynerve.sheaf.morphism import SparseMorphism, BatchedMorphismComputer

m = SparseMorphism()
m.apply(input_stalk, output_stalk)           # CPU path
m.apply_simd(input_stalk, output_stalk)      # AVX-512/AVX2 path
```

### Batched morphisms

```python
computer = BatchedMorphismComputer()
computer.add_morphism(from_sheaf, to_sheaf, morphism_matrix)
computer.compute_batch(inputs, outputs)
computer.compute_batch_simd(inputs, outputs)  # SIMD batched
```

### Composition optimizer

Shortest path in morphism DAG.

```python
from pynerve.sheaf.morphism import MorphismCompositionOptimizer
opt = MorphismCompositionOptimizer()
opt.register_chain([sheaf_a, sheaf_b, sheaf_c])
composed = opt.get_composed(sheaf_a, sheaf_c)
```

### Async morphism queue

```python
from pynerve.sheaf.morphism import AsyncMorphismQueue
queue = AsyncMorphismQueue()
queue.start(num_workers=4)
future = queue.submit(from_stalk, to_stalk, input_data)
result = future.get()
queue.stop()
```


## Morphism composition

Sheaf morphisms are chain maps: collections of linear maps f_k: C^k(S) -> C^k(T) that commute with the coboundary operators:

```
f_{k+1} * d_S^k = d_T^k * f_k
```

### Composition DAG

```python
# Build a DAG of morphisms for composition
from pynerve.sheaf.morphism import MorphismCompositionOptimizer

opt = MorphismCompositionOptimizer()

# Register morphisms between sheaves
opt.register_morphism("S -> A", morphism_sa)
opt.register_morphism("A -> B", morphism_ab)
opt.register_morphism("S -> B", morphism_sb)  # direct
opt.register_morphism("B -> C", morphism_bc)

# Find shortest composition path
# (minimizes total computation cost)
path = opt.find_shortest_path("S", "C")
# Returns ["S -> A", "A -> B", "B -> C"] if cheaper than ["S -> B", "B -> C"]

composed = opt.get_composed("S", "C")
```

### Composition cost model

The optimizer estimates cost based on:
- Stalk dimensions at each degree
- Sparsity of morphism matrices
- Number of intermediate compositions
- GPU vs CPU execution

```python
cost = opt.estimate_composition_cost(sheaf_a, sheaf_b, sheaf_c)
print(f"Direct composition cost: {cost.direct:.3f}")
print(f"Optimized path cost: {cost.optimized:.3f}")
print(f"Savings: {(1 - cost.optimized/cost.direct)*100:.1f}%")
```

## SIMD-accelerated apply

```python
# SIMD batch apply
from pynerve.sheaf.morphism import simd_apply_morphism

# Process multiple inputs in parallel
inputs = np.random.randn(1024, stalk_dim).astype(np.float64)
outputs = np.zeros_like(inputs)

m.apply_simd(inputs, outputs)
# Uses AVX-512 FMA for matrix-vector multiply
# 8x speedup for f64, 16x for f32
```

## Batched morphism pipeline

```python
from pynerve.sheaf.morphism import BatchedMorphismComputer

computer = BatchedMorphismComputer()

# Register multiple morphisms
computer.add_morphism(from_sheaf_A, to_sheaf_B, morphism_matrix)
computer.add_morphism(from_sheaf_B, to_sheaf_C, morphism_matrix)

# Compute all morphisms in one batch (GPU-efficient)
batch_inputs = torch.randn(256, total_stalk_dim)
batch_outputs = computer.compute_batch(batch_inputs)
# Also available: compute_batch_simd for CPU path
```

## Async processing

```python
from pynerve.sheaf.morphism import AsyncMorphismQueue

queue = AsyncMorphismQueue()
queue.start(num_workers=4)

# Submit morphism applications
futures = []
for i in range(100):
    future = queue.submit(from_sheaf, to_sheaf, input_data)
    futures.append(future)

# Collect results
results = [f.get() for f in futures]

queue.stop()
```

The async queue uses an internal thread pool with work-stealing for load balancing across morphism applications.

## Composition validation

```python
from pynerve.sheaf.morphism import validate_morphism_composition

# Verify that composition commutes with coboundary
valid, error = validate_morphism_composition(
    sheaf_S, sheaf_T, sheaf_U,
    morphism_ST, morphism_TU,
)

if valid:
    print("Composition is a valid chain map")
else:
    print(f"Composition error: {error}")
    # f_{k+1} * d_S^k != d_T^k * f_k at some degree
```


## FAQ

**Q: How does the composition optimizer decide the best path?**
A: The optimizer models composition as a shortest-path problem in a morphism DAG. It estimates cost based on stalk dimensions, matrix sparsity, number of intermediate compositions, and whether execution is on GPU or CPU, then finds the minimum-cost path.

**Q: When should I use SIMD vs async vs batched morphism apply?**
A: Use SIMD for single morphism applications on CPU where the stalk dimension fits vector registers. Use batched mode for applying many morphisms at once (GPU-efficient). Use the async queue when morphism workloads are irregular and you want load-balanced parallelism across a thread pool.

**Q: What is the cost model for composition and how do I inspect it?**
A: `estimate_composition_cost()` returns direct and optimized path costs as well as percentage savings. The model accounts for stalk dimensions at each degree, sparsity of morphism matrices, number of compositions, and the execution backend.


### Cross-references

- `pynerve.sheaf`: Sheaf module overview
- `pynerve.sheaf.learning`: Learning morphisms from data
- `pynerve.core.thread_pool`: Thread pool for async operations
- `pynerve.nn.simd`: SIMD operations for matrix apply
