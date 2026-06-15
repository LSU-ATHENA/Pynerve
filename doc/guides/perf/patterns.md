# Common performance patterns

### Batch processing

Reuse options across multiple runs to avoid rebuild overhead:

```python
opts = pynerve.PersistenceOptions(
    max_dim=2,
    backend=pynerve.PersistenceBackend.CUDA_HYBRID,
)

for batch in batches:
    result = pynerve.compute_persistence(batch, opts)
```

### Thread scaling

Thread scaling varies by problem size. For n of 10,000 with dim=3, a single core gives 1x baseline, 4 cores give roughly 2.5 times speedup, 8 cores give 4 times, 16 cores give 5.5 times, and 32 cores give 6 times. For n of 100,000 the scaling is better: 4 cores give 3.2 times speedup, 8 cores give 5.5 times, 16 cores give 8 times, and 32 cores give 9.5 times.

Scaling is limited by memory bandwidth for distance computation and by column reduction serial dependencies. Beyond 16 cores, batch parallelism (multiple filtrations in parallel) provides better scaling than single-filtration parallelism.

### Memory tips

1. **Use float32 for distance matrix** when float64 precision is not needed -> 2x memory savings
2. **Set max_radius tightly** -- reduces the number of edges in VR construction
3. **Use sparse mode for n > 50,000** -- memory savings dominate
4. **Avoid host<->device transfers** by keeping PyTorch tensors on GPU
5. **Set threads = number of physical cores** -- hyperthreads rarely help
6. **Enable hugepages** for large arrays (automatic but requires kernel config)
7. **Use `max_dim=1`** for initial exploration -- reduces boundary matrix size drastically
8. **Prefer cohomology** for sparse filtrations -- 2-5x faster than standard reduction

[Back to index](index.md)
