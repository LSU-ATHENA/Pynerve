# Advanced Configuration Examples

## Example: Maximum Performance (Research)

```python
config = PH5PH6Config()
config.experimental_reduction_ordering = "adaptive"
config.experimental_clearing = "approximate"
config.experimental_speculative_threads = 4
config.experimental_adaptive_pivoting = True
config.experimental_block_size = 4096
config.experimental_verify_against_ph4 = False

result = pynerve.compute_persistence_up_to_dim_6(points, max_dim=2, max_radius=0.5)
```

## Example: Maximum Correctness (Validation)

```python
config = PH5PH6Config()
config.experimental_reduction_ordering = "default"  # stable ordering
config.experimental_clearing = "exact"  # no approximations
config.experimental_speculative_threads = 1  # no speculation
config.experimental_adaptive_pivoting = True  # safe optimization
config.experimental_block_size = 0  # auto
config.experimental_verify_against_ph4 = True  # cross-check
config.enable_checksum_validation = True
config.require_bitwise_reproducibility = True

result = pynerve.compute_persistence_up_to_dim_6(points, max_dim=2)
```

## Example: Memory-Constrained

```python
config = PH5PH6Config()
config.experimental_reduction_ordering = "default"
config.experimental_clearing = "exact"
config.experimental_speculative_threads = 1  # no memory multiplier
config.experimental_adaptive_pivoting = True
config.experimental_block_size = 2048  # smaller blocks = less peak memory
config.experimental_verify_against_ph4 = False  # no redundant copy

opts = PersistenceOptions()
result = pynerve.compute_persistence_up_to_dim_6(points, opts, max_dim=2)
```


[Back to PH6 Index](index.md)
