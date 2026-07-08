# Common Pitfalls

### Pitfall 1: Enabling ALL features at once

Turning on checksum validation, stability checks, PARANOID determinism, and differentiable tracking simultaneously adds 50-100% overhead. Select only the features you need:

```python
# Overkill: 3x slowdown
config = PH5PH6Config(
    enable_checksum_validation=True,
    require_bitwise_reproducibility=True,
    enable_stability_checks=True,
)

# Minimal for production use
config = PH5PH6Config(enable_checksum_validation=True)
```

### Pitfall 2: Expecting bitwise reproducibility across hardware

STRICT and PARANOID determinism guarantee reproducibility on the *same hardware*. Different CPUs may produce different results due to:
- Different SIMD instruction set extensions (AVX-512 vs AVX2)
- Different FMA behavior (fused multiply-add is not IEEE 754 compliant)
- Different transcendental function implementations

PH5 normalizes floating-point operations to the extent possible, but cross-hardware reproducibility requires fixed-point arithmetic, which is not currently supported.

### Pitfall 3: Checksum mismatch due to metadata changes

The checksum includes library version and configuration strings. Upgrading the library will change the checksum even if the mathematical result is identical. Store checksums with the library version:

```python
expected = {
    "checksum": "abc123...",
    "library_version": "1.2.3",
    "config": "max_dim=2,max_radius=0.5"
}
```

### Pitfall 4: Gradient instability with near-degenerate filtrations

When multiple simplices have nearly identical filtration values, the pairing is unstable and gradients may not be meaningful. Use `error_tolerance` to group nearly-equal values:

```python
# Without tolerance: unstable pairing near degeneracies
result = pynerve.compute_persistence_up_to_dim_5(points, error_tolerance=0.0)

# With tolerance: stable pairing, meaningful gradients
result = pynerve.compute_persistence_up_to_dim_5(points, error_tolerance=1e-8)
```

### Pitfall 5: Stability test flakiness on multi-GPU systems

Stability tests that run across multiple GPUs may fail due to different GPU clock speeds, thermal throttling, or PCIe bus contention. PIN specific GPUs for stability testing:

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use a single GPU

engine = PH5PH6Engine()
engine.runStabilityTest(points, max_dimension=2, num_runs=10)
```

Back to [PH5 Engine Overview](index.md)
