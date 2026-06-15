# Reproducibility Checklist & Reference Results

## Reproducibility checklist

Before publishing results, verify:

- [ ] Same Pynerve version (`pynerve.__version__`)
- [ ] Same seed (`pynerve.compute_persistence(..., seed=X)`)
- [ ] Same options (`max_dim`, `max_radius`, `mode`, `backend`)
- [ ] Same input data (bit-identical floating point)
- [ ] Same hardware architecture (or enable RFA for cross-GPU)
- [ ] Same MPI rank count (or use binned accumulation for cross-count)

If any of these differ, results may not be bitwise identical.

### Environment capture

```python
import pynerve
import json

env = {
    "nerve_version": pynerve.__version__,
    "seed": None,  # set via seed= kwarg
    "options": {
        "max_dim": opts.max_dim,
        "max_radius": opts.max_radius,
        "mode": str(opts.mode),
        "backend": str(opts.backend),
    },
    "gpu_info": None,  # populated below
}

# Capture GPU info if available
try:
    import torch
    if torch.cuda.is_available():
        env["gpu_info"] = {
            "name": torch.cuda.get_device_name(),
            "capability": torch.cuda.get_device_capability(),
        }
except ImportError:
    pass

# Save for reproducibility
with open("nerve_environment.json", "w") as f:
    json.dump(env, f, indent=2)
```


## Reference results

Pynerve includes a set of reference results for validation:

The Uniform circle test uses 100 points in 2 dimensions with max_dim=1 and max_radius=1.0, expecting H0=1 and H1=1 at seed 42. The Uniform sphere test uses 200 points in 3 dimensions with max_dim=2 and max_radius=1.5, expecting H0=1 and H1=0. The Torus test uses 500 points in 3 dimensions with max_dim=2 and max_radius=2.0, expecting H0=1 and H1=2. The Figure-8 test uses 100 points in 2 dimensions with max_dim=1 and max_radius=1.0, expecting H0=1 and H1=2. The Cluster test with 3 clusters uses 300 points in 2 dimensions with max_dim=1 and max_radius=0.5, expecting H0=3 and H1=0. All tests use seed 42.

These can be verified with:

```python
import pynerve
import numpy as np

def verify_reference(name, points, max_dim, max_radius, expected_h0, expected_h1):
    result = pynerve.compute_persistence(
        points, max_dim=max_dim, max_radius=max_radius,
    )
    h0 = result.betti_numbers[0]
    h1 = result.betti_numbers[1] if len(result.betti_numbers) > 1 else 0
    assert h0 == expected_h0, f"{name}: expected H0={expected_h0}, got {h0}"
    assert h1 == expected_h1, f"{name}: expected H1={expected_h1}, got {h1}"
    print(f"PASS: {name}")
```


[Back to Correctness Index](index.md)
