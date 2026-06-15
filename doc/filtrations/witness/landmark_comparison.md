# Landmark selection comparison

```python
import time
from pynerve.nn._building_blocks_persistence import WitnessComplexPersistence

points = torch.randn(50000, 3)
n_landmarks = 500

for method in ["farthest", "random", "kmeans"]:
    t0 = time.time()
    wc = WitnessComplexPersistence(
        n_landmarks=n_landmarks,
        max_dim=2,
        method=method,
        random_seed=42,  # for reproducibility
    )
    diagram = wc(points)
    elapsed = time.time() - t0
    print(f"{method}: {elapsed:.2f}s, {diagram.mask.sum().item()} pairs")
```

### Evaluating approximation quality

To validate that the witness approximation is adequate:

1. Compute witness persistence with increasing $m$ (e.g., 100, 200, 500).
2. Compute exact VR on a random subset of the data.
3. Compare bottleneck distances between diagrams.
4. Stop when increasing $m$ no longer changes the diagram significantly.

<- [Witness Complex Overview](index.md)
