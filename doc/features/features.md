# Features

## Quick start

```python
from pynerve.features import DiagramFeatures

# Wrap a diagram tensor and access features
diagram = compute_persistence(points, max_dim=2)
features = DiagramFeatures(diagram)

# Filter by dimension
dim1 = features.by_dimension(1)
# -> DiagramFeatures with only dimension-1 pairs

# Filter by persistence threshold
persistent = features.where(persistence > 0.5)
# -> DiagramFeatures with only high-persistence pairs

# Iterate over pairs
for birth, death, dim in features:
    print(f"dim={dim} birth={birth:.3f} death={death:.3f}")
```

## Basic usage

```python
import pynerve
result = pynerve.compute_persistence(points, max_dim=2)
features = pynerve.features.DiagramFeatures(result.pairs)

# Filter by dimension
dim_1 = features.by_dimension(1)

# Filter by persistence threshold
persistent = features.where(lambda p: (p.death - p.birth) > 0.5)

# Count features in range
count = features.count_in_range(0.0, 0.5)
```

## API

```python
from pynerve.features import DiagramFeatures

class DiagramFeatures:
    def __init__(self, diagrams: np.ndarray | Tensor): ...

    # Accessors
    @property
    def births(self) -> np.ndarray: ...
    @property
    def deaths(self) -> np.ndarray: ...
    @property
    def dimensions(self) -> np.ndarray: ...
    @property
    def persistence(self) -> np.ndarray: ...

    # Filtering
    def by_dimension(self, dim: int) -> DiagramFeatures: ...
    def where(self, condition: np.ndarray | Tensor | Callable) -> DiagramFeatures: ...
    def by_persistence(self, min_pers: float = 0.0,
                       max_pers: float = inf) -> DiagramFeatures: ...
    def by_birth(self, min_b: float = -inf,
                 max_b: float = inf) -> DiagramFeatures: ...
    def by_death(self, min_d: float = -inf,
                 max_d: float = inf) -> DiagramFeatures: ...
    def count_in_range(self, lo: float, hi: float) -> int: ...

    # Iteration
    def __iter__(self): ...        # yields (birth, death, dim)
    def __len__(self) -> int: ...
    def __getitem__(self, idx): ...

    # Conversion
    def to_numpy(self) -> np.ndarray: ...
    def to_tensor(self) -> Tensor: ...
    def as_dict(self) -> dict: ...  # {"births": ..., "deaths": ..., "dimensions": ...}
    def pairs(self) -> list: ...
```

### Chaining

Every filter returns a new `DiagramFeatures` instance, enabling complex queries:

```python
top_long_living = (
    DiagramFeatures(diagram)
    .by_dimension(1)
    .by_persistence(min_pers=0.1)
)
# -> only dim-1 pairs with death - birth > 0.1
```

### Underlying container

The `PersistenceDiagram` class in `pynerve.torch._diagram` provides the batched tensor representation with mask-based access:

```python
class PersistenceDiagram:
    @property
    def diagrams(self) -> Tensor: ...
    @property
    def mask(self) -> Tensor: ...
```

Use `DiagramFeatures` for ad-hoc analysis and filtering; use `PersistenceDiagram` when working with batched tensors inside PyTorch modules.

## FAQ

**Q: How do I extract features from a computation result?**
A: Pass the `result.pairs` list to `DiagramFeatures`. Each pair is a `(birth, death, dim)` tuple that the iterator unwraps into named attributes.

**Q: Can I chain multiple filters?**
A: Yes. Every filter returns a new `DiagramFeatures` instance, so you can chain `.by_dimension(dim).where(predicate)` for complex queries.

**Q: How does `count_in_range` handle edge cases?**
A: The range is inclusive on both ends. Pairs with persistence exactly equal to `lo` or `hi` are counted. Returns 0 if no pairs fall in the range.
