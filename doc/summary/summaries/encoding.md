# Detail: field encoding

### Timestamp

The `timestamp_ns` field stores nanoseconds since Unix epoch. For cross-platform portability, use UTC. Convert from Python datetime:

```python
from datetime import datetime, timezone
ns = int(datetime.now(timezone.utc).timestamp() * 1e9)
```

### Symbol ID

The `symbol_id` field is an application-defined identifier. Use cases:
- Data source identifier (e.g., sensor ID)
- Experiment run index
- Cross-reference key for database lookup

### Params hash

The `params_hash_low` and `params_hash_high` fields store a 64-bit hash of the computation parameters (max_dim, max_radius, seed, etc.). This allows detecting when summaries were computed with different parameters:

```python
from pynerve.summary import compute_params_hash

hash_low, hash_high = compute_params_hash(
    max_dim=2, max_radius=1.0, seed=42
)
```


[Back to index](index.md)
