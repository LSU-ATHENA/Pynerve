# Reduction algorithms

Pynerve supports multiple reduction algorithms for the VR filtration,
each with different performance characteristics:

Pynerve supports multiple reduction algorithms, each with different characteristics. Standard (PH4) is best for sparse filtrations, with $O(m^3)$ worst-case time and $O(m^2)$ memory. Clearing (the default) targets dense filtrations with the same complexity. Cohomology excels at high-dimensional homology with $O(m^2)$ time and $O(m)$ memory. Twist handles large sparse matrices with $O(m^3)$ time and $O(m^2)$ memory. Spectral sequence is designed for sequential data with $O(m^3)$ time and $O(m^2)$ memory.

The clearing optimization (default) zeroes out columns that are paired
during reduction, skipping their further processing. This gives a 2-5x
speedup over standard reduction for typical VR filtrations.

### Selecting the reduction algorithm

```python
from pynerve.nn import PersistentHomology

# Clearing (default, good general purpose)
ph = PersistentHomology(max_dim=2, reduction="clearing")

# Cohomology (faster for H1 and above)
ph = PersistentHomology(max_dim=2, reduction="cohomology")

# Standard (no optimization)
ph = PersistentHomology(max_dim=2, reduction="standard")
```


<- [Vietoris-Rips Overview](index.md)
