# Advanced cup product

### Cohomology ring structure

The cup product endows cohomology with a ring structure. For cochains alpha in C^p and beta in C^q:

```
(alpha U beta)(sigma) = alpha(sigma|[v0..vp]) * beta(sigma|[vp..vp+q])
```

where sigma is a (p+q)-simplex and sigma|[v0..vp] is the front p-face.

```python
result = cup_product(points, max_dim=2, max_radius=2.0)

# Access ring structure
cup_table = result.cup_table           # [dim_p][dim_q] -> matrix
basis = result.cohomology_basis        # generators per dimension
ring_structure = result.ring_structure  # multiplication table

# Check associativity: (alpha U beta) U gamma == alpha U (beta U gamma)
from pynerve.specialized import check_associativity
ok = check_associativity(cup_table, basis)
```

### Multi-GPU cup product

For large complexes, the cup product distributes across GPUs:

```python
from pynerve.specialized import DistributedCupProduct

dist = DistributedCupProduct(num_gpus=4)
result = dist.compute(points, max_dim=3, max_radius=3.0)
# Each GPU computes a block of the cup product table
# Results merged via all-reduce
```


[Back to index](index.md)
