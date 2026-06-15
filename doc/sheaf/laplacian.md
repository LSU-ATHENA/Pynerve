## Sheaf Laplacian

Higher-order Hodge Laplacian parameterized by stalk data. Constructs the
sheaf Laplacian from node attributes, edge weights, and restriction maps.

```python
config = pynerve.sheaf.SheafConfig(
    num_attributes=4,
    sheaf_type="product",
    enable_gpu=False,
    enable_stability_certificates=True,
)
lap = pynerve.sheaf.SheafLaplacian(config)

lap.add_node(id=0, position=[0.0, 0.0], attributes=[1.0, 2.0, 3.0, 4.0],
             neighbors=[1, 2], edge_weights=[1.0, 1.0])
lap.add_nodes([...])

result = lap.build_sheaf_laplacian()
```

### Sheaf type variants

- `build_product_sheaf()` -- L = L_topology + L_attributes
- `build_cosheaf()` -- L = B^T B (coboundary^2)
- `build_generalized_sheaf()` -- L = D - A with stalk-aware weights

### Result fields

- `result.sheaf_laplacian` -- Eigen::SparseMatrix<double>
- `result.attribute_laplacian` -- attribute-only block
- `result.topological_laplacian` -- topology-only block
- `result.spectral_radius, trace, frobenius_norm`
- `result.attribute_influence, topological_influence`
- `result.isStable`

### Eigenpairs

```python
eigen = lap.compute_eigenpairs(num_eigenpairs=50, tolerance=1e-8)
# eigen.eigenvalues, eigen.eigenvectors
# eigen.attribute_contributions per eigenpair
```

### Numerical stability

- `is_stable()` -- Check numerical stability
- `get_numerical_residual()` -- Return ||L - L_expected||
- `get_stability_certificate()` -- Full stability report
- `validate_sheaf_structure()` -- Validate dimensions, edge weights


## Sheaf Laplacian variants

### Product sheaf Laplacian

```python
L = lap.build_product_sheaf()
# L = L_topological + L_attribute
# L_topological = standard graph Laplacian (block-diagonal)
# L_attribute = attribute-based coupling
```

### Cosheaf Laplacian

```python
L = lap.build_cosheaf()
# L = B^T B where B is the coboundary with stalk structure
# Useful for diffusion problems where topology constrains attributes
```

### Generalized sheaf Laplacian

```python
L = lap.build_generalized_sheaf()
# L = D - A with stalk-aware weights
# D[i] = sum_j F_ij^T F_ij
# A[i,j] = F_ij^T F_ji (if both maps exist)
```

## Eigenvalue analysis

```python
eigen = lap.compute_eigenpairs(num_eigenpairs=50, tolerance=1e-8)

# Analyze eigenvalues
print("Eigenvalue spectrum:")
for i, val in enumerate(eigen.eigenvalues[:10]):
    contrib = eigen.attribute_contributions[i]
    print(f"  lambda_{i} = {val:.6f} (attr contribution: {contrib:.3f})")

# Compute spectral gap
gap = eigen.eigenvalues[1] - eigen.eigenvalues[0]
print(f"Spectral gap: {gap:.6f}")

# Count near-zero eigenvalues (harmonic sections)
harmonic_count = sum(1 for v in eigen.eigenvalues if abs(v) < 1e-8)
print(f"Harmonic sections: {harmonic_count}")
```

## Numerical stability

```python
# Check stability
if not lap.is_stable():
    print("Warning: Sheaf Laplacian may be numerically unstable")
    print(f"Residual: {lap.get_numerical_residual():.2e}")
    cert = lap.get_stability_certificate()
    print(f"Condition number: {cert.condition_number:.2e}")
```

The stability certificate checks:
1. All restriction maps are full rank
2. No stalk is assigned zero dimension
3. Edge weights are non-negative
4. Laplacian is symmetric positive semi-definite

## When to use each sheaf type

- **Product**: L = L_topo + L_attr, best for decoupled node attributes
- **Cosheaf**: L = B^T B, best for attribute diffusion on complex
- **Generalized**: L = D - A, best for general edge-weighted sheaf


## FAQ

**Q: What is the difference between product sheaf, cosheaf, and generalized sheaf Laplacians?**
A: The product sheaf adds topological and attribute Laplacians independently (L = L_topo + L_attr), suitable when attributes do not interact with topology. The cosheaf squares the coboundary operator (L = B^T B), useful when topology constrains attribute diffusion. The generalized sheaf uses stalk-aware degree and adjacency (L = D - A), designed for arbitrary edge-weighted sheaf structures.

**Q: When should I use iterative eigensolvers vs direct computation?**
A: Use iterative eigensolvers (Lanczos) when you need only a few eigenpairs and the Laplacian is large. Direct computation is feasible for small sheaves (n * d < 5000) or when the full spectrum is required.

**Q: How do I check if my sheaf Laplacian is numerically stable?**
A: Call `is_stable()` for a quick check or `get_stability_certificate()` for a full report including condition number and spectral radius. Ensure all restriction maps are full rank, no stalk has zero dimension, and edge weights are non-negative.


### Cross-references

- `pynerve.sheaf`: Sheaf module overview
- `pynerve.spectral.laplacian`: Graph Laplacian (non-sheaf)
- `pynerve.sheaf.learning`: Learning restriction maps
- `pynerve.sheaf.gpu`: GPU-accelerated Laplacian construction
