# Common Pitfalls

### Pitfall 1: Integer Overflow in Pivot Storage

Using a signed 32-bit integer for pivot storage limits n to 2^31. For large complexes, use uint32_t or size_t. The library handles this automatically, but custom implementations may overflow.

**Fix:** Use `uint32_t` for pivot storage (supports up to ~4 billion simplices) or `size_t` for unbounded support.

### Pitfall 2: Assuming O(n^2) Always

The standard algorithm has worst-case O(n^3) complexity. For adversarial filtrations (e.g., a full simplex on 1000 vertices: n ~ 10^30, impossible; but a moderately dense complex can still trigger cubic behavior), runtime can spike unexpectedly.

**Fix:** Monitor the number of XOR operations per column. If a column exceeds `10 * average_column_ops`, consider switching to cohomology reduction or enabling approximate mode.

### Pitfall 3: Ignoring Column Density Growth

As reduction proceeds, columns tend to become *denser*, not sparser. Each XOR merges two columns, potentially increasing the non-zero count. This is the opposite of what one might intuitively expect.

**Mitigation:** Use clearing to periodically eliminate dense birth columns. If clearing is not available, prioritize processing columns that are likely deaths (they clear their births).

### Pitfall 4: Incorrect Simplex Ordering

The standard algorithm requires simplices to be ordered by:
1. Filtration value (non-decreasing)
2. Dimension (non-decreasing)

Violating this order produces incorrect persistence pairs. The library handles simplex ordering internally when given point cloud data, but custom filtrations must ensure correct ordering.

### Pitfall 5: Non-Z2 Coefficients

The standard reduction as described here works only for Z2 coefficients. For other fields, the XOR operation becomes addition modulo p, and the pivot elimination requires multiplying by the inverse of the pivot coefficient:

```python
# For field Z_p, pivot elimination becomes:
coeff = R[pivot, j]  # the pivot coefficient
inv = pow(coeff, -1, p)  # modular inverse
# Multiply column k by inv and add to column j
R[:, j] = (R[:, j] - coeff * R[:, k]) % p
```

This adds O(1) overhead per XOR but requires modular arithmetic.

### Pitfall 6: Memory Blowup in Dense Representation

Storing the boundary matrix as a dense n x n array of bytes requires n^2 bytes. For n = 10^5, this is around ten gigabytes -- likely causing OOM. Always use sparse column storage for n > 10^4.

<- [Standard Reduction Overview](index.md)
