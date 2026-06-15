# Clearing Optimization (Chen & Kerber 2011)

The clearing optimization exploits the following observation: once a column j is paired as a death with pivot p, the boundary column of the birth simplex at row p *will never be needed again* as an additive column in any future reduction step. Therefore, column p can be *cleared* -- set to all zeros -- immediately.

### Formal Description

Standard clearing adds one line after the pairing in step 8:

```
 8.          P[pivot] <- j
 9.          R[:, pivot] <- 0      // CLEAR: birth column no longer needed
```

The clearing is safe because:
- Column j (the death) is the *only* column that had pivot = p in the reduced matrix.
- No future column k > j can have pivot = p after column j is reduced, because the pivot conflict would have been resolved by adding column j (which is now reduced and has pivot p).
- Since column p is the boundary of the simplex at row p, and the pair (p, j) is already determined, column p contributes nothing to future computations.

### Clearing Walkthrough (4x4 Example)

Continuing the 4x4 example with clearing:

**j=1:** Column 1 empty. Positive. Skip clearing (no pivot).

**j=2:** Column 2 empty. Positive. Skip clearing.

**j=3:** Column 3 empty. Positive. Skip clearing.

**j=4:** Column 4 has pivot = 2. P[2] = -1, so pair (2, 4). Set P[2] = 4. **Clear column 2:** set R[:, 2] = 0.

Impact: column 2 (which was already empty) is explicitly cleared. In this tiny example clearing does nothing, but in realistic complexes with dense boundary columns, clearing eliminates expensive column XOR operations.

### When Clearing Helps Most

The clearing benefit varies by scenario. For Vietoris-Rips complexes in dimension 1, clearing provides a 10-20% benefit, affecting mostly edge columns paired with vertices. In dimension 2, the benefit rises to 30-50%, as triangle boundaries are dense and clearing their birth edges saves many operations. Alpha complexes in dimension 2 and above see 40-60% improvement, with many high-dimensional simplices paired in rapid succession. For dense filtrations, a 20-40% benefit is typical for most real-world data.

Clearing is most effective when:
1. **Many death simplices are processed** -- each death clears a birth column that would otherwise participate in many future XOR operations.
2. **Birth columns are dense** -- clearing a dense column saves more work than clearing a sparse one.
3. **The dimension is high** -- higher-dimensional simplices have longer boundaries, making their birth columns more expensive to keep.

### Clearing Variants

**Standard clearing**: Clear the birth column immediately after pairing the death. P = pivot(j). R[:, P] = 0.

**Dimension-cascading clearing** (PH5+): After clearing column p, recursively check whether the simplex at row p was itself a death (i.e., the birth for another pair). If so, continue clearing up the chain. This is more aggressive and can eliminate up to 50% more columns, but requires careful tracking to avoid clearing columns that would violate correctness.

**Clearing with compression**: When a column is cleared, its storage (sparse list, bitset) is freed. This reduces peak memory by up to 30% beyond the runtime benefit.

<- [Standard Reduction Overview](index.md)
