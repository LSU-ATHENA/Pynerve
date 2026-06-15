# Algorithm

[Back to Index](index.md)

### Pseudocode

```
Algorithm: COHOMOLOGY_REDUCTION(D, n)
Input: boundary matrix D, number of simplices n
Output: persistence pairs

 1.  delta <- coboundary matrix = transpose(D)
 2.  R <- copy of delta
 3.  pivotOf <- array of length n, initialized to -1
 4.  pairs <- empty list

 5.  // Process columns in REVERSE filtration order
 6.  for j = n down to 1:
 7.      d <- dim(simplex j)
 8.
 9.      // Eliminate pivot conflicts in dimension d+1
10.      p <- findCoboundaryPivot(R, j)
11.
12.      while p != NONE and pivotOf[p] != -1:
13.          k <- pivotOf[p]
14.          R[:, j] <- R[:, j] XOR R[:, k]    // add column k to column j
15.          p <- findCoboundaryPivot(R, j)
16.
17.      if p != NONE:
18.          pivotOf[p] <- j
19.          // Record pair: birth = j, death = p
20.          // (Note: birth/death semantics are inverted from standard reduction)
21.          pairs.append((p, j))
22.
23.  return pairs
```

Key differences from standard reduction:
1. The matrix is the coboundary (transpose of boundary).
2. The loop direction is decreasing (n down to 1).
3. The pivot dimension check is specific: we only care about pivots in the dimension d+1 (because coboundary of a d-simplex produces (d+1)-simplices).

### Emergent Pair Detection

Because we process in reverse order, a d-simplex sigma_j may be the *oldest* coface of a (d-1)-simplex sigma_i that has not yet been processed. In this case:

```
if coboundary(sigma_j) is exactly {sigma_i} for some unprocessed sigma_i:
    pair (i, j) is determined without any reduction
```

This is called an *emergent pair* in the de Silva-Morozov framework. It is detected in O(1) time by checking the coboundary column size. For sparse filtrations, emergent pairs account for 20-40% of all pairs, meaning 20-40% of columns require zero reduction steps.

### Detailed Walkthrough

Consider the same 4-simplex complex from the standard reduction documentation:

    sigma_1 = [0] (vertex, dim 0, t=0)
    sigma_2 = [1] (vertex, dim 0, t=0)
    sigma_3 = [2] (vertex, dim 0, t=0)
    sigma_4 = [0,1] (edge, dim 1, t=1)

The coboundary matrix (transpose of boundary):

    coboundary(sigma_1) = {sigma_4}  -- vertex 0 is in edge [0,1]
    coboundary(sigma_2) = {sigma_4}  -- vertex 1 is in edge [0,1]
    coboundary(sigma_3) = {}         -- vertex 2 is in no edge
    coboundary(sigma_4) = {}         -- no triangle contains edge [0,1]

Matrix (rows = coboundary targets, columns = source simplices):

         sigma_1 sigma_2 sigma_3 sigma_4
    sigma_1  0       0       0       0
    sigma_2  0       0       0       0
    sigma_3  0       0       0       0
    sigma_4  1       1       0       0

Process in reverse (j=4, then j=3, then j=2, then j=1):

**j=4 (sigma_4):** Column 4 is empty. Pivot undefined. sigma_4 is positive in cohomology.

**j=3 (sigma_3):** Column 3 is empty. Pivot undefined. sigma_3 is positive.

**j=2 (sigma_2):** Column 2 has pivot = 4 (row sigma_4 has 1). PivotOf[4] = -1. Pair: birth = sigma_4 (pivot), death = sigma_2. PivotOf[4] = 2.

In cohomology, the pairing semantics are:
- If column j has pivot p, then simplex j is the *birth* and simplex p is the *death*.
- Standard homology: if column j has pivot p, then simplex p is the *birth* and simplex j is the *death*.

For consistency, the library stores pairs as (birth, death) regardless of which algorithm was used.

The algorithm produces the same (birth, death) pairs as standard reduction.

**j=1 (sigma_1):** Column 1 has pivot = 4. PivotOf[4] = 2 (sigma_2 claimed it). Add column 2 to column 1:

    Column 1 before: [0, 0, 0, 1]
    Column 2:         [0, 0, 0, 1]
    Column 1 after:  [0, 0, 0, 0]

Empty. No pivot. sigma_1 is positive.

Result: only sigma_2 pairs with sigma_4 (same as standard reduction: pair (2, 4)). But we got sigma_4 as positive (birth) and sigma_2 as negative (death) in the cohomology reduction, which maps to pair (2, 4) in persistence terms.

The key observation: the cohomology algorithm paired the edge (sigma_4) without any column operations (emergent pair detection). The vertex columns (sigma_1, sigma_2) required one XOR between them. In the standard algorithm, all columns were trivially processed, but in larger complexes the difference is dramatic.
