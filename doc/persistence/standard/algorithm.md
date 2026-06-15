# Algorithm

The standard reduction algorithm (Zomorodian-Carlsson 2005) operates on the boundary matrix *D* over Z2. Each column *j* encodes the boundary of the *j*-th simplex in filtration order. The reduction proceeds left-to-right, adding columns to eliminate pivot conflicts.

### Boundary Matrix Construction

Let K be a filtered simplicial complex with simplices sigma_1, ..., sigma_n ordered by filtration value (and dimension as tiebreaker). The boundary matrix D is an n x n matrix over Z2 where:

    D[i, j] = 1  iff  sigma_i is a codimension-1 face of sigma_j

Concretely, for a d-simplex sigma_j = [v_0, ..., v_d], its boundary is:

    boundary(sigma_j) = sum_{k=0..d} [v_0, ..., v_{k-1}, v_{k+1}, ..., v_d]

where each term is a (d-1)-simplex. In homology over Z2, addition is XOR, so each column contains exactly d+1 ones (one for each facet of the d-simplex). The *pivot* (or *lowest-1*) of column j is the maximum row index with a non-zero entry:

    pivot(j) = max{ i | D[i, j] = 1 }

If column j is all zeros, pivot(j) is undefined (the column is empty).

### The Reduction Loop

The standard algorithm processes columns j = 1..n sequentially:

```
Algorithm: STANDARD_REDUCTION(D)
Input: boundary matrix D (size n x n over Z2)
Output: reduced matrix R, pairing array P

 1.  R <- copy of D
 2.  P <- array of size n, initialized to -1
 3.  for j = 1..n:
 4.      while pivot(R, j) is defined AND P[pivot(R, j)] != -1:
 5.          k <- P[pivot(R, j)]
 6.          R[:, j] <- R[:, j] XOR R[:, k]    // add column k to column j
 7.      if pivot(R, j) is defined:
 8.          P[pivot(R, j)] <- j                // sigma_j kills sigma_{pivot}
 9.  return R, P
```

**Key insight:** When two columns share the same lowest-1 row, XORing them cancels that row. The algorithm repeatedly eliminates pivot conflicts until either the column becomes empty (the simplex is positive -- it creates a homology class) or its pivot is unique (the simplex is negative -- it kills a class).

### Lowest-1 Tracking

The *lowest-1* (or *pivot*) of a column is the largest row index containing a 1. For a column *j*, write *pivot(j) = max{ i : D[i, j] = 1 }*. The algorithm maintains an array `lowest_one[row] = column` mapping each pivot row to the column that currently owns it.

Implementation note: after each XOR operation, the pivot of column j may change. The standard approach is to recompute the pivot by scanning from the bottom of the column upward until a 1 is found. This is O(k) per operation where k is the column length.

### Step-by-Step Walkthrough: 4x4 Example

Consider a filtration with 4 simplices:

    sigma_1 = [0]       (vertex, dim 0, filtration t=0)
    sigma_2 = [1]       (vertex, dim 0, filtration t=0)
    sigma_3 = [2]       (vertex, dim 0, filtration t=0)
    sigma_4 = [0,1]     (edge, dim 1, filtration t=1)

The boundary matrix D (simplices ordered by filtration, then dimension):

    sigma_1  sigma_2  sigma_3  sigma_4
    [ 0        0        0        0    ]  sigma_1 (vertex 0)
    [ 0        0        0        0    ]  sigma_2 (vertex 1)
    [ 0        0        0        0    ]  sigma_3 (vertex 2)
    [ 0        0        0        0    ]  sigma_4 (edge 0-1)

Wait -- edges have boundary consisting of two vertices. sigma_4 = [0,1] has boundary sigma_1 + sigma_2. The correct boundary matrix:

    sigma_1  sigma_2  sigma_3  sigma_4
    [ 0        0        0        1    ]  sigma_1 (vertex 0)
    [ 0        0        0        1    ]  sigma_2 (vertex 1)
    [ 0        0        0        0    ]  sigma_3 (vertex 2)
    [ 0        0        0        0    ]  sigma_4 (edge 0-1)

Wait -- boundary of an edge is its two endpoint vertices. So column 4 has ones at rows 1 and 2. Since we order by filtration then dimension, vertices come first (dim 0), then edges (dim 1). The boundary matrix:

Row 1 (sigma_1): entry at (1,4) = 1 because sigma_1 is a face of sigma_4.
Row 2 (sigma_2): entry at (2,4) = 1 because sigma_2 is a face of sigma_4.
Row 3 (sigma_3): no entries.
Row 4 (sigma_4): no entries (it's a 1-simplex, no boundary includes it in co-dimension 1).

But wait, rows correspond to simplices and columns to simplices. D[i,j] = 1 if sigma_i is a facet of sigma_j. So:

D[1,4] = 1 (vertex 0 is a face of edge [0,1])
D[2,4] = 1 (vertex 1 is a face of edge [0,1])

The matrix:

    sigma_1  sigma_2  sigma_3  sigma_4
    [ 0,      0,       0,       1    ]   sigma_1
    [ 0,      0,       0,       1    ]   sigma_2
    [ 0,      0,       0,       0    ]   sigma_3
    [ 0,      0,       0,       0    ]   sigma_4

Now run the reduction:

**j=1:** Column 1 is empty. Pivot undefined. sigma_1 is positive (birth).

**j=2:** Column 2 is empty. Pivot undefined. sigma_2 is positive (birth).

**j=3:** Column 3 is empty. Pivot undefined. sigma_3 is positive (birth).

**j=4:** Column 4 has pivot = 2 (row 2 has the highest 1). P[2] = -1 (unpaired). So pair (2, 4): sigma_2 (row 2) is the birth, sigma_4 (column 4) is the death. P[2] = 4.

Result: pairs = [(2, 4, dim=0)]

This says edge [0,1] kills vertex 1 at filtration value 1. Vertex 0 and vertex 2 remain unpaired (they represent H0 classes). Betti numbers: beta_0 = 2.

### Pseudocode (Formal)

PASCAL-style pseudocode with explicit data structures:

```
function standardReduction(D: Matrix[n][n] over GF(2)) -> (R: Matrix, Pairs: List)
    R := copy of D
    pivotOf := array[-1 .. n] of int, initialized to -1
    pairs := empty list

    for j := 1 to n:
        // Find current pivot of column j
        p := findLowestOne(R, j)

        // Eliminate pivot conflicts
        while p != NONE and pivotOf[p] != -1:
            k := pivotOf[p]
            addColumn(R, j, k)     // R[:,j] := R[:,j] xor R[:,k]
            p := findLowestOne(R, j)

        // Record pairing or birth
        if p != NONE:
            pivotOf[p] := j
            pairs.append((p, j))   // death at j, birth at p
        // else: j is a positive simplex (birth)

    return R, pairs

function findLowestOne(R: Matrix, j: int) -> int:
    for i := n down to 1:
        if R[i, j] == 1:
            return i
    return NONE

function addColumn(R: Matrix, j: int, k: int):
    for i := 1 to n:
        R[i, j] := R[i, j] xor R[i, k]
```

This is the canonical O(n^3) formulation. Real implementations use sparse column representations and skip the full inner loop over all rows.

### Formal Properties

The reduced matrix R satisfies:
- R is *reduced*: no two columns have the same pivot.
- Every column of R has the same pivot as the unreduced D if and only if the column's simplex is negative (a death).
- The persistence pairs are {(pivot(j), j) : j where pivot(j) is defined}.
- Unpaired simplices are births (their homology classes persist to infinity).

Theorem (Zomorodian-Carlsson 2005): The pairing produced by standard reduction is the persistence pairing. The algorithm is correct for any field of coefficients.

<- [Standard Reduction Overview](index.md)
