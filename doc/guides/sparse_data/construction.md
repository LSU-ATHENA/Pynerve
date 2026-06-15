# Sparse VR construction

The sparse Vietoris-Rips complex is built on an **epsilon-net** of landmarks. Three selection strategies are available. **Random** is a uniform random subsample; it is the fastest and adequate for dense, uniform data. **Maxmin (farthest point)** iteratively picks the farthest-from-selected points, providing better coverage and preserving global structure. **Greedy permutation** selects an epsilon-net with an approximation guarantee and provides a theoretical guarantee (epsilon-interleaving).

```python
# Farthest-point landmarks (Python API via nn module)
from pynerve.nn import farthest_point_sampling

# Select 500 landmarks from 100K points
landmarks = farthest_point_sampling(points, n_landmarks=500)

# Witness complex on landmarks
from pynerve.nn import WitnessComplexPersistence

wc = WitnessComplexPersistence(n_landmarks=500, max_dim=2)
diagram = wc(torch.tensor(points))
```

### Sparse VR construction in detail

```
Input: point cloud P = {p_1, ..., p_n}, landmark ratio r, max radius R
Output: sparse VR filtration

Step 1: Landmark selection
  k = max(ceil(r * n), max_dim + 2)  // minimum landmarks
  L = select_landmarks(P, k, strategy)

Step 2: Distance to landmarks
  For each p in P, compute d(p, L) = min_{l in L} dist(p, l)
  Store witness_distance[p] = d(p, L)

Step 3: Landmark distance matrix
  D_L = matrix of pairwise distances between landmarks (k x k)

Step 4: Edge inclusion
  For each pair (l_i, l_j) of landmarks:
      edge_distance = D_L[i][j]
      If edge_distance <= R:
          // Check witness condition using the two landmark condition:
          // An edge (l_i, l_j) is included at scale t if there exists
          // a witness point p such that max(dist(p,l_i), dist(p,l_j)) <= t
          // OR simply if edge_distance <= R (standard VR on landmarks)
          include_edge(l_i, l_j, edge_distance)

Step 5: Simplex construction
  Build flag complex on landmark graph:
  For d = 2 to max_dim:
      For each (d-1)-simplex in the complex:
          For each landmark l that forms a (d-1)-clique with all vertices:
              Add d-simplex at filtration value = max(edge_values)

Step 6: Filtration sorting
  Sort all simplices by (filtration_value, dimension, vertices)
```

### Link-time construction

For each landmark pair, the sparse VR includes an edge only when the distance is below the threshold AND both points share a significant witness relationship. This produces O(k^2) edges where k is the landmark count, versus O(n^2) for full VR.

### Witness relationship

```
A point p is a witness for landmark pair (l_i, l_j) at scale t if:
    max(dist(p, l_i), dist(p, l_j)) <= t

The edge (l_i, l_j) is included at filtration value t if
there exists at least one witness p for that pair at scale t.
```

Back to [Sparse Workflows Overview](index.md)
