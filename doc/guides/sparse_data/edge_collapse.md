# Edge collapse

Edge collapse reduces the 1-skeleton before building the complex, preserving persistence diagram up to a user-specified tolerance:

```
Original: 10^5 edges -> Collapsed: 10^3 edges -> Reduced: 10^3 columns
```

The collapse preserves:
- All persistence pairs with death - birth > collapse_tolerance
- Betti numbers exactly (for exact collapse)
- Filtration order of remaining simplices

Edge collapse is applied during `compute_persistence` when `PersistenceMode.APPROX` is used. It can reduce total simplices by 10-100x on dense point clouds with minimal diagram distortion.

### Edge collapse algorithm

```
Input: Graph G = (V, E) with edge weights w(e)
Output: Collapsed graph G' = (V', E') where persistence diagram
         is preserved up to tolerance epsilon

1. Sort edges by weight ascending: e_1, e_2, ..., e_m

2. For each edge e = (u, v) in sorted order:
   a. Let t = w(e) be the current filtration value
   b. Check if e is collapsible:
      - Both u and v are not involved in a previous collapse? -> OK
      - Or, collapsing e changes the persistence diagram by <= epsilon
        (checked via the link condition on the edge) -> OK
   c. If collapsible:
      - Mark edge as collapsed
      - Contract u into v (or v into u depending on degree)
      - Update adjacency

3. After all collapses:
   - Build VR complex on the collapsed graph G'
   - The persistence diagram of G' is epsilon-close to G

Complexity: O(|E| * alpha(|E|)) using union-find
Optimal for: dense point clouds where most edges are geometrically redundant
```

### Link condition for edge collapse

```
Edge e = (u, v) is collapsible if:
  Link(u) AND Link(v) = Link(u AND v)
  
Where Link(x) is the set of vertices adjacent to x.

If this holds, the homotopy type of the Vietoris-Rips complex
is preserved (exact collapse).

For approximate collapse (tolerance epsilon):
  The edge is collapsible if the bottleneck distance between
  the original and collapsed persistence diagrams is <= epsilon.
```

Back to [Sparse Workflows Overview](index.md)
