## Gradient field computation

The DMT gradient field is a matching between simplices of adjacent dimensions.
A pair (s, t) with dim(s) = k and dim(t) = k+1 is a gradient pair if s is a
face of t (s < t) and the pairing is acyclic.

### Algorithm

1. For each simplex, find candidate cofaces that differ by one vertex
2. Sort candidates by filtration value (ascending)
3. Greedy: pair an unmatched simplex with its best unmatched coface
4. Unpaired simplices are critical cells

### Critical cells

Simplices that remain unpaired after the greedy matching are critical cells.
They form the vertices of the reduced complex. The number of critical cells
depends on the topology and density of the input complex.

```cpp
// Extract critical cells from a MorseResult
std::vector<int> getCriticalCells(const MorseResult& result,
                                   size_t total_simplices) {
    std::vector<bool> is_paired(total_simplices, false);
    for (auto& [face, coface] : result.gradient_pairs) {
        is_paired[face] = true;
        is_paired[coface] = true;
    }
    std::vector<int> critical;
    for (size_t i = 0; i < is_paired.size(); ++i) {
        if (!is_paired[i]) critical.push_back(i);
    }
    return critical;
}
```

### Gradient path traversal

Gradient paths follow alternating face/coface edges in the gradient field.
Traversal follows the boundary graph starting from each pair's face index.

```cpp
// Traverse gradient path from a simplex
std::vector<int> traverseGradientPath(
    int start_idx,
    const std::vector<std::pair<int, int>>& gradient_pairs) {
    std::vector<int> path;
    int current = start_idx;
    while (true) {
        path.push_back(current);
        auto it = std::find_if(gradient_pairs.begin(), gradient_pairs.end(),
            [current](const auto& p) { return p.first == current; });
        if (it == gradient_pairs.end()) break;
        current = it->second;
    }
    return path;
}
```


## Gradient pair definition

A pair (s, t) with dim(s) = k and dim(t) = k+1 is a valid gradient pair if:

1. **Face relation**: s is a face of t (all vertices of s appear in t)
2. **Filtration ordering**: filtration(s) < filtration(t) (birth before death)
3. **Acyclicity**: No directed cycle exists in the gradient field

### Verification

```cpp
bool isValidGradientPair(
    const std::vector<int>& s,  // k-simplex
    const std::vector<int>& t,  // (k+1)-simplex
    float fs, float ft,
    const std::vector<std::pair<int, int>>& existing_pairs) {

    // Check face relation
    if (t.size() != s.size() + 1) return false;
    for (int v : s) {
        if (std::find(t.begin(), t.end(), v) == t.end())
            return false;
    }

    // Check filtration
    if (fs >= ft) return false;

    // Check acyclicity (no cycle involving this pair)
    return !wouldCreateCycle(s, t, existing_pairs);
}
```

### Cycle detection

```cpp
bool wouldCreateCycle(int s, int t,
    const std::vector<std::pair<int, int>>& pairs) {

    // BFS from t following gradient arrows
    // A gradient arrow goes from coface -> face in the pair
    // t -> s means we follow reverse pairing edges
    std::vector<bool> visited(max_simplex);
    std::queue<int> q;
    q.push(t);
    visited[t] = true;

    while (!q.empty()) {
        int current = q.front(); q.pop();
        if (current == s) return true;  // cycle detected

        // Follow all outgoing gradient edges
        for (auto& [face, coface] : pairs) {
            if (face == current && !visited[coface]) {
                visited[coface] = true;
                q.push(coface);
            }
        }
    }
    return false;
}
```

### Critical cell interpretation

Critical cells are unpaired simplices. They form the reduced complex:

- **Critical vertices** (dim 0): essential for connectivity
- **Critical edges** (dim 1): essential for 1-cycles
- **Critical triangles** (dim 2): essential for 2-cycles (voids)

```python
from pynerve.dmt import DMTEngine

result = engine.computeMorseComplex(simplices, filtration)
critical_by_dim = group_by_dimension(result.critical_simplices, simplices)

for dim, cells in critical_by_dim.items():
    print(f"Critical cells (dim {dim}): {len(cells)}")
```

The number of critical cells per dimension equals the Betti number (for optimal DMT) or more (for sub-optimal DMT). The difference is the number of redundant critical cells.


## Practical considerations

### Gradient path applications

Gradient paths flow from faces to cofaces. They can be used to:

1. **Compute the Morse boundary operator**: boundary of a critical (k+1)-cell is the sum of gradient paths from its boundary k-cells
2. **Cancel critical pairs**: if two critical cells are connected by a gradient path, they can be cancelled to simplify
3. **Recover the persistence pairing**: DMT gradient pairs are a subset of the persistence pairs

### Cancellation example

```cpp
// Cancel a critical pair of index (p, q) connected by a gradient path
bool cancelCriticalPair(
    int p, int q,
    std::vector<std::pair<int, int>>& gradient_pairs,
    std::vector<int>& critical_cells) {

    auto path = traverseGradientPath(p, gradient_pairs);
    if (path.back() != q) return false;  // not connected

    // Reverse the gradient along the path
    for (size_t i = 0; i < path.size() - 1; i += 2) {
        gradient_pairs.erase(
            std::remove_if(gradient_pairs.begin(), gradient_pairs.end(),
                [&](const auto& pair) {
                    return pair.first == path[i] &&
                           pair.second == path[i+1];
                }),
            gradient_pairs.end());
    }

    // Remove p, q from critical set
    critical_cells.erase(std::remove(critical_cells.begin(),
        critical_cells.end(), p), critical_cells.end());
    critical_cells.erase(std::remove(critical_cells.begin(),
        critical_cells.end(), q), critical_cells.end());

    return true;
}
```


## FAQ

**Q: How do critical cells relate to Betti numbers?**
A: For an optimal DMT reduction, the number of critical cells per dimension equals the Betti number. Sub-optimal DMT produces extra redundant critical cells beyond the true Betti numbers.

**Q: What happens if the gradient field contains a cycle?**
A: A cycle invalidates the Morse complex -- homotopy equivalence is no longer guaranteed. The greedy matching strategy is designed to be acyclic for generic filtrations. If equal filtration values exist, add a small epsilon to break ties.

**Q: How do I cancel a pair of critical cells?**
A: If two critical cells are connected by a gradient path, call `cancelCriticalPair(p, q, ...)`. The function reverses the gradient along the path and removes both cells from the critical set, further simplifying the complex.


### Cross-references

- `pynerve.dmt`: DMT module overview
- `pynerve.dmt.parallel`: Parallel gradient computation
- `pynerve.algebra`: Simplicial complex operations
- `pynerve.dmt.gpu`: GPU gradient computation
