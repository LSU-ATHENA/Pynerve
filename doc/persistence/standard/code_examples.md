# Code Examples

### Manual Reduction in Python

```python
import numpy as np

def boundary_matrix(simplices):
    """Build boundary matrix over Z2 from a list of simplices."""
    n = len(simplices)
    D = np.zeros((n, n), dtype=np.uint8)
    sigma_to_idx = {tuple(s): i for i, s in enumerate(simplices)}

    for j, sigma in enumerate(simplices):
        d = len(sigma) - 1  # dimension
        for k in range(d + 1):
            face = tuple(sigma[:k] + sigma[k+1:])
            if face in sigma_to_idx:
                i = sigma_to_idx[face]
                D[i, j] ^= 1  # XOR = addition over Z2
    return D


def standard_reduction(D):
    """Standard matrix reduction over Z2. Returns pairs."""
    n = D.shape[0]
    R = D.copy()
    pivot_of = {}  # row -> column mapping
    pairs = []

    for j in range(n):
        # Find pivot: scan from bottom
        col = R[:, j]
        # Find highest 1 in column
        nonzero = np.where(col == 1)[0]
        pivot = nonzero[-1] if len(nonzero) > 0 else -1

        # Eliminate pivot conflicts
        while pivot >= 0 and pivot in pivot_of:
            k = pivot_of[pivot]
            # Add column k to column j (XOR)
            R[:, j] ^= R[:, k]
            # Recompute pivot
            nonzero = np.where(R[:, j] == 1)[0]
            pivot = nonzero[-1] if len(nonzero) > 0 else -1

        if pivot >= 0:
            pivot_of[pivot] = j
            pairs.append((pivot, j))

    return pairs


# Example: triangle with one extra edge
simplices = [
    (0,),      # 0: vertex 0
    (1,),      # 1: vertex 1
    (2,),      # 2: vertex 2
    (3,),      # 3: vertex 3
    (0, 1),    # 4: edge 0-1
    (1, 2),    # 5: edge 1-2
    (0, 2),    # 6: edge 0-2 (triangle boundary)
    (2, 3),    # 7: edge 2-3
    (0, 1, 2), # 8: triangle 0-1-2
]

D = boundary_matrix(simplices)
pairs = standard_reduction(D)
print("Persistence pairs (birth, death):", pairs)
# Expected: pairs for dim 0 edges killing vertices, dim 2 triangle killing edge
```

### Manual Reduction in C++ (Sparse Columns)

```cpp
#include <vector>
#include <unordered_map>
#include <utility>

using Column = std::vector<size_t>;  // sorted nonzero row indices

size_t pivot(const Column& col) {
    return col.empty() ? SIZE_MAX : col.back();
}

// XOR column src into column dst (in-place)
void xor_columns(Column& dst, const Column& src) {
    Column result;
    result.reserve(std::max(dst.size(), src.size()));
    auto i = dst.begin(), j = src.begin();
    while (i != dst.end() && j != src.end()) {
        if (*i < *j) { result.push_back(*i); ++i; }
        else if (*j < *i) { result.push_back(*j); ++j; }
        else { ++i; ++j; }  // cancel: 1 xor 1 = 0
    }
    while (i != dst.end()) result.push_back(*i++);
    while (j != src.end()) result.push_back(*j++);
    dst = std::move(result);
}

std::vector<std::pair<size_t, size_t>>
standardReductionSparse(const std::vector<Column>& D) {
    size_t n = D.size();
    std::vector<Column> R = D;
    std::unordered_map<size_t, size_t> pivotOf;
    std::vector<std::pair<size_t, size_t>> pairs;

    for (size_t j = 0; j < n; ++j) {
        size_t p = pivot(R[j]);

        while (p != SIZE_MAX && pivotOf.count(p)) {
            size_t k = pivotOf[p];
            xor_columns(R[j], R[k]);
            p = pivot(R[j]);
        }

        if (p != SIZE_MAX) {
            pivotOf[p] = j;
            pairs.emplace_back(p, j);
        }
    }
    return pairs;
}
```

### Using the Python API

```python
import pynerve
import numpy as np

# Basic usage
points = np.random.rand(500, 3)
result = pynerve.compute_persistence(
    points,
    max_dim=2,
    max_radius=0.8,
)

print("Pairs:", result.pairs[:10])
print("Betti numbers:", result.betti_numbers)
print("Computation time:", result.diagnostics.get("computation_time_ms", "N/A"), "ms")

# Access all pairs by dimension
dim_0_pairs = [(b, d) for b, d, dim in result.pairs if dim == 0]
dim_1_pairs = [(b, d) for b, d, dim in result.pairs if dim == 1]
print(f"dim 0: {len(dim_0_pairs)} pairs")
print(f"dim 1: {len(dim_1_pairs)} pairs")
```

<- [Standard Reduction Overview](index.md)
