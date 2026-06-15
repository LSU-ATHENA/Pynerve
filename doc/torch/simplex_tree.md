## SimplexTree, BoundaryMatrix, Filtration

TorchScript-compatible simplex tree, sparse boundary matrix, and filtration
classes backed by ATen tensors. All operations run on GPU when tensors are
CUDA.

### SimplexTree

```python
# C++: src/include/nerve/torch/simplex_tree.hpp

tree = pynerve.torch.SimplexTree(points, max_radius=0.0)

tree.insert([0, 1, 2], filtration=0.5)
tree.insert_batch(simplices_tensor, filtration_tensor)
tree.remove([0, 1, 2])
tree.set_filtration([0, 1, 2], value=1.0)

contains = tree.contains([0, 1])
idx = tree.find([0, 1])
vertices = tree.get_vertices(node_idx)

cofaces = tree.get_cofaces([0, 1], max_dim=2)
faces = tree.get_faces([0, 1])
by_dim = tree.get_simplices_by_dimension(dim=2)
filt = tree.get_filtration([0, 1])

tree.build_vr(points, max_radius, max_dim)
tree.build_witness(points, landmarks, max_radius, max_dim)

n = tree.num_simplices()
maxd = tree.max_dimension()
fvals = tree.filtration_values()

tree.to(device)
is_cuda = tree.is_cuda()

B = tree.to_boundary_matrix(dim=2)
sorted_simplices = tree.get_sorted_simplices()
tensor_repr = tree.to_tensor()
```

**Tree structure** stored as ATen tensors:

The tree stores five arrays: `vertex_indices_` ([N] int64) holds the vertex label at each node, `parent_pointers_` ([N] int64) stores the parent index (-1 for roots), `first_child_` ([N] int64) gives the first child index (-1 for leaf), `next_sibling_` ([N] int64) tracks the next sibling index (-1 if none), and `filtration_values_` ([N] double) stores the filtration value per node.

Simplex `[0, 1, 2]` is stored as path root -> 0 -> 1 -> 2. This
representation allows O(k) insertion and O(k) traversal for a k-simplex
with all storage on GPU.

### Batch construction

```python
# Insert many simplices at once (GPU-efficient)
simplices = torch.tensor([[0, 1, -1], [0, 2, -1], [1, 2, -1]], device='cuda')
values = torch.tensor([0.5, 0.8, 1.0], device='cuda')
tree.insert_batch(simplices, values)
```

### BoundaryMatrix

```cpp
// C++: src/include/nerve/torch/boundary_matrix.hpp

enum class Format { CSR, CSC, COO };

BoundaryMatrix(tree, dimension, Format::CSR);
BoundaryMatrix(indices, indptr, data, n_rows, n_cols, Format::CSR);

y = matrix.matvec(x);
yT = matrix.matvec_transpose(x);
col = matrix.get_column(col_idx);
matrix.add_column(target_col, source_col);
matrix.reduce_column(col_idx);
matrix.reduce_all();
pivot = matrix.get_pivot(col_idx);
pivots = matrix.get_pivots();
pairs = matrix.compute_persistence_pairs();

matrix.to(device);
dense = matrix.to_dense();
matrix.convert_to_csr();
matrix.convert_to_csc();
matrix.convert_to_coo();
sparse = matrix.to_torch_sparse();
loaded = BoundaryMatrix.from_torch_sparse(sparse_tensor);
```

CSR/CSC/COO sparse matrix backed by ATen tensors. All matrix operations
(matvec, add_column, reduce) run on GPU when the tensors are CUDA.

### Filtration

```cpp
// C++: src/include/nerve/torch/filtration.hpp

Filtration(tree);
Filtration(simplices, values, dimensions);

// Factories
f = Filtration.from_vietoris_rips(points, max_radius, max_dim);
f = Filtration.from_witness(points, landmarks, max_radius, max_dim);
f = Filtration.from_alpha(points);

f.append([0, 1, 2], 0.5);
f.append_batch(vertices_list, values);
f.sort_by_filtration();
f.sort_by_dimension_and_filtration();

order = f.get_sorted_order();
by_dim = f.get_simplices_in_dimension(dim);
vals = f.get_values_in_dimension(dim);
counts = f.get_simplex_counts();

sub = f.get_sublevel_set(max_value=1.0);
dim_filt = f.get_dimension_filtration(dim);

batches = f.batch_split(batch_size);
combined = Filtration.batch_concat(batches);

hist = f.histogram(num_bins=50);
```

Tensor-backed filtration with factories for VR, witness, and alpha complexes.
Supports sublevel sets, dimension filtering, and batched splitting.


## SimplexTree tensor structure

```
vertex_indices_  [N]:  [3, 7, 12, 5, ...]
parent_pointers_ [N]:  [-1, 0, 1, 0, ...]
first_child_     [N]:  [1, 2, -1, -1, ...]
next_sibling_    [N]:  [-1, 3, -1, -1, ...]
filtration_values_ [N]: [0.0, 0.5, 0.8, 1.0, ...]
```

Root node `[0]` represents the empty simplex. Each child adds one vertex.

Search: O(k) for a k-simplex by following `first_child_` and `next_sibling_`.

Insertion: O(k) by traversing the prefix path and creating missing nodes.

```cpp
// C++ search example
int findSimpleX(const int* vertices, int dim, const at::Tensor& tree) {
    int node = 0;  // start at root
    auto ptrs = tree.accessor<int64_t, 1>();

    for (int i = 0; i <= dim; i++) {
        int v = vertices[i];
        // Search children of current node for vertex v
        int child = ptrs[first_child_offset + node];
        bool found = false;
        while (child >= 0) {
            if (ptrs[vertex_offset + child] == v) {
                node = child;
                found = true;
                break;
            }
            child = ptrs[next_sibling_offset + child];
        }
        if (!found) return -1;  // not found
    }
    return node;
}
```

## BoundaryMatrix operations

```cpp
// Column operations are optimized for sparse GPU execution

// Add column source to column target (modifies target in-place)
void BoundaryMatrix::add_column(int target_col, int source_col) {
    // CSR: scatter column entries into a hash table
    // Add source entries with XOR (char = 2)
    // Convert back to sorted CSR
}

// Reduce column to find pivot
void BoundaryMatrix::reduce_column(int col_idx) {
    int pivot = get_pivot(col_idx);
    while (pivot >= 0) {
        int source_col = pivot_to_column[pivot];
        if (source_col < 0) break;
        add_column(col_idx, source_col);
        pivot = get_pivot(col_idx);
    }
}
```

## Filtration example

```python
# Build filtration from scratch
from pynerve.torch import Filtration

# Method 1: from VR complex
f = Filtration.from_vietoris_rips(
    points, max_radius=2.0, max_dim=2
)

# Method 2: from explicit simplices
f = Filtration(
    simplices=[[0, 1], [1, 2], [0, 2], [0, 1, 2]],
    values=[0.5, 0.8, 1.0, 1.5],
    dimensions=[1, 1, 1, 2],
)

# Method 3: from alpha complex
f = Filtration.from_alpha(points)

# Access
sorted_order = f.get_sorted_order()
hist = f.histogram(num_bins=50, dim=1)
```

## Performance optimization tips

1. **Batch insertion**: Use `insert_batch` instead of individual `insert` calls. Batch insertion reduces the number of GPU kernel launches from O(k) to O(1).
2. **Boundary matrix format**: CSR is fastest for matvec; CSC is fastest for column operations; COO is most compact. `convert_to_csr()`, `convert_to_csc()`, `convert_to_coo()` convert between formats.
3. **Filtration sorting**: `sort_by_dimension_and_filtration()` sorts by dimension first, then filtration. This is the order needed for persistence computation. Calling this once before persistence is faster than sorting during computation.


## FAQ

**Q: Why use a tree-based simplex storage instead of a hash set?**
A: The tree structure allows O(k) traversal and insertion, which is optimal for the simplex operations needed during VR building (checking if all faces exist). Hash sets offer O(1) lookup but require O(k!) memory for the full face lattice.

**Q: Can the SimplexTree handle very high-dimensional simplices (dim > 10)?**
A: Yes, but insertion time grows linearly with dimension. For dim > 10, consider using the `Filtration` class directly with explicit simplex lists, or use DMT preprocessing to reduce complex size.

**Q: How does BoundaryMatrix::reduce_all compare to standard persistence?**
A: `reduce_all` performs the standard column reduction algorithm (same as PHAT or Ripser). The result is persistence pairs. The GPU version uses the same algorithm but with batched column operations on GPU.


### Cross-references

- `pynerve.torch`: PyTorch integration overview
- `pynerve.algebra.simplicial_ops`: CPU SimplicialComplex equivalent
- `pynerve.algebra.BoundaryMatrix`: CPU boundary matrix
- `pynerve.dmt`: DMT preprocessing for large complexes
