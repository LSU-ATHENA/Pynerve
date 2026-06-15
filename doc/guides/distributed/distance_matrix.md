# Distributed distance matrix

The full O(n^2) distance matrix is distributed across ranks using **Allgatherv**:

1. Each rank owns a row-slab of the input
2. Local pairwise distances computed for the slab
3. Allgatherv assembles the complete distance matrix in distributed fashion
4. Each rank retains only its slab (memory: O(n^2)/p per rank for p ranks)

```cpp
// Conceptual: allgather point coordinates, then distribute distance compute
MPI_Allgatherv(local_points, local_n * dim, MPI_DOUBLE,
               global_buffer, counts, offsets, MPI_DOUBLE, MPI_COMM_WORLD);
```

Memory scaling: `O(n^2/p)` per rank -- p ranks reduce per-node memory by p times.

### Point distribution protocol

```
Input: n points, p ranks
Output: each rank i owns local_n_i = ceil(n / p) or floor(n / p) points

1. Rank 0 computes partition sizes:
   base = n / p
   remainder = n % p
   for i in 0..p-1:
       local_n[i] = base + (i < remainder ? 1 : 0)

2. Rank 0 scatters local_n to all ranks (MPI_Scatter)

3. Points distributed via MPI_Scatterv:
   offsets[0] = 0
   for i in 1..p-1:
       offsets[i] = offsets[i-1] + local_n[i-1] * dim
   MPI_Scatterv(all_points, local_n*dim, offsets, MPI_DOUBLE,
                local_points, local_n[rank]*dim, MPI_DOUBLE, 0, comm)

4. Allgatherv point coordinates for full distance computation:
   MPI_Allgatherv(local_points, local_n[rank]*dim, MPI_DOUBLE,
                  global_buffer, local_n*dim, offsets, MPI_DOUBLE, comm)
```

### Column distribution algorithm

After the distance matrix is available, boundary matrix columns are distributed:

```
For each simplex (i0, i1, ..., id) in the filtration:

1. Compute column index = simplex's position in the filtration order

2. Determine owner rank:
   owner = column_index % p                (round-robin)
        or column_index * p / total        (chunked)

3. If owner == my_rank:
       store column locally
   Else:
       serialize column and send to owner

4. Each rank receives its columns and builds local boundary sub-matrix

5. Local reduction proceeds independently
```

<- [Distributed Computing Overview](index.md)
