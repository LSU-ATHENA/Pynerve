# Pivot exchange across ranks

When a column reduces to a pivot that lives on another rank, the pivot column is fetched via point-to-point MPI:

```cpp
// Send pivot request
MPI_Isend(&request, 3, MPI_UNSIGNED_LONG_LONG, target_rank,
          TAG_BOUNDARY_CHUNK, comm, &req);

// Receive pivot column
MPI_Irecv(buffer, max_size, MPI_UNSIGNED_LONG_LONG, source_rank,
          TAG_BOUNDARY_CHUNK, comm, &req);
```

Messages use tags `TAG_BOUNDARY_CHUNK=1`, `TAG_REDUCTION_RESULT=2`, `TAG_WORK_STEAL=3`, `TAG_CHECKPOINT=4` with chunk size 10,000 entries.

### Pivot exchange protocol

```
On rank A, reducing column c with pivot p:

1. Determine owning rank of column p: owner = p % world_size

2. If owner == my_rank:
       Fetch pivot column from local storage. Continue reduction.

3. If owner != my_rank:
       a. MPI_Isend to owner_rank requesting column p (TAG_BOUNDARY_CHUNK)
       b. MPI_Irecv from owner_rank receiving column p data
       c. MPI_Wait for the exchange to complete
       d. Apply pivot column to continue reduction of column c

4. After pivot application:
       New pivot p' = pivot of (column c XOR column p)
       If p' < p: repeat from step 1 with new pivot p'
       Else: column c is reduced. Store result.
```

### Non-blocking communication

Pynerve overlaps communication with computation using non-blocking MPI:

```cpp
// Initiate prefetch of likely-needed pivots
std::vector<MPI_Request> pending_requests;
for (auto& pivot : predicted_pivots) {
    if (pivot.rank != my_rank) {
        MPI_Request req;
        MPI_Irecv(buf, size, MPI_UNSIGNED_LONG_LONG,
                  pivot.rank, TAG_BOUNDARY_CHUNK, comm, &req);
        pending_requests.push_back(req);
    }
}

// Continue local reduction while communication is in flight
while (!pending_requests.empty()) {
    int completed;
    MPI_Testany(pending_requests.size(), pending_requests.data(),
                &completed, &flag, MPI_STATUS_IGNORE);
    if (flag) {
        // Process received pivot, continue reduction
        pending_requests.erase(pending_requests.begin() + completed);
    }
    // Do one step of local reduction between MPI_Testany checks
    if (has_local_work()) {
        reduce_next_local_column();
    }
}
```

<- [Distributed Computing Overview](index.md)
