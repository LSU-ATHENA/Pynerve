# Lock-free streaming reduction

For multi-threaded point ingestion, Pynerve provides lock-free streaming reduction:

```cpp
// src/streaming/lockfree/
// - streaming_lockfree_ops.cpp
// - streaming_lockfree_runtime_ops.cpp
// - streaming_mpi_ops.cpp
```

The lock-free architecture:
1. Producer threads push point batches into a lock-free queue
2. Consumer threads pop batches and compute per-window persistence
3. No mutex contention -- all coordination via atomic indices

```python
# Concurrent producer/consumer streaming
import asyncio

async def producer(queue):
    for batch in data_stream:
        await queue.put(batch)

async def consumer(sp, queue):
    async for result in sp.stream_compute(queue):
        process(result)

async def main():
    sp = StreamingPersistence(chunk_size=1000, max_dim=2)
    queue = asyncio.Queue(maxsize=5)
    await asyncio.gather(
        producer(queue),
        consumer(sp, queue),
    )
```

### Lock-free queue internals

```cpp
// src/streaming/lockfree/streaming_lockfree_ops.cpp
template <typename T>
class LockFreeQueue {
    struct Node {
        std::atomic<Node*> next;
        T data;
    };

    std::atomic<Node*> head;
    std::atomic<Node*> tail;
    std::atomic<size_t> size_;

public:
    void push(T item) {
        Node* node = new Node{nullptr, std::move(item)};
        Node* prev_tail = tail.exchange(node, std::memory_order_acq_rel);
        prev_tail->next.store(node, std::memory_order_release);
        size_.fetch_add(1, std::memory_order_relaxed);
    }

    bool pop(T& item) {
        Node* first = head.load(std::memory_order_acquire);
        Node* next = first->next.load(std::memory_order_acquire);
        if (next == nullptr) return false;
        item = std::move(next->data);
        head.store(next, std::memory_order_release);
        delete first;
        size_.fetch_sub(1, std::memory_order_relaxed);
        return true;
    }

    size_t size() const { return size_.load(std::memory_order_relaxed); }
};
```

### Producer-consumer protocol

```
Producer threads:
  - Read data from disk/network
  - Partition into fixed-size chunks
  - Push chunks into lock-free queue
  - Signal completion via atomic done counter

Consumer threads (CUDA stream workers):
  - Pop chunk from lock-free queue
  - If pop succeeds: process persistence for chunk
  - If pop fails and producers are done: terminate
  - If pop fails and producers are active: spin-wait with backoff

Synchronization:
  - std::atomic<size_t> producer_done_count (incremented when producer finishes)
  - std::atomic<size_t> chunks_processed (for progress tracking)
  - No mutex, no condition variable in the hot path
```

### Lock-free Persistence Reducer Accuracy

The lock-free *streaming architecture* described above handles concurrent data ingestion. Pynerve also provides a separate **lock-free persistence reducer** (`reduceMatrixLockfree` in `reduction_lockfree_ops.cpp`) for parallel matrix reduction on multi-core CPUs.

This reducer achieves **0.0000% count-level accuracy** vs the deterministic sequential ground truth -- every run produces the exact same number of persistence pairs. This is possible because the post-pass cascade operates on **shared mutable final state**: after all worker threads join, every column's reduced form is authoritative and deterministic.

At the *pair-value level*, the lock-free reducer is non-deterministic (~1.78% mismatch for dim-2, ~34% for dim-1) -- different runs produce different but topologically valid (birth, death) pairings. This is expected behavior for any parallel persistence reduction (Morozov & Nigmetov 2019). See [Determinism](../../reference/correctness_detail/determinism.md) for details.

[Back to index](index.md)
