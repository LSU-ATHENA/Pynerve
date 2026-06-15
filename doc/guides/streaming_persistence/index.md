# Streaming Persistence

Process datasets that exceed RAM by streaming point clouds in chunks, tracking topological features across sliding windows, and distributing ingestion across threads and GPUs.

## Sections

- [Windowed persistence](windowed.md)  --  Sliding-window persistence tracking across sequential point-cloud frames
- [Zigzag persistence](zigzag.md)  --  Cross-window feature birth/death tracking with lineage matching
- [Lock-free streaming reduction](lock_free.md)  --  Multi-threaded producer-consumer streaming via lock-free queue
- [GPU streaming scatter/gather](gpu_streaming.md)  --  Multi-GPU distribution of streaming persistence across devices
- [Distributed streaming](distributed.md)  --  MPI-based multi-node streaming across cluster nodes
- [Streaming processor (C++ API)](cpp_api.md)  --  C++ chunk processing API with progressive refinement
- [Input sources](inputs.md)  --  Supported file formats and custom async iterators
- [API reference](api.md)  --  Python and C++ API documentation
- [Performance tuning for streaming](performance.md)  --  Chunk sizing, GPU throughput, I/O overlap, error handling
- [FAQ](faq.md)  --  Frequently asked questions

## Quick example

```python
from pynerve import StreamingPersistence
import asyncio

async def main():
    sp = StreamingPersistence(
        chunk_size=1000,
        max_buffered_chunks=3,
        use_gpu=True,
        max_dim=2,
    )
    async for result in sp.stream_compute("large_scan.npy"):
        print(result.betti_numbers)

asyncio.run(main())
```

[Back to docs home](../../index.md)
