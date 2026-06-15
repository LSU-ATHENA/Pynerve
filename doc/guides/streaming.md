# Streaming Workflows

Process datasets that exceed RAM by streaming point clouds in chunks, tracking topological features across sliding windows, and distributing ingestion across threads and GPUs.

This page has been split into subpages:

- [Windowed persistence](streaming_persistence/windowed.md)  --  Sliding-window persistence tracking across sequential point-cloud frames
- [Zigzag persistence](streaming_persistence/zigzag.md)  --  Cross-window feature birth/death tracking with lineage matching
- [Lock-free streaming reduction](streaming_persistence/lock_free.md)  --  Multi-threaded producer-consumer streaming via lock-free queue
- [GPU streaming scatter/gather](streaming_persistence/gpu_streaming.md)  --  Multi-GPU distribution of streaming persistence across devices
- [Distributed streaming](streaming_persistence/distributed.md)  --  MPI-based multi-node streaming across cluster nodes
- [Streaming processor (C++ API)](streaming_persistence/cpp_api.md)  --  C++ chunk processing API with progressive refinement
- [Input sources](streaming_persistence/inputs.md)  --  Supported file formats and custom async iterators
- [API reference](streaming_persistence/api.md)  --  Python and C++ API documentation
- [Performance tuning for streaming](streaming_persistence/performance.md)  --  Chunk sizing, GPU throughput, I/O overlap, error handling
- [FAQ](streaming_persistence/faq.md)  --  Frequently asked questions
