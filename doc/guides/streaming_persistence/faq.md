# FAQ

**Q: What chunk size should I choose for my dataset?**
A: Chunk size depends on your hardware and latency requirements. Use 100-500 for real-time interactive streaming, 1000 for batch processing, and 5000 or more for offline analysis. Larger chunks give better topological accuracy but require more memory per chunk. A good starting point is 1000 points per chunk.

**Q: How does zigzag persistence differ from windowed persistence?**
A: Windowed persistence computes features within each sliding window independently. Zigzag persistence tracks features across windows, allowing a feature that dies in one window to be reborn in a later window. Use windowed persistence when each window is independent, and zigzag when you need to track feature lineages across the full stream.

**Q: Can I use GPU acceleration with streaming persistence?**
A: Yes. Set `use_gpu=True` in `StreamingPersistence` to distribute computation across available GPUs. The GPU scatter/gather protocol splits each chunk across devices, computes persistence in parallel, and merges results via NCCL. Multi-stream CUDA execution overlaps data transfer with computation.

**Q: What happens when a topological feature spans multiple chunks?**
A: Features that span chunk boundaries may be split or lost in per-chunk processing. Zigzag persistence mitigates this by matching features across consecutive windows via IoU of point indices. For best accuracy, choose a chunk size at least 10x the expected feature scale.

**Q: How does the lock-free queue improve multi-threaded performance?**
A: The lock-free queue eliminates mutex contention by using atomic operations (`std::atomic`) for all producer-consumer coordination. Producer threads push batches into the queue, consumer threads pop them for processing, and no mutex or condition variable is needed in the hot path, yielding higher throughput under high concurrency.

**Q: How do I handle errors during streaming computation?**
A: Wrap each iteration of `stream_compute` in a try-except block and continue on error. Common errors include `NerveIOError` (file access), `NerveMemoryError` (chunk too large), `GPUMemoryError` (GPU OOM), and `ConvergenceError` (PH timeout). Each error has a specific recovery strategy documented in the error handling section.

[Back to index](index.md)
