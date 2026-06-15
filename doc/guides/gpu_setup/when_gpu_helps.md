# When GPU helps

GPU acceleration is most beneficial for large point clouds. For n of 1,000 or fewer, CPU is fast while GPU is slowed by transfer overhead. At n of 10,000, GPU provides roughly 2 to 5 times speedup over CPU. At n of 100,000, GPU is roughly 10 to 50 times faster. For dimensions 2 to 3, CPU and GPU are comparable. For dimensions above 3, Tensor Cores give GPU an advantage. For batch processing, GPU streams allow overlapping computation, also favoring the GPU.

## Decision guide

GPU is beneficial when:
- **n > 10,000** -- transfer overhead is amortized
- **dim > 3** -- Tensor Core matmul dominates compute
- **Batch processing** -- multiple filtrations overlap on streams
- **High dimension (max_dim >= 2)** -- GPU reduction parallelism helps

CPU may be faster when:
- **n < 1,000** -- data transfer latency dominates
- **Single small computation** -- instant CPU result vs GPU init overhead
- **dim <= 3, n <= 5,000** -- SIMD dispatch is competitive


<- [Back to GPU Acceleration index](index.md)
