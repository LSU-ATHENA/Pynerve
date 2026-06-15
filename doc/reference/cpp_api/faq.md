# FAQ

**What is the memory footprint of a typical persistence computation?**
The memory usage depends on the number of points and maximum dimension. For a typical computation with thousands of points and max_dim=2, the distance matrix requires hundreds of megabytes. The DeviceMemoryPool initializes at a few gigabytes and grows as needed.

**How do I choose between CPU and GPU backends?**
Use `CPU_EXACT` for datasets with fewer than ten thousand points or when determinism is critical. Use `CUDA_HYBRID` for larger datasets or when GPU acceleration is beneficial. The `CPU_ADAPTIVE_ACCELERATION` backend provides a balanced approach for most workloads.

**Can I use mixed precision?**
Yes, through the PrecisionPolicy options. You can use float32 for distance computation with float64 for matrix reduction, which provides a good balance of speed and accuracy for most use cases.

**What happens if the GPU runs out of memory?**
The library returns an `E10_GPU_OOM` error, which propagates as `pynerve.GPUMemoryError` in Python. You can reduce the dataset size, increase the DeviceMemoryPool initial size, or switch to a CPU backend.

**Is the computation thread-safe?**
Yes. Each thread has its own ThreadLocalPool, and the slab allocators are lock-free. The GlobalPagePool and DeviceMemoryPool are shared resources and are designed for concurrent access.

<- [C++ API Overview](index.md)
