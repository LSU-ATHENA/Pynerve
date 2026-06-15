# FAQ

**Which NVIDIA GPUs are supported?**

Pynerve supports Volta (SM70) and later architectures. Turing (SM75), Ampere (SM80/SM86), Hopper (SM90), and Blackwell (SM100) each have tuned kernel parameters. Older architectures (Maxwell, Pascal) may work but lack Tensor Core support and optimal tuning.

**How do I verify my GPU is being used?**

Pass a CUDA tensor to `compute_persistence` or set `device="cuda"`. Check the diagnostics in the result dict for the backend enum value. You can also monitor GPU utilization with `nvidia-smi` or `nsys profile`.

**Why is my GPU not giving the expected speedup?**

Common causes include: small point clouds where transfer overhead dominates (n under 10,000), input data on CPU requiring device transfer, Tensor Cores unused due to dimension mismatch, or block size misconfiguration. Using FP16/BF16 input dtype and ensuring dimensions are multiples of 16 helps maximize throughput.

**Can I use multiple GPUs?**

Yes. Pynerve supports multi-GPU execution via NCCL collectives and NVLink P2P peer access. GPUs in the same NVLink domain can share columns with minimal latency. Pass point clouds placed on different CUDA devices or use MPI for multi-node distributed computation.

**Does GPU acceleration produce deterministic results?**

Yes. GPU persistence is deterministic by default with zero atomic operations. Fixed-tree warp-shuffle and shared memory reduction ensure bitwise reproducibility across runs and GPU architectures. For cross-GPU reproducibility across different GPU counts, enable RFA (reproducible floating-point accumulation).


<- [Back to GPU Acceleration index](index.md)
