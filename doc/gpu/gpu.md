## GPU module

GPU tuning and multi-GPU management infrastructure used internally by the GPU persistence engine.

### Multi-GPU support

GPU detection and multi-GPU management is handled internally by the
persistence engine. Use `pynerve.PersistenceBackend.CUDA_HYBRID` to
enable GPU acceleration. The engine auto-detects available GPUs and
NVLink connectivity.

### Device selection

Select a GPU device via the `device` parameter:

```python
result = pynerve.compute_persistence(points, max_dim=2, device="cuda:0")
```

### Tuning

Kernel launch parameters (grid/block sizes, shared memory) are auto-tuned
at runtime based on the detected GPU architecture (Turing, Ampere, Hopper,
Blackwell). No manual configuration is needed.


## FAQ

**Q: How do I choose which GPU to use?**
A: Pass `device="cuda:N"` to `compute_persistence` to select a specific GPU
by index. Device `"cuda:0"` is the default.

**Q: What GPU architectures are supported?**
A: Turing, Ampere, Hopper, and Blackwell architectures are fully supported.
Kernel parameters are auto-tuned at startup.

**Q: Can I use multiple GPUs for a single computation?**
A: Yes. The multi-GPU manager distributes work across devices using NCCL
for communication.
