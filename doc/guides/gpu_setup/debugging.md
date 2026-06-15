# Debugging GPU computations

## Common issues

A CUDA_ERROR_OUT_OF_MEMORY error occurs when the distance matrix exceeds GPU memory; the solution is to set device_memory_limit_mb or reduce max_radius. CUDA_ERROR_LAUNCH_TIMEOUT happens when a kernel exceeds the TDR timeout on Windows; increasing the TDR delay or reducing max_dim resolves this. CUDA_ERROR_ILLEGAL_ADDRESS indicates a column index out of bounds, requiring checking input data for corruption. Results differing across runs suggest the `seed=` argument was not set; pass `seed=<value>` for reproducible output. If issues persist, check that the `--fmad=false` flag is present in the build. Slow GPU utilization from a block size that is too small can be resolved by letting the auto-tuner select parameters (or deleting the tuning cache). Tensor Cores not being used typically means the dimension is not a multiple of the tile size; using auto-padding or choosing a dimension multiple of 16 addresses this.

## Debugging with Nsight

```bash
# Memory check
compute-sanitizer --tool memcheck python my_script.py

# Race condition detection
compute-sanitizer --tool racecheck python my_script.py

# Kernel profiling
nsys profile -o output python my_script.py
```


<- [Back to GPU Acceleration index](index.md)
