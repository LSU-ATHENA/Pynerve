## CUDA Kernel Launch Audit

Verify that every `<<<grid, block>>>` launch has error checking and that grid
dimensions, shared memory, and memory operations are within device limits.

```python
from pynerve.validation import audit_cuda_launches

report = audit_cuda_launches()
# report.total_kernels: int
# report.verified_kernels: int
# report.unverified_launches: list[{"kernel": str, "file": str, "line": int}]
# report.passed: bool
```

### Audit checks

- Every kernel launch is followed by `cudaPeekAtLastError()` or
  `CHECK_CUDA(cudaGetLastError())`
- Grid/block dimensions are bounded by device limits
- Shared memory size does not exceed `devProp.sharedMemPerBlock`
- All `cudaMalloc`/`cudaMemcpy` calls use the `CHECK_CUDA` wrapper

### Error checking macro

```cpp
// C++ / CUDA: used internally for every GPU call
CHECK_CUDA(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
// On failure: prints file:line, error string, and throws
```


## Audit report structure

```python
report = audit_cuda_launches()

for entry in report.unverified_launches:
    print(f"Kernel: {entry.kernel}")
    print(f"  File: {entry.file}:{entry.line}")
    print(f"  Issue: Missing error check after launch")

# Detailed per-kernel report
for kernel in report.kernel_details:
    print(f"{kernel.name}:")
    print(f"  Grid: {kernel.grid_dim}, Block: {kernel.block_dim}")
    print(f"  Shared memory: {kernel.shared_mem_bytes} bytes")
    print(f"  Has error check: {kernel.has_error_check}")
    print(f"  Grid within limits: {kernel.grid_ok}")
    print(f"  Shared within limits: {kernel.shared_ok}")
```

### Common failure modes detected

The audit detects several categories of failure, each with an associated severity level:

- **Missing `cudaPeekAtLastError`** -- launch not followed by error check (CRITICAL)
- **Grid exceeds maxGridSize** -- `gridDim.x > devProp.maxGridSize[0]` (CRITICAL)
- **Block exceeds maxThreadsPerBlock** -- `blockDim.x > devProp.maxThreadsPerBlock` (CRITICAL)
- **Shared memory exceeds limit** -- `sharedMemBytes > devProp.sharedMemPerBlock` (HIGH)
- **Unaligned memory access** -- address not a multiple of element size (MEDIUM)
- **Double-free or leak** -- `cudaMalloc` without matching `cudaFree` (HIGH)

### Source-level annotation

The audit tool can annotate source files with inline warnings:

```bash
# Generate annotated source report
python -m pynerve.validation.annotate_launches --src-dir src/cuda/ --output audit/
```

Each kernel launch in the annotated output shows:
- Line number and file
- Error check status
- Grid/block dimensions
- Shared memory usage

### Manual audit checklist

When adding new CUDA kernels, verify:

1. Every `<...grid, block...>` has a `CHECK_CUDA` or `cudaPeekAtLastError` within 3 lines
2. Grid dimensions do not exceed `deviceProp.maxGridSize`
3. Block dimensions do not exceed `deviceProp.maxThreadsPerBlock`
4. Shared memory per block does not exceed `deviceProp.sharedMemPerBlock`
5. All `cudaMalloc` calls have matching `cudaFree` in the same scope or destructor
6. `cudaMemcpy` directions are correct (HostToDevice vs DeviceToHost)
7. Stream-ordered allocations use the same stream as dependent kernels

### Integrating with CI

```bash
# In CI pipeline
python -m pynerve.validation.audit_cuda --fail-on-error
# Exits non-zero if any launch is unverified
```

Add to CMake:

```cmake
add_custom_target(audit_cuda
    COMMAND ${Python_EXECUTABLE} -m pynerve.validation.audit_cuda
    --src-dir ${CMAKE_SOURCE_DIR}/src/cuda
    --fail-on-error
    COMMENT "Auditing CUDA kernel launches..."
)
```


## GPU memory audit

Beyond launch correctness, the memory audit tracks allocations:

```python
from pynerve.validation import audit_cuda_memory

mem_report = audit_cuda_memory()
# mem_report.total_allocated, mem_report.peak_allocated
# mem_report.leaks: list of unfreed allocations
# mem_report.malloc_count, mem_report.free_count

if mem_report.leaks:
    for leak in mem_report.leaks:
        print(f"Leak: {leak.size} bytes at {leak.address}, "
              f"allocated at {leak.file}:{leak.line}")
```


## FAQ

**Q: How do I fix an unverified launch?**
A: Add `CHECK_CUDA(cudaGetLastError())` or `cudaPeekAtLastError()` immediately after the kernel launch. The audit checks for an error check within 3 lines after `<<<grid, block>>>`.

**Q: Does the audit catch all CUDA errors?**
A: It catches missing error checks, out-of-bounds grid/block dimensions, and excessive shared memory. It does not catch logic errors or data-race conditions -- those require unit tests and careful code review.

**Q: What if a false positive is reported?**
A: The audit may flag launches that use a wrapper macro for error checking instead of a direct call. Annotate the wrapper with `// CHECK_CUDA_WRAPPER` to suppress the warning, or add the wrapper to the audit's allowlist.


### Cross-references

- `pynerve.validation`: Validation overview
- `pynerve.cuda`: CUDA infrastructure
- `pynerve.validation.benchmarks`: Benchmark performance
- `pynerve.validation.determinism`: Deterministic GPU execution
