# Quick start

```python
from pynerve.exceptions import (
    NerveError, GPUError, PersistenceError,
    E10_GPU_OOM, E50_PH_ABORT,
    translate_cpp_exception,
)

# All errors inherit from NerveError
try:
    result = compute_persistence(points)
except GPUError as e:
    print(f"GPU error (code {e.error_code}): {e}")
    # -> GPU error (code 512): CUDA out of memory
except PersistenceError as e:
    print(f"Persistence error: {e}")
```

Error code taxonomy with component-aware error registry, CUDA error mapping,
pybind11 exception wrapping, and a per-component event system. Every CUDA
launch is audited for error within 3 lines of the kernel call.


[Back to index](index.md)
