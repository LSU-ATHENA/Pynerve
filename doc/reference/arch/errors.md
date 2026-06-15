# Error Propagation

### C++ error handling

All C++ functions return `ErrorResult<T>`:

```cpp
template <typename T>
class ErrorResult {
    bool is_error() const;
    bool is_ok() const;
    T& value();              // UB if is_error()
    const Error& error();    // UB if is_ok()
};

struct Error {
    ErrorCode code;
    std::string message;
    std::string file;
    int line;
    std::string function;
};
```

### Python error translation

C++ errors are translated to Python exceptions via `pybind11::register_exception`:

C++ errors are translated to Python exceptions via pybind11. `E10_GPU_OOM` maps to `pynerve.GPUMemoryError`, `E11_GPU_LAUNCH_FAIL` maps to `pynerve.GPULaunchError`, and `E20_NUM_NAN` maps to `pynerve.NumericalError`. `E30_DET_MISMATCH` becomes `pynerve.DeterminismError`, `E50_PH_ABORT` becomes `pynerve.PersistenceError`, and `E54_PH4_INVALID_INPUT` becomes `pynerve.InvalidArgumentError`. `E74_RACE_CONDITION` maps to `pynerve.Error` at runtime, `E88_INVALID_SIMPLICES` maps to `pynerve.InvalidSimplexError`, and `E94_CONVERGENCE_FAIL` maps to `pynerve.ConvergenceError`.


[Back to Architecture Index](index.md)
