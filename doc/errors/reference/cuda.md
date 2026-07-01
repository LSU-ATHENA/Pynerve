# CUDA error mapping

`src/persistence/cuda/error_mapping_cuda.cpp` maps `cudaError_t` to
`nerve::errors::ErrorCode`:

```cpp
namespace nerve::persistence::accelerated {

errors::ErrorCode mapErrorCode(cudaError_t errorCode) {
    if (errorCode == cudaErrorMemoryAllocation)
        return errors::ErrorCode::E10_GPU_OOM;
    if (errorCode == cudaErrorLaunchFailure ||
        errorCode == cudaErrorLaunchTimeout ||
        errorCode == cudaErrorLaunchOutOfResources ||
        errorCode == cudaErrorInvalidConfiguration)
        return errors::ErrorCode::E11_GPU_LAUNCH_FAIL;
    if (errorCode == cudaErrorInitializationError)
        return errors::ErrorCode::E11_GPU_LAUNCH_FAIL;
    return errors::ErrorCode::UNKNOWN;
}

}

// CUDA call audit contract: every launch audited within 3 lines
// Pattern used throughout:
auto err = cudaLaunchKernel(kernel, grid, block, args, shared, stream);
if (err != cudaSuccess) {
    auto nerve_err = mapErrorCode(err);
    // report and return ErrorResult<...>::error(nerve_err, msg)
}
```


[Back to index](index.md)
