# Error registry

Singleton `ErrorRegistry` provides:

```cpp
auto& reg = ErrorRegistry::instance();

// Query error metadata
const auto& meta = reg.getMetadata(ErrorCode::E10_GPU_OOM);
// -> {.category = GPU_COMPUTE, .severity = ERROR,
//     .name = "E10_GPU_OOM", .description = "GPU out of memory",
//     .action_hint = "Reduce batch size or free device memory"}

// Report error with context
ErrorContext ctx;
ctx.operation_name = "compute_persistence";
ctx.component_name = "ph5_engine";
reg.reportError(ErrorCode::E10_GPU_OOM, ctx);

// Check operation health
if (reg.hasOperationFailed("compute_persistence")) {
    // abort or fallback
}
```


[Back to index](index.md)
