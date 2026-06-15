# FAQ

**Q: How do I add a new error code?**
A: Add an entry to the `ErrorCode` enum in `include/nerve/errors/errors.hpp` with a unique hex value in the appropriate component range. Then register its metadata (category, severity, description, action hint) in the `ErrorRegistry` initialization. The error will automatically be available in Python bindings.

**Q: What happens when an unmapped CUDA error occurs?**
A: The `mapErrorCode` function returns `ErrorCode::UNKNOWN` for unmapped CUDA errors. The error is still reported through the `ErrorRegistry` with the original CUDA error message preserved in the context metadata.

**Q: Can I suppress specific error codes?**
A: Yes. Use `ConfigurableErrorSystemBase::setErrorEnabled(code, false)` to disable error reporting for specific codes. This is useful for expected transient failures in retry loops.


[Back to index](index.md)
