# High-dim error handling

`src/core/error/high_dim_error_handling.cpp` provides error management for
high-dimensional persistence computation (PH5, PH6 engines):

- Budget-exceeded errors when memory or time limits are hit
- Precision warnings when floating-point accuracy degrades
- Overflow detection for dimension counters exceeding internal limits


[Back to index](index.md)
