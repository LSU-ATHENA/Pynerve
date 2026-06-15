# Correctness

This section documents Pynerve's approach to correctness  --  field coefficients, floating-point guarantees, determinism system, validation infrastructure, and testing coverage.

## Topics

- [Field Coefficients](fields.md)  --  Z2 and prime field arithmetic, field selection API, complexity
- [Determinism](determinism.md)  --  Bitwise reproducibility specification, determinism API, contract levels, limitations
- [Floating-Point Assumptions & Numerical Stability](floating_point.md)  --  Compilation flags, IEEE 754 compliance, numerical tolerance, potential issues
- [GPU Determinism](gpu_determinism.md)  --  Fixed-tree reductions, GPU determinism protocol, RFA for cross-architecture
- [MPI Determinism](mpi_determinism.md)  --  Fixed-process and cross-count MPI, binned accumulation protocol
- [Validation Infrastructure, Coverage & CI](validation.md)  --  Test counts, kernel audit, property-based tests, error taxonomy, coverage by component, CI matrix, regression protocol
- [Runtime Validation & Formal Properties](runtime_validation.md)  --  Runtime assertions, formal properties of persistence diagrams
- [Precision Policies](precision.md)  --  P64, P32_DISTANCE, P32, P16_DISTANCE policies
- [Input Validation Rules](input_validation.md)  --  Point cloud, diagram, and options validation
- [Reproducibility Checklist & Reference Results](reproducibility.md)  --  Pre-publication checklist, environment capture, reference test results
- [Security Model](security.md)  --  Threat model, memory safety, non-goals
- [FAQ](faq.md)  --  Frequently asked questions


[Back to doc/reference](../correctness.md)
