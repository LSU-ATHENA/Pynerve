# Correctness

Pynerve is designed for bitwise reproducibility and numerical stability across CPU, GPU, and distributed configurations. This document is the index for the detailed correctness documentation.

## Sections

- [Field Coefficients](correctness_detail/fields.md)  --  Z2 and prime field arithmetic, field selection API, complexity
- [Determinism](correctness_detail/determinism.md)  --  Bitwise reproducibility specification, determinism API, contract levels, limitations
- [Floating-Point Assumptions & Numerical Stability](correctness_detail/floating_point.md)  --  Compilation flags, IEEE 754 compliance, numerical tolerance, potential issues
- [GPU Determinism](correctness_detail/gpu_determinism.md)  --  Fixed-tree reductions, GPU determinism protocol, RFA for cross-architecture
- [MPI Determinism](correctness_detail/mpi_determinism.md)  --  Fixed-process and cross-count MPI, binned accumulation protocol
- [Validation Infrastructure, Coverage & CI](correctness_detail/validation.md)  --  Test counts, kernel audit, property-based tests, error taxonomy, coverage by component, CI matrix, regression protocol
- [Runtime Validation & Formal Properties](correctness_detail/runtime_validation.md)  --  Runtime assertions, formal properties of persistence diagrams
- [Precision Policies](correctness_detail/precision.md)  --  P64, P32_DISTANCE, P32, P16_DISTANCE policies
- [Input Validation Rules](correctness_detail/input_validation.md)  --  Point cloud, diagram, and options validation
- [Reproducibility Checklist & Reference Results](correctness_detail/reproducibility.md)  --  Pre-publication checklist, environment capture, reference test results
- [Security Model](correctness_detail/security.md)  --  Threat model, memory safety, non-goals
- [FAQ](correctness_detail/faq.md)  --  Frequently asked questions
