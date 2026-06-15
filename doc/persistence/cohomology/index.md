# Cohomology Reduction (PH3)

> Engine: `PersistenceEngine.PH3`. This is the default algorithm used by AUTO
> for most inputs.

Reduce the *coboundary matrix* instead of the boundary matrix, processing columns in reverse filtration order. Each column kills the highest-dimensional row it can, leading to sparser intermediate matrices and faster computation.

## Sections

- [Quick Start](quickstart.md)  --  Get up and running with cohomology reduction
- [Intuition](intuition.md)  --  Why coboundary reduction is faster and the dual perspective
- [Algorithm](algorithm.md)  --  Pseudocode, emergent pair detection, and a detailed walkthrough
- [SIMD Acceleration](simd.md)  --  SSE4.1, AVX2, AVX-512 kernel selection and pivot finding
- [GPU Acceleration](gpu.md)  --  Warp-level architecture, pivot finding, column XOR, determinism
- [Complexity](complexity.md)  --  Time and memory complexity, FLOPS analysis, memory locality
- [Internal Implementation Details](implementation.md)  --  Sparse coboundary construction, bit-packed columns, clearing, GPU coherence
- [When to Use](when_to_use.md)  --  Decision guide, when cohomology wins, when standard wins, PH4 auto-selection
- [Common Pitfalls](pitfalls.md)  --  Reverse pair ordering, dimension filter, emergent pair misidentification, memory overhead, GPU transfer, determinism
- [Comparison Tables](comparison.md)  --  Algorithm characteristics and performance by complex type
- [FAQ](faq.md)  --  Frequently asked questions
- [References](references.md)  --  Foundational papers, implementation papers, GPU methods, surveys
