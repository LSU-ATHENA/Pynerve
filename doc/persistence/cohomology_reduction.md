# Cohomology Reduction (PH3)

> Engine: `PersistenceEngine.PH3`. This is the default algorithm used by AUTO
> for most inputs.

Reduce the *coboundary matrix* instead of the boundary matrix, processing columns in reverse filtration order. Each column kills the highest-dimensional row it can, leading to sparser intermediate matrices and faster computation.

Sections:
- [Quick Start](cohomology/quickstart.md) · [Intuition](cohomology/intuition.md) · [Algorithm](cohomology/algorithm.md)
- [SIMD](cohomology/simd.md) · [GPU](cohomology/gpu.md) · [Complexity](cohomology/complexity.md)
- [Implementation](cohomology/implementation.md) · [When to Use](cohomology/when_to_use.md) · [Pitfalls](cohomology/pitfalls.md)
