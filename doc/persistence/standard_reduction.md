# Standard Matrix Reduction (PH0)

> Engine: `PersistenceEngine.PH0`

Reduce a boundary matrix column-by-column over Z2 coefficients, tracking the *lowest-1* entry of each column to pair birth simplices with death simplices. This is the foundational algorithm for persistent homology computation, introduced by Zomorodian and Carlsson (2005), and all modern engines build upon it.

- [Quick Start](standard/quickstart.md) - Basic usage examples with Python and PyTorch APIs
- [Algorithm](standard/algorithm.md) - Standard reduction algorithm, boundary matrix construction, and walkthrough
- [Clearing Optimization](standard/clearing.md) - Chen & Kerber clearing optimization and its variants
- [Compression](standard/compression.md) - Skipped columns, trailing zero stripping, sparse representation
- [Complexity Analysis](standard/complexity.md) - Time and memory complexity with worst/average case analysis
- [Code Examples](standard/code_examples.md) - Manual reduction in Python and C++, Python API usage
- [Internal Implementation Details](standard/implementation.md) - Data structures, memory layout, SIMD XOR optimization
- [Common Pitfalls](standard/pitfalls.md) - Integer overflow, column density growth, simplex ordering
- [Comparison](standard/comparison.md) - Standard vs cohomology reduction
- [References](standard/references.md) - Foundational papers and further reading
- [FAQ](standard/faq.md) - Frequently asked questions
