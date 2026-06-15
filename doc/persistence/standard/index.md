# Standard Matrix Reduction (PH0)

> Engine: `PersistenceEngine.PH0`

Reduce a boundary matrix column-by-column over Z2 coefficients, tracking the *lowest-1* entry of each column to pair birth simplices with death simplices. This is the foundational algorithm for persistent homology computation, introduced by Zomorodian and Carlsson (2005), and all modern engines build upon it.

- [Quick Start](quickstart.md) - Basic usage examples with Python and PyTorch APIs
- [Algorithm](algorithm.md) - Standard reduction algorithm, boundary matrix construction, and walkthrough
- [Clearing Optimization](clearing.md) - Chen & Kerber clearing optimization and its variants
- [Compression](compression.md) - Skipped columns, trailing zero stripping, sparse representation
- [Complexity Analysis](complexity.md) - Time and memory complexity with worst/average case analysis
- [Code Examples](code_examples.md) - Manual reduction in Python and C++, Python API usage
- [Internal Implementation Details](implementation.md) - Data structures, memory layout, SIMD XOR optimization
- [Common Pitfalls](pitfalls.md) - Integer overflow, column density growth, simplex ordering
- [Comparison](comparison.md) - Standard vs cohomology reduction
- [References](references.md) - Foundational papers and further reading
- [FAQ](faq.md) - Frequently asked questions
