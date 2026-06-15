# Sparse Workflows

Sparse approximations scale persistence computation to millions of points by constructing a reduced simplicial complex that provably approximates the full VR filtration.

## Sections

- [Sparse VR Construction](construction.md)  --  Epsilon-net landmarks, witness complex, link-time construction
- [Sparse Boundary Matrix](boundary_matrix.md)  --  CSR format, column operations, memory complexity
- [Edge Collapse](edge_collapse.md)  --  Edge collapse algorithm, link condition, tolerance
- [When Sparse Helps](when_sparse.md)  --  Decision guide for sparse vs full mode
- [Approximate vs Exact](approximate_vs_exact.md)  --  Epsilon-interleaving guarantee, memory scaling
- [API Reference](api.md)  --  Full API documentation for sparse workflows
- [Performance Tuning](performance.md)  --  Landmark selection, GPU acceleration, error bounds, parameter reference
- [FAQ](faq.md)  --  Frequently asked questions
