# PH5 Engine

> Engine: `PersistenceEngine.PH5`. Unified adaptive engine combining cohomology, approximate mode, and iterative refinement. Recommended for 10K--1M points. Max_dim capped at 5.

Advanced persistence engine with extended clearing, compression, checksum validation, and stringent determinism contracts. PH5 builds on the PH4 foundation with additional correctness guarantees suitable for validated computation pipelines.

## Sections

- [Quick Start](quickstart.md)  --  Basic usage examples
- [Features](features.md)  --  Extended clearing, advanced compression, checksums, determinism contracts, stability testing, differentiable ops, structured logging
- [Complexity](complexity.md)  --  Time/memory complexity and PH5 vs PH4 overhead
- [Implementation Details](implementation.md)  --  Engine architecture, checksum, determinism contract internals
- [When to Use](when_to_use.md)  --  Recommended use cases
- [When NOT to Use](when_not_to_use.md)  --  Limitations and PH5 vs PH4 decision guide
- [Common Pitfalls](pitfalls.md)  --  Known issues and workarounds
- [References](references.md)  --  Academic references
- [FAQ](faq.md)  --  Frequently asked questions
