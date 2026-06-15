# PH6 Engine

> Engine: `PersistenceEngine.PH6`. Block-sparse speculative engine with
> cache-blocked reduction and adaptive pivoting.
> Recommended for >100K points. Max_dim capped at 6.

Experimental persistence engine incorporating the latest algorithmic innovations. PH6 shares PH5's architecture and configuration system but uses newer, potentially higher-performing reduction strategies that may not yet be fully battle-tested.

> **Status:** Experimental. The API and behavior may change between releases without a major version bump.

This document has been split into subpages for easier navigation:

- [Quick Start](ph6/quickstart.md)  --  Getting started with PH6
- [Features](ph6/features.md)  --  Experimental algorithms and graduation path
- [Compatibility](ph6/compatibility.md)  --  Configuration interface and PH6-specific options
- [Complexity](ph6/complexity.md)  --  Time and memory complexity bounds
- [When to Use](ph6/when_to_use.md)  --  PH6 vs PH4/PH5 tradeoffs
- [Migration Path](ph6/migration.md)  --  Graduation history and tracking experimental features
- [Common Pitfalls](ph6/pitfalls.md)  --  Production use, benchmarking, memory, and correctness
- [PH6 Internals](ph6/internals.md)  --  Feature registry, algorithm implementations, memory layout
- [Benchmark Methodology](ph6/benchmarking.md)  --  Reliable performance measurement
- [Advanced Configuration Examples](ph6/configuration.md)  --  Performance, correctness, and memory-constrained setups
- [FAQ](ph6/faq.md)  --  Frequently asked questions
- [References](ph6/references.md)  --  Academic and technical references
