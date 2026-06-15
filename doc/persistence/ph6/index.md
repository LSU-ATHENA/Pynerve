# PH6 Engine

> Engine: `PersistenceEngine.PH6`. Block-sparse speculative engine with
> cache-blocked reduction and adaptive pivoting.
> Recommended for >100K points. Max_dim capped at 6.

Experimental persistence engine incorporating the latest algorithmic innovations. PH6 shares PH5's architecture and configuration system but uses newer, potentially higher-performing reduction strategies that may not yet be fully battle-tested.

> **Status:** Experimental. The API and behavior may change between releases without a major version bump.

## Sections

- [Quick Start](quickstart.md)  --  Getting started with PH6
- [Features](features.md)  --  Experimental algorithms and graduation path
- [Compatibility](compatibility.md)  --  Configuration interface and PH6-specific options
- [Complexity](complexity.md)  --  Time and memory complexity bounds
- [When to Use](when_to_use.md)  --  PH6 vs PH4/PH5 tradeoffs
- [Migration Path](migration.md)  --  Graduation history and tracking experimental features
- [Common Pitfalls](pitfalls.md)  --  Production use, benchmarking, memory, and correctness
- [PH6 Internals](internals.md)  --  Feature registry, algorithm implementations, memory layout
- [Benchmark Methodology](benchmarking.md)  --  Reliable performance measurement
- [Advanced Configuration Examples](configuration.md)  --  Performance, correctness, and memory-constrained setups
- [FAQ](faq.md)  --  Frequently asked questions
- [References](references.md)  --  Academic and technical references
