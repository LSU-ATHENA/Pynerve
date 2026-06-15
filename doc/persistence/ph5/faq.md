# FAQ

### What does dimension-cascading clearing do?

Dimension-cascading clearing extends standard clearing beyond a single column. When a column is cleared, the algorithm recursively follows the pairing chain to clear additional birth columns that would otherwise participate in future pivot conflicts. On average, 2-5 columns are cleared per death, reducing active column count by an estimated 30-50% compared to standard clearing.

### How does PH5 ensure determinism?

PH5 provides four determinism levels: NONE (no guarantees), BASIC (same input, same output on same hardware), STRICT (plus RNG seed control and fixed-point comparisons), and PARANOID (internal double computation with cross-run checksum comparison). Deterministic OpenMP scheduling, seeded random number generation, and explicit seed propagation through all random processes ensure reproducibility.

### What is the checksum used for?

The SHA-256 checksum provides a compact fingerprint of the computation result, computed over a canonical serialization of the persistence pairs, Betti numbers, and metadata. It is used for cross-version reproducibility verification, distributed computation validation, and CI/CD pipeline correctness checks. The checksum is deterministic for the same input and same configuration across library versions.

### How does PH5 differ from PH4?

PH5 extends PH4 with four major additions: dimension-cascading clearing for better performance (estimated 10-20% faster on sparse data), SHA-256 checksum validation, fine-grained determinism contracts with PARANOID cross-run verification, and differentiable persistence operations for gradient-based learning. PH5 also adds advanced column compression (inter-column deduplication, run-length encoding, adaptive representation switching), structured error logging, and stability testing.

### When should I use PH5 instead of PH4 or PH6?

Use PH5 when you need checksum validation, determinism guarantees, or differentiable persistence for a validated or regulated pipeline. Use PH4 when you need maximum simplicity, minimal memory footprint, or are doing quick exploration. Use PH6 (experimental) when you want the latest algorithmic innovations and maximum throughput on very large datasets. PH5 is the recommended default for production use where correctness guarantees matter.

Back to [PH5 Engine Overview](index.md)
