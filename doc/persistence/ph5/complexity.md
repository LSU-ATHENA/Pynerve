# Complexity

The worst-case time complexity is O(n^3), while typical sparse cases run in O(n^2). Extended clearing provides an estimated 30-50% reduction in active columns. Checksum overhead is O(n) per result, and memory usage is O(n * k).

### PH5 vs PH4: Overhead Breakdown

Core reduction runs at 1x on PH4 and an estimated 0.7-0.9x on PH5, as extended clearing reduces work. Checksum computation adds an estimated 1-2% overhead on PH5 and is not applicable to PH4, making it negligible. Stability checks add an estimated 10-20% overhead on PH5 when enabled and are not applicable to PH4; they are optional. Determinism overhead is roughly 1% for PH4 and 2-5% for PH5, reflecting slightly more bookkeeping. Differentiable tracking adds an estimated 20-50% overhead on PH5 and is not applicable to PH4, only incurred when gradients are needed. Structured logging adds roughly 0.5% for PH4 and 1-2% for PH5, representing minimal impact.

For non-differentiable, non-PARANOID use, PH5 is typically 10-20% *faster* than PH4 due to extended clearing.

Back to [PH5 Engine Overview](index.md)
