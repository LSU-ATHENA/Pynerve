# FAQ

## What experimental algorithms does PH6 include?

PH6 includes five experimental algorithm families: new reduction orderings (adaptive column traversal strategies), approximate clearing (heuristic column elimination with tunable recall), speculative reduction (multi-path reduction with majority voting), adaptive pivoting (runtime pivot-strategy selection based on column density), and block-sparse reduction (cache-blocked column operations for hierarchical memory). Not all may be present in every release -- see the release notes for the specific set included in your version.

## Is PH6 safe for production use?

PH6 is classified as experimental and is not recommended for production use without careful validation. Experimental algorithms may change or be removed between minor releases, performance characteristics may regress on edge cases, and some algorithms (like approximate clearing) can produce incorrect results. If you must use PH6 in production, pin both the library version and algorithm configuration, enable cross-verification against PH4, and extensively test on your specific data.

## How does speculative reduction work?

Speculative reduction launches multiple parallel reduction threads, each using a different random column ordering of the boundary matrix. After all threads complete, the results are compared: if all threads agree on a pairing, it is accepted as correct. For the small fraction of columns (typically 1-5%) where pairings disagree, a deterministic rerun resolves the correct pairing. The approach exploits the observation that most column pairings are ordering-independent. Performance scales with thread count but shows diminishing returns beyond 4-8 threads due to synchronization overhead, memory contention, and increasing disagreement rates.

## What is adaptive pivoting?

Adaptive pivoting selects the best pivot-finding strategy at runtime based on each column's characteristics. Four strategies are available: scan from bottom (O(k), best for dense columns), binary search (O(log k), best for sorted sparse columns), SIMD max scan (best for medium bitset columns), and cached pivot (amortized O(1), best when pivot cache is maintained). The selector chooses a strategy per column using heuristics like column density, sparsity, and bitset size. Adaptive pivoting is the most stable experimental feature in PH6 and is considered safe for production use, typically providing a 5-15% speedup over any single fixed strategy.

## How does PH6 compare to PH4 and PH5?

PH6 shares PH4 and PH5's core architecture and configuration system but layers on newer, potentially higher-performing reduction strategies that are not yet fully battle-tested. PH4 and PH5 provide proven, stable algorithms suitable for production, while PH6 offers bleeding-edge variants like adaptive ordering (1.2-1.5x on sparse data), block-sparse reduction (1.1-1.3x on dense data via cache optimization), and speculative reduction (1.5-3x on multi-core systems). Successful algorithms in PH6 eventually graduate to PH5 and PH4 through a formal process requiring correctness, performance, stability, and determinism validation.


[Back to PH6 Index](index.md)
