# When to Use

- **Research**: testing new algorithmic ideas on real data
- **Benchmarking**: comparing experimental strategies against PH4/PH5 baselines
- **Development**: contributing algorithm improvements to the library
- **Cutting-edge applications**: when the latest TDA research is needed

## PH6 vs PH4/PH5: When PH6 Wins

For very sparse, high-dim scenarios, PH4/PH5 performance is baseline 1x while PH6 (experimental) achieves 1.2-1.5x using adaptive ordering, with the advantage coming from early processing of sparse columns. For dense, moderate-size scenarios, PH6 achieves 1.1-1.3x using block-sparse reduction, with better cache utilization. For multi-core systems with 32 or more cores, PH6 achieves 1.5-3x using speculative reduction via parallel exploration. For mixed density columns, PH6 achieves 1.1-1.2x using adaptive pivoting, providing optimal pivot strategy per column. For memory-bandwidth-bound scenarios, PH6 achieves 1.1-1.3x using block-sparse and compression, resulting in fewer cache misses.

## When PH6 Does NOT Win

For complexes with fewer than 10^4 simplices regardless of algorithm, PH6 overhead from adaptive selection and verification dominates. For single-threaded CPU, speculative reduction is not beneficial. For the GPU path, PH6 experimental algorithms are CPU-only. For production validation, PH6 algorithms may not be fully validated.


[Back to PH6 Index](index.md)
