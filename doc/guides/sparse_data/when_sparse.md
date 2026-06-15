# When sparse helps

For n = 10,000, full VR has 100 million edges while sparse VR with 1% landmarks has 10,000 edges. For n = 100,000, full VR has 10 billion edges while sparse VR has 1 million edges. For n = 1,000,000, full VR is not feasible (100 billion+ edges) while sparse VR has 100 million edges. When point density exceeds 10%, full VR is acceptable and sparse may not help significantly. When point density is below 1%, full VR is memory-bound while sparse provides a 100 to 1000x reduction.

### Decision guide

Use sparse mode when:
- **n > 50,000** -- full VR matrix exceeds available memory
- **Point density < 1%** -- data has low intrinsic dimensionality
- **Approximate answers acceptable** -- epsilon-interleaving guarantee
- **Rapid prototyping** -- quick landmark selection yields reasonable H1/H2

Use full (exact) mode when:
- **n < 10,000** -- full VR fits comfortably
- **Precision-critical** -- need exact persistence pairs
- **Small features matter** -- deaths within epsilon of births

Back to [Sparse Workflows Overview](index.md)
