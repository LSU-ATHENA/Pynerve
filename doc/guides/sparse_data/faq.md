# FAQ

**Q: When should I use random landmark selection vs maxmin?**
A: Random is fastest and works well for dense, uniform data. Maxmin provides better coverage for non-uniform or sparse data and is deterministic. If you need reproducible results and your data has uneven density, prefer maxmin.

**Q: How do I choose the landmark ratio for my dataset?**
A: Start with 1% for datasets up to 100K points. For larger datasets (500K+), use 0.1--0.5%. The trade-off is accuracy versus performance: higher ratios give tighter error bounds but use more memory. For near-exact results with moderate data sizes, use 5--10%.

**Q: What is the epsilon-interleaving guarantee?**
A: It guarantees that the bottleneck distance between the sparse VR persistence diagram and the full VR diagram is bounded by epsilon times the max radius. Features with persistence significantly larger than epsilon are real; smaller features may be artifacts of the approximation.

**Q: Can sparse VR be combined with edge collapse?**
A: Yes. These techniques are complementary and composable. Sparse VR reduces the landmark set while edge collapse further reduces the 1-skeleton. Together they can achieve a 10,000x reduction in simplices compared to full VR while preserving epsilon-interleaving guarantees.

**Q: Does sparse VR work on GPU?**
A: Yes. Pynerve supports a CUDA_HYBRID backend where landmark selection runs on CPU, landmark distance matrices and edge extraction run on GPU, and column reduction runs on GPU. The GPU sparse path handles up to 100K landmarks and 10 million+ simplices.

**Q: What happens when data has very high intrinsic dimensionality?**
A: Sparse VR still works but the benefits diminish. Landmark selection (especially farthest-point sampling) becomes more expensive at O(k * n * dim). For dim > 100, consider random sampling instead. The epsilon-net covering radius also grows with dimension, requiring more landmarks to maintain accuracy.

Back to [Sparse Workflows Overview](index.md)
