# FAQ

### How many landmarks should I choose?

A good default is 500 landmarks for datasets up to $10^5$ points. For noisier data, increase $m$ to capture more topological structure. For very clean data, $m$ can be as low as 100. A useful heuristic is $m \approx 50 \times 2^{d_{\text{intrinsic}}}$, where $d_{\text{intrinsic}}$ is the intrinsic dimension of the data. In general, $m$ should grow with the intrinsic dimension and the desired resolution of topological features.

### Which landmark selection strategy is best?

Farthest-point (maxmin) sampling is the recommended default. It is deterministic, maximizes coverage of the point set, and provides strong theoretical guarantees on the covering radius. Random sampling is faster and useful for very large datasets or when running multiple trials. K-means centroids work well when the data has a natural cluster structure but are more expensive and non-deterministic.

### How does the witness approximation compare to exact VR?

The witness complex persistence diagram is a 3-approximation of the full VR diagram in bottleneck distance under mild sampling conditions (Lipschitz density on a compact Riemannian manifold). In practice, this means the topological features visible in the VR diagram appear in the witness diagram at slightly different scales, but the overall structure is preserved. The approximation improves as the number of landmarks increases, with the error scaling as $O(m^{-1/d_{\text{intrinsic}}})$.

### What is the weak vs strong witness difference?

Weak witnesses require that all vertices of a simplex be within $\epsilon + m_k(w)$ of some witness $w$, where $m_k(w)$ is the distance to the $(k+1)$-th nearest landmark. Strong witnesses require a stricter condition: the furthest vertex must be within $\epsilon$ and all other vertices strictly closer than the $(k+1)$-th nearest landmark. Weak witnesses produce a richer filtration with a proven 3-approximation ratio, while strong witnesses give a tighter 2-approximation but may miss features. Pynerve uses weak witnesses by default.

### Can I use witness complex for streaming data?

Yes. Since the witness complex operates on a fixed set of landmarks, you can process data in chunks: select landmarks from an initial sample, then process each subsequent chunk independently against the same landmarks. The PyTorch API supports batched inputs, and the chunked processing pattern shown in the distance computation section can be adapted for streaming scenarios. This makes the witness complex well-suited for online or out-of-core applications.

<- [Witness Complex Overview](index.md)
