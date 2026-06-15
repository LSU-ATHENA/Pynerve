# FAQ

[Back to index](index.md)

### How does the epsilon parameter affect accuracy vs speed?

The epsilon parameter $\varepsilon$ directly controls the tradeoff between accuracy and computational cost. Smaller values (e.g., 0.1) yield more landmarks, tighter interleaving bounds, and higher accuracy, but increase memory and runtime. Larger values (e.g., 0.5) produce fewer landmarks, faster construction, and lower memory usage, at the cost of looser approximation. The bottleneck error bound grows as $-\log(1-\varepsilon)$, so the accuracy penalty accelerates as $\varepsilon$ approaches 1. In practice, $\varepsilon = 0.3$ offers a good default balance.

### When should I use sparse VR instead of exact VR?

Use sparse VR when the dataset exceeds roughly 10,000 points, especially in higher dimensions where exact VR becomes prohibitively expensive. It is also the right choice in memory-constrained environments, for exploratory analysis where approximate results are acceptable, or when processing streaming data. For datasets under 10,000 points, or when exact results are required (e.g., for verification or publication-quality diagrams), exact VR is preferable.

### What is the interleaving guarantee in practice?

The $\frac{1}{1-\varepsilon}$ interleaving guarantee means that every homology class appearing in the true VR filtration also appears in the sparse VR filtration (possibly at a slightly later scale), and no spurious class persists longer than the interleaving bound. In practice, the observed bottleneck distance is typically 40-60% of the theoretical worst-case bound for random data, so the guarantee is conservative for most real-world datasets.

### How are landmarks selected?

Landmarks are selected via greedy farthest-point sampling (maxmin). Starting from an arbitrary initial point, each successive landmark is chosen as the point farthest from all previously selected landmarks. This produces a nested sequence of epsilon-nets with decreasing covering radii. The ordering matters: earlier landmarks cover larger regions, while later landmarks fill in fine-scale structure. The covering radius at which each landmark was added determines its role in the sparse distance computation.

### Can sparse VR be used for streaming data?

Yes. The sparse VR construction supports incremental updates: as new points arrive, any point farther than the current covering radius becomes a new landmark, and the sparse filtration is recomputed on the updated landmark set. This makes it suitable for infinite or very large streams where storing the entire history is infeasible. The streaming variant maintains the same interleaving guarantee, though the landmark count grows over time with the expanding domain.
