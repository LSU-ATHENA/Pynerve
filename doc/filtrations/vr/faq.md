# FAQ

### Why does VR get so expensive for large point clouds?

The VR complex builds simplices from all subsets of points whose pairwise distances fall below a threshold. The number of $k$-simplices grows as $O(n^{k+1})$, so even modest increases in the number of points $n$ or the maximum dimension $k$ cause a combinatorial explosion. For $n = 10^4$ and $k = 4$, the total number of possible simplices exceeds $10^{17}$. Additionally, the all-pairs distance computation costs $O(n^2)$, and matrix reduction can be $O(m^3)$ in the worst case. This makes exact VR impractical beyond a few thousand points at moderate dimensions.

### When should I use sparse VR instead of exact VR?

Use sparse VR when the point count exceeds roughly ten thousand, or when you need fast exploratory results and can tolerate a controlled approximation. Sparse VR uses epsilon-net sampling to reduce the point set, then builds the VR complex on the landmarks. The output diagram is guaranteed to be $\frac{1}{1-\varepsilon}$-interleaved with the exact diagram in bottleneck distance. It is also useful when memory is constrained, since the landmark set is typically 10 to 100 times smaller than the original data.

### Can VR handle non-Euclidean metrics?

Yes. The VR complex is defined for any metric space --- it requires only a pairwise distance function. Pynerve supports Euclidean, Manhattan, Chebyshev, Minkowski, cosine, correlation, and Hamming metrics out of the box. For custom metrics, you can precompute a distance matrix and pass it via `pynerve.torch.persistence_from_matrix`. The VR construction works the same regardless of whether the distances satisfy the triangle inequality, though the geometric interpretation of the resulting diagram may be less intuitive for non-metric dissimilarities.

### What dimension should I compute?

For most applications, computing homology up to dimension 2 (H0, H1, H2) is sufficient. Dimension 2 captures loops and voids that correspond to meaningful topological features in many real-world datasets. Computing higher dimensions (3 and above) dramatically increases the number of simplices and the computational cost. For point clouds with more than a few thousand points, dimension 2 is often the practical maximum. If you need higher-dimensional features, consider using sparse VR, a witness complex, or a different filtration entirely.

### How do I interpret the birth/death times in the diagram?

Each persistence pair $(b, d)$ represents a topological feature that is born at scale $b$ and dies at scale $d$. In the VR filtration, the scale parameter is the distance threshold $\epsilon$. A feature born at $b$ and dying at $d$ means it appears when the proximity graph connects points at distance $b$ and disappears when the surrounding structure fills in at distance $d$. The persistence $d - b$ measures how long the feature survives; longer persistence generally indicates more robust topological signal, while short-lived features are often considered noise. H0 pairs (connected components) have birth time 0 (all points born at the start) and death times equal to the merger scale; H1 pairs correspond to loops, and H2 pairs to voids.

<- [Vietoris-Rips Overview](index.md)
