# FAQ

### What does the lowest-1 (pivot) represent?

The lowest-1 of a column is the largest row index containing a non-zero entry. In the context of persistent homology, the pivot identifies the *youngest* face in the boundary of a simplex -- the face that appeared most recently in the filtration. When the pivot of column j is row p, it means simplex sigma_p is a face of simplex sigma_j and sigma_p appeared later in the filtration than any other face in sigma_j's boundary. The pairing (p, j) indicates that sigma_p creates a homology class that sigma_j later kills.

### Why does the algorithm have O(n^3) worst-case complexity?

The worst-case bound arises from three nested levels: the outer loop iterates over n columns (one per simplex), the inner while-loop may require up to O(n) XOR operations per column when many columns share the same pivot, and each column XOR takes O(k) time where k is the column length. In the worst case (e.g., a full simplex on d vertices with all simplices present), k = O(n), giving O(n * n * n) = O(n^3). In practice, sparse column representations and the clearing optimization dramatically reduce this cost.

### How does clearing optimization work?

Clearing exploits the observation that once a column j is paired as a death with pivot p, the boundary column of the birth simplex at row p will never be needed again. The birth column can therefore be set to all zeros immediately after the pairing, which saves it from participating in future XOR operations. This eliminates many expensive column operations, particularly when birth columns are dense. The optimization was introduced by Chen and Kerber (2011) and is safe because no future column can have the same pivot after the death column is reduced.

### What is the difference between positive and negative simplices?

A positive simplex (or birth) creates a new homology class -- its reduced column becomes all zeros, meaning its boundary cycles were not already boundaries of existing simplices. A negative simplex (or death) kills an existing homology class -- its reduced column has a unique pivot that matches the birth simplex it kills. In the pairing (p, j) produced by standard reduction, sigma_p is the positive simplex (birth) and sigma_j is the negative simplex (death).

### When should I use standard reduction vs cohomology?

Standard reduction is best for dense filtrations, small complexes (n < 10^5), and when computational resources are not a primary concern. Cohomology is preferred for sparse filtrations, large complexes (n > 10^5), and GPU-accelerated computation. If the boundary matrix columns are expected to grow dense during reduction, standard reduction with clearing and compression is a solid choice. If columns stay sparse (common in high-dimensional sparse data), cohomology typically achieves 2-10x speedup over standard reduction.

<- [Standard Reduction Overview](index.md)
