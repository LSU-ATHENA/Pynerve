# Comparison to other approximations

[Back to index](index.md)

Sparse VR, witness complexes, and alpha complexes differ along several axes. For error bounds, sparse VR guarantees $\frac{1}{1-\varepsilon}$ interleaving, witness provides a 3-approximation, and alpha is exact up to homotopy equivalence. Sparse VR and witness complexes work with any metric, while alpha is restricted to Euclidean data. All three support any dimensionality, though alpha is typically limited to $d \leq 3$ in practice. In terms of memory scaling, sparse VR is $O(m^{k+1})$, witness is $O(m^{k+1} + nm)$, and alpha is $O(n^{\lceil d/2 \rceil})$. Landmark selection differs as well: sparse VR uses farthest-point sampling, witness supports farthest, random, or kmeans selection, and alpha uses all points without landmark selection.

Sparse VR provides the **strongest approximation guarantee** among
sampling-based methods (a multiplicative interleaving bound), while
witness provides a looser additive bound. Alpha is exact but restricted
to Euclidean data in $\dim \leq 3$.
