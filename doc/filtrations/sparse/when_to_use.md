# When to use sparse VR

[Back to index](index.md)

Use sparse VR when $n$ exceeds 50,000 in any dimension, starting with $\varepsilon = 0.3$. It is also recommended when the data density is below 1\% of the total volume, meaning points are sparse in the ambient space. In memory-constrained environments, sparse VR with a reduced landmark count is the best choice. For streaming pipelines, apply sparse VR independently on each chunk. For exploratory analysis, a looser $\varepsilon$ of 0.4 to 0.5 provides fast initial results.
