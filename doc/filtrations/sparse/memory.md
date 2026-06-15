# Memory savings analysis

[Back to index](index.md)

### Simplex count comparison

The sparse VR's memory advantage comes from reducing both the number of
points and the edge density. For $n = 10{,}000$ points and dimension $k = 2$, the full VR complex has approximately $1.7 \times 10^{11}$ simplices while sparse VR with $\varepsilon = 0.3$ has $3.4 \times 10^5$, a savings factor of $5 \times 10^5$. At dimension $k = 3$, VR has $4.2 \times 10^{14}$ simplices versus sparse VR's $3.4 \times 10^7$, saving $1.2 \times 10^7$x. For $n = 50{,}000$ at $k = 2$, VR has $2.1 \times 10^{13}$ simplices versus sparse VR's $4.3 \times 10^6$, saving $4.9 \times 10^6$x. For $n = 100{,}000$ at $k = 2$, VR has $1.7 \times 10^{14}$ simplices versus sparse VR's $1.4 \times 10^7$, saving $1.2 \times 10^7$x.

### Memory breakdown

For $n = 100,000$, $\varepsilon = 0.3$, $m = 3000$: the full VR requires a distance matrix on the order of tens of gigabytes, while sparse VR does not store the full distance matrix at all. Landmark distances consume only a few tens of megabytes ($m^2 \times 4B$). The edge list in VR occupies tens of gigabytes, while sparse VR uses tens of megabytes ($m \times 4B \times \text{degree}$). The boundary matrix for VR requires hundreds of gigabytes or more, whereas sparse VR needs hundreds of megabytes. Total memory for VR is hundreds of gigabytes or more, compared to hundreds of megabytes for sparse VR.

### Scaling laws

- VR memory scales as $O(n^{k+1})$ with point count and dimension.
- Sparse VR memory scales as $O(m^{k+1})$ with $m = O(\varepsilon^{-d} \log n)$.
- For fixed $\varepsilon$, memory grows linearly with $n$ (via $m$ growth),
  not exponentially.
