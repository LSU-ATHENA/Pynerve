# Complexity

### Construction cost

The construction cost breaks down as follows. Edge enumeration requires $O(n^2)$ all-pairs distance computations in $\mathbb{R}^d$. The $k$-simplex count grows as $O(n^{k+1})$ in the worst case. Filtration sorting takes $O(m_k \log m_k)$ where $m_k$ is the total number of simplices. Boundary matrix fill via CSR construction costs $O(m_k \cdot k)$. Matrix reduction is $O(m_k^3)$ in the worst case, though typically $O(m_k^{\omega})$ with the clearing optimization.

### Practical limits

Practical computation limits depend on the number of points and the maximum homology dimension. For 1,000 points at dimension 2, runtime is under a second and memory stays under a hundred megabytes. At 5,000 points and dimension 2, runtime is 1 to 5 seconds with a few hundred megabytes of RAM. At 10,000 points and dimension 2, runtime increases to 5 to 30 seconds and memory usage is hundreds of megabytes to a couple of gigabytes. At 20,000 points and dimension 2, runtime reaches 30 to 120 seconds with a few to several gigabytes. For higher dimensions, 5,000 points at dimension 3 requires 10 to 60 seconds and one to several gigabytes, while 10,000 points at dimension 3 takes 60 to 300 seconds and a few to tens of gigabytes.

Beyond $n \approx 10^4$ at dim 3, exact VR becomes impractical on a single
machine. For high-dimensional data ($d > 10$), the curse of dimensionality
makes all-pairs distances meaningful only up to $n \approx 10^3$.

### Memory blowup in high dimensions

The boundary matrix is stored as an array of column vectors. Each column
represents a simplex and stores its boundary (list of codimension-1 faces).
The total number of simplices in a full VR complex up to dimension $k$ is

$$
\sum_{i=0}^{k} \binom{n}{i+1} \approx O(n^{k+1}).
$$

For $n = 10^4$ and $k = 4$, this exceeds $10^{17}$ possible simplices --
infeasible on any hardware. Practical computation limits the dimension to
$k \leq 3$ for $n > 2000$, or uses approximation algorithms.


<- [Vietoris-Rips Overview](index.md)
