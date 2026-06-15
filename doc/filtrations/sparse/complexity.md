# Complexity

[Back to index](index.md)

Landmark selection via greedy farthest-point sampling costs $O(n \cdot m)$. Sparse distance computation is $O(m^2)$ worst-case but typically $O(m)$ thanks to edge threshold pruning. The total simplex count is $O(m^k)$, controlled by the landmark count. Memory usage is $O(m^2 + m^k)$, bounded by the landmark set size. The bottleneck error is $-\log(1-\varepsilon)$, adjustable via the epsilon parameter.

Typical landmark ratios: $m/n \approx 0.01$-$0.05$ for
$1/(1-\varepsilon) \approx 1.1$-$1.5$.
