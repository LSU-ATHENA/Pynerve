# Don Sheehy's epsilon-net theory

[Back to index](index.md)

The sparse VR construction is based on the theory of **epsilon-nets**
developed by Don Sheehy (2013). The key insight is that a point cloud
can be approximated by a subset of points (an $\varepsilon$-net) while
preserving the persistence diagram up to a bounded error.

### Epsilon-net construction

Let $(X, d)$ be a finite metric space. An **$\varepsilon$-net** is a subset
$L \subseteq X$ such that:

1. **Packing:** For any $\ell_i \neq \ell_j \in L$, $d(\ell_i, \ell_j) \geq \varepsilon$.
2. **Covering:** For every $x \in X$, there exists $\ell \in L$ with
   $d(x, \ell) \leq \varepsilon$.

The epsilon-net is built via **farthest-point sampling** (also called
maxmin sampling or greedy permutation):

```python
def farthest_point_sample(X, m):
    """Select m landmarks via farthest-point sampling."""
    n = len(X)
    distances = np.full(n, np.inf)
    landmarks = []
    # Pick the first point (e.g., the point with maximum norm)
    idx = np.argmax(np.linalg.norm(X, axis=1))
    landmarks.append(idx)
    for _ in range(1, m):
        # Update distances to nearest landmark
        d = np.linalg.norm(X - X[idx], axis=1)
        distances = np.minimum(distances, d)
        # Select the farthest point
        idx = np.argmax(distances)
        landmarks.append(idx)
    return np.array(landmarks)
```

This produces a sequence $L_1 \subset L_2 \subset \dots \subset L_m$ of
nested epsilon-nets, where each $L_i$ is an $\varepsilon_i$-net with
$\varepsilon_i$ decreasing as $i$ increases.

### Sparse distance matrix

The sparse distance $d_L$ is defined as:

$$
d_L(\ell_i, \ell_j) = \begin{cases}
d(\ell_i, \ell_j) & \text{if } \ell_i, \ell_j \text{ are ``close''} \\
\infty & \text{otherwise}
\end{cases}
$$

where "close" means the distance is within a threshold determined by the
epsilon-net radii. Specifically, for a pair of landmarks $(\ell_i, \ell_j)$:

- Let $r_i$ be the radius at which $\ell_i$ was added to the net.
- Compute the **scale factor** $s = \max(r_i, r_j) / \varepsilon$.
- If $d(\ell_i, \ell_j) \leq s \cdot (1 + \varepsilon)$, include the edge.

This produces a **sparse graph** with $O(m)$ edges instead of $O(m^2)$,
dramatically reducing the VR complex size.
