# Definition

Let $L \subset X$ be a set of **landmark** points (typically $|L| = m \ll n$)
and $W = X$ the **witness** set (all points). For each witness $w \in W$,
let $d(w, \ell)$ be the distance to a landmark $\ell \in L$.

### Weak witness

Let $m_k(w)$ be the distance from witness $w$ to its $(k+1)$-th nearest
landmark. A simplex $[\ell_0, \dots, \ell_k] \subset L$ belongs to the
**weak witness complex** at scale $\epsilon$ if there exists a witness
$w \in W$ such that every vertex $\ell_i$ is within distance
$\epsilon + m_k(w)$ of $w$:

$$
\max_{i = 0, \dots, k} d(w, \ell_i) \leq \epsilon + m_k(w)
$$

Equivalently, $\sigma$ appears when some witness $w$ has all vertices of
$\sigma$ within a $\delta$ that is $\epsilon$ beyond $w$'s $(k+1)$-th
nearest landmark distance.

### Strong witness

A stronger condition requires that the **furthest** vertex of $\sigma$ is
within $\epsilon$ of $w$, and all other vertices are strictly closer than
the $(k+1)$-th nearest landmark:

$$
\max_{i = 0, \dots, k} d(w, \ell_i) \leq \epsilon
\quad\text{and}\quad
\max_{i = 0, \dots, k-1} d(w, \ell_i) < d(w, \ell_k)
$$

where $\ell_k$ is the $(k+1)$-th nearest landmark to $w$.

### Pynerve's implementation

Pynerve uses the **weak witness** construction. This provides the
theoretical guarantee that the resulting persistence diagram is a
**3-approximation** of the full VR diagram under mild sampling conditions
(Lipschitz density on a compact Riemannian manifold).

The weak witness is strictly more inclusive than the strong witness:
every strong witness simplex is also a weak witness simplex, but not
vice versa. Weak witnesses produce a richer filtration (more simplices)
while maintaining the approximation guarantee.

<- [Witness Complex Overview](index.md)
