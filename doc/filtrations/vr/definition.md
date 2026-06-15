# Definition

Let $(X, d)$ be a finite metric space. For a scale parameter $\epsilon \geq 0$,
the **Vietoris-Rips complex** $\operatorname{VR}_\epsilon(X)$ is the simplicial
complex whose $k$-simplices are all subsets $\{x_0, \dots, x_k\} \subseteq X$
with diameter at most $\epsilon$:

$$
\operatorname{VR}_\epsilon(X) = \{\sigma \subseteq X : \max_{x, y \in \sigma} d(x, y) \leq \epsilon\}.
$$

Equivalently, $\operatorname{VR}_\epsilon(X)$ is the **clique complex** (flag
complex) of the $\epsilon$-proximity graph $G_\epsilon = (X, E_\epsilon)$ where

$$
E_\epsilon = \{\{x, y\} : d(x, y) \leq \epsilon\}.
$$

A simplex $\sigma$ belongs to $\operatorname{VR}_\epsilon(X)$ iff every edge
of $\sigma$ belongs to $G_\epsilon$.

As $\epsilon$ increases from $0$ to $\infty$, the complexes form a **filtration**

$$
\operatorname{VR}_{\epsilon_1}(X) \subseteq \operatorname{VR}_{\epsilon_2}(X)
\subseteq \cdots \subseteq \operatorname{VR}_{\epsilon_\ell}(X)
$$

with $\epsilon_1 < \epsilon_2 < \dots < \epsilon_\ell$. Persistent homology
tracks the birth and death of homology classes across this filtration.

### Relationship to Cech complex

The VR complex is an approximation of the Cech complex:

$$
\operatorname{Cech}_\epsilon(X) \subseteq \operatorname{VR}_\epsilon(X)
\subseteq \operatorname{Cech}_{2\epsilon}(X)
$$

where $\operatorname{Cech}_\epsilon(X)$ contains a simplex $[x_0, \dots, x_k]$
whenever the intersection of $\epsilon$-balls around these points is
non-empty. This 2-parameter interleaving gives the VR complex its theoretical
guarantees: persistence diagrams computed from VR are at most a factor of 2
from the true Cech diagram in bottleneck distance.


<- [Vietoris-Rips Overview](index.md)
