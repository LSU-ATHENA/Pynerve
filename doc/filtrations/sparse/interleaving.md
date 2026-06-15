# 1/(1-epsilon) interleaving guarantee

[Back to index](index.md)

Let $\operatorname{Dgm}_\text{VR}$ and $\operatorname{Dgm}_\text{sparse}$
be the persistence diagrams of the full VR and sparse VR filtrations. Then:

$$
d_B(\operatorname{Dgm}_\text{VR}, \operatorname{Dgm}_\text{sparse})
\leq \log\left(\frac{1}{1-\varepsilon}\right)
$$

where $d_B$ is the **bottleneck distance**. Equivalently, the two diagrams
are $\frac{1}{1-\varepsilon}$-interleaved.

### Interleaving interpretation

The $\frac{1}{1-\varepsilon}$ interleaving means:

$$
\operatorname{VR}_t(X) \subseteq \operatorname{SparseVR}_{t}(L)
\subseteq \operatorname{VR}_{\frac{t}{1-\varepsilon}}(X)
$$

for all $t \geq 0$. Every homology class of the true VR is captured by the
sparse VR (possibly with a delayed birth), and no spurious classes persist
beyond the bound.

### Parameter selection

The epsilon parameter controls the accuracy-sparsity tradeoff. With $\varepsilon = 0.1$, the bottleneck bound is $0.105$, the typical landmark ratio $m/n$ ranges from 0.10 to 0.20, and the use case is high accuracy with moderate sparsity. With $\varepsilon = 0.2$, the bound is $0.223$, $m/n$ ranges from 0.05 to 0.10, and the use case is balanced. The default $\varepsilon = 0.3$ gives a bound of $0.357$, $m/n$ from 0.02 to 0.05, and good accuracy with high sparsity. With $\varepsilon = 0.4$, the bound is $0.511$, $m/n$ ranges from 0.01 to 0.03, suitable for exploratory analysis. With $\varepsilon = 0.5$, the bound is $0.693$, $m/n$ ranges from 0.005 to 0.02, suitable for quick sketches.
