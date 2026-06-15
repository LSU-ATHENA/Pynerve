# Interleaving guarantee proof sketch

[Back to index](index.md)

Define the sparse VR filtration as:

$$
\operatorname{SparseVR}_t(L) = \operatorname{VR}_{d_L(t)}(L)
$$

where $d_L(t) = \inf\{s \geq 0 : \text{edge } (\ell_i, \ell_j) \text{ present}\}$.
The scaling function $d_L$ satisfies:

$$
d(x, y) \leq d_L(\pi(x), \pi(y)) \leq \frac{1}{1-\varepsilon} \cdot d(x, y)
$$

where $\pi: X \to L$ maps each point to its nearest landmark. This gives
the interleaving:

$$
\operatorname{VR}_t(X) \subseteq \operatorname{SparseVR}_t(L)
\subseteq \operatorname{VR}_{\frac{t}{1-\varepsilon}}(X)
$$

which implies the bottleneck bound:

$$
d_B(\operatorname{Dgm}_\text{VR}, \operatorname{Dgm}_\text{sparse})
\leq \log\left(\frac{1}{1-\varepsilon}\right)
$$
