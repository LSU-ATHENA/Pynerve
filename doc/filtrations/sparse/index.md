# Sparse Vietoris-Rips complex

Don Sheehy's sparse approximation of the Vietoris-Rips filtration. The sparse
VR complex produces a persistence diagram that is guaranteed to be
$\frac{1}{1-\varepsilon}$-interleaved with the true VR diagram, while using
$O(m)$ simplices where $m$ is the number of landmarks.

## Sections

- [Definition](definition.md)  --  Three-step construction overview
- [Don Sheehy's epsilon-net theory](epsilon_net.md)  --  Epsilon-net construction and sparse distance matrix
- [1/(1-epsilon) interleaving guarantee](interleaving.md)  --  Bottleneck distance bound and parameter selection
- [Landmark selection details](landmarks.md)  --  Greedy permutation algorithm and stopping criteria
- [Memory savings analysis](memory.md)  --  Simplex count comparison and scaling laws
- [Complexity](complexity.md)  --  Time, memory, and error complexity
- [Failure modes for non-uniform data](failure_modes.md)  --  Poor coverage, approximation error, numerical instability
- [Code examples](code_examples.md)  --  Basic usage, nn.Module, accuracy levels, epsilon comparison
- [API](api.md)  --  SparsePH, SparseRipsPersistence, C++ engine
- [When to use sparse VR](when_to_use.md)  --  Recommended use cases
- [When not to use sparse VR](when_not_to_use.md)  --  Limitations and alternatives
- [Performance analysis](performance.md)  --  Scaling with epsilon, point count, and approximation quality
- [Interleaving guarantee proof sketch](proof_sketch.md)  --  Mathematical derivation
- [Comparison to other approximations](comparison.md)  --  Sparse VR vs witness vs alpha
- [Algorithm details](algorithm_details.md)  --  Sparse graph construction and edge pruning
- [Extensions](extensions.md)  --  Streaming and distributed sparse VR
- [FAQ](faq.md)  --  Frequently asked questions
