# Fields

### Top lifetimes

The 10 longest-lived persistence pairs, ordered by persistence. Each `Lifetime`:
- `birth`, `death`, `dimension`, `persistence`

### Betti counts

Betti numbers for dimensions 0-4 (or up to MAX_BETTI_DIM).

### Top eigenvalues

Top 10 Laplacian eigenvalues with multiplicities.

### Entropy measures

- **Persistence entropy** is computed as -sum(p_i * log(p_i)) where p_i = L_i / total_L, representing the spread of lifetimes.
- **Betti entropy** is computed as -sum(b_i * log(b_i)), representing the spread of topological activity across dimensions.
- **Spectral entropy** is computed as -sum(lambda_i * log(lambda_i)), representing the spread of Laplacian eigenvalues.


[Back to index](index.md)
