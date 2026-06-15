# Distance computation details

### Memory-optimized distance matrix

The witness-landmark distance matrix is stored in **packed triangular**
format or **row-major** depending on the API:

```python
# Row-major: (n_witnesses, n_landmarks) -- default
# Each row = distances from a witness to all landmarks

# Memory: 4 bytes * 10^5 * 500 = megabytes (float32)
```

For very large witness sets ($n > 10^6$), the distance matrix can be
computed in chunks:

```python
def chunked_witness_persistence(points, n_landmarks=500, chunk_size=50000):
    landmarks = select_landmarks(points[:chunk_size], n_landmarks)
    all_pairs = []
    for i in range(0, len(points), chunk_size):
        chunk = points[i:i + chunk_size]
        pairs = compute_witness_persistence(
            landmarks, chunk, max_dim=2,
        )
        all_pairs.append(pairs)
    return np.vstack(all_pairs)
```

### Landmark-distance ordering

For each witness $w$, the landmarks are sorted by distance. This ordering
determines the sequence of simplices that $w$ witnesses:

$$
d(w, \ell_{(0)}) \leq d(w, \ell_{(1)}) \leq \dots \leq d(w, \ell_{(m-1)})
$$

The $(k+1)$-th nearest distance $m_k(w) = d(w, \ell_{(k)})$ is cached for
each witness and dimension. The simplex $[\ell_{(0)}, \dots, \ell_{(k)}]$
enters the filtration at birth time $\max_i d(w, \ell_{(i)}) - m_k(w)$.

<- [Witness Complex Overview](index.md)
