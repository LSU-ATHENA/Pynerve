# Detail: entropy formulas

### Persistence entropy

```
H_pers = -sum_i (L_i / L_total) * log(L_i / L_total)
```

where L_i is the persistence of the i-th pair, and L_total = sum L_i.

Properties:
- H_pers = 0 when only one pair has non-zero persistence
- H_pers = log(N) when all N pairs have equal persistence
- Higher H_pers = more uniform lifetime distribution

### Betti entropy

```
H_betti = -sum_k (beta_k / beta_total) * log(beta_k / beta_total)
```

where beta_k is the k-th Betti number.

Properties:
- H_betti = 0 when only one dimension has non-zero Betti numbers
- Higher H_betti = topological activity spread across more dimensions

### Spectral entropy

```
H_spec = -sum_i (lambda_i / Z) * log(lambda_i / Z)
```

where lambda_i are Laplacian eigenvalues and Z = sum lambda_i.

Properties:
- H_spec = 0 when only one eigenvalue is non-zero
- Higher H_spec = more uniform spectral distribution


[Back to index](index.md)
