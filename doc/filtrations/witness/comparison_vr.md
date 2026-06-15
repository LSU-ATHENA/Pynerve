# Comparison to VR on large datasets

The witness complex differs from exact VR in several key aspects. VR operates on all $n$ points while the witness complex uses $n$ witnesses and $m$ landmarks. VR simplex generation cost is $O(n^{k+1})$ whereas the witness complex is $O(m^{k+1})$. VR requires an $n \times n$ distance matrix ($O(n^2)$) while the witness complex only needs an $n \times m$ matrix ($O(nm)$). VR memory usage scales as $O(n^{k+1})$ compared to $O(m^{k+1} + nm)$ for the witness complex. VR provides exact accuracy while the witness complex accuracy is $O(m^{-1/d_{\text{intrinsic}}})$. In practice, VR is limited to roughly $10^4$ points while the witness complex can handle over $10^6$ points.

### Scaling comparison

```
Dataset: 100K points in 3D, dim = 2

VR:             ~5 x 10^9 possible 2-simplices
                ~gigabytes memory (unfeasible)

Witness (m=500):
                ~500^3 / 6 = 20.8M possible 2-simplices
                ~megabytes memory (feasible)
                ~125x fewer simplices than VR
```

### When the witness complex is faster

The witness complex wins on three fronts:

1. **Distance computation.** Only compute $n \times m$ witness-landmark
   distances instead of $n \times n$ all-pairs distances. For $n = 10^5$
   and $m = 500$, this is a 200x reduction.
2. **Simplex enumeration.** Only enumerate simplices on the $m$ landmarks.
   For $k = 2$, this is $O(m^3)$ vs $O(n^3)$.
3. **Matrix reduction.** The boundary matrix is $O(m^{k+1})$ columns
   instead of $O(n^{k+1})$.

<- [Witness Complex Overview](index.md)
