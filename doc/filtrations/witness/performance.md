# Performance analysis

### Scaling with landmarks

The following benchmarks show scaling with the number of landmarks on a dataset of $50,000$ points in 3D with $\max\_dim = 2$. With 100 landmarks, approximately 50 H0 pairs, 10 H1 pairs, 2 H2 pairs, and 0.5 seconds are expected. With 200 landmarks, approximately 100 H0 pairs, 25 H1 pairs, 5 H2 pairs, and 1.2 seconds. With 500 landmarks, approximately 200 H0 pairs, 50 H1 pairs, 12 H2 pairs, and 5.0 seconds. With 1000 landmarks, approximately 350 H0 pairs, 80 H1 pairs, 20 H2 pairs, and 18.0 seconds.

### Scaling with point count

The following benchmarks show scaling with point count. For $10,000$ points with 200 landmarks, runtime is about 0.8 seconds and memory usage is a couple hundred megabytes. For $50,000$ points with 200 landmarks, runtime is about 3.5 seconds and memory usage is a few hundred megabytes. For $100,000$ points with 500 landmarks, runtime is about 12.0 seconds and memory usage is around a gigabyte. For $500,000$ points with 500 landmarks, runtime is about 45.0 seconds and memory usage is a few gigabytes. For $1,000,000$ points with 1000 landmarks, runtime is about 120.0 seconds and memory usage is around ten gigabytes.

### Bottleneck errors for intrinsic dimension 2

```
Landmarks m     Covering radius delta     Bottleneck bound (3 * delta)
100             0.10                      0.30
200             0.07                      0.21
500             0.045                     0.14
1000            0.032                     0.10
```

<- [Witness Complex Overview](index.md)
