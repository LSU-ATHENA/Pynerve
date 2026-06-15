# Complexity

Performance varies by scenario. For dense, small complexes with n < 10^4 the worst-case complexity is O(n^3) with baseline typical speed. For sparse, large complexes with n >= 10^4 the typical complexity is O(n^2), running 2-5x over standard. Approximate mode has complexity O(m^3) where m is the number of witnesses, achieving 10-50x over exact. GPU hybrid mode provides 5-15x over CPU.

Memory usage scales with the number of non-zero entries in the boundary/coboundary matrix. PH4 uses sparse column storage by default.

### Performance Characteristics at Different Scales

**Small scale (n < 10^3, N < 100 points)**:
- Computation time: < 10 ms
- Memory: under a few tens of megabytes
- Standard reduction is optimal (no cohomology overhead)
- Clearing provides 10-20% benefit

**Medium scale (10^3 < n < 10^5, 100 < N < 1000 points)**:
- Computation time: 10 ms - 1 s
- Memory: from tens to hundreds of megabytes
- Cohomology becomes competitive at n ~ 10^4
- Clearing provides 20-40% benefit
- AVX-512 acceleration helps significantly

**Large scale (10^5 < n < 10^7, 1000 < N < 10000 points)**:
- Computation time: 1 s - 100 s
- Memory: from hundreds of megabytes to tens of gigabytes
- Cohomology is strongly preferred (3-10x over standard)
- Approximate mode may be necessary for n > 10^6
- GPU acceleration provides 5-15x speedup
- Clearing provides 30-50% benefit

**Extreme scale (n > 10^7, N > 10000 points)**:
- Computation time: 100 s - hours
- Memory: tens of gigabytes or more
- Approximate mode or streaming strongly recommended
- GPU acceleration almost essential for interactive use
- Memory budget should be explicitly configured

### Memory Usage Breakdown

For a typical Vietoris-Rips complex on N=2000 points, max_dim=2:

```
Component                    Memory (megabytes)   Percentage
Boundary/coboundary columns  312           52%
Coface index (cohomology)    120           20%
Pivot table                   8             1%
Working buffers               80            13%
Thread-local storage          24            4%
Filtration data               60            10%
Total                        604           100%
```

For a larger complex (N=10000 points, same parameters, exact mode would be several gigabytes):

```
Component                    Memory (gigabytes)   Percentage
Boundary/coboundary columns  4.2            52%
Coface index (cohomology)    1.6            20%
Pivot table                   0.04           0.5%
Working buffers               1.1            14%
Thread-local storage          0.3            4%
Filtration data               0.8            10%
Total                        8.04           100%
```

At this scale, approximate mode with 200 landmarks reduces total memory to a few hundred megabytes (less than 3 percent of exact).

Back to [PH4 Engine Overview](index.md)
