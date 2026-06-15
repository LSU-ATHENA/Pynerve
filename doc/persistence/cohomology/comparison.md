# Comparison Tables

[Back to Index](index.md)

### Algorithm Characteristics

Standard reduction uses the boundary matrix, processes in forward order (1 to n), and column density increases during reduction. Cohomology uses the coboundary matrix, processes in reverse order (n down to 1), and columns stay sparse. Cohomology supports emergent pair detection, accounting for 20-40% of pairs, while standard reduction does not. Cohomology maps naturally to GPU with a parallel per-warp approach, while standard reduction is sequential. SIMD benefits are high for cohomology and only moderate for standard reduction. Cohomology applies a pivot dimension filter restricted to dim+1 only, while standard reduction has no such filter. Cohomology requires one extra index per simplex for the coface index, while standard reduction has no coindex memory overhead. Standard reduction clears the birth column, while cohomology clears the death column.

### Performance by Complex Type

For a Rips complex with N=500 and radius 0.5, there are 2.1 x 10^4 simplices; standard reduction takes 50 ms and cohomology takes 15 ms, a 3.3x ratio. For Rips with N=2000 and radius 0.3, there are 4.2 x 10^5 simplices; standard takes 2.1 seconds and cohomology takes 0.4 seconds, a 5.3x ratio. For Rips with N=10000 and radius 0.2, there are 8.1 x 10^6 simplices; standard takes 95 seconds and cohomology takes 12 seconds, a 7.9x ratio. For an Alpha complex with N=1000, there are 1.5 x 10^5 simplices; standard takes 0.8 seconds and cohomology takes 0.3 seconds, a 2.7x ratio. For a Witness complex with N=5000 and L=500, there are 3.2 x 10^5 simplices; standard takes 1.5 seconds and cohomology takes 0.6 seconds, a 2.5x ratio. A full simplex on N=30 has 5.4 x 10^8 simplices and is too large for practical computation.

Note: "Full simplex on N=30" has 2^31-1 simplices (2.1 billion), which is too many for any practical computation. The comparisons above focus on feasible complex sizes.
