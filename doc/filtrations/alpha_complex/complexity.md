# Complexity analysis

[Back to index](index.md)

### Construction cost

The construction phases and their complexities are as follows. Delaunay
triangulation in 2D runs in $O(n \log n)$ expected time using randomized
incremental insertion. In 3D, Delaunay triangulation has $O(n^2)$ worst-case
but $O(n \log n)$ typical performance, depending on point distribution.
Circumradius computation costs $O(m)$ per simplex and
$O(n^{\lceil d/2 \rceil})$ total. Filtration sorting runs in $O(m \log m)$
where $m$ is the total number of simplices. Boundary matrix reduction costs
$O(m^\omega)$ with $\omega \approx 2$ using clearing.

### Simplex counts

For 10,000 points randomly distributed in a unit square, VR produces
approximately $5 \times 10^7$ simplices at dimension 2 while Alpha produces
approximately $6 \times 10^4$. For 10,000 points in a unit cube, VR again
produces approximately $5 \times 10^7$ simplices while Alpha produces
approximately $6 \times 10^5$. For 10,000 points on a sphere $S^2$, VR
produces approximately $5 \times 10^7$ while Alpha produces approximately
$6 \times 10^4$.

Alpha typically produces **10-100x fewer simplices** than VR for the same
point set, translating to proportional speedups in matrix reduction.

### Memory usage

For 10,000 points in 2D, Alpha produces roughly 60K simplices and a boundary
matrix of about 60K columns, requiring under a hundred megabytes of RAM. For
50,000 points, there are around 300K simplices and 300K columns, needing a
few hundred megabytes. For 100,000 points, roughly 600K simplices and 600K
columns require hundreds of megabytes to a gigabyte. For 1,000,000 points,
about 6 million simplices and 6 million columns need a few to several
gigabytes of RAM.

Alpha on 1M points in 2D uses less memory than VR on 10K points.
