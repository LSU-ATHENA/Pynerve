# Complexity

The complexity of each algorithm is as follows. The cup product at dimension (p,q) runs in O(s * n_p * n_q) where s is the number of (p+q)-simplices. The GPU variant runs in O(s / cores) with a single grid launch. The Reeb graph runs in O(n log n + m) using union-find and sorting. Zigzag persistence runs in O(m * r) via extended persistence reduction, and its GPU variant runs in O(m * r / cores) with batched reduction.


[Back to index](index.md)
