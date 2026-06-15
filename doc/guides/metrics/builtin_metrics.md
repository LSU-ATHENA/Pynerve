# Available built-in metric strings

`"euclidean"` provides the L2 norm with SIMD and Tensor Core GPU support as the default. `"manhattan"` provides the L1 norm with SIMD and GPU support. `"cosine"` provides 1 minus cosine similarity with SIMD and GPU support. `"chebyshev"` provides the L-infinity norm with SIMD support. `"minkowski"` provides the Lp norm (p configurable) with SIMD support. `"canberra"` provides the Canberra distance with SIMD support. `"braycurtis"` provides the Bray-Curtis distance with SIMD support. `"correlation"` provides 1 minus Pearson correlation with SIMD support. `"precomputed"` uses a user-provided matrix (dense or CSR). A callable accepts any custom function without SIMD or GPU support.

[Back to index](index.md)
