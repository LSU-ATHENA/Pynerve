# When not to use sparse VR

[Back to index](index.md)

For $n$ below 10,000, exact VR is fast enough and introduces no approximation error. On geometric data in dimension 3 or lower, the alpha complex is exact and faster than sparse VR. When high precision is required, use exact VR (determinism is always enabled). For highly non-uniform data, the witness complex may give better coverage.
