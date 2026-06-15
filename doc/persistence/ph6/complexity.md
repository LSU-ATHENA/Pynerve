# Complexity

The worst-case complexity is O(n^3), with expected typical performance at O(n^2) or better. Experimental benefit varies by algorithm -- benchmark on your data. Memory usage is O(n * k).

## Algorithm-Specific Complexity

For standard (baseline) reduction, best-case time is O(n^2), typical is O(n^2 * k), worst is O(n^3), with O(n * k) memory. Cohomology (baseline) achieves O(n^2) best and typical, O(n^3) worst, with O(n * k) memory. Adaptive ordering runs at O(n^2 log n) best and typical, O(n^3 log n) worst, with O(n * k) memory. Approximate clearing achieves O(n * k) best, O(n * sqrt(n)) typical, O(n^2) worst, with O(n * k) memory. Speculative reduction with k threads achieves O(n^2 / k) best and typical, O(n^3 / k + n^2) worst, with O(n * k * k) memory. Adaptive pivoting achieves O(n^2) best and typical, O(n^3) worst, with O(n * k) memory. Block-sparse reduction achieves O(n^2) best and typical, O(n^3) worst, with O(n * k) + O(block^2) memory.

Complexity bounds for experimental algorithms are not yet rigorously characterized. Profile on representative data before deploying.


[Back to PH6 Index](index.md)
