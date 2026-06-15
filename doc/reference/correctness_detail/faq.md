# FAQ

**Is Pynerve guaranteed to produce bitwise identical results across different machines?**
Within the same CPU or GPU architecture family, yes -- Pynerve enforces bitwise reproducibility through fixed-tree reductions, canonical ordering, and precise floating-point flags. Across different GPU architectures, the opt-in Reduction Fusion Algorithm (RFA) provides cross-architecture bit-identity.

**What happens if my input contains NaN or infinity values?**
NaN values are detected at input validation and produce a `NumericalError` (code `E20_NUM_NAN`). Infinity values in distance computation are ignored during filtration but are permitted in precomputed matrices. Both are handled gracefully rather than causing undefined behavior.

**How does Pynerve handle floating-point tolerance?**
The default numerical tolerance is 1e-12 for double precision and 1e-6 for single precision. Values below the tolerance are treated as zero, pairs with persistence below the tolerance are filtered out, and iterative algorithms stop when the change falls below the tolerance.

**Can I use mixed precision for faster computation?**
Yes. Pynerve supports precision policies such as `P32_DISTANCE` (float32 distance with float64 reduction) and `P16_DISTANCE` (half-precision via Tensor Cores with float64 reduction). These provide substantial speedups while maintaining full reduction accuracy.

**How are regression tests managed?**
When a bug is found, a failing test is created to reproduce the exact issue, a fix is implemented with a comment referencing the test, and a regression test is added to the permanent regression suite. The fix is then verified on all CI platforms before merging.


[Back to Correctness Index](index.md)
