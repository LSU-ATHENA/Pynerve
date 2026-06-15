# Field Coefficients

The default field is Z2 (mod 2) using unsigned integer arithmetic. Configurable fields include any prime field Zp for p in the set {2, 3, 5, 7, ..., 2^31-1}. Internally, matrix coefficients are represented as `double` with modulo reduction performed via `nerve::algebra::FiniteField`.

Field selection is configured at the C++ level via `FiniteField`:

```cpp
#include <nerve/math/finite_field.hpp>

using Z2 = nerve::algebra::FiniteField<2>;         // default
using Z3 = nerve::algebra::FiniteField<3>;
using ZP = nerve::algebra::FiniteField<7919>;       // large prime
```

### Field arithmetic guarantees

For Z2, addition is XOR, multiplication is AND, negation is the identity (since a equals negative a), and inversion is also the identity (since a equals its own inverse). Zero is represented as `0` and one as `1`. For Zp with p greater than 2, addition uses modular addition via `(a + b) % p`, multiplication uses modular multiplication via `(a * b) % p`, negation is `(p - a) % p`, and inversion uses the Extended Euclidean algorithm. Zero and one are `0` and `1` respectively.

For Z2, all field operations are O(1) bitwise operations. For Zp, operations are O(1) integer arithmetic with a single modulo. These are used inside the reduction kernel inner loops and are not a performance bottleneck.


[Back to Correctness Index](index.md)
