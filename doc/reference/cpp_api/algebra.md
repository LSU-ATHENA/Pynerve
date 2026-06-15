# `nerve::algebra` -- Algebraic Types

```cpp
#include <nerve/math/finite_field.hpp>

namespace nerve::algebra;

template <int p>
class FiniteField {
public:
    using Element = int;

    static Element add(Element a, Element b);
    static Element sub(Element a, Element b);
    static Element mul(Element a, Element b);
    static Element inv(Element a);          // multiplicative inverse
    static Element zero();
    static Element one();
    static bool isZero(Element a);
    static bool isOne(Element a);
};

using Z2 = FiniteField<2>;
using Z3 = FiniteField<3>;

// Simplicial complex types
class SimplexCollection;
class ChainComplex;
class BoundaryMatrix;
```

**Cost:** All field operations are O(1) with small primes.

<- [C++ API Overview](index.md)
