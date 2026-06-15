#include <cuda_runtime.h>

namespace nerve::spectral::gpu
{

struct CliffordElement
{
    float scalar;
    float e1;
    float e2;
    float e3;
    float e12;
    float e13;
    float e23;
    float e123;
};

class CliffordProduct3D
{
public:
    static void multiply(const CliffordElement &a, const CliffordElement &b,
                         CliffordElement &result)
    {
        result.scalar = a.scalar * b.scalar + a.e1 * b.e1 + a.e2 * b.e2 + a.e3 * b.e3 -
                        a.e12 * b.e12 - a.e13 * b.e13 - a.e23 * b.e23 - a.e123 * b.e123;

        result.e1 =
            a.scalar * b.e1 + a.e1 * b.scalar - a.e12 * b.e2 - a.e13 * b.e3 + a.e23 * b.e123;
        result.e2 =
            a.scalar * b.e2 + a.e2 * b.scalar + a.e12 * b.e1 - a.e23 * b.e3 - a.e13 * b.e123;
        result.e3 =
            a.scalar * b.e3 + a.e3 * b.scalar + a.e13 * b.e1 + a.e23 * b.e2 + a.e12 * b.e123;

        result.e12 =
            a.scalar * b.e12 + a.e12 * b.scalar + a.e1 * b.e2 - a.e2 * b.e1 - a.e123 * b.e3;
        result.e13 =
            a.scalar * b.e13 + a.e13 * b.scalar + a.e1 * b.e3 - a.e3 * b.e1 + a.e123 * b.e2;
        result.e23 =
            a.scalar * b.e23 + a.e23 * b.scalar + a.e2 * b.e3 - a.e3 * b.e2 - a.e123 * b.e1;

        result.e123 = a.scalar * b.e123 + a.e123 * b.scalar + a.e1 * b.e23 - a.e2 * b.e13 +
                      a.e3 * b.e12 + a.e12 * b.e3 - a.e13 * b.e2 + a.e23 * b.e1;
    }
};

} // namespace nerve::spectral::gpu
