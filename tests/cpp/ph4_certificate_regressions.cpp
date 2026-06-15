#include "nerve/persistence/kernels/ph4_ops.hpp"

#include <cassert>
#include <limits>

int main()
{
    nerve::persistence::StabilityCertificate cert(1.0, 0.5, true);
    assert(cert.isValid());

    nerve::persistence::StabilityCertificate infinite(std::numeric_limits<double>::infinity(), 0.5,
                                                      true);
    assert(!infinite.isValid());

    nerve::persistence::StabilityCertificate nan_residual(
        1.0, std::numeric_limits<double>::quiet_NaN(), true);
    assert(!nan_residual.isValid());

    return 0;
}
