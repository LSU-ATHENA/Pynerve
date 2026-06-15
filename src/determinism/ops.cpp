#include "nerve/determinism.hpp"

#include <algorithm>

namespace nerve::determinism
{

thread_local uint64_t tls_seed = 0;

bool cross_count = false;

void set_cross_count_reproducible(bool v)
{
    cross_count = v;
}

} // namespace nerve::determinism
