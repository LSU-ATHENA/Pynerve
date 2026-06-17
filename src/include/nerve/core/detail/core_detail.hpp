#pragma once

#include "nerve/core/detail/error_event_extensions.hpp"
#include "nerve/core/error/error_event.hpp"
#include "nerve/core/memory/memory_pool.hpp"
#include "nerve/core/memory/numa_aware_pool.hpp"
#include "nerve/core/rng/determinism_contract.hpp"
#include "nerve/core/rng/random.hpp"
#include "nerve/core/rng/rng.hpp"
#include "nerve/core_types.hpp"

#include <memory>
#include <random>

namespace nerve::core
{
// SIMD ops
namespace simd
{
void memcpy(void *dst, const void *src, size_t n);
void memset(void *dst, int value, size_t n);
double reduceSum(const double *data, size_t n);
} // namespace simd

// RNG determinism
uint64_t deterministicSeed(uint64_t base_seed);
std::vector<uint64_t> splitSeed(uint64_t seed, size_t count);
bool validateDeterminism(const std::vector<uint64_t> &sequence1,
                         const std::vector<uint64_t> &sequence2);

// RNG random ops
uint64_t boundedUniform(uint64_t seed, uint64_t bound);
std::vector<uint64_t> generatePermutation(uint64_t seed, size_t n);

// Halton sequence
class HaltonSequence
{
public:
    explicit HaltonSequence(size_t dim);
    std::vector<double> next();
};

} // namespace nerve::core
