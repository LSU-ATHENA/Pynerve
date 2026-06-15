#pragma once
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

// NUMA allocator
class NUMAallocator
{
public:
    NUMAallocator();
    void *allocate(size_t bytes, int numa_node = -1);
    void deallocate(void *ptr, size_t bytes);
    size_t getAlignment() const;
};

// Memory pool
class DeterministicMemoryPool
{
public:
    DeterministicMemoryPool(size_t block_size);
    void *allocate();
    void deallocate(void *ptr);
    void reset();
};

// RNG determinism
uint64_t deterministicSeed(uint64_t base_seed);
std::vector<uint64_t> splitSeed(uint64_t seed, size_t count);
bool validateDeterminism(const std::vector<uint64_t> &sequence1,
                         const std::vector<uint64_t> &sequence2);

// RNG factory
class RNGfactory
{
public:
    std::mt19937_64 createEngine(uint64_t seed);
    std::uniform_real_distribution<double> createUniform(double min, double max);
    std::normal_distribution<double> createNormal(double mean, double stddev);
};

// RNG random ops
uint64_t boundedUniform(uint64_t seed, uint64_t bound);
std::vector<uint64_t> generatePermutation(uint64_t seed, size_t n);

// Error high-dim ops
class HighDimensionError
{
public:
    HighDimensionError(size_t actual_dim, size_t max_dim);
    bool isValid() const;
    std::string message() const;
};

// Halton sequence
class HaltonSequence
{
public:
    explicit HaltonSequence(size_t dim);
    std::vector<double> next();
};
} // namespace nerve::core
