#include "nerve/determinism.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

namespace
{

constexpr double kTol = 1e-10;

bool check_seed_get_seed_next_seed_cycle()
{
    nerve::determinism::seed(42);
    auto val = nerve::determinism::get_seed();
    if (val != 42)
    {
        std::cerr << "seed(42) then get_seed() should return 42, got " << val << "\n";
        return false;
    }
    uint64_t n1 = nerve::determinism::next_seed();
    (void)n1;
    return true;
}

bool check_same_seed_same_sequence()
{
    nerve::determinism::seed(42);
    std::vector<uint64_t> seq1;
    for (int i = 0; i < 100; ++i)
        seq1.push_back(nerve::determinism::next_seed());

    nerve::determinism::seed(42);
    std::vector<uint64_t> seq2;
    for (int i = 0; i < 100; ++i)
        seq2.push_back(nerve::determinism::next_seed());

    for (int i = 0; i < 100; ++i)
    {
        if (seq1[i] != seq2[i])
        {
            std::cerr << "same seeds produced different values at " << i << "\n";
            return false;
        }
    }
    return true;
}

bool check_different_seeds_different_values()
{
    nerve::determinism::seed(42);
    uint64_t v1 = nerve::determinism::next_seed();

    nerve::determinism::seed(99);
    uint64_t v2 = nerve::determinism::next_seed();

    if (v1 == v2)
    {
        std::cerr << "different seeds should produce different first values\n";
        return false;
    }
    return true;
}

bool check_determinism_validation_valid()
{
    nerve::determinism::seed(12345);
    uint64_t v1 = nerve::determinism::next_seed();

    nerve::determinism::seed(12345);
    uint64_t v2 = nerve::determinism::next_seed();

    if (v1 != v2)
    {
        std::cerr << "determinism validation: same seed should produce same value\n";
        return false;
    }
    return true;
}

} // namespace

int main()
{
    if (!check_seed_get_seed_next_seed_cycle())
    {
        std::cerr << "FAIL: seed/get_seed/next_seed cycle\n";
        return 1;
    }
    if (!check_same_seed_same_sequence())
    {
        std::cerr << "FAIL: same seed same sequence\n";
        return 1;
    }
    if (!check_different_seeds_different_values())
    {
        std::cerr << "FAIL: different seeds different values\n";
        return 1;
    }
    if (!check_determinism_validation_valid())
    {
        std::cerr << "FAIL: determinism validation\n";
        return 1;
    }
    return 0;
}
