#include "nerve/core_types.hpp"
#include "nerve/encoders/simd_encoder.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

namespace
{

using nerve::Size;

bool check_simd_encoder_configuration()
{
    std::vector<double> input(16, 1.0);
    std::vector<double> output(16, 0.0);
    nerve::encoders::simdEncodeBatch(input.data(), 4, 4, output.data());
    for (size_t i = 0; i < output.size(); ++i)
    {
        if (!std::isfinite(output[i]))
        {
            std::cerr << "encode output non-finite at " << i << "\n";
            return false;
        }
    }
    return true;
}

bool check_forward_pass_minimal()
{
    std::vector<double> input = {0.0, 1.0, 2.0, 3.0};
    std::vector<double> output(4, 0.0);
    nerve::encoders::simdEncodeBatch(input.data(), 1, 4, output.data());
    for (size_t i = 0; i < 4; ++i)
    {
        if (!std::isfinite(output[i]))
        {
            std::cerr << "forward pass non-finite output\n";
            return false;
        }
    }
    return true;
}

bool check_decode_roundtrip()
{
    std::vector<double> original = {0.5, 1.5, 2.5, 3.5};
    std::vector<double> encoded(4, 0.0);
    std::vector<double> decoded(4, 0.0);
    nerve::encoders::simdEncodeBatch(original.data(), 1, 4, encoded.data());
    nerve::encoders::simdDecodeBatch(encoded.data(), 1, 4, decoded.data());
    for (size_t i = 0; i < 4; ++i)
    {
        if (!std::isfinite(decoded[i]))
        {
            std::cerr << "decode roundtrip non-finite at " << i << "\n";
            return false;
        }
    }
    return true;
}

#ifdef NERVE_HAS_AVX512
bool check_simd_vs_scalar_equivalence()
{
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(-10.0, 10.0);
    std::vector<double> input(64);
    for (auto &v : input)
        v = dist(rng);
    std::vector<double> simd_output(64, 0.0);
    std::vector<double> scalar_output(64, 0.0);
    nerve::encoders::simdEncodeBatch(input.data(), 8, 8, simd_output.data());
    for (size_t i = 0; i < 64; ++i)
    {
        scalar_output[i] = 2.0 * input[i] + 0.5;
    }
    for (size_t i = 0; i < 64; ++i)
    {
        if (std::abs(simd_output[i] - scalar_output[i]) > 1e-12)
        {
            std::cerr << "SIMD vs scalar mismatch at " << i << "\n";
            return false;
        }
    }
    return true;
}
#endif

} // namespace

int main()
{
    if (!check_simd_encoder_configuration())
    {
        std::cerr << "FAIL: simd_encoder_configuration\n";
        return 1;
    }
    if (!check_forward_pass_minimal())
    {
        std::cerr << "FAIL: forward_pass_minimal\n";
        return 1;
    }
    if (!check_decode_roundtrip())
    {
        std::cerr << "FAIL: decode_roundtrip\n";
        return 1;
    }
#ifdef NERVE_HAS_AVX512
    if (!check_simd_vs_scalar_equivalence())
    {
        std::cerr << "FAIL: simd_vs_scalar_equivalence\n";
        return 1;
    }
#endif
    return 0;
}
