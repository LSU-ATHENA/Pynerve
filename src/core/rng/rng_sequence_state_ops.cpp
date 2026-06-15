#include "nerve/core/rng/rng.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace nerve::core
{
namespace
{

[[nodiscard]] bool multiplyWouldOverflow(Size lhs, Size rhs) noexcept
{
    return rhs != 0 && lhs > std::numeric_limits<Size>::max() / rhs;
}

[[nodiscard]] Size checkedProduct(Size lhs, Size rhs, const char *context)
{
    if (multiplyWouldOverflow(lhs, rhs))
    {
        throw std::length_error(context);
    }
    return lhs * rhs;
}

std::vector<uint64_t> packEngineState(uint64_t seed, const std::string &text_state)
{
    std::vector<uint64_t> state;
    state.reserve(2 + (text_state.size() + 7) / 8);
    state.push_back(seed);
    state.push_back(static_cast<uint64_t>(text_state.size()));
    for (std::size_t offset = 0; offset < text_state.size(); offset += 8)
    {
        uint64_t word = 0;
        const std::size_t chunk = std::min<std::size_t>(8, text_state.size() - offset);
        for (std::size_t i = 0; i < chunk; ++i)
        {
            word |= static_cast<uint64_t>(static_cast<unsigned char>(text_state[offset + i]))
                    << (8 * i);
        }
        state.push_back(word);
    }
    return state;
}

std::string unpackEngineState(const ConstSpan<uint64_t> &state)
{
    const uint64_t byte_count = state[1];
    if (byte_count > static_cast<uint64_t>(std::numeric_limits<std::size_t>::max()))
    {
        throw std::invalid_argument("RNG state payload is malformed");
    }
    const uint64_t payload_words = (byte_count / 8) + ((byte_count % 8) == 0 ? 0 : 1);
    if (payload_words > std::numeric_limits<uint64_t>::max() - 2)
    {
        throw std::invalid_argument("RNG state payload is malformed");
    }
    const uint64_t expected_words = 2 + payload_words;
    if (expected_words != state.size())
    {
        throw std::invalid_argument("RNG state payload is malformed");
    }
    std::string text_state;
    text_state.reserve(static_cast<std::size_t>(byte_count));
    for (std::size_t word_index = 2; word_index < state.size(); ++word_index)
    {
        uint64_t word = state[word_index];
        for (std::size_t byte = 0; byte < 8 && text_state.size() < byte_count; ++byte)
        {
            text_state.push_back(static_cast<char>(word & 0xffU));
            word >>= 8;
        }
    }
    return text_state;
}

} // namespace

std::vector<double> RNG::haltonSequence(Size n, Size base)
{
    if (base < 2)
    {
        throw std::invalid_argument("Halton base must be at least 2");
    }
    std::vector<double> sequence;
    sequence.reserve(n);
    for (Size i = 0; i < n; ++i)
    {
        sequence.push_back(halton(i, base));
    }
    return sequence;
}

std::vector<double> RNG::sobolSequence(Size n, Size dimension)
{
    if (dimension == 0)
    {
        throw std::invalid_argument("Sobol dimension must be positive");
    }
    if (dimension > std::numeric_limits<Size>::max() - 2)
    {
        throw std::length_error("Sobol dimension overflow");
    }
    const Size total_values = checkedProduct(n, dimension, "Sobol sequence size overflow");
    std::vector<double> sequence(total_values);
    for (Size i = 0; i < n; ++i)
    {
        for (Size d = 0; d < dimension; ++d)
        {
            sequence[i * dimension + d] = halton(i, 2 + d);
        }
    }
    return sequence;
}

void RNG::jumpAhead(Size steps)
{
    generator_.discard(steps);
    updateDeterminismMetadata();
}

RNG RNG::split()
{
    const uint64_t new_seed = generator_();
    RNG child(new_seed);
    child.is_deterministic_ = is_deterministic_;
    child.determinism_contract_ = determinism_contract_;
    child.determinism_contract_.setRngSeed(new_seed);
    child.updateDeterminismMetadata();
    updateDeterminismMetadata();
    return child;
}

std::vector<uint64_t> RNG::getState() const
{
    std::ostringstream out;
    out << generator_;
    return packEngineState(seed_, out.str());
}

void RNG::setState(const ConstSpan<uint64_t> &state)
{
    if (state.empty())
    {
        return;
    }
    if (state.size() == 1)
    {
        seed(state[0]);
        return;
    }
    const std::string text_state = unpackEngineState(state);
    std::istringstream input(text_state);
    std::mt19937_64 restored;
    input >> restored;
    if (input.fail())
    {
        throw std::invalid_argument("RNG engine state is malformed");
    }
    seed_ = state[0];
    generator_ = restored;
    determinism_contract_.setRngSeed(seed_);
    initializeDistributions();
    is_deterministic_ = true;
    determinism_metadata_.warnings.clear();
    updateDeterminismMetadata();
}

double RNG::halton(Size index, Size base)
{
    if (base < 2)
    {
        throw std::invalid_argument("Halton base must be at least 2");
    }
    double result = 0.0;
    double f = 1.0;
    Size i = index;
    while (i > 0)
    {
        f = f / static_cast<double>(base);
        result = result + f * static_cast<double>(i % base);
        i = i / base;
    }
    return result;
}

} // namespace nerve::core
