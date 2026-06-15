#include "nerve/core/rng/rng.hpp"

#include <cassert>
#include <chrono>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

int main()
{
    nerve::core::DeterminismContract strict(nerve::core::DeterminismLevel::STRICT, "rng");
    strict.fail_on_non_deterministic = false;

    nerve::core::RNG first(strict);
    nerve::core::RNG second(strict);

    const double first_value = first.uniform();
    (void)second;
    const auto metadata = first.getDeterminismMetadata();
    assert(metadata.achieved_level >= nerve::core::DeterminismLevel::NONE);

    nerve::core::DeterminismMetadata valid_metadata;
    valid_metadata.params_hash.fill(1);
    valid_metadata.result_checksum.fill(2);
    valid_metadata.rng_seed_used.fill(3);
    valid_metadata.achieved_level = nerve::core::DeterminismLevel::STRICT;
    valid_metadata.actual_execution_time = std::chrono::milliseconds(7);
    assert(valid_metadata.isValid());

    auto invalid_metadata = valid_metadata;
    invalid_metadata.achieved_level = static_cast<nerve::core::DeterminismLevel>(255);
    assert(!invalid_metadata.isValid());

    invalid_metadata = valid_metadata;
    invalid_metadata.actual_execution_time = std::chrono::milliseconds(-1);
    assert(!invalid_metadata.isValid());

    auto invalid_level_wire = valid_metadata.serialize();
    invalid_level_wire[32 + 32 + 16] = 0xFF;
    nerve::core::DeterminismMetadata decoded_invalid_level;
    assert(!decoded_invalid_level.deserialize(invalid_level_wire));

    invalid_metadata = valid_metadata;
    invalid_metadata.actual_execution_time = std::chrono::milliseconds(-1);
    const auto clamped_negative_time = invalid_metadata.serialize();
    nerve::core::DeterminismMetadata decoded_negative_time;
    assert(decoded_negative_time.deserialize(clamped_negative_time));
    assert(decoded_negative_time.actual_execution_time.count() == 0);

    auto oversized_time = valid_metadata.serialize();
    const std::size_t time_offset = 32 + 32 + 16 + 1;
    for (std::size_t i = 0; i < 8; ++i)
    {
        oversized_time[time_offset + i] = 0xFF;
    }
    nerve::core::DeterminismMetadata decoded_oversized_time;
    assert(!decoded_oversized_time.deserialize(oversized_time));

    bool rejected_malformed_state = false;
    try
    {
        first.setState(std::vector<uint64_t>{first.seed(), std::numeric_limits<uint64_t>::max()});
    }
    catch (const std::invalid_argument &)
    {
        rejected_malformed_state = true;
    }
    assert(rejected_malformed_state);

    nerve::core::DeterminismContract audit(nerve::core::DeterminismLevel::AUDIT, "rng");
    bool rejected = false;
    try
    {
        nerve::core::RNG must_fail(audit);
        (void)must_fail;
    }
    catch (const std::runtime_error &)
    {
        rejected = true;
    }
    assert(rejected);

    return 0;
}
