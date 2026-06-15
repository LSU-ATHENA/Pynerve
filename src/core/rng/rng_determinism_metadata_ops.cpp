
#include "nerve/core/rng/determinism_contract.hpp"
#include "rng_wire_format.hpp"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <limits>
#include <stdexcept>

namespace nerve::core
{

namespace
{

[[nodiscard]] uint32_t checkedLength(std::size_t size, const char *field)
{
    if (size > std::numeric_limits<uint32_t>::max())
    {
        throw std::length_error(field);
    }
    return static_cast<uint32_t>(size);
}

auto atOffset(const std::vector<uint8_t> &data, std::size_t offset)
{
    return std::next(data.begin(), static_cast<std::ptrdiff_t>(offset));
}

template <typename OutputIt>
void copyFromOffset(const std::vector<uint8_t> &data, std::size_t offset, std::size_t count,
                    OutputIt output)
{
    std::copy_n(atOffset(data, offset), static_cast<std::ptrdiff_t>(count), output);
}

} // namespace

[[nodiscard]] bool DeterminismMetadata::isValid() const
{
    if (achieved_level > DeterminismLevel::AUDIT)
    {
        return false;
    }
    if (actual_execution_time.count() < 0)
    {
        return false;
    }
    if (params_hash == std::array<uint8_t, 32>{})
    {
        return false;
    }
    if (result_checksum == std::array<uint8_t, 32>{})
    {
        return false;
    }
    if (rng_seed_used == std::array<uint8_t, 16>{})
    {
        return false;
    }
    return true;
}

[[nodiscard]] std::vector<uint8_t> DeterminismMetadata::serialize() const
{
    std::vector<uint8_t> data;
    data.insert(data.end(), params_hash.begin(), params_hash.end());
    data.insert(data.end(), result_checksum.begin(), result_checksum.end());
    data.insert(data.end(), rng_seed_used.begin(), rng_seed_used.end());
    data.push_back(static_cast<uint8_t>(achieved_level));

    const auto time_count = actual_execution_time.count();
    uint64_t time_ms = time_count > 0 ? static_cast<uint64_t>(time_count) : 0;
    detail::appendUint64LittleEndian(data, time_ms);

    uint64_t memory_mb = actual_memory_usage_mb;
    detail::appendUint64LittleEndian(data, memory_mb);

    uint8_t flags = 0;
    flags |= static_cast<uint8_t>(was_deterministic) << 0;
    data.push_back(flags);

    uint32_t num_warnings = checkedLength(warnings.size(), "Too many metadata warnings");
    detail::appendUint32LittleEndian(data, num_warnings);

    for (const auto &warning : warnings)
    {
        uint32_t warning_len = checkedLength(warning.length(), "Metadata warning too long");
        detail::appendUint32LittleEndian(data, warning_len);
        data.insert(data.end(), warning.begin(), warning.end());
    }

    uint32_t error_len = checkedLength(error_message.length(), "Metadata error message too long");
    detail::appendUint32LittleEndian(data, error_len);
    data.insert(data.end(), error_message.begin(), error_message.end());
    return data;
}

bool DeterminismMetadata::deserialize(const std::vector<uint8_t> &data)
{
    if (data.empty())
    {
        return false;
    }

    size_t offset = 0;
    if (offset + params_hash.size() > data.size())
    {
        return false;
    }
    copyFromOffset(data, offset, params_hash.size(), params_hash.begin());
    offset += params_hash.size();

    if (offset + result_checksum.size() > data.size())
    {
        return false;
    }
    copyFromOffset(data, offset, result_checksum.size(), result_checksum.begin());
    offset += result_checksum.size();

    if (offset + rng_seed_used.size() > data.size())
    {
        return false;
    }
    copyFromOffset(data, offset, rng_seed_used.size(), rng_seed_used.begin());
    offset += rng_seed_used.size();

    if (offset >= data.size())
    {
        return false;
    }
    achieved_level = static_cast<DeterminismLevel>(data[offset++]);
    if (achieved_level > DeterminismLevel::AUDIT)
    {
        return false;
    }

    uint64_t time_ms = 0;
    if (!detail::readUint64LittleEndian(data, &offset, &time_ms))
    {
        return false;
    }
    if (time_ms > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
    {
        return false;
    }
    actual_execution_time = std::chrono::milliseconds(static_cast<int64_t>(time_ms));

    uint64_t memory_mb = 0;
    if (!detail::readUint64LittleEndian(data, &offset, &memory_mb))
    {
        return false;
    }
    if (memory_mb > static_cast<uint64_t>(std::numeric_limits<size_t>::max()))
    {
        return false;
    }
    actual_memory_usage_mb = static_cast<size_t>(memory_mb);

    if (offset >= data.size())
    {
        return false;
    }
    uint8_t flags = data[offset++];
    was_deterministic = (flags & (1 << 0)) != 0;

    uint32_t num_warnings = 0;
    if (!detail::readUint32LittleEndian(data, &offset, &num_warnings))
    {
        return false;
    }

    warnings.clear();
    for (uint32_t i = 0; i < num_warnings; ++i)
    {
        uint32_t warning_len = 0;
        if (!detail::readUint32LittleEndian(data, &offset, &warning_len))
        {
            return false;
        }

        if (offset + warning_len > data.size())
        {
            return false;
        }
        warnings.emplace_back(reinterpret_cast<const char *>(data.data() + offset), warning_len);
        offset += warning_len;
    }

    uint32_t error_len = 0;
    if (!detail::readUint32LittleEndian(data, &offset, &error_len))
    {
        return false;
    }

    if (offset + error_len > data.size())
    {
        return false;
    }
    error_message.assign(reinterpret_cast<const char *>(data.data() + offset), error_len);
    return true;
}

} // namespace nerve::core
