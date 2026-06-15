#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace nerve::core::detail
{

inline void appendUint32LittleEndian(std::vector<uint8_t> &output, uint32_t value)
{
    output.push_back(static_cast<uint8_t>(value & 0xFFU));
    output.push_back(static_cast<uint8_t>((value >> 8U) & 0xFFU));
    output.push_back(static_cast<uint8_t>((value >> 16U) & 0xFFU));
    output.push_back(static_cast<uint8_t>((value >> 24U) & 0xFFU));
}

inline void appendUint64LittleEndian(std::vector<uint8_t> &output, uint64_t value)
{
    for (unsigned shift = 0; shift < 64; shift += 8)
    {
        output.push_back(static_cast<uint8_t>((value >> shift) & 0xFFU));
    }
}

inline bool canRead(const std::vector<uint8_t> &data, std::size_t offset, std::size_t count)
{
    return offset <= data.size() && data.size() - offset >= count;
}

inline bool readUint32LittleEndian(const std::vector<uint8_t> &data, std::size_t *offset,
                                   uint32_t *value)
{
    if (offset == nullptr || value == nullptr || !canRead(data, *offset, sizeof(uint32_t)))
    {
        return false;
    }
    const auto *cursor = data.data() + *offset;
    *value = static_cast<uint32_t>(cursor[0]) | (static_cast<uint32_t>(cursor[1]) << 8U) |
             (static_cast<uint32_t>(cursor[2]) << 16U) | (static_cast<uint32_t>(cursor[3]) << 24U);
    *offset += sizeof(uint32_t);
    return true;
}

inline bool readUint64LittleEndian(const std::vector<uint8_t> &data, std::size_t *offset,
                                   uint64_t *value)
{
    if (offset == nullptr || value == nullptr || !canRead(data, *offset, sizeof(uint64_t)))
    {
        return false;
    }
    const auto *cursor = data.data() + *offset;
    uint64_t parsed = 0;
    for (unsigned shift = 0; shift < 64; shift += 8)
    {
        parsed |= static_cast<uint64_t>(*cursor++) << shift;
    }
    *value = parsed;
    *offset += sizeof(uint64_t);
    return true;
}

} // namespace nerve::core::detail
