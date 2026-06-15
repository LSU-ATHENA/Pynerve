#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace nerve::distributed::detail
{

inline constexpr std::size_t kUint32WireSize = 4;
inline constexpr std::size_t kUint64WireSize = 8;

inline void appendUint32LittleEndian(std::vector<std::uint8_t> &output, std::uint32_t value)
{
    output.push_back(static_cast<std::uint8_t>(value & 0xFFU));
    output.push_back(static_cast<std::uint8_t>((value >> 8U) & 0xFFU));
    output.push_back(static_cast<std::uint8_t>((value >> 16U) & 0xFFU));
    output.push_back(static_cast<std::uint8_t>((value >> 24U) & 0xFFU));
}

inline void appendUint64LittleEndian(std::vector<std::uint8_t> &output, std::uint64_t value)
{
    for (unsigned shift = 0; shift < 64; shift += 8)
    {
        output.push_back(static_cast<std::uint8_t>((value >> shift) & 0xFFU));
    }
}

inline void appendInt32LittleEndian(std::vector<std::uint8_t> &output, std::int32_t value)
{
    appendUint32LittleEndian(output, std::bit_cast<std::uint32_t>(value));
}

inline void appendFloatLittleEndian(std::vector<std::uint8_t> &output, float value)
{
    appendUint32LittleEndian(output, std::bit_cast<std::uint32_t>(value));
}

inline bool canRead(const std::vector<std::uint8_t> &data, std::size_t offset, std::size_t count)
{
    return offset <= data.size() && data.size() - offset >= count;
}

inline int checkedMpiByteCount(std::size_t size, const char *context)
{
    if (size > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::length_error(context);
    }
    return static_cast<int>(size);
}

inline int buildMpiDisplacements(const std::vector<int> &recv_sizes,
                                 std::vector<int> *displacements, const char *context)
{
    if (displacements == nullptr)
    {
        throw std::invalid_argument("MPI displacement output cannot be null");
    }
    displacements->assign(recv_sizes.size(), 0);

    std::size_t total_bytes = 0;
    const auto max_mpi_count = static_cast<std::size_t>(std::numeric_limits<int>::max());
    for (std::size_t i = 0; i < recv_sizes.size(); ++i)
    {
        const int recv_size = recv_sizes[i];
        if (recv_size < 0)
        {
            throw std::length_error(context);
        }
        const auto recv_size_bytes = static_cast<std::size_t>(recv_size);
        if (total_bytes > max_mpi_count - recv_size_bytes)
        {
            throw std::length_error(context);
        }
        (*displacements)[i] = static_cast<int>(total_bytes);
        total_bytes += recv_size_bytes;
    }
    return static_cast<int>(total_bytes);
}

inline bool readUint32LittleEndian(const std::vector<std::uint8_t> &data, std::size_t *offset,
                                   std::uint32_t *value)
{
    if (offset == nullptr || value == nullptr || !canRead(data, *offset, kUint32WireSize))
    {
        return false;
    }
    const auto *cursor = data.data() + *offset;
    *value = static_cast<std::uint32_t>(cursor[0]) | (static_cast<std::uint32_t>(cursor[1]) << 8U) |
             (static_cast<std::uint32_t>(cursor[2]) << 16U) |
             (static_cast<std::uint32_t>(cursor[3]) << 24U);
    *offset += kUint32WireSize;
    return true;
}

inline bool readUint64LittleEndian(const std::vector<std::uint8_t> &data, std::size_t *offset,
                                   std::uint64_t *value)
{
    if (offset == nullptr || value == nullptr || !canRead(data, *offset, kUint64WireSize))
    {
        return false;
    }
    const auto *cursor = data.data() + *offset;
    std::uint64_t parsed = 0;
    for (unsigned shift = 0; shift < 64; shift += 8)
    {
        parsed |= static_cast<std::uint64_t>(*cursor++) << shift;
    }
    *value = parsed;
    *offset += kUint64WireSize;
    return true;
}

inline bool readInt32LittleEndian(const std::vector<std::uint8_t> &data, std::size_t *offset,
                                  std::int32_t *value)
{
    std::uint32_t raw = 0;
    if (!readUint32LittleEndian(data, offset, &raw))
    {
        return false;
    }
    *value = std::bit_cast<std::int32_t>(raw);
    return true;
}

inline bool readFloatLittleEndian(const std::vector<std::uint8_t> &data, std::size_t *offset,
                                  float *value)
{
    std::uint32_t raw = 0;
    if (!readUint32LittleEndian(data, offset, &raw))
    {
        return false;
    }
    *value = std::bit_cast<float>(raw);
    return true;
}

} // namespace nerve::distributed::detail
