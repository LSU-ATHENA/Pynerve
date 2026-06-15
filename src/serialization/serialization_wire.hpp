#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace nerve::serialization::detail
{

inline constexpr std::size_t kUint32WireSize = 4;
inline constexpr std::size_t kUint16WireSize = 2;

inline bool fitsUint32(std::size_t value)
{
    return value <= std::numeric_limits<std::uint32_t>::max();
}

inline void appendUint32LittleEndian(std::vector<std::uint8_t> &output, std::uint32_t value)
{
    output.push_back(static_cast<std::uint8_t>(value & 0xFFU));
    output.push_back(static_cast<std::uint8_t>((value >> 8U) & 0xFFU));
    output.push_back(static_cast<std::uint8_t>((value >> 16U) & 0xFFU));
    output.push_back(static_cast<std::uint8_t>((value >> 24U) & 0xFFU));
}

inline void appendUint16LittleEndian(std::vector<std::uint8_t> &output, std::uint16_t value)
{
    output.push_back(static_cast<std::uint8_t>(value & 0xFFU));
    output.push_back(static_cast<std::uint8_t>((value >> 8U) & 0xFFU));
}

inline bool readUint32LittleEndian(const std::uint8_t *&cursor, const std::uint8_t *end,
                                   std::uint32_t &value)
{
    if (cursor > end || static_cast<std::size_t>(end - cursor) < kUint32WireSize)
    {
        return false;
    }
    value = static_cast<std::uint32_t>(cursor[0]) | (static_cast<std::uint32_t>(cursor[1]) << 8U) |
            (static_cast<std::uint32_t>(cursor[2]) << 16U) |
            (static_cast<std::uint32_t>(cursor[3]) << 24U);
    cursor += kUint32WireSize;
    return true;
}

inline bool readUint16LittleEndian(const std::uint8_t *&cursor, const std::uint8_t *end,
                                   std::uint16_t &value)
{
    if (cursor > end || static_cast<std::size_t>(end - cursor) < kUint16WireSize)
    {
        return false;
    }
    value = static_cast<std::uint16_t>(cursor[0]) |
            static_cast<std::uint16_t>(static_cast<std::uint16_t>(cursor[1]) << 8U);
    cursor += kUint16WireSize;
    return true;
}

inline char jsonHexDigit(unsigned value)
{
    return static_cast<char>(value < 10U ? ('0' + value) : ('A' + (value - 10U)));
}

inline bool jsonHexValue(char ch, std::uint32_t &value)
{
    if (ch >= '0' && ch <= '9')
    {
        value = static_cast<std::uint32_t>(ch - '0');
        return true;
    }
    if (ch >= 'a' && ch <= 'f')
    {
        value = static_cast<std::uint32_t>(ch - 'a' + 10);
        return true;
    }
    if (ch >= 'A' && ch <= 'F')
    {
        value = static_cast<std::uint32_t>(ch - 'A' + 10);
        return true;
    }
    return false;
}

inline bool appendJsonCodePoint(std::string &output, std::uint32_t code_point)
{
    if (code_point <= 0x7FU)
    {
        output.push_back(static_cast<char>(code_point));
        return true;
    }
    if (code_point <= 0x7FFU)
    {
        output.push_back(static_cast<char>(0xC0U | (code_point >> 6U)));
        output.push_back(static_cast<char>(0x80U | (code_point & 0x3FU)));
        return true;
    }
    if (code_point >= 0xD800U && code_point <= 0xDFFFU)
    {
        return false;
    }
    if (code_point <= 0xFFFFU)
    {
        output.push_back(static_cast<char>(0xE0U | (code_point >> 12U)));
        output.push_back(static_cast<char>(0x80U | ((code_point >> 6U) & 0x3FU)));
        output.push_back(static_cast<char>(0x80U | (code_point & 0x3FU)));
        return true;
    }
    if (code_point <= 0x10FFFFU)
    {
        output.push_back(static_cast<char>(0xF0U | (code_point >> 18U)));
        output.push_back(static_cast<char>(0x80U | ((code_point >> 12U) & 0x3FU)));
        output.push_back(static_cast<char>(0x80U | ((code_point >> 6U) & 0x3FU)));
        output.push_back(static_cast<char>(0x80U | (code_point & 0x3FU)));
        return true;
    }
    return false;
}

inline std::string escapeJsonString(std::string_view input)
{
    std::string escaped;
    escaped.reserve(input.size());
    for (unsigned char ch : input)
    {
        switch (ch)
        {
            case '"':
                escaped += "\\\"";
                break;
            case '\\':
                escaped += "\\\\";
                break;
            case '\b':
                escaped += "\\b";
                break;
            case '\f':
                escaped += "\\f";
                break;
            case '\n':
                escaped += "\\n";
                break;
            case '\r':
                escaped += "\\r";
                break;
            case '\t':
                escaped += "\\t";
                break;
            default:
                if (ch < 0x20U)
                {
                    escaped += "\\u00";
                    escaped.push_back(jsonHexDigit((ch >> 4U) & 0x0FU));
                    escaped.push_back(jsonHexDigit(ch & 0x0FU));
                }
                else
                {
                    escaped.push_back(static_cast<char>(ch));
                }
                break;
        }
    }
    return escaped;
}

inline bool readJsonStringField(const std::string &json, std::string_view key, std::string &value)
{
    const std::string field = "\"" + std::string(key) + "\":\"";
    std::size_t pos = json.find(field);
    if (pos == std::string::npos)
    {
        return false;
    }
    pos += field.size();

    std::string decoded;
    while (pos < json.size())
    {
        const unsigned char ch = static_cast<unsigned char>(json[pos++]);
        if (ch == '"')
        {
            value = std::move(decoded);
            return true;
        }
        if (ch != '\\')
        {
            if (ch < 0x20U)
            {
                return false;
            }
            decoded.push_back(static_cast<char>(ch));
            continue;
        }
        if (pos >= json.size())
        {
            return false;
        }

        const char escape = json[pos++];
        switch (escape)
        {
            case '"':
            case '\\':
            case '/':
                decoded.push_back(escape);
                break;
            case 'b':
                decoded.push_back('\b');
                break;
            case 'f':
                decoded.push_back('\f');
                break;
            case 'n':
                decoded.push_back('\n');
                break;
            case 'r':
                decoded.push_back('\r');
                break;
            case 't':
                decoded.push_back('\t');
                break;
            case 'u':
            {
                if (json.size() - pos < 4U)
                {
                    return false;
                }
                std::uint32_t code_point = 0;
                for (std::size_t i = 0; i < 4U; ++i)
                {
                    std::uint32_t digit = 0;
                    if (!jsonHexValue(json[pos + i], digit))
                    {
                        return false;
                    }
                    code_point = (code_point << 4U) | digit;
                }
                pos += 4U;
                if (!appendJsonCodePoint(decoded, code_point))
                {
                    return false;
                }
                break;
            }
            default:
                return false;
        }
    }
    return false;
}

} // namespace nerve::serialization::detail
