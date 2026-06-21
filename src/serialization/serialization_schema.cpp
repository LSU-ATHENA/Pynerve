
#include "nerve/serialization/serialization_manager.hpp"

#include <chrono>
#include <limits>
#include <regex>
#include <sstream>

namespace nerve::serialization
{
namespace
{

bool parseVersionComponent(const std::string &component, uint32_t &value)
{
    if (component.empty())
    {
        return false;
    }

    uint32_t parsed = 0;
    for (unsigned char ch : component)
    {
        if (ch < '0' || ch > '9')
        {
            return false;
        }
        const uint32_t digit = static_cast<uint32_t>(ch - '0');
        if (parsed > (std::numeric_limits<uint32_t>::max() - digit) / 10U)
        {
            return false;
        }
        parsed = parsed * 10U + digit;
    }
    value = parsed;
    return true;
}

} // namespace

std::string SchemaVersion::toString() const
{
    return std::to_string(major) + "." + std::to_string(minor) + "." + std::to_string(patch);
}
bool SchemaVersion::isCompatibleWith(const SchemaVersion &other) const
{
    if (major != other.major)
    {
        return false;
    }
    if (minor < other.minor)
    {
        return false;
    }
    return true;
}
bool SchemaVersion::operator==(const SchemaVersion &other) const
{
    return major == other.major && minor == other.minor && patch == other.patch;
}
bool SchemaVersion::operator<(const SchemaVersion &other) const
{
    if (major != other.major)
        return major < other.major;
    if (minor != other.minor)
        return minor < other.minor;
    return patch < other.patch;
}
bool SchemaVersion::operator<=(const SchemaVersion &other) const
{
    return *this < other || *this == other;
}
bool SchemaVersion::operator>=(const SchemaVersion &other) const
{
    return !(*this < other);
}
SchemaVersion SchemaVersion::fromString(const std::string &version_str)
{
    static const std::regex versionRegex(R"(^(\d+)\.(\d+)\.(\d+)$)");
    std::smatch match;
    if (std::regex_match(version_str, match, versionRegex))
    {
        uint32_t major = 0;
        uint32_t minor = 0;
        uint32_t patch = 0;
        if (parseVersionComponent(match[1].str(), major) &&
            parseVersionComponent(match[2].str(), minor) &&
            parseVersionComponent(match[3].str(), patch))
        {
            return SchemaVersion(major, minor, patch);
        }
    }
    return SchemaVersion(1, 0, 0);
}
bool SchemaMetadata::isVersionCompatible(const SchemaVersion &check_version) const
{
    return check_version >= minCompatibleVersion && check_version <= maxCompatibleVersion;
}
std::string SchemaMetadata::toString() const
{
    std::ostringstream oss;
    oss << "Schema: " << schema_name << "\n";
    oss << "Version: " << version.toString() << "\n";
    oss << "Min Compatible: " << minCompatibleVersion.toString() << "\n";
    oss << "Max Compatible: " << maxCompatibleVersion.toString() << "\n";
    oss << "Description: " << description << "\n";
    return oss.str();
}
std::unordered_map<std::string, std::string> SchemaMetadata::toMap() const
{
    std::unordered_map<std::string, std::string> result;
    result["schema_name"] = schema_name;
    result["version"] = version.toString();
    result["min_compatible_version"] = minCompatibleVersion.toString();
    result["max_compatible_version"] = maxCompatibleVersion.toString();
    result["description"] = description;
    auto time_t = std::chrono::system_clock::to_time_t(createdAt);
    result["created_at"] = std::to_string(time_t);
    return result;
}

} // namespace nerve::serialization
