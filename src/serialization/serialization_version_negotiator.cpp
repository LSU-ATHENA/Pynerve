
#include "nerve/serialization/serialization_manager.hpp"

#include <algorithm>

namespace nerve::serialization
{
namespace
{

constexpr uint32_t kMaxGeneratedVersionMinorSpan = 1000;

} // namespace

VersionNegotiator::VersionNegotiator() = default;
void VersionNegotiator::registerSchema(const SchemaMetadata &metadata)
{
    std::lock_guard<std::mutex> lock(mutex_);
    schemas_[metadata.schema_name] = metadata;
}
void VersionNegotiator::unregisterSchema(const std::string &schema_name)
{
    std::lock_guard<std::mutex> lock(mutex_);
    schemas_.erase(schema_name);
}
VersionNegotiationResult
VersionNegotiator::negotiateVersion(const std::string &schema_name,
                                    const SchemaVersion &requested_version,
                                    const std::vector<SchemaVersion> &supported_versions)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = schemas_.find(schema_name);
    if (it == schemas_.end())
    {
        return VersionNegotiationResult::errorResult("Schema not found: " + schema_name);
    }
    const SchemaMetadata &metadata = it->second;
    if (!metadata.isVersionCompatible(requested_version))
    {
        return VersionNegotiationResult::errorResult(
            "Requested version " + requested_version.toString() +
            " is not compatible with schema " + schema_name);
    }
    SchemaVersion best_version = requested_version;
    bool needs_conversion = false;
    std::string conversion_strategy;
    for (const auto &supported : supported_versions)
    {
        if (metadata.isVersionCompatible(supported))
        {
            if (supported == requested_version)
            {
                return VersionNegotiationResult::successResult(supported, false);
            }
            if (supported.major == requested_version.major &&
                supported.minor >= requested_version.minor)
            {
                if (best_version.major != requested_version.major ||
                    supported.minor < best_version.minor)
                {
                    best_version = supported;
                    needs_conversion = true;
                    conversion_strategy = "upgrade_minor_version";
                }
            }
        }
    }
    if (needs_conversion)
    {
        return VersionNegotiationResult::successResult(best_version, true, conversion_strategy);
    }
    return VersionNegotiationResult::errorResult("No compatible version found for schema " +
                                                 schema_name);
}
VersionNegotiationResult VersionNegotiator::negotiateVersion(const std::string &schema_name,
                                                             const SchemaVersion &requested_version)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = schemas_.find(schema_name);
    if (it == schemas_.end())
    {
        return VersionNegotiationResult::errorResult("Schema not found: " + schema_name);
    }
    const SchemaMetadata &metadata = it->second;
    if (metadata.isVersionCompatible(requested_version))
    {
        return VersionNegotiationResult::successResult(requested_version, false);
    }
    if (metadata.version.isCompatibleWith(requested_version))
    {
        return VersionNegotiationResult::successResult(metadata.version, true,
                                                       "use_latest_compatible");
    }
    return VersionNegotiationResult::errorResult("Requested version " +
                                                 requested_version.toString() +
                                                 " is not compatible with schema " + schema_name);
}
SchemaMetadata VersionNegotiator::getSchemaMetadata(const std::string &schema_name) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = schemas_.find(schema_name);
    if (it != schemas_.end())
    {
        return it->second;
    }
    return SchemaMetadata{};
}
std::vector<std::string> VersionNegotiator::getRegisteredSchemas() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> schemas;
    schemas.reserve(schemas_.size());
    for (const auto &[name, metadata] : schemas_)
    {
        schemas.push_back(name);
    }
    return schemas;
}
std::vector<SchemaVersion>
VersionNegotiator::getSupportedVersions(const std::string &schema_name) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = schemas_.find(schema_name);
    if (it != schemas_.end())
    {
        const SchemaMetadata &metadata = it->second;
        if (metadata.minCompatibleVersion.major != metadata.version.major ||
            metadata.maxCompatibleVersion.major != metadata.version.major ||
            metadata.minCompatibleVersion.minor > metadata.maxCompatibleVersion.minor)
        {
            return {};
        }
        const uint32_t minor_span =
            metadata.maxCompatibleVersion.minor - metadata.minCompatibleVersion.minor;
        if (minor_span > kMaxGeneratedVersionMinorSpan)
        {
            return {};
        }
        std::vector<SchemaVersion> versions;
        versions.reserve(static_cast<std::size_t>(minor_span) + 1U);
        for (uint32_t offset = 0; offset <= minor_span; ++offset)
        {
            versions.emplace_back(metadata.version.major,
                                  metadata.minCompatibleVersion.minor + offset, 0);
        }
        return versions;
    }
    return {};
}
bool VersionNegotiator::isVersionSupported(const std::string &schema_name,
                                           const SchemaVersion &version) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = schemas_.find(schema_name);
    if (it != schemas_.end())
    {
        return it->second.isVersionCompatible(version);
    }
    return false;
}
bool VersionNegotiator::areVersionsCompatible(const std::string &schema_name,
                                              const SchemaVersion &version1,
                                              const SchemaVersion &version2) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = schemas_.find(schema_name);
    if (it != schemas_.end())
    {
        const SchemaMetadata &metadata = it->second;
        return metadata.isVersionCompatible(version1) && metadata.isVersionCompatible(version2) &&
               version1.major == version2.major;
    }
    return version1.isCompatibleWith(version2);
}
VersionNegotiationResult VersionNegotiationResult::successResult(const SchemaVersion &version,
                                                                 bool needs_conversion,
                                                                 const std::string &strategy)
{
    VersionNegotiationResult result;
    result.success = true;
    result.negotiated_version = version;
    result.requiresConversion = needs_conversion;
    result.conversion_strategy = strategy;
    return result;
}
VersionNegotiationResult VersionNegotiationResult::errorResult(const std::string &error)
{
    VersionNegotiationResult result;
    result.success = false;
    result.error_message = error;
    return result;
}

} // namespace nerve::serialization
