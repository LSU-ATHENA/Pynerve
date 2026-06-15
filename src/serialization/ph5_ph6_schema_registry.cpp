
#include "nerve/serialization/ph5_ph6_schema_serializer.hpp"
#include "serialization_wire.hpp"

#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace nerve::serialization
{
namespace
{

bool parseBoolOption(const std::unordered_map<std::string, std::string> &options,
                     const std::string &key, bool default_value)
{
    const auto it = options.find(key);
    if (it == options.end())
    {
        return default_value;
    }
    return it->second == "1" || it->second == "true" || it->second == "TRUE" || it->second == "yes";
}

uint8_t parseDimensionOption(const std::unordered_map<std::string, std::string> &options,
                             const std::string &key, uint8_t default_value)
{
    const auto it = options.find(key);
    if (it == options.end())
    {
        return default_value;
    }
    try
    {
        const unsigned long value = std::stoul(it->second);
        return static_cast<uint8_t>(std::min<unsigned long>(value, 255UL));
    }
    catch (const std::exception &e)
    {
        fprintf(stderr, "error: %s\n", e.what());
        return default_value;
    }
}

int hexValue(char c)
{
    if (c >= '0' && c <= '9')
    {
        return c - '0';
    }
    if (c >= 'a' && c <= 'f')
    {
        return c - 'a' + 10;
    }
    if (c >= 'A' && c <= 'F')
    {
        return c - 'A' + 10;
    }
    return -1;
}

bool parseHexPayload(const std::string &encoded, std::vector<uint8_t> *payload)
{
    payload->clear();
    if (encoded.size() % 2 != 0)
    {
        return false;
    }
    payload->reserve(encoded.size() / 2);
    for (size_t i = 0; i < encoded.size(); i += 2)
    {
        const int high = hexValue(encoded[i]);
        const int low = hexValue(encoded[i + 1]);
        if (high < 0 || low < 0)
        {
            payload->clear();
            return false;
        }
        payload->push_back(static_cast<uint8_t>((high << 4) | low));
    }
    return true;
}

bool contextIsSupported(const SerializationContext &context, std::string *error)
{
    if (context.format != SerializationFormat::FLATBUFFERS)
    {
        *error = "PH5/PH6 migration requires FLATBUFFERS serialization context";
        return false;
    }
    if (context.schemaVersion.major == 0 || context.schemaVersion.major > 1)
    {
        *error = "Unsupported PH5/PH6 source schema version: " + context.schemaVersion.toString();
        return false;
    }
    return true;
}

PH5PH6ArtifactMetadata metadataFromContext(const SerializationContext &context,
                                           const std::string &default_type,
                                           const std::string &default_variant, bool default_highdim,
                                           uint8_t default_max_dimension)
{
    PH5PH6ArtifactMetadata metadata;
    metadata.schema_version = SchemaVersion(1, 1, 0);
    metadata.min_compatible_version = context.schemaVersion;
    metadata.artifact_type = default_type;
    metadata.algorithm_variant = default_variant;
    metadata.has_highdim_extension = default_highdim;
    metadata.max_supported_dimension = default_max_dimension;

    if (const auto it = context.options.find("artifact_type");
        it != context.options.end() && !it->second.empty())
    {
        metadata.artifact_type = it->second;
    }
    if (const auto it = context.options.find("algorithm_variant");
        it != context.options.end() && !it->second.empty())
    {
        metadata.algorithm_variant = it->second;
    }
    metadata.has_highdim_extension =
        parseBoolOption(context.options, "has_highdim_extension", metadata.has_highdim_extension);
    metadata.max_supported_dimension = parseDimensionOption(
        context.options, "max_supported_dimension", metadata.max_supported_dimension);
    if (const auto it = context.options.find("extension_field");
        it != context.options.end() && !it->second.empty())
    {
        metadata.extension_fields.push_back(it->second);
    }
    return metadata;
}

bool expectedMetadataHeaderMatches(const std::vector<uint8_t> &data,
                                   const PH5PH6ArtifactMetadata &expected_metadata)
{
    if (expected_metadata.artifact_type.empty())
    {
        return true;
    }
    const size_t header_offset = sizeof(uint32_t) + 1;
    if (data.size() < header_offset)
    {
        return false;
    }
    const uint8_t header_size = data[sizeof(uint32_t)];
    if (header_size < PH5PH6SchemaSerializer::MIN_HEADER_SIZE_EXTENDED ||
        header_size > PH5PH6SchemaSerializer::HEADER_SIZE_EXTENDED ||
        data.size() < header_offset + header_size)
    {
        return false;
    }
    std::vector<uint8_t> expected_header = expected_metadata.serializeMetadata();
    if (expected_header.size() > header_size)
    {
        return false;
    }
    expected_header.resize(header_size, 0);
    return std::equal(expected_header.begin(), expected_header.end(),
                      data.begin() + static_cast<std::ptrdiff_t>(header_offset));
}

} // namespace

PH5PH6SchemaRegistry::PH5PH6SchemaRegistry()
{
    registerArtifactType("PH5", createDefaultPh5Metadata());
    registerArtifactType("PH6", createDefaultPh6Metadata());
    registerArtifactType("CompactSummary", createDefaultCompactSummaryMetadata());
}
void PH5PH6SchemaRegistry::registerArtifactType(const std::string &artifact_type,
                                                const PH5PH6ArtifactMetadata &metadata)
{
    std::lock_guard<std::mutex> lock(mutex_);
    artifact_types_[artifact_type] = metadata;
}
void PH5PH6SchemaRegistry::unregisterArtifactType(const std::string &artifact_type)
{
    std::lock_guard<std::mutex> lock(mutex_);
    artifact_types_.erase(artifact_type);
}
VersionNegotiationResult
PH5PH6SchemaRegistry::negotiateVersion(const std::string &artifact_type,
                                       const SchemaVersion &requested_version) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = artifact_types_.find(artifact_type);
    if (it == artifact_types_.end())
    {
        return VersionNegotiationResult::errorResult("Unknown artifact type: " + artifact_type);
    }
    const auto &metadata = it->second;
    if (requested_version.major > metadata.schema_version.major)
    {
        return VersionNegotiationResult::errorResult(
            "Requested major version " + requested_version.toString() +
            " is newer than supported " + metadata.schema_version.toString());
    }
    if (requested_version.major < metadata.min_compatible_version.major)
    {
        return VersionNegotiationResult::errorResult(
            "Requested major version " + requested_version.toString() +
            " is below minimum compatible " + metadata.min_compatible_version.toString());
    }
    if (requested_version < metadata.schema_version)
    {
        return VersionNegotiationResult::successResult(metadata.schema_version, true,
                                                       "Upgrade recommended");
    }
    return VersionNegotiationResult::successResult(requested_version, false, "Version compatible");
}
PH5PH6ArtifactMetadata
PH5PH6SchemaRegistry::getArtifactMetadata(const std::string &artifact_type) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = artifact_types_.find(artifact_type);
    return (it != artifact_types_.end()) ? it->second : PH5PH6ArtifactMetadata{};
}
std::vector<std::string> PH5PH6SchemaRegistry::getRegisteredArtifactTypes() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> types;
    types.reserve(artifact_types_.size());
    for (const auto &[type, metadata] : artifact_types_)
    {
        types.push_back(type);
    }
    return types;
}
bool PH5PH6SchemaRegistry::isArtifactSupported(const std::string &artifact_type,
                                               const SchemaVersion &version) const
{
    auto result = negotiateVersion(artifact_type, version);
    return result.success;
}
std::unique_ptr<PH5PH6SchemaSerializer> PH5PH6SchemaRegistry::createSerializer() const
{
    return std::make_unique<PH5PH6SchemaSerializer>();
}
std::unique_ptr<Serializer> PH5PH6SchemaRegistry::createCompatibleSerializer() const
{
    return std::make_unique<PH5PH6SchemaSerializer>();
}
PH5PH6ArtifactMetadata PH5PH6SchemaRegistry::createDefaultPh5Metadata() const
{
    PH5PH6ArtifactMetadata metadata;
    metadata.schema_version = SchemaVersion(1, 1, 0);
    metadata.artifact_type = "PH5";
    metadata.algorithm_variant = "witness";
    metadata.has_highdim_extension = true;
    metadata.max_supported_dimension = 5;
    metadata.extension_fields = {"highdim_betti_top8", "highdim_lifetime_stats"};
    metadata.min_compatible_version = SchemaVersion(1, 0, 0);
    return metadata;
}
PH5PH6ArtifactMetadata PH5PH6SchemaRegistry::createDefaultPh6Metadata() const
{
    PH5PH6ArtifactMetadata metadata;
    metadata.schema_version = SchemaVersion(1, 1, 0);
    metadata.artifact_type = "PH6";
    metadata.algorithm_variant = "hierarchical";
    metadata.has_highdim_extension = true;
    metadata.max_supported_dimension = 6;
    metadata.extension_fields = {"highdim_betti_top8", "highdim_lifetime_stats",
                                 "dimension_complexity"};
    metadata.min_compatible_version = SchemaVersion(1, 0, 0);
    return metadata;
}
PH5PH6ArtifactMetadata PH5PH6SchemaRegistry::createDefaultCompactSummaryMetadata() const
{
    PH5PH6ArtifactMetadata metadata;
    metadata.schema_version = SchemaVersion(1, 1, 0);
    metadata.artifact_type = "CompactSummary";
    metadata.algorithm_variant = "default_value";
    metadata.has_highdim_extension = true;
    metadata.max_supported_dimension = 8;
    metadata.extension_fields = {"highdim_betti_top8", "highdim_lifetime_stats",
                                 "budget_utilization"};
    metadata.min_compatible_version = SchemaVersion(1, 0, 0);
    return metadata;
}
PH5PH6SchemaMigrator::MigrationResult
PH5PH6SchemaMigrator::migrateToExtended(const std::vector<uint8_t> &data,
                                              const SerializationContext &context)
{
    MigrationResult result;
    try
    {
        std::string context_error;
        if (!contextIsSupported(context, &context_error))
        {
            result.error_message = context_error;
            return result;
        }
        if (!validateData(data))
        {
            result.error_message = "Invalid legacy data format";
            return result;
        }
        PH5PH6ArtifactMetadata extended_metadata =
            metadataFromContext(context, "migrated", "legacy", false, 5);
        result.migrated_data = updateExtendedHeader(data, extended_metadata);
        result.new_metadata = extended_metadata;
        result.success = true;
    }
    catch (const std::exception &e)
    {
        result.success = false;
        result.error_message = "Migration failed: " + std::string(e.what());
    }
    return result;
}
PH5PH6SchemaMigrator::MigrationResult
PH5PH6SchemaMigrator::migrateWithExtension(const std::vector<uint8_t> &data,
                                           const PH5PH6ArtifactMetadata &extension_metadata,
                                           const SerializationContext &context)
{
    MigrationResult result;
    try
    {
        std::string context_error;
        if (!contextIsSupported(context, &context_error))
        {
            result.error_message = context_error;
            return result;
        }
        if (!validateData(data))
        {
            result.error_message = "Invalid legacy data format";
            return result;
        }
        result.new_metadata = extension_metadata;
        result.new_metadata.schema_version = SchemaVersion(1, 1, 0);
        result.new_metadata.min_compatible_version = context.schemaVersion;
        std::vector<uint8_t> extension_payload;
        if (result.new_metadata.has_highdim_extension)
        {
            const auto it = context.options.find("highdim_extension_payload_hex");
            if (it == context.options.end() || !parseHexPayload(it->second, &extension_payload))
            {
                result.error_message = "High-dimensional migration requires context option "
                                       "highdim_extension_payload_hex";
                return result;
            }
        }
        result.migrated_data =
            result.new_metadata.has_highdim_extension
                ? addHighdimExtension(data, result.new_metadata, extension_payload)
                : updateExtendedHeader(data, result.new_metadata);
        result.success = true;
    }
    catch (const std::exception &e)
    {
        result.success = false;
        result.error_message = "Extension migration failed: " + std::string(e.what());
    }
    return result;
}
bool PH5PH6SchemaMigrator::validateData(const std::vector<uint8_t> &data)
{
    if (data.size() < sizeof(uint32_t))
        return false;
    uint32_t magic = 0;
    const auto *cursor = data.data();
    const auto *end = data.data() + data.size();
    if (!detail::readUint32LittleEndian(cursor, end, magic))
    {
        return false;
    }
    return magic == PH5PH6SchemaSerializer::PH5PH6_MAGIC_V1;
}
bool PH5PH6SchemaMigrator::validateExtendedData(const std::vector<uint8_t> &data,
                                                const PH5PH6ArtifactMetadata &expected_metadata)
{
    if (data.size() < sizeof(uint32_t))
        return false;
    uint32_t magic = 0;
    const auto *cursor = data.data();
    const auto *end = data.data() + data.size();
    if (!detail::readUint32LittleEndian(cursor, end, magic))
    {
        return false;
    }
    if (magic != PH5PH6SchemaSerializer::PH5PH6_MAGIC_EXTENDED)
        return false;
    if (data.size() < sizeof(uint32_t) + 1)
        return false;
    uint8_t header_size = data[sizeof(uint32_t)];
    if (header_size < PH5PH6SchemaSerializer::MIN_HEADER_SIZE_EXTENDED ||
        header_size > PH5PH6SchemaSerializer::HEADER_SIZE_EXTENDED)
    {
        return false;
    }
    return expectedMetadataHeaderMatches(data, expected_metadata);
}
std::vector<uint8_t>
PH5PH6SchemaMigrator::addHighdimExtension(const std::vector<uint8_t> &data,
                                                  const PH5PH6ArtifactMetadata &metadata,
                                                  const std::vector<uint8_t> &extension_payload)
{
    std::vector<uint8_t> result = updateExtendedHeader(data, metadata);
    result.insert(result.end(), extension_payload.begin(), extension_payload.end());
    return result;
}
std::vector<uint8_t>
PH5PH6SchemaMigrator::updateExtendedHeader(const std::vector<uint8_t> &data,
                                           const PH5PH6ArtifactMetadata &new_metadata)
{
    std::vector<uint8_t> result;
    result.reserve(sizeof(uint32_t) + 1 + PH5PH6SchemaSerializer::HEADER_SIZE_EXTENDED +
                   data.size());
    detail::appendUint32LittleEndian(result, PH5PH6SchemaSerializer::PH5PH6_MAGIC_EXTENDED);
    auto metadata_bytes = new_metadata.serializeMetadata();
    if (metadata_bytes.size() > PH5PH6SchemaSerializer::HEADER_SIZE_EXTENDED)
    {
        throw std::length_error("PH5/PH6 migration metadata does not fit the extended header");
    }
    result.push_back(PH5PH6SchemaSerializer::HEADER_SIZE_EXTENDED);
    metadata_bytes.resize(PH5PH6SchemaSerializer::HEADER_SIZE_EXTENDED, 0);
    result.insert(result.end(), metadata_bytes.begin(), metadata_bytes.end());
    if (data.size() > sizeof(uint32_t))
    {
        result.insert(result.end(),
                      data.begin() + static_cast<std::ptrdiff_t>(sizeof(uint32_t)),
                      data.end());
    }
    return result;
}
} // namespace nerve::serialization
