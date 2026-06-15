
#include "nerve/serialization/ph5_ph6_schema_serializer.hpp"
#include "serialization_wire.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

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

bool contextIsSupported(const SerializationContext &context)
{
    return context.format == SerializationFormat::FLATBUFFERS && context.schemaVersion.major > 0 &&
           context.schemaVersion.major <= 1;
}

void applyContextOptions(PH5PH6ArtifactMetadata &metadata, const SerializationContext &context,
                         bool allow_artifact_type_override)
{
    metadata.min_compatible_version = context.schemaVersion;
    if (allow_artifact_type_override)
    {
        if (const auto it = context.options.find("artifact_type");
            it != context.options.end() && !it->second.empty())
        {
            metadata.artifact_type = it->second;
        }
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
}

} // namespace

std::vector<uint8_t> PH5PH6ArtifactMetadata::serializeMetadata() const
{
    std::vector<uint8_t> bytes;
    auto appendString = [&bytes](const std::string &value) {
        if (value.size() > std::numeric_limits<uint16_t>::max())
        {
            throw std::length_error("PH5/PH6 metadata string exceeds uint16 length field");
        }
        detail::appendUint16LittleEndian(bytes, static_cast<uint16_t>(value.size()));
        bytes.insert(bytes.end(), value.begin(), value.end());
    };
    detail::appendUint32LittleEndian(bytes, schema_version.major);
    detail::appendUint32LittleEndian(bytes, schema_version.minor);
    detail::appendUint32LittleEndian(bytes, schema_version.patch);
    detail::appendUint32LittleEndian(bytes, min_compatible_version.major);
    detail::appendUint32LittleEndian(bytes, min_compatible_version.minor);
    detail::appendUint32LittleEndian(bytes, min_compatible_version.patch);
    appendString(artifact_type);
    appendString(algorithm_variant);
    bytes.push_back(has_highdim_extension ? 1U : 0U);
    bytes.push_back(max_supported_dimension);
    if (extension_fields.size() > std::numeric_limits<uint16_t>::max())
    {
        throw std::length_error("PH5/PH6 metadata extension field count exceeds uint16");
    }
    detail::appendUint16LittleEndian(bytes, static_cast<uint16_t>(extension_fields.size()));
    for (const auto &field : extension_fields)
    {
        appendString(field);
    }
    return bytes;
}

bool PH5PH6ArtifactMetadata::deserializeMetadata(const std::vector<uint8_t> &data)
{
    std::size_t offset = 0;
    auto readU32 = [&data, &offset](uint32_t &value) -> bool {
        if (offset + sizeof(uint32_t) > data.size())
        {
            return false;
        }
        const auto *cursor = data.data() + offset;
        const auto *end = data.data() + data.size();
        if (!detail::readUint32LittleEndian(cursor, end, value))
        {
            return false;
        }
        offset += sizeof(uint32_t);
        return true;
    };
    auto read_u16 = [&data, &offset](uint16_t &value) -> bool {
        if (offset + sizeof(uint16_t) > data.size())
        {
            return false;
        }
        const auto *cursor = data.data() + offset;
        const auto *end = data.data() + data.size();
        if (!detail::readUint16LittleEndian(cursor, end, value))
        {
            return false;
        }
        offset += sizeof(uint16_t);
        return true;
    };

    if (!readU32(schema_version.major) || !readU32(schema_version.minor) ||
        !readU32(schema_version.patch))
    {
        return false;
    }
    if (!readU32(min_compatible_version.major) || !readU32(min_compatible_version.minor) ||
        !readU32(min_compatible_version.patch))
    {
        return false;
    }

    uint16_t text_size = 0;
    if (!read_u16(text_size) || offset + text_size > data.size())
    {
        return false;
    }
    artifact_type.assign(reinterpret_cast<const char *>(data.data() + offset), text_size);
    offset += text_size;

    if (!read_u16(text_size) || offset + text_size > data.size())
    {
        return false;
    }
    algorithm_variant.assign(reinterpret_cast<const char *>(data.data() + offset), text_size);
    offset += text_size;

    if (offset + 2 > data.size())
    {
        return false;
    }
    has_highdim_extension = data[offset++] != 0U;
    max_supported_dimension = data[offset++];

    uint16_t extension_count = 0;
    if (!read_u16(extension_count))
    {
        return false;
    }
    extension_fields.clear();
    for (uint16_t i = 0; i < extension_count; ++i)
    {
        uint16_t field_size = 0;
        if (!read_u16(field_size) || offset + field_size > data.size())
        {
            return false;
        }
        extension_fields.emplace_back(reinterpret_cast<const char *>(data.data() + offset),
                                      field_size);
        offset += field_size;
    }
    return true;
}

std::string PH5PH6ArtifactMetadata::toString() const
{
    return std::string("PH5PH6Artifact{version=") + schema_version.toString() +
           ",type=" + artifact_type + ",variant=" + algorithm_variant +
           ",highdim=" + (has_highdim_extension ? "true" : "false") +
           ",max_dim=" + std::to_string(max_supported_dimension) + "}";
}

PH5PH6SchemaSerializer::PH5PH6SchemaSerializer() = default;

ErrorResult<std::vector<uint8_t>>
PH5PH6SchemaSerializer::serialize(const void *data, size_t size,
                                  const SerializationContext &context)
{
    SchemaMetadata metadata;
    metadata.schema_name = "PH5PH6";
    metadata.description = "Default PH5/PH6 artifact";
    metadata.version = getSchemaVersion();
    metadata.minCompatibleVersion = getMinCompatibleVersion();
    metadata.maxCompatibleVersion = getSchemaVersion();
    return serializeWithMetadata(data, size, metadata, context);
}

ErrorResult<std::vector<uint8_t>>
PH5PH6SchemaSerializer::serializeWithMetadata(const void *data, size_t size,
                                              const SchemaMetadata &schema_metadata,
                                              const SerializationContext &context)
{
    if (!contextIsSupported(context))
    {
        return ErrorResult<std::vector<uint8_t>>::error(ErrorCode::E31_SCHEMA_VERSION);
    }
    PH5PH6ArtifactMetadata metadata;
    metadata.schema_version = schema_metadata.version;
    metadata.min_compatible_version = schema_metadata.minCompatibleVersion.major == 0
                                          ? getMinCompatibleVersion()
                                          : schema_metadata.minCompatibleVersion;
    metadata.artifact_type = schema_metadata.schema_name;
    metadata.algorithm_variant = "default";
    applyContextOptions(metadata, context, true);
    return serializeArtifactWithMetadata(data, size, metadata);
}

ErrorResult<std::vector<uint8_t>>
PH5PH6SchemaSerializer::deserialize(const std::vector<uint8_t> &data,
                                    const SerializationContext &context)
{
    auto parsed = deserializeWithMetadata(data, context);
    if (parsed.isError())
    {
        return ErrorResult<std::vector<uint8_t>>::error(parsed.errorCode());
    }
    return ErrorResult<std::vector<uint8_t>>::success(std::move(parsed.value().first));
}

ErrorResult<std::pair<std::vector<uint8_t>, SchemaMetadata>>
PH5PH6SchemaSerializer::deserializeWithMetadata(const std::vector<uint8_t> &data,
                                                const SerializationContext &context)
{
    if (!contextIsSupported(context))
    {
        return ErrorResult<std::pair<std::vector<uint8_t>, SchemaMetadata>>::error(
            ErrorCode::E31_SCHEMA_VERSION);
    }
    if (data.size() < sizeof(uint32_t))
    {
        return ErrorResult<std::pair<std::vector<uint8_t>, SchemaMetadata>>::error(
            ErrorCode::E31_SCHEMA_VERSION);
    }
    uint32_t magic = 0;
    const auto *magic_cursor = data.data();
    const auto *data_end = data.data() + data.size();
    if (!detail::readUint32LittleEndian(magic_cursor, data_end, magic))
    {
        return ErrorResult<std::pair<std::vector<uint8_t>, SchemaMetadata>>::error(
            ErrorCode::E31_SCHEMA_VERSION);
    }
    if (magic != PH5PH6_MAGIC_EXTENDED && magic != PH5PH6_MAGIC_V1)
    {
        return ErrorResult<std::pair<std::vector<uint8_t>, SchemaMetadata>>::error(
            ErrorCode::E31_SCHEMA_VERSION);
    }

    if (magic == PH5PH6_MAGIC_V1)
    {
        auto payload_result = deserializePayload(data, sizeof(uint32_t));
        if (payload_result.isError())
        {
            return ErrorResult<std::pair<std::vector<uint8_t>, SchemaMetadata>>::error(
                payload_result.errorCode());
        }
        SchemaMetadata metadata;
        metadata.version = SchemaVersion(1, 0, 0);
        metadata.minCompatibleVersion = SchemaVersion(1, 0, 0);
        metadata.maxCompatibleVersion = getSchemaVersion();
        metadata.schema_name = "PH5PH6";
        metadata.description = "Legacy PH5/PH6 artifact";
        return ErrorResult<std::pair<std::vector<uint8_t>, SchemaMetadata>>::success(
            std::make_pair(payload_result.value(), metadata));
    }

    if (data.size() < sizeof(uint32_t) + 1)
    {
        return ErrorResult<std::pair<std::vector<uint8_t>, SchemaMetadata>>::error(
            ErrorCode::E31_SCHEMA_VERSION);
    }
    const uint8_t header_size = data[sizeof(uint32_t)];
    if (header_size < MIN_HEADER_SIZE_EXTENDED || header_size > HEADER_SIZE_EXTENDED ||
        data.size() < sizeof(uint32_t) + 1 + header_size)
    {
        return ErrorResult<std::pair<std::vector<uint8_t>, SchemaMetadata>>::error(
            ErrorCode::E31_SCHEMA_VERSION);
    }

    std::size_t offset = sizeof(uint32_t) + 1;
    PH5PH6ArtifactMetadata metadata = deserializeHeader(data, offset);
    if (!validateHeader(metadata))
    {
        return ErrorResult<std::pair<std::vector<uint8_t>, SchemaMetadata>>::error(
            ErrorCode::E31_SCHEMA_VERSION);
    }

    auto payload_result = (magic == PH5PH6_MAGIC_EXTENDED)
                              ? deserializeExtendedPayload(data, offset)
                              : deserializePayload(data, offset);
    if (payload_result.isError())
    {
        return ErrorResult<std::pair<std::vector<uint8_t>, SchemaMetadata>>::error(
            payload_result.errorCode());
    }

    SchemaMetadata schema_metadata;
    schema_metadata.version = metadata.schema_version;
    schema_metadata.minCompatibleVersion = metadata.min_compatible_version;
    schema_metadata.maxCompatibleVersion = getSchemaVersion();
    schema_metadata.schema_name = metadata.artifact_type;
    schema_metadata.description = metadata.toString();
    return ErrorResult<std::pair<std::vector<uint8_t>, SchemaMetadata>>::success(
        std::make_pair(payload_result.value(), schema_metadata));
}

ErrorResult<std::vector<uint8_t>>
PH5PH6SchemaSerializer::serializePh5Artifact(const void *data, size_t size,
                                             const PH5PH6ArtifactMetadata &artifact_metadata,
                                             const SerializationContext &context)
{
    if (!contextIsSupported(context))
    {
        return ErrorResult<std::vector<uint8_t>>::error(ErrorCode::E31_SCHEMA_VERSION);
    }
    PH5PH6ArtifactMetadata metadata = artifact_metadata;
    metadata.artifact_type = "PH5";
    metadata.schema_version = getSchemaVersion();
    metadata.min_compatible_version = context.schemaVersion;
    return serializeArtifactWithMetadata(data, size, metadata);
}

ErrorResult<std::vector<uint8_t>>
PH5PH6SchemaSerializer::serializePh6Artifact(const void *data, size_t size,
                                             const PH5PH6ArtifactMetadata &artifact_metadata,
                                             const SerializationContext &context)
{
    if (!contextIsSupported(context))
    {
        return ErrorResult<std::vector<uint8_t>>::error(ErrorCode::E31_SCHEMA_VERSION);
    }
    PH5PH6ArtifactMetadata metadata = artifact_metadata;
    metadata.artifact_type = "PH6";
    metadata.schema_version = getSchemaVersion();
    metadata.min_compatible_version = context.schemaVersion;
    metadata.max_supported_dimension = std::max<uint8_t>(6, metadata.max_supported_dimension);
    return serializeArtifactWithMetadata(data, size, metadata);
}

ErrorResult<std::vector<uint8_t>>
PH5PH6SchemaSerializer::serializeCompactSummary(const nerve::summary::CompactSummary &summary,
                                                const PH5PH6ArtifactMetadata &artifact_metadata,
                                                const SerializationContext &context)
{
    std::vector<uint8_t> payload = summary.serialize();
    return serializePh6Artifact(payload.data(), payload.size(), artifact_metadata, context);
}

ErrorResult<std::pair<nerve::summary::CompactSummary, PH5PH6ArtifactMetadata>>
PH5PH6SchemaSerializer::deserializeCompactSummary(const std::vector<uint8_t> &data,
                                                  const SerializationContext &context)
{
    auto parsed = deserializeWithMetadata(data, context);
    if (parsed.isError())
    {
        return ErrorResult<std::pair<nerve::summary::CompactSummary,
                                     PH5PH6ArtifactMetadata>>::error(parsed.errorCode());
    }
    nerve::summary::CompactSummary summary;
    if (!summary.deserialize(parsed.value().first))
    {
        return ErrorResult<std::pair<nerve::summary::CompactSummary,
                                     PH5PH6ArtifactMetadata>>::error(ErrorCode::E31_SCHEMA_VERSION);
    }
    PH5PH6ArtifactMetadata metadata;
    metadata.schema_version = parsed.value().second.version;
    metadata.artifact_type = parsed.value().second.schema_name;
    metadata.algorithm_variant = "deserialized";
    return ErrorResult<std::pair<nerve::summary::CompactSummary, PH5PH6ArtifactMetadata>>::success(
        std::make_pair(summary, metadata));
}

bool PH5PH6SchemaSerializer::isVersionCompatible(const SchemaVersion &version) const
{
    return version >= getMinCompatibleVersion() && version.major <= getSchemaVersion().major;
}

ErrorResult<std::vector<uint8_t>>
PH5PH6SchemaSerializer::serializeArtifactWithMetadata(const void *data, size_t size,
                                                      const PH5PH6ArtifactMetadata &metadata) const
{
    if (!validateHeader(metadata) || !validatePayload(data, size, metadata))
    {
        return ErrorResult<std::vector<uint8_t>>::error(ErrorCode::E31_SCHEMA_VERSION);
    }

    std::vector<uint8_t> output;
    detail::appendUint32LittleEndian(output, PH5PH6_MAGIC_EXTENDED);

    std::vector<uint8_t> header = serializeHeader(metadata);
    output.push_back(static_cast<uint8_t>(header.size()));
    output.insert(output.end(), header.begin(), header.end());

    std::vector<uint8_t> payload = serializeExtendedPayload(data, size);
    output.insert(output.end(), payload.begin(), payload.end());
    return ErrorResult<std::vector<uint8_t>>::success(std::move(output));
}

std::vector<uint8_t>
PH5PH6SchemaSerializer::serializeHeader(const PH5PH6ArtifactMetadata &metadata) const
{
    std::vector<uint8_t> raw = metadata.serializeMetadata();
    if (raw.size() > HEADER_SIZE_EXTENDED)
    {
        throw std::length_error("PH5/PH6 metadata does not fit the extended header");
    }
    raw.resize(HEADER_SIZE_EXTENDED, 0);
    return raw;
}

std::vector<uint8_t> PH5PH6SchemaSerializer::serializePayload(const void *data,
                                                                    size_t size) const
{
    if (size == 0)
    {
        return {};
    }
    const auto *ptr = static_cast<const uint8_t *>(data);
    return std::vector<uint8_t>(ptr, ptr + size);
}

std::vector<uint8_t> PH5PH6SchemaSerializer::serializeExtendedPayload(const void *data,
                                                                      size_t size) const
{
    return serializePayload(data, size);
}

PH5PH6ArtifactMetadata PH5PH6SchemaSerializer::deserializeHeader(const std::vector<uint8_t> &data,
                                                                 size_t &offset) const
{
    PH5PH6ArtifactMetadata metadata;
    const uint8_t header_size = data[sizeof(uint32_t)];
    if (header_size >= MIN_HEADER_SIZE_EXTENDED && offset + header_size <= data.size())
    {
        const std::vector<uint8_t> header(data.begin() + static_cast<std::ptrdiff_t>(offset),
                                          data.begin() +
                                              static_cast<std::ptrdiff_t>(offset + header_size));
        PH5PH6ArtifactMetadata parsed;
        if (parsed.deserializeMetadata(header))
        {
            metadata = std::move(parsed);
        }
        offset += header_size;
    }
    return metadata;
}

ErrorResult<std::vector<uint8_t>>
PH5PH6SchemaSerializer::deserializePayload(const std::vector<uint8_t> &data,
                                                 size_t offset) const
{
    if (offset > data.size())
    {
        return ErrorResult<std::vector<uint8_t>>::error(ErrorCode::E31_SCHEMA_VERSION);
    }
    return ErrorResult<std::vector<uint8_t>>::success(
        std::vector<uint8_t>(data.begin() + static_cast<std::ptrdiff_t>(offset), data.end()));
}

ErrorResult<std::vector<uint8_t>>
PH5PH6SchemaSerializer::deserializeExtendedPayload(const std::vector<uint8_t> &data,
                                                   size_t offset) const
{
    return deserializePayload(data, offset);
}

bool PH5PH6SchemaSerializer::validateHeader(const PH5PH6ArtifactMetadata &metadata) const
{
    try
    {
        return metadata.schema_version.major >= 1 && metadata.min_compatible_version.major >= 1 &&
               !metadata.artifact_type.empty() &&
               metadata.serializeMetadata().size() <= HEADER_SIZE_EXTENDED;
    }
    catch (const std::exception &e)
    {
        fprintf(stderr, "error: %s\n", e.what());
        return false;
    }
}

bool PH5PH6SchemaSerializer::validatePayload(const void *data, size_t size,
                                             const PH5PH6ArtifactMetadata &metadata) const
{
    if (metadata.has_highdim_extension && metadata.max_supported_dimension < 5)
    {
        return false;
    }
    return data != nullptr || size == 0;
}

} // namespace nerve::serialization
