
#include "nerve/serialization/serialization_manager.hpp"

#include <cstring>
#include <vector>

namespace nerve::serialization
{

// Schema Version Constants
constexpr uint32_t MAX_SCHEMA_VERSION_MINOR = 1000;
constexpr uint32_t MAX_SCHEMA_VERSION_PATCH = 1000;

ErrorCode serializationErrorToErrorCode(errors::SerializationErrorCode ser_error)
{
    switch (ser_error)
    {
        case errors::SerializationErrorCode::SUCCESS:
            return ErrorCode::SUCCESS;
        case errors::SerializationErrorCode::INCOMPATIBLE_SCHEMA_VERSION:
        case errors::SerializationErrorCode::UNSUPPORTED_SCHEMA_VERSION:
        case errors::SerializationErrorCode::SCHEMA_VERSION_NEGOTIATION_FAILED:
        case errors::SerializationErrorCode::METADATA_MISSING:
        case errors::SerializationErrorCode::METADATA_CORRUPTED:
        case errors::SerializationErrorCode::METADATA_VERSION_MISMATCH:
        case errors::SerializationErrorCode::VERSION_CONVERSION_FAILED:
        case errors::SerializationErrorCode::CONVERSION_NOT_SUPPORTED:
        case errors::SerializationErrorCode::CONVERSION_DATA_LOSS:
            return ErrorCode::E31_SCHEMA_VERSION;
        case errors::SerializationErrorCode::SCHEMA_NOT_FOUND:
        case errors::SerializationErrorCode::UNSUPPORTED_SERIALIZATION_FORMAT:
        case errors::SerializationErrorCode::SERIALIZATION_FORMAT_MISMATCH:
            return ErrorCode::E41_RESOURCE_LIMIT;
        case errors::SerializationErrorCode::INVALID_SERIALIZATION_DATA:
        case errors::SerializationErrorCode::SERIALIZATION_DATA_CORRUPTED:
        case errors::SerializationErrorCode::DESERIALIZATION_FAILED:
        case errors::SerializationErrorCode::BUFFER_TOO_SMALL:
            return ErrorCode::E01_IO_CORRUPT;
    }
    return ErrorCode::E01_IO_CORRUPT;
}
std::string serialization_format_to_string(SerializationFormat format)
{
    switch (format)
    {
        case SerializationFormat::FLATBUFFERS:
            return "FlatBuffers";
        case SerializationFormat::ARROW:
            return "Apache Arrow";
        case SerializationFormat::JSON:
            return "JSON";
        case SerializationFormat::BINARY:
            return "Binary";
        case SerializationFormat::PROTOBUF:
            return "Protocol Buffers";
        default:
            return "Unknown";
    }
}
SerializationFormat stringToSerializationFormat(const std::string &format_str)
{
    if (format_str == "FlatBuffers" || format_str == "flatbuffers")
    {
        return SerializationFormat::FLATBUFFERS;
    }
    else if (format_str == "Apache Arrow" || format_str == "arrow")
    {
        return SerializationFormat::ARROW;
    }
    else if (format_str == "JSON" || format_str == "json")
    {
        return SerializationFormat::JSON;
    }
    else if (format_str == "Binary" || format_str == "binary")
    {
        return SerializationFormat::BINARY;
    }
    else if (format_str == "Protocol Buffers" || format_str == "protobuf")
    {
        return SerializationFormat::PROTOBUF;
    }
    return SerializationFormat::FLATBUFFERS;
}
bool isSchemaVersionValid(const SchemaVersion &version)
{
    return version.major > 0 && version.minor <= MAX_SCHEMA_VERSION_MINOR &&
           version.patch <= MAX_SCHEMA_VERSION_PATCH;
}
bool isVersionCompatible(const SchemaVersion &current, const SchemaVersion &target)
{
    return current.isCompatibleWith(target);
}
ErrorResult<bool> testRoundTripCompatibility(const std::string &schema_name, const void *data,
                                             size_t size, const SchemaVersion &version,
                                             SerializationFormat format)
{
    auto &manager = SerializationManager::instance();
    SerializationContext context(format, version);
    auto serialize_result = manager.serialize(schema_name, data, size, context);
    if (serialize_result.isError())
    {
        return ErrorResult<bool>::error(serialize_result.errorCode());
    }
    auto deserialize_result = manager.deserialize(schema_name, serialize_result.value(), context);
    if (deserialize_result.isError())
    {
        return ErrorResult<bool>::error(deserialize_result.errorCode());
    }
    if (deserialize_result.value().size() != size ||
        std::memcmp(deserialize_result.value().data(), data, size) != 0)
    {
        return ErrorResult<bool>::error(ErrorCode::E01_IO_CORRUPT, "Round-trip data mismatch");
    }
    return ErrorResult<bool>::success(true);
}
ErrorResult<std::vector<uint8_t>> migrateSchema(const std::vector<uint8_t> &data,
                                                const SchemaVersion &from_version,
                                                const SchemaVersion &to_version,
                                                const std::string &schema_name)
{
    if (from_version.major != to_version.major)
    {
        return ErrorResult<std::vector<uint8_t>>::error(
            serializationErrorToErrorCode(
                errors::SerializationErrorCode::VERSION_CONVERSION_FAILED),
            "Schema migration across major versions is unsupported for " + schema_name);
    }

    if (from_version == to_version)
    {
        return ErrorResult<std::vector<uint8_t>>::success(std::vector<uint8_t>(data));
    }

    // Current migration path preserves payload bytes and relies on
    // versioned metadata converters in serializers.
    std::vector<uint8_t> migrated = data;
    return ErrorResult<std::vector<uint8_t>>::success(std::move(migrated));
}

} // namespace nerve::serialization
