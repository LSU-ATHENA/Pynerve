
#include "nerve/serialization/serialization_manager.hpp"
#include "serialization_wire.hpp"

#include <cstdint>
#include <cstring>
#include <sstream>

namespace nerve::serialization
{
FlatBuffersSerializer::FlatBuffersSerializer() = default;
ErrorResult<std::vector<uint8_t>>
FlatBuffersSerializer::serialize(const void *data, size_t size, const SerializationContext &context)
{
    if (!detail::fitsUint32(size))
    {
        return ErrorResult<std::vector<uint8_t>>::error(ErrorCode::E41_RESOURCE_LIMIT,
                                                        "Invalid FlatBuffers data: too large");
    }
    std::vector<uint8_t> result;
    const uint32_t magic = 0x46424C31;
    detail::appendUint32LittleEndian(result, magic);
    const uint32_t version_data = (context.schemaVersion.major << 24) |
                                  (context.schemaVersion.minor << 16) |
                                  (context.schemaVersion.patch << 8);
    detail::appendUint32LittleEndian(result, version_data);
    const uint32_t data_size = static_cast<uint32_t>(size);
    detail::appendUint32LittleEndian(result, data_size);
    if (size > 0)
    {
        if (data == nullptr)
        {
            return ErrorResult<std::vector<uint8_t>>::error(
                ErrorCode::E41_RESOURCE_LIMIT, "Invalid FlatBuffers data: null payload");
        }
        const auto *bytes = static_cast<const uint8_t *>(data);
        result.insert(result.end(), bytes, bytes + size);
    }
    return ErrorResult<std::vector<uint8_t>>::success(std::move(result));
}
ErrorResult<std::vector<uint8_t>>
FlatBuffersSerializer::serializeWithMetadata(const void *data, size_t size,
                                             const SchemaMetadata &schema_metadata,
                                             const SerializationContext &context)
{
    auto metadataData = packMetadata(schema_metadata);
    if (!detail::fitsUint32(metadataData.size()))
    {
        return ErrorResult<std::vector<uint8_t>>::error(ErrorCode::E41_RESOURCE_LIMIT,
                                                        "Invalid FlatBuffers metadata: too large");
    }
    auto basic_result = serialize(data, size, context);
    if (basic_result.isError())
    {
        return ErrorResult<std::vector<uint8_t>>::error(basic_result.errorCode());
    }
    std::vector<uint8_t> result;
    const uint32_t metadata_size = static_cast<uint32_t>(metadataData.size());
    detail::appendUint32LittleEndian(result, metadata_size);
    result.insert(result.end(), metadataData.begin(), metadataData.end());
    result.insert(result.end(), basic_result.value().begin(), basic_result.value().end());
    return ErrorResult<std::vector<uint8_t>>::success(std::move(result));
}
ErrorResult<std::vector<uint8_t>>
FlatBuffersSerializer::deserialize(const std::vector<uint8_t> &data,
                                   const SerializationContext &context)
{
    if (data.size() < detail::kUint32WireSize * 3)
    {
        return ErrorResult<std::vector<uint8_t>>::error(ErrorCode::E41_RESOURCE_LIMIT,
                                                        "Invalid FlatBuffers data: too small");
    }
    const uint8_t *ptr = data.data();
    const uint8_t *end = data.data() + data.size();
    uint32_t magic = 0;
    if (!detail::readUint32LittleEndian(ptr, end, magic))
    {
        return ErrorResult<std::vector<uint8_t>>::error(ErrorCode::E41_RESOURCE_LIMIT,
                                                        "Invalid FlatBuffers data: too small");
    }
    if (magic != 0x46424C31)
    {
        return ErrorResult<std::vector<uint8_t>>::error(ErrorCode::E31_SCHEMA_VERSION,
                                                        "Invalid FlatBuffers magic number");
    }
    uint32_t version_data = 0;
    if (!detail::readUint32LittleEndian(ptr, end, version_data))
    {
        return ErrorResult<std::vector<uint8_t>>::error(ErrorCode::E41_RESOURCE_LIMIT,
                                                        "Invalid FlatBuffers data: too small");
    }
    SchemaVersion extractedVersion((version_data >> 24) & 0xFF, (version_data >> 16) & 0xFF,
                                   (version_data >> 8) & 0xFF);
    if (!extractedVersion.isCompatibleWith(context.schemaVersion))
    {
        return ErrorResult<std::vector<uint8_t>>::error(
            serializationErrorToErrorCode(
                errors::SerializationErrorCode::INCOMPATIBLE_SCHEMA_VERSION),
            "Schema version mismatch: expected " + context.schemaVersion.toString() + ", got " +
                extractedVersion.toString());
    }
    uint32_t data_size = 0;
    if (!detail::readUint32LittleEndian(ptr, end, data_size))
    {
        return ErrorResult<std::vector<uint8_t>>::error(ErrorCode::E41_RESOURCE_LIMIT,
                                                        "Invalid FlatBuffers data: too small");
    }
    if (static_cast<size_t>(data_size) > data.size() - detail::kUint32WireSize * 3)
    {
        return ErrorResult<std::vector<uint8_t>>::error(ErrorCode::E41_RESOURCE_LIMIT,
                                                        "Invalid FlatBuffers data: size mismatch");
    }
    std::vector<uint8_t> result(ptr, ptr + data_size);
    return ErrorResult<std::vector<uint8_t>>::success(std::move(result));
}
ErrorResult<std::pair<std::vector<uint8_t>, SchemaMetadata>>
FlatBuffersSerializer::deserializeWithMetadata(const std::vector<uint8_t> &data,
                                               const SerializationContext &context)
{
    if (data.size() < sizeof(uint32_t))
    {
        return ErrorResult<std::pair<std::vector<uint8_t>, SchemaMetadata>>::error(
            ErrorCode::E41_RESOURCE_LIMIT, "Invalid data: too small for metadata");
    }
    const uint8_t *ptr = data.data();
    const uint8_t *end = data.data() + data.size();
    uint32_t metadata_size = 0;
    if (!detail::readUint32LittleEndian(ptr, end, metadata_size))
    {
        return ErrorResult<std::pair<std::vector<uint8_t>, SchemaMetadata>>::error(
            ErrorCode::E41_RESOURCE_LIMIT, "Invalid data: too small for metadata");
    }
    if (static_cast<size_t>(metadata_size) > data.size() - detail::kUint32WireSize)
    {
        return ErrorResult<std::pair<std::vector<uint8_t>, SchemaMetadata>>::error(
            ErrorCode::E41_RESOURCE_LIMIT, "Invalid data: metadata size mismatch");
    }
    std::vector<uint8_t> metadataData(ptr, ptr + metadata_size);
    SchemaMetadata metadata = unpackMetadata(metadataData);
    ptr += metadata_size;
    std::vector<uint8_t> remaining_data(ptr, data.data() + data.size());
    auto deserialization_result = deserialize(remaining_data, context);
    if (deserialization_result.isError())
    {
        return ErrorResult<std::pair<std::vector<uint8_t>, SchemaMetadata>>::error(
            deserialization_result.errorCode());
    }
    return ErrorResult<std::pair<std::vector<uint8_t>, SchemaMetadata>>::success(
        std::make_pair(deserialization_result.value(), metadata));
}
std::vector<uint8_t> FlatBuffersSerializer::packMetadata(const SchemaMetadata &metadata) const
{
    std::string json = "{";
    json += "\"schema_name\":\"" + detail::escapeJsonString(metadata.schema_name) + "\",";
    json += "\"version\":\"" + detail::escapeJsonString(metadata.version.toString()) + "\",";
    json += "\"min_compatible\":\"" +
            detail::escapeJsonString(metadata.minCompatibleVersion.toString()) + "\",";
    json += "\"max_compatible\":\"" +
            detail::escapeJsonString(metadata.maxCompatibleVersion.toString()) + "\",";
    json += "\"description\":\"" + detail::escapeJsonString(metadata.description) + "\"";
    json += "}";
    return std::vector<uint8_t>(json.begin(), json.end());
}
SchemaMetadata FlatBuffersSerializer::unpackMetadata(const std::vector<uint8_t> &metadataData) const
{
    SchemaMetadata metadata;
    std::string json(metadataData.begin(), metadataData.end());
    std::string field;
    if (detail::readJsonStringField(json, "schema_name", field))
    {
        metadata.schema_name = field;
    }
    if (detail::readJsonStringField(json, "version", field))
    {
        metadata.version = SchemaVersion::fromString(field);
    }
    if (detail::readJsonStringField(json, "min_compatible", field))
    {
        metadata.minCompatibleVersion = SchemaVersion::fromString(field);
    }
    if (detail::readJsonStringField(json, "max_compatible", field))
    {
        metadata.maxCompatibleVersion = SchemaVersion::fromString(field);
    }
    if (detail::readJsonStringField(json, "description", field))
    {
        metadata.description = field;
    }
    return metadata;
}

} // namespace nerve::serialization
