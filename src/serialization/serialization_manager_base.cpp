
#include "nerve/io/mmap_io.hpp"
#include "nerve/serialization/serialization_manager.hpp"

#include <algorithm>
#include <fstream>

namespace nerve::serialization
{

namespace
{

class BinaryPassthroughSerializer : public Serializer
{
public:
    BinaryPassthroughSerializer() = default;
    ErrorResult<std::vector<uint8_t>> serialize(const void *data, size_t size,
                                                const SerializationContext &) override
    {
        const auto *bytes = static_cast<const uint8_t *>(data);
        return ErrorResult<std::vector<uint8_t>>::success(
            std::vector<uint8_t>(bytes, bytes + size));
    }
    ErrorResult<std::vector<uint8_t>>
    serializeWithMetadata(const void *data, size_t size, const SchemaMetadata &,
                          const SerializationContext &context) override
    {
        return serialize(data, size, context);
    }
    ErrorResult<std::vector<uint8_t>> deserialize(const std::vector<uint8_t> &data,
                                                  const SerializationContext &) override
    {
        return ErrorResult<std::vector<uint8_t>>::success(data);
    }
    ErrorResult<std::pair<std::vector<uint8_t>, SchemaMetadata>>
    deserializeWithMetadata(const std::vector<uint8_t> &data,
                            const SerializationContext &context) override
    {
        auto result = deserialize(data, context);
        if (result.isError())
        {
            return ErrorResult<std::pair<std::vector<uint8_t>, SchemaMetadata>>::error(
                result.errorCode());
        }
        SchemaMetadata default_meta;
        return ErrorResult<std::pair<std::vector<uint8_t>, SchemaMetadata>>::success(
            {result.moveValue(), std::move(default_meta)});
    }
    std::string getFormatName() const override { return "BinaryPassthrough"; }
    SerializationFormat getFormat() const override { return SerializationFormat::BINARY; }
};

} // namespace

SerializationManager::SerializationManager()
{
    registerSerializer(std::make_unique<BinaryPassthroughSerializer>());
    registerSerializer(std::make_unique<FlatBuffersSerializer>());
}

SerializationManager &SerializationManager::instance()
{
    static SerializationManager instance;
    return instance;
}
void SerializationManager::registerSerializer(std::unique_ptr<Serializer> serializer)
{
    std::lock_guard<std::mutex> lock(mutex_);
    serializers_[serializer->getFormat()] = std::move(serializer);
}
void SerializationManager::unregisterSerializer(SerializationFormat format)
{
    std::lock_guard<std::mutex> lock(mutex_);
    serializers_.erase(format);
}
ErrorResult<std::vector<uint8_t>>
SerializationManager::serialize(const std::string &schema_name, const void *data, size_t size,
                                const SerializationContext &context)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = serializers_.find(context.format);
    if (it == serializers_.end())
    {
        return ErrorResult<std::vector<uint8_t>>::error(
            serializationErrorToErrorCode(
                errors::SerializationErrorCode::UNSUPPORTED_SERIALIZATION_FORMAT),
            "Unsupported serialization format");
    }
    SchemaMetadata metadata = version_negotiator_.getSchemaMetadata(schema_name);
    if (metadata.schema_name.empty())
    {
        return ErrorResult<std::vector<uint8_t>>::error(
            serializationErrorToErrorCode(errors::SerializationErrorCode::SCHEMA_NOT_FOUND),
            "Schema not found: " + schema_name);
    }
    if (!metadata.isVersionCompatible(context.schemaVersion))
    {
        return ErrorResult<std::vector<uint8_t>>::error(
            serializationErrorToErrorCode(
                errors::SerializationErrorCode::INCOMPATIBLE_SCHEMA_VERSION),
            "Schema version not compatible: " + schema_name);
    }
    return it->second->serialize(data, size, context);
}
ErrorResult<std::vector<uint8_t>>
SerializationManager::serializeWithMetadata(const std::string &schema_name, const void *data,
                                            size_t size, const SchemaVersion &schema_version,
                                            const SerializationContext &context)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = serializers_.find(context.format);
    if (it == serializers_.end())
    {
        return ErrorResult<std::vector<uint8_t>>::error(
            serializationErrorToErrorCode(
                errors::SerializationErrorCode::UNSUPPORTED_SERIALIZATION_FORMAT),
            "Unsupported serialization format");
    }
    auto metadata = version_negotiator_.getSchemaMetadata(schema_name);
    if (metadata.schema_name.empty())
    {
        return ErrorResult<std::vector<uint8_t>>::error(
            serializationErrorToErrorCode(errors::SerializationErrorCode::SCHEMA_NOT_FOUND),
            "Schema not found: " + schema_name);
    }
    metadata.version = schema_version;
    return it->second->serializeWithMetadata(data, size, metadata, context);
}
ErrorResult<std::vector<uint8_t>>
SerializationManager::deserialize(const std::string &schema_name, const std::vector<uint8_t> &data,
                                  const SerializationContext &context)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = serializers_.find(context.format);
    if (it == serializers_.end())
    {
        return ErrorResult<std::vector<uint8_t>>::error(
            serializationErrorToErrorCode(
                errors::SerializationErrorCode::UNSUPPORTED_SERIALIZATION_FORMAT),
            "Unsupported serialization format");
    }
    if (!version_negotiator_.isVersionSupported(schema_name, context.schemaVersion))
    {
        return ErrorResult<std::vector<uint8_t>>::error(
            serializationErrorToErrorCode(errors::SerializationErrorCode::SCHEMA_NOT_FOUND),
            "Schema not found or incompatible: " + schema_name);
    }
    return it->second->deserialize(data, context);
}
ErrorResult<std::pair<std::vector<uint8_t>, SchemaMetadata>>
SerializationManager::deserializeWithMetadata(const std::string &schema_name,
                                              const std::vector<uint8_t> &data,
                                              const SerializationContext &context)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = serializers_.find(context.format);
    if (it == serializers_.end())
    {
        return ErrorResult<std::pair<std::vector<uint8_t>, SchemaMetadata>>::error(
            serializationErrorToErrorCode(
                errors::SerializationErrorCode::UNSUPPORTED_SERIALIZATION_FORMAT),
            "Unsupported serialization format");
    }
    auto result = it->second->deserializeWithMetadata(data, context);
    if (result.isError())
    {
        return result;
    }
    if (result.value().second.schema_name != schema_name ||
        !version_negotiator_.isVersionSupported(schema_name, result.value().second.version))
    {
        return ErrorResult<std::pair<std::vector<uint8_t>, SchemaMetadata>>::error(
            serializationErrorToErrorCode(errors::SerializationErrorCode::SCHEMA_NOT_FOUND),
            "Serialized schema does not match requested schema: " + schema_name);
    }
    return result;
}
VersionNegotiationResult
SerializationManager::negotiateVersion(const std::string &schema_name,
                                       const SchemaVersion &requested_version,
                                       SerializationFormat format)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = serializers_.find(format);
    if (it == serializers_.end())
    {
        return VersionNegotiationResult::errorResult("Unsupported serialization format");
    }
    return version_negotiator_.negotiateVersion(schema_name, requested_version);
}
void SerializationManager::registerSchema(const SchemaMetadata &metadata)
{
    std::lock_guard<std::mutex> lock(mutex_);
    version_negotiator_.registerSchema(metadata);
}
SchemaMetadata SerializationManager::getSchemaMetadata(const std::string &schema_name) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return version_negotiator_.getSchemaMetadata(schema_name);
}
std::vector<SerializationFormat> SerializationManager::getSupportedFormats() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<SerializationFormat> formats;
    formats.reserve(serializers_.size());
    for (const auto &[format, serializer] : serializers_)
    {
        formats.push_back(format);
    }
    return formats;
}
bool SerializationManager::isFormatSupported(SerializationFormat format) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return serializers_.find(format) != serializers_.end();
}

ErrorResult<std::vector<uint8_t>> SerializationManager::loadFromFile(const std::string &path,
                                                                     const std::string &schema_name,
                                                                     SerializationFormat format)
{
    auto file = io::mmapReadFile(path);
    if (!file.valid())
    {
        return ErrorResult<std::vector<uint8_t>>::error(
            serializationErrorToErrorCode(
                errors::SerializationErrorCode::INVALID_SERIALIZATION_DATA),
            "Cannot read file: " + path);
    }
    std::vector<uint8_t> data(file.bytes(), file.bytes() + file.size);
    return deserialize(schema_name, data, SerializationContext(format));
}

ErrorResult<bool> SerializationManager::saveToFile(const std::string &path,
                                                   const std::string &schema_name, const void *data,
                                                   size_t size, SerializationFormat format)
{
    auto result = serialize(schema_name, data, size, SerializationContext(format));
    if (!result.isSuccess())
    {
        return ErrorResult<bool>::error(result.error().code, result.error().message);
    }
    auto &serialized = result.value();
    auto file = io::mmapWriteFile(path, serialized.size());
    if (!file.valid())
    {
        std::ofstream ofs(path, std::ios::binary);
        if (!ofs)
        {
            return ErrorResult<bool>::error(
                serializationErrorToErrorCode(
                    errors::SerializationErrorCode::INVALID_SERIALIZATION_DATA),
                "Cannot write file: " + path);
        }
        ofs.write(reinterpret_cast<const char *>(serialized.data()),
                  static_cast<std::streamsize>(serialized.size()));
        return ErrorResult<bool>::success(true);
    }
    std::memcpy(file.mutableBytes(), serialized.data(), serialized.size());
    return ErrorResult<bool>::success(true);
}

} // namespace nerve::serialization
