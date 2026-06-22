
#pragma once
#include "nerve/errors/errors.hpp"

#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace nerve::serialization
{
using nerve::errors::ErrorCode;
template <typename T>
using ErrorResult = nerve::errors::ErrorResult<T>;

struct SchemaVersion
{
    uint32_t major;
    uint32_t minor;
    uint32_t patch;
    SchemaVersion(uint32_t major = 1, uint32_t minor = 0, uint32_t patch = 0)
        : major(major)
        , minor(minor)
        , patch(patch)
    {}
    std::string toString() const;
    bool isCompatibleWith(const SchemaVersion &other) const;
    bool operator==(const SchemaVersion &other) const;
    bool operator<(const SchemaVersion &other) const;
    bool operator<=(const SchemaVersion &other) const;
    bool operator>=(const SchemaVersion &other) const;
    static SchemaVersion fromString(const std::string &version_str);
};
struct SchemaMetadata
{
    SchemaVersion version;
    SchemaVersion minCompatibleVersion;
    SchemaVersion maxCompatibleVersion;
    std::string schema_name;
    std::string description;
    std::chrono::system_clock::time_point createdAt;
    std::unordered_map<std::string, std::string> custom_fields;
    SchemaMetadata()
        : version(1, 0, 0)
        , minCompatibleVersion(1, 0, 0)
        , maxCompatibleVersion(1, 0, 0)
        , schema_name("default")
        , createdAt(std::chrono::system_clock::now())
    {}
    bool isVersionCompatible(const SchemaVersion &check_version) const;
    std::string toString() const;
    std::unordered_map<std::string, std::string> toMap() const;
};
enum class SerializationFormat : uint8_t
{
    FLATBUFFERS = 0,
    ARROW = 1,
    JSON = 2,
    BINARY = 3,
    PROTOBUF = 4
};
struct SerializationContext
{
    SerializationFormat format;
    SchemaVersion schemaVersion;
    std::unordered_map<std::string, std::string> options;
    SerializationContext(SerializationFormat format = SerializationFormat::FLATBUFFERS,
                         SchemaVersion version = SchemaVersion(1, 0, 0))
        : format(format)
        , schemaVersion(version)
    {}
};
struct VersionNegotiationResult
{
    bool success;
    SchemaVersion negotiated_version;
    std::string error_message;
    bool requiresConversion;
    std::string conversion_strategy;
    VersionNegotiationResult()
        : success(false)
        , requiresConversion(false)
    {}
    static VersionNegotiationResult successResult(const SchemaVersion &version,
                                                  bool needs_conversion = false,
                                                  const std::string &strategy = "");
    static VersionNegotiationResult errorResult(const std::string &error);
};
class VersionNegotiator
{
public:
    VersionNegotiator();
    ~VersionNegotiator() = default;
    void registerSchema(const SchemaMetadata &metadata);
    void unregisterSchema(const std::string &schema_name);
    VersionNegotiationResult negotiateVersion(const std::string &schema_name,
                                              const SchemaVersion &requested_version,
                                              const std::vector<SchemaVersion> &supported_versions);
    VersionNegotiationResult negotiateVersion(const std::string &schema_name,
                                              const SchemaVersion &requested_version);
    SchemaMetadata getSchemaMetadata(const std::string &schema_name) const;
    std::vector<std::string> getRegisteredSchemas() const;
    std::vector<SchemaVersion> getSupportedVersions(const std::string &schema_name) const;
    bool isVersionSupported(const std::string &schema_name, const SchemaVersion &version) const;
    bool areVersionsCompatible(const std::string &schema_name, const SchemaVersion &version1,
                               const SchemaVersion &version2) const;

private:
    std::unordered_map<std::string, SchemaMetadata> schemas_;
    mutable std::mutex mutex_;
};
class Serializer
{
public:
    virtual ~Serializer() = default;
    virtual ErrorResult<std::vector<uint8_t>> serialize(const void *data, size_t size,
                                                        const SerializationContext &context) = 0;
    virtual ErrorResult<std::vector<uint8_t>>
    serializeWithMetadata(const void *data, size_t size, const SchemaMetadata &schema_metadata,
                          const SerializationContext &context) = 0;
    virtual ErrorResult<std::vector<uint8_t>> deserialize(const std::vector<uint8_t> &data,
                                                          const SerializationContext &context) = 0;
    virtual ErrorResult<std::pair<std::vector<uint8_t>, SchemaMetadata>>
    deserializeWithMetadata(const std::vector<uint8_t> &data,
                            const SerializationContext &context) = 0;
    virtual std::string getFormatName() const = 0;
    virtual SerializationFormat getFormat() const = 0;
};
class FlatBuffersSerializer : public Serializer
{
public:
    FlatBuffersSerializer();
    ~FlatBuffersSerializer() = default;
    ErrorResult<std::vector<uint8_t>> serialize(const void *data, size_t size,
                                                const SerializationContext &context) override;
    ErrorResult<std::vector<uint8_t>>
    serializeWithMetadata(const void *data, size_t size, const SchemaMetadata &schema_metadata,
                          const SerializationContext &context) override;
    ErrorResult<std::vector<uint8_t>> deserialize(const std::vector<uint8_t> &data,
                                                  const SerializationContext &context) override;
    ErrorResult<std::pair<std::vector<uint8_t>, SchemaMetadata>>
    deserializeWithMetadata(const std::vector<uint8_t> &data,
                            const SerializationContext &context) override;
    std::string getFormatName() const override { return "FlatBuffers"; }
    SerializationFormat getFormat() const override { return SerializationFormat::FLATBUFFERS; }

private:
    std::vector<uint8_t> packMetadata(const SchemaMetadata &metadata) const;
    SchemaMetadata unpackMetadata(const std::vector<uint8_t> &metadata_data) const;
};
class ArrowSerializer : public Serializer
{
public:
    ArrowSerializer();
    ~ArrowSerializer() = default;
    ErrorResult<std::vector<uint8_t>> serialize(const void *data, size_t size,
                                                const SerializationContext &context) override;
    ErrorResult<std::vector<uint8_t>>
    serializeWithMetadata(const void *data, size_t size, const SchemaMetadata &schema_metadata,
                          const SerializationContext &context) override;
    ErrorResult<std::vector<uint8_t>> deserialize(const std::vector<uint8_t> &data,
                                                  const SerializationContext &context) override;
    ErrorResult<std::pair<std::vector<uint8_t>, SchemaMetadata>>
    deserializeWithMetadata(const std::vector<uint8_t> &data,
                            const SerializationContext &context) override;
    std::string getFormatName() const override { return "Apache Arrow"; }
    SerializationFormat getFormat() const override { return SerializationFormat::ARROW; }

private:
    std::vector<uint8_t> packMetadata(const SchemaMetadata &metadata) const;
    SchemaMetadata unpackMetadata(const std::vector<uint8_t> &metadata_data) const;
};
class SerializationManager
{
public:
    static SerializationManager &instance();
    void registerSerializer(std::unique_ptr<Serializer> serializer);
    void unregisterSerializer(SerializationFormat format);
    ErrorResult<std::vector<uint8_t>> serialize(const std::string &schema_name, const void *data,
                                                size_t size, const SerializationContext &context);
    ErrorResult<std::vector<uint8_t>> serializeWithMetadata(const std::string &schema_name,
                                                            const void *data, size_t size,
                                                            const SchemaVersion &schemaVersion,
                                                            const SerializationContext &context);
    ErrorResult<std::vector<uint8_t>> deserialize(const std::string &schema_name,
                                                  const std::vector<uint8_t> &data,
                                                  const SerializationContext &context);
    ErrorResult<std::pair<std::vector<uint8_t>, SchemaMetadata>>
    deserializeWithMetadata(const std::string &schema_name, const std::vector<uint8_t> &data,
                            const SerializationContext &context);
    ErrorResult<std::vector<uint8_t>> loadFromFile(const std::string &path,
                                                   const std::string &schema_name,
                                                   SerializationFormat format);
    ErrorResult<bool> saveToFile(const std::string &path, const std::string &schema_name,
                                 const void *data, size_t size, SerializationFormat format);
    VersionNegotiationResult negotiateVersion(const std::string &schema_name,
                                              const SchemaVersion &requested_version,
                                              SerializationFormat format);
    void registerSchema(const SchemaMetadata &metadata);
    SchemaMetadata getSchemaMetadata(const std::string &schema_name) const;
    std::vector<SerializationFormat> getSupportedFormats() const;
    bool isFormatSupported(SerializationFormat format) const;

private:
    SerializationManager();
    std::unordered_map<SerializationFormat, std::unique_ptr<Serializer>> serializers_;
    VersionNegotiator version_negotiator_;
    mutable std::mutex mutex_;
};
namespace errors
{
enum class SerializationErrorCode : uint32_t
{
    SUCCESS = 0x00000000,
    INCOMPATIBLE_SCHEMA_VERSION = 0x00010000,
    UNSUPPORTED_SCHEMA_VERSION = 0x00010001,
    SCHEMA_VERSION_NEGOTIATION_FAILED = 0x00010002,
    SCHEMA_NOT_FOUND = 0x00010003,
    UNSUPPORTED_SERIALIZATION_FORMAT = 0x00020000,
    SERIALIZATION_FORMAT_MISMATCH = 0x00020001,
    INVALID_SERIALIZATION_DATA = 0x00020002,
    SERIALIZATION_DATA_CORRUPTED = 0x00030000,
    DESERIALIZATION_FAILED = 0x00030001,
    BUFFER_TOO_SMALL = 0x00030002,
    METADATA_MISSING = 0x00040000,
    METADATA_CORRUPTED = 0x00040001,
    METADATA_VERSION_MISMATCH = 0x00040002,
    VERSION_CONVERSION_FAILED = 0x00050000,
    CONVERSION_NOT_SUPPORTED = 0x00050001,
    CONVERSION_DATA_LOSS = 0x00050002
};
}
ErrorCode serializationErrorToErrorCode(errors::SerializationErrorCode ser_error);
std::string serialization_format_to_string(SerializationFormat format);
SerializationFormat stringToSerializationFormat(const std::string &format_str);
bool isSchemaVersionValid(const SchemaVersion &version);
bool isVersionCompatible(const SchemaVersion &current, const SchemaVersion &target);
ErrorResult<bool> testRoundTripCompatibility(const std::string &schema_name, const void *data,
                                             size_t size, const SchemaVersion &version,
                                             SerializationFormat format);
ErrorResult<std::vector<uint8_t>> migrateSchema(const std::vector<uint8_t> &data,
                                                const SchemaVersion &from_version,
                                                const SchemaVersion &to_version,
                                                const std::string &schema_name);
#define TOPOLOGIB_SERIALIZE(schema_name, data, size, format)                                       \
    nerve::serialization::SerializationManager::instance().serialize(                              \
        schema_name, data, size, nerve::serialization::SerializationContext(format))
#define TOPOLOGIB_DESERIALIZE(schema_name, data, format)                                           \
    nerve::serialization::SerializationManager::instance().deserialize(                            \
        schema_name, data, nerve::serialization::SerializationContext(format))
#define TOPOLOGIB_SERIALIZE_WITH_VERSION(schema_name, data, size, version, format)                 \
    nerve::serialization::SerializationManager::instance().serializeWithMetadata(                  \
        schema_name, data, size, version, nerve::serialization::SerializationContext(format))
#define TOPOLOGIB_DESERIALIZE_WITH_METADATA(schema_name, data, format)                             \
    nerve::serialization::SerializationManager::instance().deserializeWithMetadata(                \
        schema_name, data, nerve::serialization::SerializationContext(format))
#define NERVE_SERIALIZE(schema_name, data, size, format)                                           \
    nerve::serialization::SerializationManager::instance().serialize(                              \
        schema_name, data, size, nerve::serialization::SerializationContext(format))
#define NERVE_DESERIALIZE(schema_name, data, format)                                               \
    nerve::serialization::SerializationManager::instance().deserialize(                            \
        schema_name, data, nerve::serialization::SerializationContext(format))
#define NERVE_SERIALIZE_WITH_VERSION(schema_name, data, size, version, format)                     \
    nerve::serialization::SerializationManager::instance().serializeWithMetadata(                  \
        schema_name, data, size, version, nerve::serialization::SerializationContext(format))
#define NERVE_DESERIALIZE_WITH_METADATA(schema_name, data, format)                                 \
    nerve::serialization::SerializationManager::instance().deserializeWithMetadata(                \
        schema_name, data, nerve::serialization::SerializationContext(format))
#define NERVE_LOAD_FROM_FILE(path, schema_name, format)                                            \
    nerve::serialization::SerializationManager::instance().loadFromFile(path, schema_name, format)
#define NERVE_SAVE_TO_FILE(path, schema_name, data, size, format)                                  \
    nerve::serialization::SerializationManager::instance().saveToFile(path, schema_name, data,     \
                                                                      size, format)
} // namespace nerve::serialization
