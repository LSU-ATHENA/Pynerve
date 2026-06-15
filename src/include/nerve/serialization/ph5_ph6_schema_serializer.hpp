
#pragma once
#include "nerve/serialization/serialization_manager.hpp"
#include "nerve/summary/compact_summary.hpp"

#include <unordered_map>
#include <vector>

namespace nerve::serialization
{
struct PH5PH6ArtifactMetadata
{
    SchemaVersion schema_version = SchemaVersion(1, 1, 0);
    std::string artifact_type;
    std::string algorithm_variant;
    bool has_highdim_extension = false;
    uint8_t max_supported_dimension = 5;
    std::vector<std::string> extension_fields;
    SchemaVersion min_compatible_version = SchemaVersion(1, 0, 0);
    std::vector<std::string> deprecated_fields;
    std::vector<std::string> new_fields;
    std::vector<uint8_t> serializeMetadata() const;
    bool deserializeMetadata(const std::vector<uint8_t> &data);
    std::string toString() const;
};
class PH5PH6SchemaSerializer : public Serializer
{
public:
    PH5PH6SchemaSerializer();
    ~PH5PH6SchemaSerializer() = default;
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
    ErrorResult<std::vector<uint8_t>>
    serializePh5Artifact(const void *data, size_t size,
                         const PH5PH6ArtifactMetadata &artifact_metadata,
                         const SerializationContext &context);
    ErrorResult<std::vector<uint8_t>>
    serializePh6Artifact(const void *data, size_t size,
                         const PH5PH6ArtifactMetadata &artifact_metadata,
                         const SerializationContext &context);
    ErrorResult<std::vector<uint8_t>>
    serializeCompactSummary(const nerve::summary::CompactSummary &summary,
                            const PH5PH6ArtifactMetadata &artifact_metadata,
                            const SerializationContext &context);
    ErrorResult<std::pair<nerve::summary::CompactSummary, PH5PH6ArtifactMetadata>>
    deserializeCompactSummary(const std::vector<uint8_t> &data,
                              const SerializationContext &context);
    std::string getFormatName() const override { return "PH5PH6SchemaSerializer"; }
    SerializationFormat getFormat() const override { return SerializationFormat::FLATBUFFERS; }
    bool isVersionCompatible(const SchemaVersion &version) const;
    SchemaVersion getMinCompatibleVersion() const { return SchemaVersion(1, 0, 0); }
    SchemaVersion getSchemaVersion() const { return SchemaVersion(1, 1, 0); }
    static constexpr uint32_t PH5PH6_MAGIC_V1 = 0x50483531;
    static constexpr uint32_t PH5PH6_MAGIC_EXTENDED = 0x50483532;
    static constexpr uint8_t HEADER_SIZE_V1 = 16;
    static constexpr uint8_t MIN_HEADER_SIZE_EXTENDED = 32;
    static constexpr uint8_t HEADER_SIZE_EXTENDED = 128;

private:
    ErrorResult<std::vector<uint8_t>>
    serializeArtifactWithMetadata(const void *data, size_t size,
                                  const PH5PH6ArtifactMetadata &metadata) const;
    std::vector<uint8_t> serializeHeader(const PH5PH6ArtifactMetadata &metadata) const;
    std::vector<uint8_t> serializePayload(const void *data, size_t size) const;
    std::vector<uint8_t> serializeExtendedPayload(const void *data, size_t size) const;
    PH5PH6ArtifactMetadata deserializeHeader(const std::vector<uint8_t> &data,
                                             size_t &offset) const;
    ErrorResult<std::vector<uint8_t>> deserializePayload(const std::vector<uint8_t> &data,
                                                         size_t offset) const;
    ErrorResult<std::vector<uint8_t>> deserializeExtendedPayload(const std::vector<uint8_t> &data,
                                                                 size_t offset) const;
    bool validateHeader(const PH5PH6ArtifactMetadata &metadata) const;
    bool validatePayload(const void *data, size_t size,
                         const PH5PH6ArtifactMetadata &metadata) const;
};
class PH5PH6SchemaRegistry
{
public:
    PH5PH6SchemaRegistry();
    ~PH5PH6SchemaRegistry() = default;
    void registerArtifactType(const std::string &artifact_type,
                              const PH5PH6ArtifactMetadata &metadata);
    void unregisterArtifactType(const std::string &artifact_type);
    VersionNegotiationResult negotiateVersion(const std::string &artifact_type,
                                              const SchemaVersion &requested_version) const;
    PH5PH6ArtifactMetadata getArtifactMetadata(const std::string &artifact_type) const;
    std::vector<std::string> getRegisteredArtifactTypes() const;
    bool isArtifactSupported(const std::string &artifact_type, const SchemaVersion &version) const;
    std::unique_ptr<PH5PH6SchemaSerializer> createSerializer() const;
    std::unique_ptr<Serializer> createCompatibleSerializer() const;

private:
    std::unordered_map<std::string, PH5PH6ArtifactMetadata> artifact_types_;
    mutable std::mutex mutex_;
    PH5PH6ArtifactMetadata createDefaultPh5Metadata() const;
    PH5PH6ArtifactMetadata createDefaultPh6Metadata() const;
    PH5PH6ArtifactMetadata createDefaultCompactSummaryMetadata() const;
};
class PH5PH6SchemaMigrator
{
public:
    struct MigrationResult
    {
        bool success = false;
        std::string error_message;
        std::vector<uint8_t> migrated_data;
        PH5PH6ArtifactMetadata new_metadata;
    };
    static MigrationResult migrateToExtended(const std::vector<uint8_t> &data,
                                             const SerializationContext &context);
    static MigrationResult migrateWithExtension(const std::vector<uint8_t> &data,
                                                const PH5PH6ArtifactMetadata &extension_metadata,
                                                const SerializationContext &context);
    static bool validateData(const std::vector<uint8_t> &data);
    static bool validateExtendedData(const std::vector<uint8_t> &data,
                                     const PH5PH6ArtifactMetadata &expected_metadata);

private:
    static std::vector<uint8_t> addHighdimExtension(const std::vector<uint8_t> &data,
                                                    const PH5PH6ArtifactMetadata &metadata,
                                                    const std::vector<uint8_t> &extension_payload);
    static std::vector<uint8_t> updateExtendedHeader(const std::vector<uint8_t> &data,
                                                     const PH5PH6ArtifactMetadata &new_metadata);
};
} // namespace nerve::serialization
