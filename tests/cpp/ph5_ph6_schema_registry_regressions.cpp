#include "nerve/persistence/kernels/ph5_ph6_ops.hpp"
#include "nerve/serialization/ph5_ph6_schema_serializer.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace
{

using nerve::serialization::PH5PH6SchemaRegistry;
using nerve::serialization::PH5PH6ArtifactMetadata;
using nerve::serialization::SchemaVersion;
using nerve::serialization::VersionNegotiationResult;

bool check_schema_registration()
{
    PH5PH6SchemaRegistry registry;
    auto types = registry.getRegisteredArtifactTypes();
    if (types.empty())
    {
        std::cerr << "registry should have registered types\n";
        return false;
    }
    return true;
}

bool check_schema_lookup_by_name()
{
    PH5PH6SchemaRegistry registry;
    PH5PH6ArtifactMetadata meta = registry.getArtifactMetadata("PH5");
    if (meta.artifact_type != "PH5")
    {
        std::cerr << "expected PH5 artifact type\n";
        return false;
    }
    meta = registry.getArtifactMetadata("PH6");
    if (meta.artifact_type != "PH6")
    {
        std::cerr << "expected PH6 artifact type\n";
        return false;
    }
    return true;
}

bool check_schema_version_compatibility()
{
    PH5PH6SchemaRegistry registry;
    SchemaVersion v1_0(1, 0, 0);
    bool supported = registry.isArtifactSupported("PH5", v1_0);
    (void)supported;
    SchemaVersion v1_1(1, 1, 0);
    supported = registry.isArtifactSupported("PH5", v1_1);
    (void)supported;
    VersionNegotiationResult result = registry.negotiateVersion("PH5", v1_1);
    if (!result.success)
    {
        std::cerr << "version negotiation should succeed\n";
        return false;
    }
    return true;
}

bool check_schema_metadata_validation()
{
    PH5PH6ArtifactMetadata meta;
    meta.artifact_type = "PH5";
    meta.schema_version = SchemaVersion(1, 1, 0);
    meta.min_compatible_version = SchemaVersion(1, 0, 0);
    meta.algorithm_variant = "cohomology";
    meta.has_highdim_extension = false;
    meta.max_supported_dimension = 5;
    auto serialized = meta.serializeMetadata();
    if (serialized.empty())
    {
        std::cerr << "serialized metadata should not be empty\n";
        return false;
    }
    PH5PH6ArtifactMetadata deserialized;
    bool ok = deserialized.deserializeMetadata(serialized);
    if (!ok)
    {
        std::cerr << "deserialize metadata should succeed\n";
        return false;
    }
    if (deserialized.artifact_type != meta.artifact_type)
    {
        std::cerr << "artifact type mismatch\n";
        return false;
    }
    return true;
}

bool check_schema_register_custom()
{
    PH5PH6SchemaRegistry registry;
    PH5PH6ArtifactMetadata custom;
    custom.artifact_type = "CustomType";
    custom.schema_version = SchemaVersion(2, 0, 0);
    registry.registerArtifactType("CustomType", custom);
    PH5PH6ArtifactMetadata retrieved = registry.getArtifactMetadata("CustomType");
    if (retrieved.artifact_type != "CustomType")
    {
        std::cerr << "custom type mismatch\n";
        return false;
    }
    registry.unregisterArtifactType("CustomType");
    return true;
}

} // namespace

int main()
{
    if (!check_schema_registration())
    {
        std::cerr << "FAIL: schema_registration\n";
        return 1;
    }
    if (!check_schema_lookup_by_name())
    {
        std::cerr << "FAIL: schema_lookup_by_name\n";
        return 1;
    }
    if (!check_schema_version_compatibility())
    {
        std::cerr << "FAIL: schema_version_compatibility\n";
        return 1;
    }
    if (!check_schema_metadata_validation())
    {
        std::cerr << "FAIL: schema_metadata_validation\n";
        return 1;
    }
    if (!check_schema_register_custom())
    {
        std::cerr << "FAIL: schema_register_custom\n";
        return 1;
    }
    return 0;
}
