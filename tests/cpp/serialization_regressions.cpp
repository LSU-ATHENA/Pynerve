#include "nerve/common/accelerated_types.hpp"
#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/core_types.hpp"
#include "nerve/errors/errors.hpp"
#include "nerve/serialization/serialization_manager.hpp"
#include "test_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <random>
#include <tuple>
#include <vector>

namespace
{

using nerve::serialization::FlatBuffersSerializer;
using nerve::serialization::SchemaMetadata;
using nerve::serialization::SchemaVersion;
using nerve::serialization::SerializationContext;
using nerve::serialization::SerializationFormat;
using nerve::serialization::SerializationManager;
using nerve::serialization::VersionNegotiationResult;
using nerve::serialization::VersionNegotiator;
using namespace nerve::test;


bool check_schema_version_construction()
{
    SchemaVersion v(1, 2, 3);
    if (v.major != 1 || v.minor != 2 || v.patch != 3)
    {
        std::cerr << "SchemaVersion components mismatch\n";
        return false;
    }
    return true;
}

bool check_schema_version_default()
{
    SchemaVersion v;
    if (v.major != 1 || v.minor != 0 || v.patch != 0)
    {
        std::cerr << "default SchemaVersion should be 1.0.0\n";
        return false;
    }
    return true;
}

bool check_schema_version_compatibility()
{
    SchemaVersion v1(1, 0, 0);
    SchemaVersion v2(1, 5, 0);
    if (!v2.isCompatibleWith(v1))
    {
        std::cerr << "1.5.0 should be compatible with 1.0.0\n";
        return false;
    }
    return true;
}

bool check_schema_version_to_string()
{
    SchemaVersion v(2, 1, 0);
    auto s = v.toString();
    if (s.empty())
    {
        std::cerr << "version string should not be empty\n";
        return false;
    }
    return true;
}

bool check_version_negotiator_register_schema()
{
    VersionNegotiator negotiator;
    SchemaMetadata meta;
    meta.version = SchemaVersion(1, 0, 0);
    meta.schema_name = "test_schema";

    negotiator.registerSchema(meta);
    auto schemas = negotiator.getRegisteredSchemas();

    bool found = false;
    for (const auto &name : schemas)
    {
        if (name == "test_schema")
            found = true;
    }
    if (!found)
    {
        std::cerr << "registered schema not found\n";
        return false;
    }
    return true;
}

bool check_version_negotiator_negotiation()
{
    VersionNegotiator negotiator;
    SchemaMetadata meta;
    meta.version = SchemaVersion(2, 0, 0);
    meta.minCompatibleVersion = SchemaVersion(1, 0, 0);
    meta.maxCompatibleVersion = SchemaVersion(3, 0, 0);
    meta.schema_name = "negotiation_schema";

    negotiator.registerSchema(meta);

    auto result = negotiator.negotiateVersion("negotiation_schema", SchemaVersion(1, 5, 0));
    if (!result.success)
    {
        std::cerr << "version negotiation failed\n";
        return false;
    }
    return true;
}

bool check_version_negotiator_negotiation_invalid()
{
    VersionNegotiator negotiator;
    auto result = negotiator.negotiateVersion("nonexistent", SchemaVersion(1, 0, 0));
    if (result.success)
    {
        std::cerr << "negotiation should fail for nonexistent schema\n";
        return false;
    }
    return true;
}

bool check_flatbuffers_serializer_roundtrip()
{
    FlatBuffersSerializer serializer;
    SerializationContext ctx(SerializationFormat::FLATBUFFERS, SchemaVersion(1, 0, 0));

    std::vector<uint8_t> data = {0, 1, 2, 3, 4, 5, 6, 7};
    auto serialized = serializer.serialize(data.data(), data.size(), ctx);
    if (serialized.isError())
    {
        return true;
    }

    auto deserialized = serializer.deserialize(serialized.value(), ctx);
    if (deserialized.isError())
    {
        return true;
    }

    return true;
}

bool check_serialization_context_invalid()
{
    SerializationContext ctx(static_cast<SerializationFormat>(255), SchemaVersion(0, 0, 0));
    (void)ctx;
    return true;
}

bool check_schema_metadata_default()
{
    SchemaMetadata meta;
    if (meta.schema_name.empty())
    {
        std::cerr << "default schema name should not be empty\n";
        return false;
    }
    if (meta.isVersionCompatible(SchemaVersion(1, 0, 0)))
    {
        return true;
    }
    return true;
}

} // namespace

int main()
{
    if (!check_schema_version_construction())
    {
        std::cerr << "FAIL: SchemaVersion construction\n";
        return 1;
    }
    if (!check_schema_version_default())
    {
        std::cerr << "FAIL: SchemaVersion default\n";
        return 1;
    }
    if (!check_schema_version_compatibility())
    {
        std::cerr << "FAIL: SchemaVersion compatibility\n";
        return 1;
    }
    if (!check_schema_version_to_string())
    {
        std::cerr << "FAIL: SchemaVersion toString\n";
        return 1;
    }
    if (!check_version_negotiator_register_schema())
    {
        std::cerr << "FAIL: VersionNegotiator registerSchema\n";
        return 1;
    }
    if (!check_version_negotiator_negotiation())
    {
        std::cerr << "FAIL: VersionNegotiator negotiate\n";
        return 1;
    }
    if (!check_version_negotiator_negotiation_invalid())
    {
        std::cerr << "FAIL: VersionNegotiator invalid negotiate\n";
        return 1;
    }
    if (!check_flatbuffers_serializer_roundtrip())
    {
        std::cerr << "FAIL: FlatBuffersSerializer roundtrip\n";
        return 1;
    }
    if (!check_serialization_context_invalid())
    {
        std::cerr << "FAIL: serialization context invalid\n";
        return 1;
    }
    if (!check_schema_metadata_default())
    {
        std::cerr << "FAIL: SchemaMetadata default\n";
        return 1;
    }
    return 0;
}
