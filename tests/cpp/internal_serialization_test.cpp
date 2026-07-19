
#include "nerve/serialization/detail/serialization_detail.hpp"
#include "nerve/serialization/serialization_manager.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace
{

bool check_byte_swapping()
{
    (void)sizeof(nerve::serialization::SchemaVersion);
    return true;
}

bool check_serialization_format_to_string()
{
    std::string fb = nerve::serialization::serialization_format_to_string(
        nerve::serialization::SerializationFormat::FLATBUFFERS);
    if (fb != "FlatBuffers")
    {
        return false;
    }
    std::string arrow = nerve::serialization::serialization_format_to_string(
        nerve::serialization::SerializationFormat::ARROW);
    if (arrow != "Apache Arrow")
    {
        return false;
    }
    return true;
}

bool check_string_to_format()
{
    auto fmt = nerve::serialization::stringToSerializationFormat("arrow");
    if (fmt != nerve::serialization::SerializationFormat::ARROW)
    {
        return false;
    }
    fmt = nerve::serialization::stringToSerializationFormat("FlatBuffers");
    return fmt == nerve::serialization::SerializationFormat::FLATBUFFERS;
}

bool check_schema_version_construction()
{
    nerve::serialization::SchemaVersion v1(1, 2, 3);
    if (v1.major != 1 || v1.minor != 2 || v1.patch != 3)
    {
        return false;
    }
    std::string s = v1.toString();
    return s == "1.2.3";
}

bool check_schema_version_from_string()
{
    auto v = nerve::serialization::SchemaVersion::fromString("2.5.1");
    return v.major == 2 && v.minor == 5 && v.patch == 1;
}

bool check_schema_version_compatibility()
{
    nerve::serialization::SchemaVersion current(2, 3, 0);
    nerve::serialization::SchemaVersion older(2, 1, 0);
    nerve::serialization::SchemaVersion newer(2, 5, 0);
    nerve::serialization::SchemaVersion different_major(3, 0, 0);

    if (!current.isCompatibleWith(older))
    {
        return false;
    }
    if (current.isCompatibleWith(newer))
    {
        return false;
    }
    if (current.isCompatibleWith(different_major))
    {
        return false;
    }
    return true;
}

bool check_schema_version_comparison()
{
    nerve::serialization::SchemaVersion a(1, 0, 0);
    nerve::serialization::SchemaVersion b(1, 2, 0);
    nerve::serialization::SchemaVersion c(1, 2, 3);
    nerve::serialization::SchemaVersion d(1, 2, 3);

    if (!(a < b))
    {
        return false;
    }
    if (!(b < c))
    {
        return false;
    }
    if (!(c == d))
    {
        return false;
    }
    if (c < d)
    {
        return false;
    }
    if (!(c <= d))
    {
        return false;
    }
    if (!(d >= c))
    {
        return false;
    }
    return true;
}

bool check_schema_metadata_version_compatibility()
{
    nerve::serialization::SchemaMetadata meta;
    meta.version = nerve::serialization::SchemaVersion(2, 0, 0);
    meta.minCompatibleVersion = nerve::serialization::SchemaVersion(2, 0, 0);
    meta.maxCompatibleVersion = nerve::serialization::SchemaVersion(2, 5, 0);

    nerve::serialization::SchemaVersion valid(2, 3, 0);
    nerve::serialization::SchemaVersion invalid(2, 6, 0);

    if (!meta.isVersionCompatible(valid))
    {
        return false;
    }
    if (meta.isVersionCompatible(invalid))
    {
        return false;
    }
    return true;
}

bool check_schema_metadata_to_string()
{
    nerve::serialization::SchemaMetadata meta;
    meta.schema_name = "test_schema";
    meta.version = nerve::serialization::SchemaVersion(1, 0, 0);
    std::string s = meta.toString();
    return s.find("test_schema") != std::string::npos;
}

bool check_schema_metadata_to_map()
{
    nerve::serialization::SchemaMetadata meta;
    meta.schema_name = "map_test";
    auto m = meta.toMap();
    auto it = m.find("schema_name");
    return it != m.end() && it->second == "map_test";
}

bool check_version_negotiator_register_and_find()
{
    nerve::serialization::VersionNegotiator negotiator;

    nerve::serialization::SchemaMetadata meta;
    meta.schema_name = "test_schema";
    meta.version = nerve::serialization::SchemaVersion(1, 0, 0);
    meta.minCompatibleVersion = nerve::serialization::SchemaVersion(1, 0, 0);
    meta.maxCompatibleVersion = nerve::serialization::SchemaVersion(1, 5, 0);

    negotiator.registerSchema(meta);

    auto retrieved = negotiator.getSchemaMetadata("test_schema");
    if (retrieved.schema_name != "test_schema")
    {
        return false;
    }

    auto schemas = negotiator.getRegisteredSchemas();
    return std::find(schemas.begin(), schemas.end(), "test_schema") != schemas.end();
}

bool check_version_negotiator_unregister()
{
    nerve::serialization::VersionNegotiator negotiator;
    nerve::serialization::SchemaMetadata meta;
    meta.schema_name = "temp_schema";
    meta.version = nerve::serialization::SchemaVersion(1, 0, 0);
    meta.minCompatibleVersion = nerve::serialization::SchemaVersion(1, 0, 0);
    meta.maxCompatibleVersion = nerve::serialization::SchemaVersion(1, 0, 0);

    negotiator.registerSchema(meta);
    negotiator.unregisterSchema("temp_schema");

    auto schemas = negotiator.getRegisteredSchemas();
    return std::find(schemas.begin(), schemas.end(), "temp_schema") == schemas.end();
}

bool check_version_negotiator_negotiate_exact()
{
    nerve::serialization::VersionNegotiator negotiator;

    nerve::serialization::SchemaMetadata meta;
    meta.schema_name = "exact_test";
    meta.version = nerve::serialization::SchemaVersion(2, 0, 0);
    meta.minCompatibleVersion = nerve::serialization::SchemaVersion(2, 0, 0);
    meta.maxCompatibleVersion = nerve::serialization::SchemaVersion(2, 0, 0);

    negotiator.registerSchema(meta);

    auto result =
        negotiator.negotiateVersion("exact_test", nerve::serialization::SchemaVersion(2, 0, 0));
    if (!result.success)
    {
        return false;
    }
    return result.negotiated_version == nerve::serialization::SchemaVersion(2, 0, 0);
}

bool check_version_negotiator_negotiate_fail()
{
    nerve::serialization::VersionNegotiator negotiator;

    nerve::serialization::SchemaMetadata meta;
    meta.schema_name = "fail_test";
    meta.version = nerve::serialization::SchemaVersion(1, 0, 0);
    meta.minCompatibleVersion = nerve::serialization::SchemaVersion(1, 0, 0);
    meta.maxCompatibleVersion = nerve::serialization::SchemaVersion(1, 0, 0);

    negotiator.registerSchema(meta);

    auto result =
        negotiator.negotiateVersion("fail_test", nerve::serialization::SchemaVersion(2, 0, 0));
    return !result.success;
}

bool check_version_negotiator_supported_versions()
{
    nerve::serialization::VersionNegotiator negotiator;

    nerve::serialization::SchemaMetadata meta;
    meta.schema_name = "range_test";
    meta.version = nerve::serialization::SchemaVersion(1, 0, 0);
    meta.minCompatibleVersion = nerve::serialization::SchemaVersion(1, 0, 0);
    meta.maxCompatibleVersion = nerve::serialization::SchemaVersion(1, 3, 0);

    negotiator.registerSchema(meta);

    auto versions = negotiator.getSupportedVersions("range_test");
    return versions.size() == 4;
}

bool check_version_compatibility_check()
{
    nerve::serialization::SchemaVersion v1(1, 0, 0);
    nerve::serialization::SchemaVersion v2(3, 0, 0);
    return nerve::serialization::isVersionCompatible(v1, v2) == false;
}

bool check_schema_version_is_valid()
{
    nerve::serialization::SchemaVersion valid(1, 500, 500);
    if (!nerve::serialization::isSchemaVersionValid(valid))
    {
        return false;
    }
    nerve::serialization::SchemaVersion invalid(0, 1, 0);
    return !nerve::serialization::isSchemaVersionValid(invalid);
}

bool check_serialization_manager_singleton()
{
    auto &m1 = nerve::serialization::SerializationManager::instance();
    auto &m2 = nerve::serialization::SerializationManager::instance();
    return &m1 == &m2;
}

bool check_serialization_manager_format_support()
{
    auto &mgr = nerve::serialization::SerializationManager::instance();
    if (!mgr.isFormatSupported(nerve::serialization::SerializationFormat::BINARY))
    {
        std::cerr << "BINARY format should be supported\n";
        return false;
    }
    if (!mgr.isFormatSupported(nerve::serialization::SerializationFormat::FLATBUFFERS))
    {
        std::cerr << "FLATBUFFERS format should be supported\n";
        return false;
    }
    auto formats = mgr.getSupportedFormats();
    if (formats.empty())
    {
        std::cerr << "no formats registered\n";
        return false;
    }
    return true;
}

bool check_round_trip_schema_migration()
{
    std::vector<uint8_t> data = {0x01, 0x02, 0x03, 0x04};
    nerve::serialization::SchemaVersion from(1, 0, 0);
    nerve::serialization::SchemaVersion to(1, 2, 0);
    auto result = nerve::serialization::migrateSchema(data, from, to, "test");
    if (result.isError())
    {
        return false;
    }
    return result.value() == data;
}

bool check_round_trip_cross_major_fails()
{
    std::vector<uint8_t> data = {0x01};
    nerve::serialization::SchemaVersion from(1, 0, 0);
    nerve::serialization::SchemaVersion to(2, 0, 0);
    auto result = nerve::serialization::migrateSchema(data, from, to, "test");
    return result.isError();
}

bool check_error_code_conversion()
{
    auto ec = nerve::serialization::serializationErrorToErrorCode(
        nerve::serialization::errors::SerializationErrorCode::SUCCESS);
    if (ec != nerve::serialization::ErrorCode::SUCCESS)
    {
        return false;
    }
    ec = nerve::serialization::serializationErrorToErrorCode(
        nerve::serialization::errors::SerializationErrorCode::INCOMPATIBLE_SCHEMA_VERSION);
    return ec == nerve::serialization::ErrorCode::E31_SCHEMA_VERSION;
}

bool check_negotiator_is_version_supported()
{
    nerve::serialization::VersionNegotiator negotiator;
    nerve::serialization::SchemaMetadata meta;
    meta.schema_name = "support_test";
    meta.version = nerve::serialization::SchemaVersion(3, 0, 0);
    meta.minCompatibleVersion = nerve::serialization::SchemaVersion(3, 0, 0);
    meta.maxCompatibleVersion = nerve::serialization::SchemaVersion(3, 5, 0);
    negotiator.registerSchema(meta);

    if (!negotiator.isVersionSupported("support_test",
                                       nerve::serialization::SchemaVersion(3, 2, 0)))
    {
        return false;
    }
    if (negotiator.isVersionSupported("nonexistent", nerve::serialization::SchemaVersion(1, 0, 0)))
    {
        return false;
    }
    return true;
}

bool check_negotiation_result_factory()
{
    auto success = nerve::serialization::VersionNegotiationResult::successResult(
        nerve::serialization::SchemaVersion(2, 0, 0));
    if (!success.success)
    {
        return false;
    }
    if (success.negotiated_version != nerve::serialization::SchemaVersion(2, 0, 0))
    {
        return false;
    }

    auto error = nerve::serialization::VersionNegotiationResult::errorResult("oops");
    if (error.success)
    {
        return false;
    }
    return error.error_message == "oops";
}

} // namespace

int main()
{
    int failures = 0;

    auto run = [&](const char *name, bool ok) {
        if (!ok)
        {
            std::cerr << "FAIL: " << name << "\n";
            ++failures;
        }
        else
        {
            std::cout << "PASS: " << name << "\n";
        }
    };

    run("serialization_format_to_string", check_serialization_format_to_string());
    run("string_to_format", check_string_to_format());
    run("schema_version_construction", check_schema_version_construction());
    run("schema_version_from_string", check_schema_version_from_string());
    run("schema_version_compatibility", check_schema_version_compatibility());
    run("schema_version_comparison", check_schema_version_comparison());
    run("schema_metadata_version_compatibility", check_schema_metadata_version_compatibility());
    run("schema_metadata_to_string", check_schema_metadata_to_string());
    run("schema_metadata_to_map", check_schema_metadata_to_map());
    run("version_negotiator_register_and_find", check_version_negotiator_register_and_find());
    run("version_negotiator_unregister", check_version_negotiator_unregister());
    run("version_negotiator_negotiate_exact", check_version_negotiator_negotiate_exact());
    run("version_negotiator_negotiate_fail", check_version_negotiator_negotiate_fail());
    run("version_negotiator_supported_versions", check_version_negotiator_supported_versions());
    run("version_compatibility_check", check_version_compatibility_check());
    run("schema_version_is_valid", check_schema_version_is_valid());
    run("serialization_manager_singleton", check_serialization_manager_singleton());
    run("serialization_manager_format_support", check_serialization_manager_format_support());
    run("round_trip_schema_migration", check_round_trip_schema_migration());
    run("round_trip_cross_major_fails", check_round_trip_cross_major_fails());
    run("error_code_conversion", check_error_code_conversion());
    run("negotiator_is_version_supported", check_negotiator_is_version_supported());
    run("negotiation_result_factory", check_negotiation_result_factory());

    return failures > 0 ? 1 : 0;
}
