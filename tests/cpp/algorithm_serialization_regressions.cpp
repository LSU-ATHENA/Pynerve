#include "nerve/determinism.hpp"
#include "nerve/serialization/serialization_manager.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

int main()
{
    using namespace nerve::serialization;

    assert(SchemaVersion::fromString("2.3.4") == SchemaVersion(2, 3, 4));
    assert(SchemaVersion::fromString("not-a-version") == SchemaVersion(1, 0, 0));
    assert(SchemaVersion::fromString("4294967296.0.0") == SchemaVersion(1, 0, 0));
    assert(SchemaVersion::fromString("999999999999999999999.0.0") == SchemaVersion(1, 0, 0));
    assert(SchemaVersion::fromString("4294967295.0.0") ==
           SchemaVersion(std::numeric_limits<std::uint32_t>::max(), 0, 0));

    VersionNegotiator negotiator;
    SchemaMetadata bounded_schema;
    bounded_schema.schema_name = "bounded";
    bounded_schema.version = SchemaVersion(1, 3, 0);
    bounded_schema.minCompatibleVersion = SchemaVersion(1, 1, 0);
    bounded_schema.maxCompatibleVersion = SchemaVersion(1, 3, 0);
    negotiator.registerSchema(bounded_schema);
    const auto bounded_versions = negotiator.getSupportedVersions("bounded");
    assert(bounded_versions.size() == 3);
    assert(bounded_versions[0] == SchemaVersion(1, 1, 0));
    assert(bounded_versions[2] == SchemaVersion(1, 3, 0));

    SchemaMetadata huge_schema = bounded_schema;
    huge_schema.schema_name = "huge";
    huge_schema.maxCompatibleVersion =
        SchemaVersion(1, std::numeric_limits<std::uint32_t>::max(), 0);
    negotiator.registerSchema(huge_schema);
    assert(negotiator.getSupportedVersions("huge").empty());

    SchemaMetadata wrapped_schema = bounded_schema;
    wrapped_schema.schema_name = "wrapped";
    wrapped_schema.minCompatibleVersion =
        SchemaVersion(1, std::numeric_limits<std::uint32_t>::max() - 1U, 0);
    wrapped_schema.maxCompatibleVersion =
        SchemaVersion(1, std::numeric_limits<std::uint32_t>::max(), 0);
    negotiator.registerSchema(wrapped_schema);
    const auto wrapped_versions = negotiator.getSupportedVersions("wrapped");
    assert(wrapped_versions.size() == 2);
    assert(wrapped_versions[0] ==
           SchemaVersion(1, std::numeric_limits<std::uint32_t>::max() - 1U, 0));
    assert(wrapped_versions[1] == SchemaVersion(1, std::numeric_limits<std::uint32_t>::max(), 0));

    const std::vector<std::uint8_t> payload{'t', 'd', 'a'};
    const SerializationContext context(SerializationFormat::FLATBUFFERS, SchemaVersion(1, 2, 3));
    FlatBuffersSerializer serializer;

    const auto serialized = serializer.serialize(payload.data(), payload.size(), context);
    assert(!serialized.isError());
    const auto &bytes = serialized.value();
    assert(bytes.size() == payload.size() + 12);
    assert(bytes[0] == 0x31);
    assert(bytes[1] == 0x4C);
    assert(bytes[2] == 0x42);
    assert(bytes[3] == 0x46);
    assert(bytes[4] == 0x00);
    assert(bytes[5] == 0x03);
    assert(bytes[6] == 0x02);
    assert(bytes[7] == 0x01);
    assert(bytes[8] == payload.size());
    assert(bytes[9] == 0x00);
    assert(bytes[10] == 0x00);
    assert(bytes[11] == 0x00);

    const auto decoded = serializer.deserialize(bytes, context);
    assert(!decoded.isError());
    assert(decoded.value() == payload);

    SchemaMetadata metadata;
    metadata.schema_name = "schema\"name\\flat";
    metadata.description = "desc\nquote\"slash\\";
    metadata.version = SchemaVersion(2, 3, 4);
    metadata.minCompatibleVersion = SchemaVersion(2, 0, 0);
    metadata.maxCompatibleVersion = SchemaVersion(2, 9, 0);
    const auto serialized_with_metadata =
        serializer.serializeWithMetadata(payload.data(), payload.size(), metadata, context);
    assert(!serialized_with_metadata.isError());
    const auto &metadata_bytes = serialized_with_metadata.value();
    assert(metadata_bytes.size() >= 4);
    const std::uint32_t metadata_size = static_cast<std::uint32_t>(metadata_bytes[0]) |
                                        (static_cast<std::uint32_t>(metadata_bytes[1]) << 8U) |
                                        (static_cast<std::uint32_t>(metadata_bytes[2]) << 16U) |
                                        (static_cast<std::uint32_t>(metadata_bytes[3]) << 24U);
    assert(metadata_bytes.size() >= 4 + metadata_size);
    const std::string metadata_json(metadata_bytes.begin() + 4,
                                    metadata_bytes.begin() + 4 + metadata_size);
    assert(metadata_json.find("\\\"") != std::string::npos);
    assert(metadata_json.find("\\\\") != std::string::npos);
    assert(metadata_json.find("\\n") != std::string::npos);

    const auto decoded_with_metadata = serializer.deserializeWithMetadata(metadata_bytes, context);
    assert(!decoded_with_metadata.isError());
    assert(decoded_with_metadata.value().first == payload);
    assert(decoded_with_metadata.value().second.schema_name == metadata.schema_name);
    assert(decoded_with_metadata.value().second.description == metadata.description);
    assert(decoded_with_metadata.value().second.version == metadata.version);
    assert(decoded_with_metadata.value().second.minCompatibleVersion ==
           metadata.minCompatibleVersion);
    assert(decoded_with_metadata.value().second.maxCompatibleVersion ==
           metadata.maxCompatibleVersion);

    auto corrupt_size = bytes;
    corrupt_size[8] = 0xFF;
    corrupt_size[9] = 0xFF;
    corrupt_size[10] = 0xFF;
    corrupt_size[11] = 0xFF;
    assert(serializer.deserialize(corrupt_size, context).isError());
}

{
    using namespace nerve::serialization;

    const std::vector<std::uint8_t> payload{'p', 'h'};
    const SerializationContext context(SerializationFormat::ARROW, SchemaVersion(4, 5, 6));
    ArrowSerializer serializer;

    const auto serialized = serializer.serialize(payload.data(), payload.size(), context);
    assert(!serialized.isError());
    const auto &bytes = serialized.value();
    assert(bytes.size() == payload.size() + 12);
    assert(bytes[0] == 0xFF);
    assert(bytes[1] == 0xFF);
    assert(bytes[2] == 'A');
    assert(bytes[3] == 'R');
    assert(bytes[8] == 0x00);
    assert(bytes[9] == 0x06);
    assert(bytes[10] == 0x05);
    assert(bytes[11] == 0x04);

    const auto decoded = serializer.deserialize(bytes, context);
    assert(!decoded.isError());
    assert(decoded.value() == payload);

    SchemaMetadata metadata;
    metadata.schema_name = "schema\"name\\arrow";
    metadata.description = "desc\nquote\"slash\\";
    metadata.version = SchemaVersion(4, 7, 8);
    metadata.minCompatibleVersion = SchemaVersion(4, 0, 0);
    metadata.maxCompatibleVersion = SchemaVersion(4, 9, 0);
    const auto serialized_with_metadata =
        serializer.serializeWithMetadata(payload.data(), payload.size(), metadata, context);
    assert(!serialized_with_metadata.isError());
    const auto &metadata_bytes = serialized_with_metadata.value();
    assert(metadata_bytes.size() >= 4);
    const std::uint32_t metadata_size = static_cast<std::uint32_t>(metadata_bytes[0]) |
                                        (static_cast<std::uint32_t>(metadata_bytes[1]) << 8U) |
                                        (static_cast<std::uint32_t>(metadata_bytes[2]) << 16U) |
                                        (static_cast<std::uint32_t>(metadata_bytes[3]) << 24U);
    assert(metadata_bytes.size() >= 4 + metadata_size);
    const std::string metadata_json(metadata_bytes.begin() + 4,
                                    metadata_bytes.begin() + 4 + metadata_size);
    assert(metadata_json.find("\\\"") != std::string::npos);
    assert(metadata_json.find("\\\\") != std::string::npos);
    assert(metadata_json.find("\\n") != std::string::npos);

    const auto decoded_with_metadata = serializer.deserializeWithMetadata(metadata_bytes, context);
    assert(!decoded_with_metadata.isError());
    assert(decoded_with_metadata.value().first == payload);
    assert(decoded_with_metadata.value().second.schema_name == metadata.schema_name);
    assert(decoded_with_metadata.value().second.description == metadata.description);
    assert(decoded_with_metadata.value().second.version == metadata.version);
    assert(decoded_with_metadata.value().second.minCompatibleVersion ==
           metadata.minCompatibleVersion);
    assert(decoded_with_metadata.value().second.maxCompatibleVersion ==
           metadata.maxCompatibleVersion);
}

{
    using namespace nerve::serialization;

    PH5PH6ArtifactMetadata metadata;
    metadata.schema_version = SchemaVersion(1, 2, 3);
    metadata.min_compatible_version = SchemaVersion(4, 5, 6);
    metadata.artifact_type = "PH5";
    metadata.algorithm_variant = "alg";
    metadata.has_highdim_extension = true;
    metadata.max_supported_dimension = 6;
    metadata.extension_fields = {"x"};

    const auto metadata_bytes = metadata.serializeMetadata();
    assert(metadata_bytes[0] == 0x01);
    assert(metadata_bytes[1] == 0x00);
    assert(metadata_bytes[4] == 0x02);
    assert(metadata_bytes[8] == 0x03);
    assert(metadata_bytes[12] == 0x04);
    assert(metadata_bytes[16] == 0x05);
    assert(metadata_bytes[20] == 0x06);
    assert(metadata_bytes[24] == 0x03);
    assert(metadata_bytes[25] == 0x00);

    PH5PH6ArtifactMetadata decoded_metadata;
    assert(decoded_metadata.deserializeMetadata(metadata_bytes));
    assert(decoded_metadata.schema_version == SchemaVersion(1, 2, 3));
    assert(decoded_metadata.min_compatible_version == SchemaVersion(4, 5, 6));
    assert(decoded_metadata.artifact_type == "PH5");
    assert(decoded_metadata.algorithm_variant == "alg");
    assert(decoded_metadata.has_highdim_extension);
    assert(decoded_metadata.max_supported_dimension == 6);
    assert(decoded_metadata.extension_fields.size() == 1);
    assert(decoded_metadata.extension_fields[0] == "x");

    const std::vector<std::uint8_t> payload{'h', 'd'};
    const SerializationContext context(SerializationFormat::FLATBUFFERS, SchemaVersion(1, 0, 0));
    PH5PH6SchemaSerializer serializer;
    const auto serialized =
        serializer.serializePh5Artifact(payload.data(), payload.size(), metadata, context);
    assert(!serialized.isError());
    const auto &bytes = serialized.value();
    assert(bytes[0] == 0x32);
    assert(bytes[1] == 0x35);
    assert(bytes[2] == 0x48);
    assert(bytes[3] == 0x50);
    assert(bytes[4] == PH5PH6SchemaSerializer::HEADER_SIZE_EXTENDED);
    assert(bytes[5] == 0x01);
    assert(bytes[9] == 0x01);

    const auto decoded = serializer.deserialize(bytes, context);
    assert(!decoded.isError());
    assert(decoded.value() == payload);

    const std::vector<std::uint8_t> legacy{0x31, 0x35, 0x48, 0x50, 'o', 'k'};
    assert(PH5PH6SchemaMigrator::validateData(legacy));
    const auto migrated = PH5PH6SchemaMigrator::migrateToExtended(legacy, context);
    assert(migrated.success);
    assert(
        PH5PH6SchemaMigrator::validateExtendedData(migrated.migrated_data, migrated.new_metadata));
}

{
    nerve::summary::CompactSummary summary{};
    summary.lifetime_count = 1;
    summary.top_lifetimes[0] = nerve::summary::CompactSummary::Lifetime{1.0F, 2.0F, 7U, 3.0F};
    summary.betti_dimension_count = 1;
    summary.betti_counts[0] = 513U;
    summary.eigenvalue_count = 1;
    summary.top_eigenvalues[0] = nerve::summary::CompactSummary::Eigenvalue{4.0F, 7U};
    summary.persistence_entropy = 0.5F;
    summary.betti_entropy = 0.25F;
    summary.spectral_entropy = 0.125F;
    summary.timestamp_ns = 123456789;
    summary.symbol_id = 42;
    summary.computation_time_us = 77;
    summary.data_points_count = 9;
    summary.noise_level = 0.03125F;
    assert(summary.isValid());

    const auto bytes = summary.serialize();
    assert(bytes[0] == 0x01);
    assert(bytes[1] == 0x00);
    assert(bytes[2] == 0x00);
    assert(bytes[3] == 0x80);
    assert(bytes[4] == 0x3F);
    assert(bytes[5] == 0x00);
    assert(bytes[8] == 0x40);
    assert(bytes[9] == 0x07);
    assert(bytes[10] == 0x00);
    assert(bytes[11] == 0x00);
    assert(bytes[12] == 0x00);
    assert(bytes[13] == 0x00);
    assert(bytes[16] == 0x40);
    assert(bytes[17] == 0x01);
    assert(bytes[18] == 0x01);
    assert(bytes[19] == 0x02);
    assert(bytes[20] == 0x01);
    assert(bytes[25] == 0x07);
    assert(bytes[26] == 0x00);
    assert(bytes[27] == 0x00);
    assert(bytes[28] == 0x00);

    nerve::summary::CompactSummary decoded{};
    assert(decoded.deserialize(bytes));
    assert(decoded.lifetime_count == 1);
    assert(decoded.top_lifetimes[0].birth == 1.0F);
    assert(decoded.top_lifetimes[0].death == 2.0F);
    assert(decoded.top_lifetimes[0].dimension == 7U);
    assert(decoded.top_lifetimes[0].persistence == 3.0F);
    assert(decoded.betti_counts[0] == 513U);
    assert(decoded.top_eigenvalues[0].value == 4.0F);
    assert(decoded.top_eigenvalues[0].multiplicity == 7U);

    auto corrupt = bytes;
    corrupt[0] = 0xFF;
    nerve::summary::CompactSummary rejected{};
    assert(!rejected.deserialize(corrupt));

    nerve::summary::CompactSummary invalid = summary;
    invalid.persistence_entropy = std::numeric_limits<float>::quiet_NaN();
    assert(!invalid.isValid());

    invalid = summary;
    invalid.top_lifetimes[0].death = std::numeric_limits<float>::quiet_NaN();
    assert(!invalid.isValid());

    invalid = summary;
    invalid.top_lifetimes[0].death = std::numeric_limits<float>::infinity();
    invalid.top_lifetimes[0].persistence = std::numeric_limits<float>::infinity();
    assert(invalid.isValid());

    invalid = summary;
    invalid.top_eigenvalues[0].value = std::numeric_limits<float>::infinity();
    assert(!invalid.isValid());

    invalid = summary;
    invalid.has_highdim_extension = true;
    invalid.highdim_extension.dimension_complexity.fill(0.0F);
    invalid.highdim_extension.dimension_complexity[0] = std::numeric_limits<float>::infinity();
    assert(!invalid.isValid());
}

{
    using namespace nerve::core;

    DeterminismContract contract(DeterminismLevel::STRICT, "component");
    contract.max_execution_time = std::chrono::milliseconds(0x0102030405060708LL);
    contract.max_memory_usage_mb = static_cast<size_t>(0x1122334455667788ULL);
    contract.params_hash.fill(0xAB);
    contract.params_hash_valid = true;

    const auto bytes = contract.serialize();
    assert(bytes[50] == 0x08);
    assert(bytes[57] == 0x01);
    assert(bytes[58] == 0x88);
    assert(bytes[65] == 0x11);
    assert(bytes[66] == 0x09);
    assert(bytes[67] == 0x00);
    assert(bytes[68] == 0x00);
    assert(bytes[69] == 0x00);

    DeterminismContract decoded;
    assert(decoded.deserialize(bytes));
    assert(decoded.level == contract.level);
    assert(decoded.max_execution_time == contract.max_execution_time);
    assert(decoded.max_memory_usage_mb == contract.max_memory_usage_mb);
    assert(decoded.component_name == "component");

    DeterminismMetadata metadata;
    metadata.params_hash.fill(0x01);
    metadata.result_checksum.fill(0x02);
    metadata.rng_seed_used.fill(0x03);
    metadata.achieved_level = DeterminismLevel::AUDIT;
    metadata.actual_execution_time = std::chrono::milliseconds(0x0102030405060708LL);
    metadata.actual_memory_usage_mb = static_cast<size_t>(0x1122334455667788ULL);
    metadata.was_deterministic = true;
    metadata.warnings = {"w"};
    metadata.error_message = "e";

    const auto metadata_bytes = metadata.serialize();
    assert(metadata_bytes[81] == 0x08);
    assert(metadata_bytes[88] == 0x01);
    assert(metadata_bytes[89] == 0x88);
    assert(metadata_bytes[96] == 0x11);
    assert(metadata_bytes[98] == 0x01);
    assert(metadata_bytes[99] == 0x00);
    assert(metadata_bytes[100] == 0x00);
    assert(metadata_bytes[101] == 0x00);

    DeterminismMetadata decoded_metadata;
    assert(decoded_metadata.deserialize(metadata_bytes));
    assert(decoded_metadata.achieved_level == DeterminismLevel::AUDIT);
    assert(decoded_metadata.actual_execution_time == metadata.actual_execution_time);
    assert(decoded_metadata.actual_memory_usage_mb == metadata.actual_memory_usage_mb);
    assert(decoded_metadata.was_deterministic);
    assert(decoded_metadata.warnings == metadata.warnings);
    assert(decoded_metadata.error_message == "e");
}

{
    using namespace nerve::persistence::perfect;

    PerfectPivotMap map;
    const std::vector<int> keys{4, 8, 12};
    const std::vector<int> values{40, 80, 120};
    assert(map.build(keys, values));
    assert(map.lookup(8) == 80);
    assert(!map.contains(7));

    const char *valid_path = "nerve_perfect_hash_regression.bin";
    map.save(valid_path);
    PerfectPivotMap loaded;
    assert(loaded.load(valid_path));
    assert(loaded.lookup(12) == 120);
    std::remove(valid_path);

    const char *corrupt_path = "nerve_perfect_hash_corrupt.bin";
    {
        std::FILE *file = std::fopen(corrupt_path, "wb");
        assert(file != nullptr);
        const std::uint32_t magic = 0x54504831U;
        const std::uint32_t version = 1;
        const std::uint64_t seed = 0;
        const std::size_t num_keys = 1;
        const int min_key = 0;
        const std::uint8_t direct = 0;
        const std::size_t impossible_vector_size =
            std::numeric_limits<std::size_t>::max() / sizeof(std::uint8_t);
        assert(std::fwrite(&magic, sizeof(magic), 1, file) == 1);
        assert(std::fwrite(&version, sizeof(version), 1, file) == 1);
        assert(std::fwrite(&seed, sizeof(seed), 1, file) == 1);
        assert(std::fwrite(&seed, sizeof(seed), 1, file) == 1);
        assert(std::fwrite(&num_keys, sizeof(num_keys), 1, file) == 1);
        assert(std::fwrite(&min_key, sizeof(min_key), 1, file) == 1);
        assert(std::fwrite(&direct, sizeof(direct), 1, file) == 1);
        assert(std::fwrite(&impossible_vector_size, sizeof(impossible_vector_size), 1, file) == 1);
        std::fclose(file);
    }
    PerfectPivotMap rejected;
    assert(!rejected.load(corrupt_path));
    std::remove(corrupt_path);

    FastStaticMap static_map;
    assert(static_map.build(keys, values));
    assert(static_map.find(4).value() == 40);
    assert(!static_map.build({1, 1}, {10, 20}));
}

{
    bool rejected_bloom_nan_fpp = false;
    try
    {
        nerve::persistence::bloom::BloomFilter bloom(128, std::numeric_limits<double>::quiet_NaN());
        (void)bloom;
    }
    catch (const std::invalid_argument &)
    {
        rejected_bloom_nan_fpp = true;
    }
    assert(rejected_bloom_nan_fpp);

    bool rejected_bloom_oversize = false;
    try
    {
        nerve::persistence::bloom::BloomFilter bloom(std::numeric_limits<size_t>::max(), 0.01);
        (void)bloom;
    }
    catch (const std::length_error &)
    {
        rejected_bloom_oversize = true;
    }
    assert(rejected_bloom_oversize);

    nerve::persistence::bloom::BloomFilter clamped_bloom(128, 0.0);
    assert(clamped_bloom.numHashFunctions() > 0);
}

{
    nerve::persistence::robin_hood::RobinHoodHashMap<int, int> pivot_map(1024);
    for (int i = 0; i < 300; ++i)
    {
        pivot_map.insert(i * 1024, i);
    }
    assert(pivot_map.size() == 300);
    for (int i = 0; i < 300; ++i)
    {
        const auto value = pivot_map.find(i * 1024);
        assert(value.has_value());
        assert(value.value() == i);
    }

    using namespace nerve::persistence::bitparallel;

    std::vector<BitColumn> columns;
    columns.push_back(buildBitColumn({1, 511}, 512));
    columns.push_back(buildBitColumn({2, 511}, 512));

    BitParallelConfig config;
    config.use_avx512 = true;
    const auto result = reduceMatrixBitParallel(columns, config, {});

    assert(result.xor_operations == 1);
    assert(std::isfinite(result.speedup_estimate));
    assert(result.speedup_estimate >= 1.0);
    assert(columns[1].pivot == 2);
    assert(columns[1].words.size() == 8);
}

{
    using namespace nerve::persistence::bitparallel;

    std::vector<BitColumn> columns;
    columns.push_back(buildBitColumn({}, 3));
    columns.push_back(buildBitColumn({}, 3));
    columns.push_back(buildBitColumn({0, 1}, 3));

    BitParallelConfig config;
    config.use_clearing = false;
    const auto result = reduceMatrixBitParallel(columns, config, {0.0, 0.0, 2.0});

    bool found_finite_pair = false;
    bool found_stale_infinite_pair = false;
    for (const auto &pair : result.pairs)
    {
        found_finite_pair = found_finite_pair || (pair.birth_index == 1 && pair.death_index == 2 &&
                                                  pair.birth_time == 0.0 && pair.death_time == 2.0);
        found_stale_infinite_pair =
            found_stale_infinite_pair || (pair.birth_index == 1 && pair.death_index < 0);
    }
    assert(found_finite_pair);
    assert(!found_stale_infinite_pair);
}

{
    nerve::persistence::DistilledVRConfig config;
    nerve::persistence::DistilledVRFiltration filtration(config);
    const std::vector<double> points{
        0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
    };
    const auto distilled = filtration.distill(points, 2, 2.0);
    assert(!distilled.landmark_indices.empty());

    bool rejected_distilled_nan_ratio = false;
    try
    {
        auto invalid_config = config;
        invalid_config.target_reduction_ratio = std::numeric_limits<double>::quiet_NaN();
        nerve::persistence::DistilledVRFiltration invalid_filtration(invalid_config);
        (void)invalid_filtration.distill(points, 2, 2.0);
    }
    catch (const std::invalid_argument &)
    {
        rejected_distilled_nan_ratio = true;
    }
    assert(rejected_distilled_nan_ratio);

    bool rejected_distilled_nan_radius = false;
    try
    {
        (void)filtration.distill(points, 2, std::numeric_limits<double>::quiet_NaN());
    }
    catch (const std::invalid_argument &)
    {
        rejected_distilled_nan_radius = true;
    }
    assert(rejected_distilled_nan_radius);

    bool rejected_distilled_nan_point = false;
    try
    {
        const std::vector<double> invalid_points{0.0, 0.0, std::numeric_limits<double>::quiet_NaN(),
                                                 1.0};
        (void)filtration.distill(invalid_points, 2, 2.0);
    }
    catch (const std::invalid_argument &)
    {
        rejected_distilled_nan_point = true;
    }
    assert(rejected_distilled_nan_point);

    bool rejected_distilled_nan_weight = false;
    try
    {
        nerve::persistence::DistilledVRResult invalid_distilled;
        invalid_distilled.landmark_indices = {0, 1};
        invalid_distilled.landmark_edges = {{0, 1}};
        invalid_distilled.landmark_edge_weights = {std::numeric_limits<double>::quiet_NaN()};
        (void)filtration.computeApproximatePersistence(invalid_distilled);
    }
    catch (const std::invalid_argument &)
    {
        rejected_distilled_nan_weight = true;
    }
    assert(rejected_distilled_nan_weight);
}

{
    using namespace nerve::persistence::distilled;

    DistilledFiltration filtration;
    for (int i = 0; i < 1001; ++i)
    {
        DistilledSimplex simplex;
        simplex.vertices = {i};
        simplex.dimension = 0;
        simplex.filtration_value = 0.0;
        simplex.original_index = i;
        filtration.simplices.push_back(simplex);
    }
    DistilledSimplex edge;
    edge.vertices = {0, 1};
    edge.dimension = 1;
    edge.filtration_value = 2.0;
    edge.original_index = 1001;
    filtration.simplices.push_back(edge);
    filtration.distilled_size = static_cast<int>(filtration.simplices.size());

    DistilledVRConfig config;
    config.use_bit_parallel = true;
    const auto result = computePersistenceDistilled(filtration, config);

    bool found_distilled_pair = false;
    for (const auto &pair : result.pairs)
    {
        if (pair.death_index >= 0)
        {
            assert(pair.birth_time <= pair.death_time);
        }
        found_distilled_pair =
            found_distilled_pair || (pair.birth_index == 1 && pair.death_index == 1001 &&
                                     pair.birth_time == 0.0 && pair.death_time == 2.0);
    }
    assert(result.used_bit_parallel);
    assert(found_distilled_pair);
}

return 0;
}

return 0;
}
