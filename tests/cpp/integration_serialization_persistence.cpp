#include "nerve/common/accelerated_types.hpp"
#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/core_types.hpp"
#include "nerve/persistence/core/core_types.hpp"
#include "nerve/persistence/vr/vr_fast_ops.hpp"
#include "nerve/serialization/serialization_manager.hpp"
#include "test_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <limits>
#include <tuple>
#include <vector>

namespace
{

using nerve::common::VRConfig;
using nerve::persistence::Pair;
using nerve::persistence::VRAlgorithmSelection;
using namespace nerve::test;

constexpr double kTol = 1e-10;


bool assert_same_pairs(const std::vector<Pair> &expected, const std::vector<Pair> &actual)
{
    const auto c1 = canonical(expected);
    const auto c2 = canonical(actual);
    if (c1.size() != c2.size())
    {
        std::cerr << "pair count mismatch: " << c1.size() << " vs " << c2.size() << "\n";
        return false;
    }
    for (std::size_t i = 0; i < c1.size(); ++i)
    {
        if (!pairs_equal(c1[i], c2[i]))
        {
            std::cerr << "pair " << i << " differs: dim=" << c1[i].dimension
                      << " birth=" << c1[i].birth << " death=" << c1[i].death
                      << " vs dim=" << c2[i].dimension << " birth=" << c2[i].birth
                      << " death=" << c2[i].death << "\n";
            return false;
        }
    }
    return true;
}

bool check_serialize_deserialize_pairs_round_trip()
{
    const std::vector<double> pts = {0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0};
    VRConfig cfg;
    cfg.max_radius = 2.0;
    cfg.max_dim = 2;
    cfg.algorithm = VRAlgorithmSelection::EXACT_STANDARD;

    const auto pairs = nerve::persistence::computeVrPersistenceFast(view_of(pts), 2, cfg);
    if (pairs.empty())
    {
        std::cerr << "serialize: pairs should not be empty\n";
        return false;
    }

    nerve::serialization::SchemaMetadata schema;
    schema.schema_name = "nerve::persistence::Pair";
    nerve::serialization::SerializationManager::instance().registerSchema(schema);

    auto &mgr = nerve::serialization::SerializationManager::instance();

    auto ser_result =
        mgr.serialize("nerve::persistence::Pair", pairs.data(), pairs.size() * sizeof(Pair),
                      nerve::serialization::SerializationContext(
                          nerve::serialization::SerializationFormat::BINARY));
    if (ser_result.isError())
    {
        std::cerr << "serialize pairs failed\n";
        return false;
    }
    auto serialized = ser_result.moveValue();

    if (serialized.empty())
    {
        std::cerr << "serialized data is empty\n";
        return false;
    }

    auto deser_result = mgr.deserialize("nerve::persistence::Pair", serialized,
                                        nerve::serialization::SerializationContext(
                                            nerve::serialization::SerializationFormat::BINARY));
    if (deser_result.isError())
    {
        std::cerr << "deserialize pairs failed\n";
        return false;
    }
    auto deserialized_bytes = deser_result.moveValue();

    if (deserialized_bytes.size() != pairs.size() * sizeof(Pair))
    {
        std::cerr << "deserialized size mismatch: " << deserialized_bytes.size() << " vs "
                  << pairs.size() * sizeof(Pair) << "\n";
        return false;
    }

    const auto *deserialized_pairs = reinterpret_cast<const Pair *>(deserialized_bytes.data());
    std::vector<Pair> roundtrip(deserialized_pairs, deserialized_pairs + pairs.size());

    if (!assert_same_pairs(pairs, roundtrip))
    {
        std::cerr << "FAIL: pairs round-trip mismatch\n";
        return false;
    }
    return true;
}

bool check_serialize_deserialize_diagram_round_trip()
{
    const std::vector<double> pts = {0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.5};
    VRConfig cfg;
    cfg.max_radius = 2.0;
    cfg.max_dim = 2;
    cfg.algorithm = VRAlgorithmSelection::EXACT_STANDARD;

    const auto pairs = nerve::persistence::computeVrPersistenceFast(view_of(pts), 2, cfg);
    nerve::persistence::Diagram diagram(pairs);

    nerve::serialization::SchemaMetadata schema;
    schema.schema_name = "nerve::persistence::Diagram";
    nerve::serialization::SerializationManager::instance().registerSchema(schema);

    auto &mgr = nerve::serialization::SerializationManager::instance();

    auto ser_result = mgr.serialize("nerve::persistence::Diagram", diagram.getPairs().data(),
                                    diagram.getPairs().size() * sizeof(Pair),
                                    nerve::serialization::SerializationContext(
                                        nerve::serialization::SerializationFormat::BINARY));
    if (ser_result.isError())
    {
        std::cerr << "serialize diagram failed\n";
        return false;
    }
    auto serialized = ser_result.moveValue();

    auto deser_result = mgr.deserialize("nerve::persistence::Diagram", serialized,
                                        nerve::serialization::SerializationContext(
                                            nerve::serialization::SerializationFormat::BINARY));
    if (deser_result.isError())
    {
        std::cerr << "deserialize diagram failed\n";
        return false;
    }
    auto deserialized_bytes = deser_result.moveValue();

    const auto *deserialized_pairs = reinterpret_cast<const Pair *>(deserialized_bytes.data());
    std::vector<Pair> roundtrip(deserialized_pairs, deserialized_pairs + pairs.size());
    nerve::persistence::Diagram roundtrip_diagram(roundtrip);

    if (roundtrip_diagram.count() != diagram.count())
    {
        std::cerr << "diagram count mismatch after round-trip: " << roundtrip_diagram.count()
                  << " vs " << diagram.count() << "\n";
        return false;
    }

    auto betti_orig = diagram.computeBetti();
    auto betti_rt = roundtrip_diagram.computeBetti();
    if (betti_orig.size() != betti_rt.size())
    {
        std::cerr << "Betti numbers size mismatch after round-trip\n";
        return false;
    }
    for (std::size_t i = 0; i < betti_orig.size(); ++i)
    {
        if (betti_orig[i] != betti_rt[i])
        {
            std::cerr << "Betti[" << i << "] mismatch after round-trip: " << betti_orig[i] << " vs "
                      << betti_rt[i] << "\n";
            return false;
        }
    }

    if (!assert_same_pairs(pairs, roundtrip))
    {
        std::cerr << "FAIL: diagram round-trip mismatch\n";
        return false;
    }
    return true;
}

bool check_serialization_with_different_formats()
{
    const std::vector<double> pts = {0.0, 0.0, 1.0, 0.0, 0.0, 1.0};
    VRConfig cfg;
    cfg.max_radius = 2.0;
    cfg.max_dim = 2;
    cfg.algorithm = VRAlgorithmSelection::EXACT_STANDARD;

    const auto pairs = nerve::persistence::computeVrPersistenceFast(view_of(pts), 2, cfg);

    nerve::serialization::SchemaMetadata schema;
    schema.schema_name = "nerve::persistence::Pair";
    nerve::serialization::SerializationManager::instance().registerSchema(schema);

    auto &mgr = nerve::serialization::SerializationManager::instance();

    nerve::serialization::SerializationFormat formats[] = {
        nerve::serialization::SerializationFormat::BINARY,
        nerve::serialization::SerializationFormat::FLATBUFFERS};

    for (auto fmt : formats)
    {
        auto ser_result =
            mgr.serialize("nerve::persistence::Pair", pairs.data(), pairs.size() * sizeof(Pair),
                          nerve::serialization::SerializationContext(fmt));
        if (ser_result.isError())
        {
            std::cerr << "serialize failed for format " << static_cast<int>(fmt) << "\n";
            return false;
        }
        auto serialized = ser_result.moveValue();

        auto deser_result = mgr.deserialize("nerve::persistence::Pair", serialized,
                                            nerve::serialization::SerializationContext(fmt));
        if (deser_result.isError())
        {
            std::cerr << "deserialize failed for format " << static_cast<int>(fmt) << "\n";
            return false;
        }
        auto deserialized_bytes = deser_result.moveValue();

        if (deserialized_bytes.size() != pairs.size() * sizeof(Pair))
        {
            std::cerr << "size mismatch for format " << static_cast<int>(fmt) << "\n";
            return false;
        }

        const auto *deserialized_pairs = reinterpret_cast<const Pair *>(deserialized_bytes.data());
        std::vector<Pair> roundtrip(deserialized_pairs, deserialized_pairs + pairs.size());

        if (!assert_same_pairs(pairs, roundtrip))
        {
            std::cerr << "pair mismatch for format " << static_cast<int>(fmt) << "\n";
            return false;
        }
    }

    return true;
}

} // namespace

int main()
{
    if (!check_serialize_deserialize_pairs_round_trip())
    {
        std::cerr << "FAIL: serialize/deserialize pairs round-trip\n";
        return 1;
    }
    if (!check_serialize_deserialize_diagram_round_trip())
    {
        std::cerr << "FAIL: serialize/deserialize diagram round-trip\n";
        return 1;
    }
    if (!check_serialization_with_different_formats())
    {
        std::cerr << "FAIL: serialization with different formats\n";
        return 1;
    }
    return 0;
}
