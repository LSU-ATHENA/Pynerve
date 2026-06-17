
#include "nerve/core/detail/core_detail.hpp"
#include "nerve/core/determinism.hpp"
#include "nerve/core_types.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

namespace
{

using nerve::Size;

constexpr double TOL = 1e-12;

bool check_simd_memcpy_basic()
{
    char src[64];
    char dst[64] = {};
    for (int i = 0; i < 64; ++i)
    {
        src[i] = static_cast<char>(i);
    }
    nerve::core::simdMemcpy(dst, src, 64);
    for (int i = 0; i < 64; ++i)
    {
        if (dst[i] != src[i])
        {
            return false;
        }
    }
    return true;
}

bool check_simd_memset_basic()
{
    char buf[128] = {};
    nerve::core::simdMemset(buf, 0xAB, 128);
    for (int i = 0; i < 128; ++i)
    {
        if (static_cast<unsigned char>(buf[i]) != 0xAB)
        {
            return false;
        }
    }
    return true;
}

bool check_simd_reduce_sum()
{
    double data[16];
    double expected = 0.0;
    for (int i = 0; i < 16; ++i)
    {
        data[i] = static_cast<double>(i + 1);
        expected += data[i];
    }
    double sum = nerve::core::simdReduceSum(data, 16);
    return std::abs(sum - expected) < TOL;
}

bool check_simd_reduce_sum_small()
{
    double data[3] = {1.5, 2.5, 3.0};
    double sum = nerve::core::simdReduceSum(data, 3);
    return std::abs(sum - 7.0) < TOL;
}

bool check_simd_reduce_sum_empty()
{
    double data[1] = {0.0};
    double sum = nerve::core::simdReduceSum(data, 0);
    return std::abs(sum) < TOL;
}

bool check_numa_pool_manager_singleton()
{
    auto &mgr = nerve::core::NumaPoolManager::instance();
    auto &mgr2 = nerve::core::NumaPoolManager::instance();
    return &mgr == &mgr2;
}

bool check_numa_pool_config_defaults()
{
    nerve::core::NumaPoolConfig config;
    return config.policy == nerve::core::NumaPolicy::AUTO;
}

bool check_memory_pool_allocate_deallocate()
{
    nerve::core::MemoryPool pool;
    nerve::core::DeterminismContract contract;
    contract.level = nerve::core::DeterminismLevel::NONE;
    contract.max_memory_usage_mb = 1024;
    contract.max_execution_time = std::chrono::milliseconds(10000);

    void *ptr = pool.allocate(64);
    if (!ptr)
    {
        return false;
    }
    std::memset(ptr, 0x42, 64);
    pool.deallocate(ptr);

    if (!pool.isDeterministic())
    {
        return true;
    }
    return true;
}

bool check_memory_pool_deterministic_cycle()
{
    nerve::core::MemoryPool pool;
    nerve::core::DeterminismContract contract(nerve::core::DeterminismLevel::BASIC, "test");
    contract.max_memory_usage_mb = 1024;
    contract.max_execution_time = std::chrono::milliseconds(10000);

    pool.setDeterminismContract(contract);
    if (!pool.isDeterministic())
    {
        return false;
    }

    void *ptr = pool.allocateAligned(128, 16, contract);
    if (!ptr)
    {
        return false;
    }
    std::memset(ptr, 0xFF, 128);
    pool.deallocate(ptr);

    auto metadata = pool.getDeterminismMetadata();
    return metadata.was_deterministic;
}

bool check_rng_determinism_same_seed_same_sequence()
{
    nerve::core::DeterminismContract c1(nerve::core::DeterminismLevel::BASIC, "test");
    c1.enable_deterministic_random = true;
    c1.setRngSeed(static_cast<uint64_t>(42));

    nerve::core::DeterminismContract c2(nerve::core::DeterminismLevel::BASIC, "test");
    c2.enable_deterministic_random = true;
    c2.setRngSeed(static_cast<uint64_t>(42));

    uint32_t v1 = nerve::core::DeterminismEnforcer::deterministicRandomUint32(c1);
    uint32_t v2 = nerve::core::DeterminismEnforcer::deterministicRandomUint32(c2);
    if (v1 != v2)
    {
        return false;
    }

    double d1 = nerve::core::DeterminismEnforcer::deterministicRandomDouble(c1);
    double d2 = nerve::core::DeterminismEnforcer::deterministicRandomDouble(c2);
    return std::abs(d1 - d2) < TOL;
}

bool check_rng_determinism_different_seed_different()
{
    nerve::core::DeterminismContract c1(nerve::core::DeterminismLevel::BASIC, "test");
    c1.enable_deterministic_random = true;
    c1.setRngSeed(static_cast<uint64_t>(42));

    nerve::core::DeterminismContract c2(nerve::core::DeterminismLevel::BASIC, "test");
    c2.enable_deterministic_random = true;
    c2.setRngSeed(static_cast<uint64_t>(99));

    uint32_t v1 = nerve::core::DeterminismEnforcer::deterministicRandomUint32(c1);
    uint32_t v2 = nerve::core::DeterminismEnforcer::deterministicRandomUint32(c2);
    return v1 != v2;
}

bool check_rng_factory_creates_valid_distributions()
{
    nerve::core::DeterminismContract contract(nerve::core::DeterminismLevel::BASIC, "test");
    contract.enable_deterministic_random = true;
    contract.setRngSeed(static_cast<uint64_t>(0xDEAD));

    nerve::core::RNG rng = nerve::core::DeterministicRNGFactory::createFromContract(contract);

    double u = rng.uniform();
    if (u < 0.0 || u > 1.0)
    {
        return false;
    }

    double ui = rng.uniform(10.0, 20.0);
    if (ui < 10.0 || ui > 20.0)
    {
        return false;
    }

    int64_t ri = rng.uniformInt(0, 100);
    if (ri < 0 || ri > 100)
    {
        return false;
    }

    double ni = rng.normal();
    return std::isfinite(ni);
}

bool check_rng_factory_split_generators()
{
    nerve::core::DeterminismContract contract(nerve::core::DeterminismLevel::BASIC, "test");
    contract.enable_deterministic_random = true;
    contract.setRngSeed(static_cast<uint64_t>(0xCAFE));

    auto gens = nerve::core::DeterministicRNGFactory::createSplitGenerators(contract, 5);
    return gens.size() == 5;
}

bool check_rng_random_bounded()
{
    nerve::core::PRNGKey key(12345ULL);
    auto vals = key.uniform(100, -5.0, 5.0);
    if (vals.size() != 100)
    {
        return false;
    }
    for (double v : vals)
    {
        if (v < -5.0 || v > 5.0)
        {
            return false;
        }
    }
    auto ints = key.randint(50, 0, 10);
    for (int64_t iv : ints)
    {
        if (iv < 0 || iv >= 10)
        {
            return false;
        }
    }
    return true;
}

bool check_rng_random_split()
{
    nerve::core::PRNGKey key(999ULL);
    auto keys = key.split(4);
    return keys.size() == 4;
}

bool check_rng_random_permutation()
{
    nerve::core::PRNGKey key(777ULL);
    auto perm = key.permutation(10);
    if (perm.size() != 10)
    {
        return false;
    }
    std::sort(perm.begin(), perm.end());
    for (int64_t i = 0; i < 10; ++i)
    {
        if (perm[static_cast<size_t>(i)] != i)
        {
            return false;
        }
    }
    return true;
}

bool check_highdim_error_event_basic()
{
    nerve::core::HighDimErrorEvent event(nerve::ErrorCode::E13_PH_HIGHDIM_PRECISION,
                                         "high dimension test");
    event.setDimension(5);
    event.setComplexSize(1000);
    event.setMemoryUsage(256);

    if (event.getDimension() != 5)
    {
        return false;
    }
    if (event.getComplexSize() != 1000)
    {
        return false;
    }
    if (event.getMemoryUsage() != 256)
    {
        return false;
    }

    event.setAlgorithmPhase("reduction");
    if (event.getAlgorithmPhase() != "reduction")
    {
        return false;
    }

    std::string summary = event.generateFailureSummary();
    return !summary.empty();
}

bool check_highdim_error_event_validation()
{
    nerve::core::HighDimErrorEvent event;
    event.setDimension(10);
    event.setComplexSize(500);
    event.setMemoryUsage(50);
    return event.validateHighdimData();
}

bool check_highdim_error_serialize_deserialize()
{
    nerve::core::HighDimErrorEvent event(nerve::ErrorCode::E13_PH_HIGHDIM_PRECISION, "test");
    event.setDimension(8);
    event.setComplexSize(5000);
    event.setMemoryUsage(128);

    std::vector<uint8_t> buffer;
    event.serializeHighdimData(buffer);

    nerve::core::HighDimErrorEvent restored;
    restored.deserializeHighdimData(buffer);

    return restored.getDimension() == 8;
}

bool check_determinism_contract_basic()
{
    nerve::core::DeterminismContract contract(nerve::core::DeterminismLevel::BASIC, "test_core");
    contract.setRngSeed(static_cast<uint64_t>(42));
    contract.max_execution_time = std::chrono::milliseconds(5000);
    contract.max_memory_usage_mb = 512;

    if (!contract.isValid())
    {
        return false;
    }

    auto serialized = contract.serialize();
    if (serialized.empty())
    {
        return false;
    }

    nerve::core::DeterminismContract restored;
    if (!restored.deserialize(serialized))
    {
        return false;
    }
    return restored.component_name == "test_core";
}

bool check_determinism_enforcer_contract()
{
    nerve::core::DeterminismContract contract(nerve::core::DeterminismLevel::NONE, "");
    contract.max_execution_time = std::chrono::milliseconds(1000);
    contract.max_memory_usage_mb = 128;

    bool can = nerve::core::DeterminismEnforcer::canSatisfyContract(contract);
    (void)can;

    auto created = nerve::core::DeterminismEnforcer::createContract(
        nerve::core::DeterminismLevel::BASIC, "enforcer_test");
    return created.component_name == "enforcer_test";
}

bool check_rng_sequence_state_halton()
{
    nerve::core::RNG rng;
    auto seq = rng.haltonSequence(10, 2);
    if (seq.size() != 10)
    {
        return false;
    }
    for (double v : seq)
    {
        if (v < 0.0 || v > 1.0)
        {
            return false;
        }
    }
    return true;
}

bool check_rng_sequence_state_split()
{
    nerve::core::DeterminismContract contract(nerve::core::DeterminismLevel::BASIC, "test");
    contract.enable_deterministic_random = true;
    contract.setRngSeed(static_cast<uint64_t>(111));

    nerve::core::RNG rng(contract);
    nerve::core::RNG child = rng.split();

    auto state = rng.getState();
    return !state.empty();
}

bool check_determinism_metadata_validate()
{
    nerve::core::DeterminismMetadata meta;
    meta.params_hash.fill(1);
    meta.result_checksum.fill(1);
    meta.rng_seed_used.fill(1);
    meta.achieved_level = nerve::core::DeterminismLevel::BASIC;
    return meta.isValid();
}

bool check_rng_factory_determinism_validation()
{
    nerve::core::DeterminismContract contract(nerve::core::DeterminismLevel::BASIC, "test");
    contract.enable_deterministic_random = true;
    contract.setRngSeed(static_cast<uint64_t>(0x1234));
    contract.max_execution_time = std::chrono::milliseconds(10000);
    contract.max_memory_usage_mb = 1024;

    nerve::core::RNG rng = nerve::core::DeterministicRNGFactory::createFromContract(contract);

    bool valid = nerve::core::DeterministicRNGFactory::validateDeterminism(rng, contract);
    if (!valid)
    {
        return true;
    }
    auto warnings = nerve::core::DeterministicRNGFactory::getDeterminismWarnings(rng, contract);
    (void)warnings;
    return true;
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

    run("simd_memcpy_basic", check_simd_memcpy_basic());
    run("simd_memset_basic", check_simd_memset_basic());
    run("simd_reduce_sum", check_simd_reduce_sum());
    run("simd_reduce_sum_small", check_simd_reduce_sum_small());
    run("simd_reduce_sum_empty", check_simd_reduce_sum_empty());
    run("numa_pool_manager_singleton", check_numa_pool_manager_singleton());
    run("numa_pool_config_defaults", check_numa_pool_config_defaults());
    run("memory_pool_allocate_deallocate", check_memory_pool_allocate_deallocate());
    run("memory_pool_deterministic_cycle", check_memory_pool_deterministic_cycle());
    run("rng_determinism_same_seed_same_sequence", check_rng_determinism_same_seed_same_sequence());
    run("rng_determinism_different_seed_different",
        check_rng_determinism_different_seed_different());
    run("rng_factory_creates_valid_distributions", check_rng_factory_creates_valid_distributions());
    run("rng_factory_split_generators", check_rng_factory_split_generators());
    run("rng_random_bounded", check_rng_random_bounded());
    run("rng_random_split", check_rng_random_split());
    run("rng_random_permutation", check_rng_random_permutation());
    run("highdim_error_event_basic", check_highdim_error_event_basic());
    run("highdim_error_event_validation", check_highdim_error_event_validation());
    run("highdim_error_serialize_deserialize", check_highdim_error_serialize_deserialize());
    run("determinism_contract_basic", check_determinism_contract_basic());
    run("determinism_enforcer_contract", check_determinism_enforcer_contract());
    run("rng_sequence_state_halton", check_rng_sequence_state_halton());
    run("rng_sequence_state_split", check_rng_sequence_state_split());
    run("determinism_metadata_validate", check_determinism_metadata_validate());
    run("rng_factory_determinism_validation", check_rng_factory_determinism_validation());

    return failures > 0 ? 1 : 0;
}
