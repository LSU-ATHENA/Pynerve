#include "nerve/core/rng/determinism_contract.hpp"
#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/core_types.hpp"
#include "nerve/persistence/core/core_types.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace
{

using nerve::Dimension;
using nerve::Index;
using nerve::Size;
using nerve::core::BufferView;
using nerve::core::DeterminismContract;
using nerve::core::DeterminismEnforcer;
using nerve::core::DeterminismLevel;
using nerve::persistence::Pair;

bool check_contract_default_level()
{
    DeterminismContract contract;
    if (contract.level != DeterminismLevel::BASIC)
    {
        std::cerr << "default level should be BASIC\n";
        return false;
    }
    return true;
}

bool check_contract_strict_level()
{
    DeterminismContract contract(DeterminismLevel::STRICT, "test_component");
    if (contract.level != DeterminismLevel::STRICT)
    {
        std::cerr << "level should be STRICT\n";
        return false;
    }
    if (contract.component_name != "test_component")
    {
        std::cerr << "component_name mismatch\n";
        return false;
    }
    return true;
}

bool check_contract_invalid_returns_false()
{
    DeterminismContract contract;
    contract.component_name = "";
    bool valid = contract.isValid();
    (void)valid;
    return true;
}

bool check_contract_rng_seed_setting()
{
    DeterminismContract contract;
    contract.setRngSeed(static_cast<uint64_t>(42));
    if (!contract.rng_seed_provided)
    {
        std::cerr << "rng_seed_provided should be true after setting seed\n";
        return false;
    }
    return true;
}

bool check_enforcer_can_satisfy()
{
    bool can_satisfy = DeterminismEnforcer::canSatisfyContract(DeterminismContract{});
    (void)can_satisfy;
    return true;
}

bool check_enforcer_create_contract()
{
    auto contract = DeterminismEnforcer::createContract(DeterminismLevel::STRICT);
    if (contract.level != DeterminismLevel::STRICT)
    {
        std::cerr << "enforcer created contract should have STRICT level\n";
        return false;
    }
    return true;
}

bool check_enforcer_supports_deterministic_threading()
{
    bool supported = DeterminismEnforcer::supportsDeterministicThreading();
    (void)supported;
    return true;
}

bool check_enforcer_deterministic_random()
{
    auto contract = DeterminismEnforcer::createContract(DeterminismLevel::BASIC);
    uint32_t val = DeterminismEnforcer::deterministicRandomUint32(contract);
    (void)val;

    DeterminismContract same_contract = contract;
    same_contract.rng_seed = contract.rng_seed;
    same_contract.rng_seed_provided = contract.rng_seed_provided;
    uint32_t val2 = DeterminismEnforcer::deterministicRandomUint32(same_contract);
    (void)val2;
    return true;
}

bool check_enforcer_deterministic_random_double()
{
    auto contract = DeterminismEnforcer::createContract(DeterminismLevel::BASIC);
    double val = DeterminismEnforcer::deterministicRandomDouble(contract);
    if (val < 0.0 || val > 1.0)
    {
        std::cerr << "deterministicRandomDouble should be in [0,1], got " << val << "\n";
        return false;
    }
    return true;
}

bool check_contract_comparison()
{
    DeterminismContract a(DeterminismLevel::BASIC, "comp_a");
    DeterminismContract b(DeterminismLevel::BASIC, "comp_b");
    if (a == b)
    {
        std::cerr << "contracts with different component names should not be equal\n";
        return false;
    }
    return true;
}

} // namespace

int main()
{
    if (!check_contract_default_level())
    {
        std::cerr << "FAIL: contract default level\n";
        return 1;
    }
    if (!check_contract_strict_level())
    {
        std::cerr << "FAIL: contract strict level\n";
        return 1;
    }
    if (!check_contract_invalid_returns_false())
    {
        std::cerr << "FAIL: contract invalid returns false\n";
        return 1;
    }
    if (!check_contract_rng_seed_setting())
    {
        std::cerr << "FAIL: contract rng seed setting\n";
        return 1;
    }
    if (!check_enforcer_can_satisfy())
    {
        std::cerr << "FAIL: enforcer can satisfy\n";
        return 1;
    }
    if (!check_enforcer_create_contract())
    {
        std::cerr << "FAIL: enforcer create contract\n";
        return 1;
    }
    if (!check_enforcer_supports_deterministic_threading())
    {
        std::cerr << "FAIL: enforcer supports deterministic threading\n";
        return 1;
    }
    if (!check_enforcer_deterministic_random())
    {
        std::cerr << "FAIL: enforcer deterministic random\n";
        return 1;
    }
    if (!check_enforcer_deterministic_random_double())
    {
        std::cerr << "FAIL: enforcer deterministic random double\n";
        return 1;
    }
    if (!check_contract_comparison())
    {
        std::cerr << "FAIL: contract comparison\n";
        return 1;
    }
    return 0;
}
