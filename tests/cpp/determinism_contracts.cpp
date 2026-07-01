#include "nerve/common/accelerated_types.hpp"
#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/core/rng/determinism_contract.hpp"
#include "nerve/core/rng/rng.hpp"
#include "nerve/core_types.hpp"

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

using nerve::core::BufferView;

constexpr double kTol = 1e-10;

bool check_same_seed_same_sequence()
{
    nerve::core::RNG rng1(42);
    nerve::core::RNG rng2(42);

    std::vector<double> seq1, seq2;
    for (int i = 0; i < 100; ++i)
    {
        seq1.push_back(rng1.uniform());
        seq2.push_back(rng2.uniform());
    }

    for (int i = 0; i < 100; ++i)
    {
        if (std::abs(seq1[i] - seq2[i]) > kTol)
        {
            std::cerr << "same seeds produced different values at " << i << "\n";
            return false;
        }
    }

    return true;
}

bool check_different_seed_different_sequence()
{
    nerve::core::RNG rng1(42);
    nerve::core::RNG rng2(99);

    std::vector<double> seq1, seq2;
    for (int i = 0; i < 10; ++i)
    {
        seq1.push_back(rng1.uniform());
        seq2.push_back(rng2.uniform());
    }

    bool all_same = true;
    for (int i = 0; i < 10; ++i)
    {
        if (std::abs(seq1[i] - seq2[i]) > kTol)
        {
            all_same = false;
            break;
        }
    }

    if (all_same)
    {
        std::cerr << "different seeds should produce different sequences\n";
        return false;
    }

    return true;
}

bool check_rng_seed_get_set()
{
    nerve::core::RNG rng(42);

    auto seed_val = rng.seed();
    if (seed_val != 42)
    {
        std::cerr << "expected seed 42, got " << seed_val << "\n";
        return false;
    }

    rng.seed(100);
    if (rng.seed() != 100)
    {
        std::cerr << "seed should be 100 after reseeding\n";
        return false;
    }

    return true;
}

bool check_rng_uniform_range()
{
    nerve::core::RNG rng(42);

    for (int i = 0; i < 1000; ++i)
    {
        double val = rng.uniform(-5.0, 5.0);
        if (val < -5.0 || val > 5.0)
        {
            std::cerr << "uniform value out of range: " << val << "\n";
            return false;
        }
    }

    return true;
}

bool check_rng_normal_distribution()
{
    nerve::core::RNG rng(42);

    for (int i = 0; i < 100; ++i)
    {
        double val = rng.normal(0.0, 1.0);
        if (!std::isfinite(val))
        {
            std::cerr << "normal produced non-finite value\n";
            return false;
        }
    }

    return true;
}

bool check_rng_uniform_int()
{
    nerve::core::RNG rng(42);

    for (int i = 0; i < 100; ++i)
    {
        auto val = rng.uniformInt(0, 10);
        if (val < 0 || val > 10)
        {
            std::cerr << "uniformInt out of range: " << val << "\n";
            return false;
        }
    }

    return true;
}

bool check_determinism_contract_construction()
{
    nerve::core::DeterminismContract contract;
    if (contract.level != nerve::core::DeterminismLevel::BASIC)
    {
        std::cerr << "default level should be BASIC\n";
        return false;
    }
    if (!contract.isValid())
    {
        std::cerr << "default contract should be valid\n";
        return false;
    }

    nerve::core::DeterminismContract strict(nerve::core::DeterminismLevel::STRICT, "test");
    if (strict.level != nerve::core::DeterminismLevel::STRICT)
    {
        std::cerr << "level should be STRICT\n";
        return false;
    }
    if (strict.component_name != "test")
    {
        std::cerr << "component name should be 'test'\n";
        return false;
    }

    return true;
}

bool check_determinism_contract_rng_seed()
{
    nerve::core::DeterminismContract contract;
    contract.setRngSeed(uint64_t(12345));

    if (!contract.rng_seed_provided)
    {
        std::cerr << "rng_seed_provided should be true after setRngSeed\n";
        return false;
    }

    if (contract.validationErrors().empty())
    {
        nerve::core::RNG rng(contract);
        if (rng.seed() != 12345)
        {
            std::cerr << "contract-based RNG seed mismatch\n";
            return false;
        }

        if (!rng.isDeterministic())
        {
            std::cerr << "contract-based RNG should be deterministic\n";
            return false;
        }
    }

    return true;
}

bool check_determinism_enforcer()
{
    auto contract = nerve::core::DeterminismEnforcer::createContract(
        nerve::core::DeterminismLevel::STRICT, "test_component");

    if (contract.level != nerve::core::DeterminismLevel::STRICT)
    {
        std::cerr << "enforcer created wrong level\n";
        return false;
    }

    auto violations = nerve::core::DeterminismEnforcer::getContractViolations(contract);
    static_cast<void>(violations);

    auto capable = nerve::core::DeterminismEnforcer::canSatisfyContract(contract);
    static_cast<void>(capable);

    return true;
}

bool check_determinism_context()
{
    nerve::core::DeterminismContract contract(nerve::core::DeterminismLevel::AUDIT, "audit_test");

    nerve::core::DeterminismContext ctx(contract);

    if (ctx.contract().level != nerve::core::DeterminismLevel::AUDIT)
    {
        std::cerr << "context contract level mismatch\n";
        return false;
    }

    if (ctx.contract().component_name != "audit_test")
    {
        std::cerr << "context component name mismatch\n";
        return false;
    }

    return true;
}

} // namespace

int main()
{
    if (!check_same_seed_same_sequence())
    {
        std::cerr << "FAIL: same seed same sequence\n";
        return 1;
    }
    if (!check_different_seed_different_sequence())
    {
        std::cerr << "FAIL: different seed different sequence\n";
        return 1;
    }
    if (!check_rng_seed_get_set())
    {
        std::cerr << "FAIL: rng seed get set\n";
        return 1;
    }
    if (!check_rng_uniform_range())
    {
        std::cerr << "FAIL: rng uniform range\n";
        return 1;
    }
    if (!check_rng_normal_distribution())
    {
        std::cerr << "FAIL: rng normal distribution\n";
        return 1;
    }
    if (!check_rng_uniform_int())
    {
        std::cerr << "FAIL: rng uniform int\n";
        return 1;
    }
    if (!check_determinism_contract_construction())
    {
        std::cerr << "FAIL: determinism contract construction\n";
        return 1;
    }
    if (!check_determinism_contract_rng_seed())
    {
        std::cerr << "FAIL: determinism contract rng seed\n";
        return 1;
    }
    if (!check_determinism_enforcer())
    {
        std::cerr << "FAIL: determinism enforcer\n";
        return 1;
    }
    if (!check_determinism_context())
    {
        std::cerr << "FAIL: determinism context\n";
        return 1;
    }
    return 0;
}
