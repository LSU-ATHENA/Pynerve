#include "nerve/precision/precision_policy.hpp"

#include <iostream>
#include <string>

namespace
{

bool check_algorithm_policy_construction()
{
    nerve::precision::AlgorithmPolicy policy("TEST", nerve::precision::PrecisionLevel::BALANCED,
                                             nerve::precision::NumericalPrecision::FP64,
                                             nerve::precision::NumericalPrecision::FP32, true, true,
                                             1e-10, true);
    if (policy.algorithmName != "TEST")
    {
        std::cerr << "algorithm name should be TEST\n";
        return false;
    }
    if (policy.defaultLevel != nerve::precision::PrecisionLevel::BALANCED)
    {
        std::cerr << "default level should be BALANCED\n";
        return false;
    }
    if (policy.maxPrecision != nerve::precision::NumericalPrecision::FP64)
    {
        std::cerr << "max precision should be FP64\n";
        return false;
    }
    if (policy.minPrecision != nerve::precision::NumericalPrecision::FP32)
    {
        std::cerr << "min precision should be FP32\n";
        return false;
    }
    if (policy.allowMixedPrecision != true)
    {
        std::cerr << "allowMixedPrecision should be true\n";
        return false;
    }
    if (policy.allowAutomaticDowngrade != true)
    {
        std::cerr << "allowAutomaticDowngrade should be true\n";
        return false;
    }
    if (policy.stabilityThreshold != 1e-10)
    {
        std::cerr << "stabilityThreshold should be 1e-10\n";
        return false;
    }
    if (policy.requiresValidation != true)
    {
        std::cerr << "requiresValidation should be true\n";
        return false;
    }
    return true;
}

bool check_precision_level_enum_ordering()
{
    int conservative = static_cast<int>(nerve::precision::PrecisionLevel::CONSERVATIVE);
    int balanced = static_cast<int>(nerve::precision::PrecisionLevel::BALANCED);
    int performance = static_cast<int>(nerve::precision::PrecisionLevel::PERFORMANCE);
    int advanced = static_cast<int>(nerve::precision::PrecisionLevel::ADVANCED);

    if (conservative >= balanced || balanced >= performance || performance >= advanced)
    {
        std::cerr << "PrecisionLevel should be ordered CONSERVATIVE < BALANCED < PERFORMANCE < "
                     "ADVANCED\n";
        return false;
    }
    return true;
}

bool check_numerical_precision_enum_ordering()
{
    int fp64 = static_cast<int>(nerve::precision::NumericalPrecision::FP64);
    int fp32 = static_cast<int>(nerve::precision::NumericalPrecision::FP32);
    int fp16 = static_cast<int>(nerve::precision::NumericalPrecision::FP16);
    int mixed = static_cast<int>(nerve::precision::NumericalPrecision::MIXED);
    int adaptive = static_cast<int>(nerve::precision::NumericalPrecision::ADAPTIVE);

    if (fp64 >= fp32 || fp32 >= fp16 || fp16 >= mixed || mixed >= adaptive)
    {
        std::cerr << "NumericalPrecision should be FP64 < FP32 < FP16 < MIXED < ADAPTIVE\n";
        return false;
    }
    return true;
}

bool check_downgrade_reason_enum_valid()
{
    int none = static_cast<int>(nerve::precision::DowngradeReason::NONE);
    int mem = static_cast<int>(nerve::precision::DowngradeReason::MEMORY_PRESSURE);
    int perf = static_cast<int>(nerve::precision::DowngradeReason::PERFORMANCE_REQUIREMENT);
    int num = static_cast<int>(nerve::precision::DowngradeReason::NUMERICAL_INSTABILITY);
    int user = static_cast<int>(nerve::precision::DowngradeReason::USER_REQUEST);
    int algo = static_cast<int>(nerve::precision::DowngradeReason::ALGORITHM_LIMITATION);
    int budget = static_cast<int>(nerve::precision::DowngradeReason::BUDGET_CONSTRAINT);

    if (none < 0 || mem < 0 || perf < 0 || num < 0 || user < 0 || algo < 0 || budget < 0)
    {
        std::cerr << "all DowngradeReason values should be non-negative\n";
        return false;
    }
    return true;
}

bool check_policy_getters_return_correct_defaults()
{
    nerve::precision::PrecisionPolicyManager manager;
    if (manager.getGlobalLevel() != nerve::precision::PrecisionLevel::CONSERVATIVE)
    {
        std::cerr << "default global level should be CONSERVATIVE\n";
        return false;
    }
    if (manager.isStrictFpMode() != false)
    {
        std::cerr << "default strict fp mode should be false\n";
        return false;
    }

    auto prec = manager.getAlgorithmPrecision("PH4_EXACT");
    if (prec != nerve::precision::NumericalPrecision::FP64)
    {
        std::cerr << "PH4_EXACT default precision should be FP64\n";
        return false;
    }

    auto state = manager.getPrecisionState("PH4_EXACT");
    if (state.current_level != nerve::precision::PrecisionLevel::CONSERVATIVE)
    {
        std::cerr << "PH4_EXACT default level should be CONSERVATIVE\n";
        return false;
    }
    return true;
}

} // namespace

int main()
{
    if (!check_algorithm_policy_construction())
    {
        std::cerr << "FAIL: algorithm policy construction\n";
        return 1;
    }
    if (!check_precision_level_enum_ordering())
    {
        std::cerr << "FAIL: precision level enum ordering\n";
        return 1;
    }
    if (!check_numerical_precision_enum_ordering())
    {
        std::cerr << "FAIL: numerical precision enum ordering\n";
        return 1;
    }
    if (!check_downgrade_reason_enum_valid())
    {
        std::cerr << "FAIL: downgrade reason enum valid\n";
        return 1;
    }
    if (!check_policy_getters_return_correct_defaults())
    {
        std::cerr << "FAIL: policy getters return correct defaults\n";
        return 1;
    }
    return 0;
}
