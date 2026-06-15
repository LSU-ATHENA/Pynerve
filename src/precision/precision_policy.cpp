
#include "nerve/precision/precision_policy.hpp"

#include <algorithm>
#include <stdexcept>

namespace nerve::precision
{
PrecisionPolicyManager::PrecisionPolicyManager()
    : global_level_(PrecisionLevel::CONSERVATIVE)
    , strict_fp_mode_(false)
{
    initializeDefaultPolicies();
}
void PrecisionPolicyManager::initializeDefaultPolicies()
{
    registerAlgorithmPolicy(AlgorithmPolicy("PH4_EXACT", PrecisionLevel::CONSERVATIVE,
                                            NumericalPrecision::FP64, NumericalPrecision::FP64,
                                            false, false, 1e-15, true));
    registerAlgorithmPolicy(AlgorithmPolicy("PH4_SPARSE", PrecisionLevel::BALANCED,
                                            NumericalPrecision::FP64, NumericalPrecision::FP32,
                                            true, true, 1e-12, true));
    registerAlgorithmPolicy(AlgorithmPolicy("PH4_WITNESS", PrecisionLevel::PERFORMANCE,
                                            NumericalPrecision::FP32, NumericalPrecision::FP16,
                                            true, true, 1e-10, true));
    registerAlgorithmPolicy(AlgorithmPolicy("ZIGZAG_PH", PrecisionLevel::BALANCED,
                                            NumericalPrecision::FP64, NumericalPrecision::FP32,
                                            false, true, 1e-11, true));
    registerAlgorithmPolicy(AlgorithmPolicy("INCREMENTAL_PH", PrecisionLevel::CONSERVATIVE,
                                            NumericalPrecision::FP64, NumericalPrecision::FP64,
                                            false, false, 1e-15, true));
    registerAlgorithmPolicy(AlgorithmPolicy("STREAMING_PH", PrecisionLevel::BALANCED,
                                            NumericalPrecision::FP64, NumericalPrecision::FP32,
                                            true, true, 1e-12, true));
    registerAlgorithmPolicy(AlgorithmPolicy("LAPLACIAN_EIGEN", PrecisionLevel::BALANCED,
                                            NumericalPrecision::FP64, NumericalPrecision::FP32,
                                            true, true, 1e-13, true));
    registerAlgorithmPolicy(AlgorithmPolicy("DISTANCE_METRICS", PrecisionLevel::PERFORMANCE,
                                            NumericalPrecision::FP32, NumericalPrecision::FP16,
                                            true, true, 1e-8, true));
    registerAlgorithmPolicy(AlgorithmPolicy("BOUNDARY_MATRIX", PrecisionLevel::CONSERVATIVE,
                                            NumericalPrecision::FP64, NumericalPrecision::FP64,
                                            false, false, 1e-14, true));
}
void PrecisionPolicyManager::setGlobalPrecisionLevel(PrecisionLevel level)
{
    global_level_ = level;
    for (auto &[name, policy] : algorithm_policies_)
    {
        auto &state = precision_states_[name];
        state.current_level = level;
        state.current_precision = precisionForLevel(level);
        if (state.current_precision < policy.maxPrecision)
        {
            recordDowngrade(name, DowngradeReason::USER_REQUEST, policy.maxPrecision,
                            state.current_precision);
        }
    }
}
void PrecisionPolicyManager::setStrictFpMode(bool strict)
{
    strict_fp_mode_ = strict;
    if (strict)
    {
        for (auto &[name, policy] : algorithm_policies_)
        {
            auto &state = precision_states_[name];
            NumericalPrecision max_prec = policy.maxPrecision;
            if (state.current_precision < max_prec)
            {
                recordDowngrade(name, DowngradeReason::USER_REQUEST, state.current_precision,
                                max_prec);
                state.current_precision = max_prec;
            }
        }
    }
}
void PrecisionPolicyManager::registerAlgorithmPolicy(const AlgorithmPolicy &policy)
{
    algorithm_policies_.insert_or_assign(policy.algorithmName, policy);
    PrecisionState state;
    state.current_level = global_level_;
    state.current_precision = precisionForLevel(global_level_);
    state.last_downgrade_reason = DowngradeReason::NONE;
    state.numerical_stability_estimate = policy.stabilityThreshold;
    state.strict_fp_mode = false;
    state.downgrade_count = 0;
    precision_states_[policy.algorithmName] = state;
}
NumericalPrecision
PrecisionPolicyManager::getAlgorithmPrecision(const std::string &algorithm_name) const
{
    auto it = precision_states_.find(algorithm_name);
    if (it != precision_states_.end())
    {
        return it->second.current_precision;
    }
    return NumericalPrecision::FP64;
}
PrecisionPolicyManager::PrecisionState
PrecisionPolicyManager::getPrecisionState(const std::string &algorithm_name) const
{
    auto it = precision_states_.find(algorithm_name);
    if (it != precision_states_.end())
    {
        return it->second;
    }
    PrecisionState default_state;
    default_state.current_precision = NumericalPrecision::FP64;
    default_state.current_level = PrecisionLevel::CONSERVATIVE;
    default_state.last_downgrade_reason = DowngradeReason::NONE;
    default_state.numerical_stability_estimate = 1e-15;
    default_state.strict_fp_mode = false;
    default_state.downgrade_count = 0;
    return default_state;
}
std::vector<PrecisionPolicyManager::PolicyViolation>
PrecisionPolicyManager::getPolicyViolations() const
{
    return policy_violations_;
}
void PrecisionPolicyManager::updateStabilityCertificateForDowngrade(
    const std::string &algorithm_name, persistence::StabilityCertificate &certificate,
    DowngradeReason reason)
{
    auto it = precision_states_.find(algorithm_name);
    if (it == precision_states_.end())
    {
        return;
    }
    const auto &state = it->second;
    const auto &policy = algorithm_policies_.at(algorithm_name);
    double stability_impact =
        computeStabilityImpact(reason, policy.maxPrecision, state.current_precision);
    if (!certificate.isValid() ||
        stability_impact + certificate.getNumericalResidual() > policy.stabilityThreshold)
    {
        PolicyViolation violation;
        violation.algorithmName = algorithm_name;
        violation.requested_precision = state.current_precision;
        violation.actual_precision = state.current_precision;
        violation.reason = reason;
        violation.description = "Precision downgrade exceeds stability certificate tolerance";
        policy_violations_.push_back(violation);
    }
}
bool PrecisionPolicyManager::validatePrecisionUsage(const std::string &algorithm_name,
                                                    NumericalPrecision used_precision) const
{
    auto it = algorithm_policies_.find(algorithm_name);
    if (it == algorithm_policies_.end())
    {
        return true;
    }
    const auto &policy = it->second;
    if (used_precision < policy.minPrecision || used_precision > policy.maxPrecision)
    {
        return false;
    }
    if (used_precision == NumericalPrecision::MIXED && !policy.allowMixedPrecision)
    {
        return false;
    }
    return true;
}
void PrecisionPolicyManager::enforcePrecisionPolicy(const std::string &algorithm_name,
                                                    NumericalPrecision requested_precision)
{
    auto it = algorithm_policies_.find(algorithm_name);
    if (it == algorithm_policies_.end())
    {
        return;
    }
    const auto &policy = it->second;
    auto &state = precision_states_[algorithm_name];
    if (!isPrecisionAllowed(algorithm_name, requested_precision))
    {
        PolicyViolation violation;
        violation.algorithmName = algorithm_name;
        violation.requested_precision = requested_precision;
        violation.actual_precision = state.current_precision;
        violation.reason = DowngradeReason::ALGORITHM_LIMITATION;
        violation.description = "Requested precision not allowed by algorithm policy";
        policy_violations_.push_back(violation);
        if (requested_precision > policy.maxPrecision)
        {
            recordDowngrade(algorithm_name, DowngradeReason::ALGORITHM_LIMITATION,
                            requested_precision, policy.maxPrecision);
            state.current_precision = policy.maxPrecision;
        }
        return;
    }
    if (requested_precision < state.current_precision)
    {
        if (!policy.allowAutomaticDowngrade && !strict_fp_mode_)
        {
            PolicyViolation violation;
            violation.algorithmName = algorithm_name;
            violation.requested_precision = requested_precision;
            violation.actual_precision = state.current_precision;
            violation.reason = DowngradeReason::ALGORITHM_LIMITATION;
            violation.description = "Automatic precision downgrade not allowed";
            policy_violations_.push_back(violation);
            return;
        }
        recordDowngrade(algorithm_name, DowngradeReason::PERFORMANCE_REQUIREMENT,
                        state.current_precision, requested_precision);
        state.current_precision = requested_precision;
    }
    else
    {
        state.current_precision = requested_precision;
        state.last_downgrade_reason = DowngradeReason::NONE;
    }
}
NumericalPrecision PrecisionPolicyManager::precisionForLevel(PrecisionLevel level) const
{
    switch (level)
    {
        case PrecisionLevel::CONSERVATIVE:
            return NumericalPrecision::FP64;
        case PrecisionLevel::BALANCED:
            return NumericalPrecision::MIXED;
        case PrecisionLevel::PERFORMANCE:
            return NumericalPrecision::FP32;
        case PrecisionLevel::ADVANCED:
            return NumericalPrecision::FP16;
        default:
            return NumericalPrecision::FP64;
    }
}
bool PrecisionPolicyManager::isPrecisionAllowed(const std::string &algorithm_name,
                                                NumericalPrecision precision) const
{
    auto it = algorithm_policies_.find(algorithm_name);
    if (it == algorithm_policies_.end())
    {
        return true;
    }
    const auto &policy = it->second;
    if (precision < policy.minPrecision || precision > policy.maxPrecision)
    {
        return false;
    }
    if (precision == NumericalPrecision::MIXED && !policy.allowMixedPrecision)
    {
        return false;
    }
    return true;
}
void PrecisionPolicyManager::recordDowngrade(const std::string &algorithm_name,
                                             DowngradeReason reason,
                                             NumericalPrecision from_precision,
                                             NumericalPrecision to_precision)
{
    auto &state = precision_states_[algorithm_name];
    state.last_downgrade_reason = reason;
    state.downgrade_count++;
    double stability_impact = computeStabilityImpact(reason, from_precision, to_precision);
    state.numerical_stability_estimate += stability_impact;
}
double PrecisionPolicyManager::computeStabilityImpact(DowngradeReason reason,
                                                      NumericalPrecision from_precision,
                                                      NumericalPrecision to_precision) const
{
    double impact = 0.0;
    switch (reason)
    {
        case DowngradeReason::MEMORY_PRESSURE:
            impact = 1e-8;
            break;
        case DowngradeReason::PERFORMANCE_REQUIREMENT:
            impact = 1e-10;
            break;
        case DowngradeReason::NUMERICAL_INSTABILITY:
            impact = 1e-6;
            break;
        case DowngradeReason::USER_REQUEST:
            impact = 0.0;
            break;
        case DowngradeReason::ALGORITHM_LIMITATION:
            impact = 1e-9;
            break;
        case DowngradeReason::BUDGET_CONSTRAINT:
            impact = 1e-7;
            break;
        case DowngradeReason::NONE:
        default:
            impact = 0.0;
            break;
    }
    int precision_diff = static_cast<int>(from_precision) - static_cast<int>(to_precision);
    impact *= std::abs(precision_diff);
    return impact;
}
} // namespace nerve::precision
