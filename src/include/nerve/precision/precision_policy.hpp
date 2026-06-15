
#pragma once
#include "nerve/errors/errors.hpp"
#include "nerve/persistence/kernels/ph4_ops.hpp"

#include <cassert>
#include <memory>
#include <string>
#include <unordered_map>

namespace nerve::precision
{
enum class PrecisionLevel
{
    CONSERVATIVE = 0,
    BALANCED = 1,
    PERFORMANCE = 2,
    ADVANCED = 3
};
enum class NumericalPrecision
{
    FP64 = 0,
    FP32 = 1,
    FP16 = 2,
    MIXED = 3,
    ADAPTIVE = 4
};
enum class DowngradeReason
{
    NONE = 0,
    MEMORY_PRESSURE = 1,
    PERFORMANCE_REQUIREMENT = 2,
    NUMERICAL_INSTABILITY = 3,
    USER_REQUEST = 4,
    ALGORITHM_LIMITATION = 5,
    BUDGET_CONSTRAINT = 6
};
struct AlgorithmPolicy
{
    std::string algorithmName;
    PrecisionLevel defaultLevel;
    NumericalPrecision maxPrecision;
    NumericalPrecision minPrecision;
    bool allowMixedPrecision;
    bool allowAutomaticDowngrade;
    double stabilityThreshold;
    bool requiresValidation;
    AlgorithmPolicy(const std::string &name, PrecisionLevel default_lvl,
                    NumericalPrecision max_prec, NumericalPrecision min_prec,
                    bool allow_mixed = false, bool allow_auto_downgrade = false,
                    double stability_thresh = 1e-12, bool needs_validation = true)
        : algorithmName(name)
        , defaultLevel(default_lvl)
        , maxPrecision(max_prec)
        , minPrecision(min_prec)
        , allowMixedPrecision(allow_mixed)
        , allowAutomaticDowngrade(allow_auto_downgrade)
        , stabilityThreshold(stability_thresh)
        , requiresValidation(needs_validation)
    {}
};
class PrecisionPolicyManager
{
public:
    struct PrecisionState
    {
        NumericalPrecision current_precision;
        PrecisionLevel current_level;
        DowngradeReason last_downgrade_reason;
        double numerical_stability_estimate;
        bool strict_fp_mode;
        size_t downgrade_count;
    };
    struct PolicyViolation
    {
        std::string algorithmName;
        NumericalPrecision requested_precision;
        NumericalPrecision actual_precision;
        DowngradeReason reason;
        std::string description;
    };
    PrecisionPolicyManager();
    void setGlobalPrecisionLevel(PrecisionLevel level);
    void setStrictFpMode(bool strict);
    void registerAlgorithmPolicy(const AlgorithmPolicy &policy);
    NumericalPrecision getAlgorithmPrecision(const std::string &algorithmName) const;
    PrecisionState getPrecisionState(const std::string &algorithmName) const;
    std::vector<PolicyViolation> getPolicyViolations() const;
    void updateStabilityCertificateForDowngrade(const std::string &algorithmName,
                                                persistence::StabilityCertificate &certificate,
                                                DowngradeReason reason);
    bool validatePrecisionUsage(const std::string &algorithmName,
                                NumericalPrecision used_precision) const;
    void enforcePrecisionPolicy(const std::string &algorithmName,
                                NumericalPrecision requested_precision);
    PrecisionLevel getGlobalLevel() const { return global_level_; }
    bool isStrictFpMode() const { return strict_fp_mode_; }

private:
    PrecisionLevel global_level_;
    bool strict_fp_mode_;
    std::unordered_map<std::string, AlgorithmPolicy> algorithm_policies_;
    std::unordered_map<std::string, PrecisionState> precision_states_;
    std::vector<PolicyViolation> policy_violations_;
    void initializeDefaultPolicies();
    NumericalPrecision precisionForLevel(PrecisionLevel level) const;
    bool isPrecisionAllowed(const std::string &algorithmName, NumericalPrecision precision) const;
    void recordDowngrade(const std::string &algorithmName, DowngradeReason reason,
                         NumericalPrecision from_precision, NumericalPrecision to_precision);
    double computeStabilityImpact(DowngradeReason reason, NumericalPrecision from_precision,
                                  NumericalPrecision to_precision) const;
};
template <typename ComputationType>
class PrecisionAwareComputation
{
public:
    using ResultType = typename ComputationType::ResultType;
    PrecisionAwareComputation(const std::string &algorithmName,
                              std::unique_ptr<ComputationType> computation);
    ResultType
    executeWithPrecisionPolicy(const typename ComputationType::InputType &input,
                               NumericalPrecision requested_precision = NumericalPrecision::FP64);
    ResultType executeWithValidation(const typename ComputationType::InputType &input,
                                     bool validate_result = true);

private:
    std::string algorithm_name_;
    std::unique_ptr<ComputationType> computation_;
    PrecisionPolicyManager &policy_manager_;
    NumericalPrecision determineExecutionPrecision(const typename ComputationType::InputType &input,
                                                   NumericalPrecision requested_precision) const;
    bool validateComputationResult(const ResultType &result) const;
};
} // namespace nerve::precision
