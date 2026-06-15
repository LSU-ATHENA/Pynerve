
#pragma once
#include <array>
#include <cmath>
#include <cstdint>

namespace nerve::instrumentation
{
struct alignas(32) StabilityCertificate
{
    float bottleneck_upper_bound;
    float wasserstein_upper_bound;
    float numerical_residual;
    uint8_t confidence_bucket;
    uint8_t convergence_iterations;
    uint8_t stability_level;
    uint8_t flags;
    uint32_t computation_hash;
    uint64_t timestamp_ns;
    uint64_t computation_time_ns;
    float highdim_condition_estimate;
    float effective_rank_estimate;
    uint16_t max_dimension_computed;
    uint16_t precision_event_count;
    float compression_ratio;
    float memory_efficiency_score;
    bool isValid() const
    {
        return std::isfinite(bottleneck_upper_bound) && std::isfinite(wasserstein_upper_bound) &&
               std::isfinite(numerical_residual) && std::isfinite(highdim_condition_estimate) &&
               std::isfinite(effective_rank_estimate) && std::isfinite(compression_ratio) &&
               std::isfinite(memory_efficiency_score) && bottleneck_upper_bound >= 0.0f &&
               wasserstein_upper_bound >= 0.0f && numerical_residual >= 0.0f &&
               highdim_condition_estimate >= 1.0f && effective_rank_estimate >= 0.0f &&
               compression_ratio >= 0.0f && memory_efficiency_score >= 0.0f &&
               memory_efficiency_score <= 1.0f && timestamp_ns > 0;
    }
    bool isHighQuality() const
    {
        return confidence_bucket >= 200 && numerical_residual < 1e-6f && stability_level >= 200 &&
               highdim_condition_estimate < 1e12f && memory_efficiency_score > 0.8f;
    }
    bool isAcceptable() const
    {
        return confidence_bucket >= 100 && numerical_residual < 1e-4f && stability_level >= 100 &&
               highdim_condition_estimate < 1e15f && memory_efficiency_score > 0.5f;
    }
    bool hasGoodConditioning() const
    {
        return highdim_condition_estimate < 1e10f &&
               effective_rank_estimate > 0.1f * static_cast<float>(max_dimension_computed);
    }
    bool hasEfficientCompression() const
    {
        return compression_ratio > 0.1f && memory_efficiency_score > 0.7f;
    }
    static constexpr uint8_t FLAG_HAS_NAN_VALUES = 0x01;
    static constexpr uint8_t FLAG_HAS_INF_VALUES = 0x02;
    static constexpr uint8_t FLAG_APPROXIMATE = 0x04;
    static constexpr uint8_t FLAG_PARTIAL_RESULT = 0x08;
    static constexpr uint8_t FLAG_TIME_LIMITED = 0x10;
    static constexpr uint8_t FLAG_MEMORY_LIMITED = 0x20;
    static constexpr uint8_t FLAG_PRECISION_DOWNGRADED = 0x40;
    static constexpr uint8_t FLAG_HIGH_DIMENSIONAL = 0x80;
    bool hasNanValues() const { return flags & FLAG_HAS_NAN_VALUES; }
    bool hasInfValues() const { return flags & FLAG_HAS_INF_VALUES; }
    bool isApproximate() const { return flags & FLAG_APPROXIMATE; }
    bool isPartialResult() const { return flags & FLAG_PARTIAL_RESULT; }
    bool isTimeLimited() const { return flags & FLAG_TIME_LIMITED; }
    bool isMemoryLimited() const { return flags & FLAG_MEMORY_LIMITED; }
    bool wasPrecisionDowngraded() const { return flags & FLAG_PRECISION_DOWNGRADED; }
    bool isHighDimensional() const { return flags & FLAG_HIGH_DIMENSIONAL; }
    void setFlag(uint8_t flag) { flags |= flag; }
    void clearFlag(uint8_t flag) { flags &= ~flag; }
};
static_assert(sizeof(StabilityCertificate) == 64,
              "StabilityCertificate must be 64 bytes with PH5/PH6 extensions");
class CertificateFactory
{
public:
    static StabilityCertificate
    createPersistenceCertificate(float bottleneck_bound, float wasserstein_bound,
                                 float numerical_residual, uint8_t confidence_bucket,
                                 uint32_t computation_hash, uint64_t computation_time_ns);
    static StabilityCertificate
    createPh5Ph6Certificate(float bottleneck_bound, float wasserstein_bound,
                            float numerical_residual, uint8_t confidence_bucket,
                            uint32_t computation_hash, uint64_t computation_time_ns,
                            float highdim_condition_estimate, float effective_rank_estimate,
                            uint16_t max_dimension_computed, uint16_t precision_event_count,
                            float compression_ratio, float memory_efficiency_score);
    static StabilityCertificate createEigenpairCertificate(float numerical_residual,
                                                           uint8_t convergence_iterations,
                                                           uint8_t confidence_bucket,
                                                           uint32_t computation_hash,
                                                           uint64_t computation_time_ns);
    static StabilityCertificate createStreamingCertificate(float update_error_bound,
                                                           float stability_metric,
                                                           uint8_t confidence_bucket,
                                                           uint32_t computation_hash,
                                                           uint64_t computation_time_ns);
    static StabilityCertificate createApproximateCertificate(float approximation_error,
                                                             float theoretical_bound,
                                                             uint8_t confidence_bucket,
                                                             uint32_t computation_hash,
                                                             uint64_t computation_time_ns);
    static StabilityCertificate createMinimalCertificate(uint32_t computation_hash,
                                                         uint64_t computation_time_ns);

private:
    static uint64_t getTimestampNs();
    static uint8_t computeConfidenceBucket(float error, float bound);
    static uint8_t computeStabilityLevel(float numerical_residual, uint8_t iterations);
};
class CertificateValidator
{
public:
    struct ValidationRules
    {
        float min_confidence_bucket = 100.0f;
        float max_numerical_residual = 1e-4f;
        float min_stability_level = 100.0f;
        uint64_t max_computation_time_ns = 100000000ULL;
        bool allow_approximate_for_ml = true;
        ValidationRules() = default;
    };
    explicit CertificateValidator();
    enum class ValidationResult
    {
        ACCEPT_HIGH_QUALITY,
        ACCEPT_STANDARD,
        ACCEPT_DEGRADED,
        REJECT_INSUFFICIENT,
        REJECT_UNSTABLE
    };
    ValidationResult validateForMlTraining(const StabilityCertificate &cert) const;
    ValidationResult validateForResearch(const StabilityCertificate &cert) const;
    ValidationResult validateForDebugging(const StabilityCertificate &cert) const;
    ValidationResult validate(const StabilityCertificate &cert) const;
    const ValidationRules &getRules() const { return rules_; }
    void setRules(const ValidationRules &rules) { rules_ = rules; }

private:
    ValidationRules rules_;
    bool meetsMinimumQuality(const StabilityCertificate &cert) const;
    bool meetsMlRequirements(const StabilityCertificate &cert) const;
};
class CertificateAggregator
{
public:
    static constexpr size_t MAX_CERTIFICATES = 16;
    explicit CertificateAggregator();
    bool addCertificate(const StabilityCertificate &cert);
    StabilityCertificate getAggregatedCertificate() const;
    struct AggregationStats
    {
        size_t total_certificates;
        float average_confidence;
        float average_numerical_residual;
        float worst_numerical_residual;
        size_t high_quality_count;
        size_t acceptable_count;
        size_t degraded_count;
        size_t rejected_count;
    };
    AggregationStats getStats() const;
    void clear();
    bool isValid() const { return certificate_count_ > 0; }

private:
    std::array<StabilityCertificate, MAX_CERTIFICATES> certificates_;
    size_t certificate_count_;
    float computeAverageConfidence() const;
    float computeAverageNumericalResidual() const;
    float computeWorstNumericalResidual() const;
    void countQualityLevels(size_t &high, size_t &acceptable, size_t &degraded,
                            size_t &rejected) const;
};
#define CREATE_PERSISTENCE_CERTIFICATE(bottleneck, wasserstein, residual, confidence, hash,        \
                                       time_ns)                                                    \
    nerve::instrumentation::CertificateFactory::createPersistenceCertificate(                      \
        bottleneck, wasserstein, residual, confidence, hash, time_ns)
#define CREATE_PH5_PH6_CERTIFICATE(bottleneck, wasserstein, residual, confidence, hash, time_ns,   \
                                   condition, rank, max_dim, precision_count, compression,         \
                                   memory_eff)                                                     \
    nerve::instrumentation::CertificateFactory::createPh5Ph6Certificate(                           \
        bottleneck, wasserstein, residual, confidence, hash, time_ns, condition, rank, max_dim,    \
        precision_count, compression, memory_eff)
#define CREATE_EIGENPAIR_CERTIFICATE(residual, iterations, confidence, hash, time_ns)              \
    nerve::instrumentation::CertificateFactory::createEigenpairCertificate(                        \
        residual, iterations, confidence, hash, time_ns)
#define CREATE_STREAMING_CERTIFICATE(error, stability, confidence, hash, time_ns)                  \
    nerve::instrumentation::CertificateFactory::createStreamingCertificate(                        \
        error, stability, confidence, hash, time_ns)
#define VALIDATE_FOR_ML(cert)                                                                      \
    nerve::instrumentation::CertificateValidator().validateForMlTraining(cert)
} // namespace nerve::instrumentation
