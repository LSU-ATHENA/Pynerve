
#include "nerve/instrumentation/stability_certificates.hpp"

#include <algorithm>
#include <chrono>
#include <limits>

namespace nerve::instrumentation
{
namespace
{

float finiteNonnegative(float value, float fallback)
{
    if (!std::isfinite(value))
    {
        return fallback;
    }
    return std::max(0.0f, value);
}

float finiteAtLeast(float value, float minimum, float fallback)
{
    if (!std::isfinite(value))
    {
        return fallback;
    }
    return std::max(minimum, value);
}

float finiteUnit(float value, float fallback)
{
    if (!std::isfinite(value))
    {
        return fallback;
    }
    return std::clamp(value, 0.0f, 1.0f);
}

uint8_t computeActualIterations(float residual)
{
    if (!std::isfinite(residual) || residual <= 0.0f)
    {
        return 255;
    }
    const double scaled = std::log10(static_cast<double>(residual) + 1e-16);
    const int iterations = static_cast<int>(std::clamp(-scaled * 16.0, 1.0, 255.0));
    return static_cast<uint8_t>(iterations);
}

StabilityCertificate makeBaseCertificate(float bottleneck_bound, float wasserstein_bound,
                                         float numerical_residual, uint8_t confidence_bucket,
                                         uint32_t computation_hash, uint64_t computation_time_ns)
{
    StabilityCertificate cert{};
    cert.bottleneck_upper_bound = finiteNonnegative(bottleneck_bound, 0.0f);
    cert.wasserstein_upper_bound = finiteNonnegative(wasserstein_bound, 0.0f);
    cert.numerical_residual = finiteNonnegative(numerical_residual, 0.0f);
    cert.confidence_bucket = confidence_bucket;
    cert.convergence_iterations = computeActualIterations(numerical_residual);
    cert.stability_level = 0;
    cert.flags = 0;
    cert.computation_hash = computation_hash;
    cert.timestamp_ns = 0;
    cert.computation_time_ns = computation_time_ns;
    cert.highdim_condition_estimate = 1.0f;
    cert.effective_rank_estimate = 0.0f;
    cert.max_dimension_computed = 0;
    cert.precision_event_count = 0;
    cert.compression_ratio = 0.0f;
    cert.memory_efficiency_score = 1.0f;
    return cert;
}

} // namespace

StabilityCertificate CertificateFactory::createPersistenceCertificate(
    float bottleneck_bound, float wasserstein_bound, float numerical_residual,
    uint8_t confidence_bucket, uint32_t computation_hash, uint64_t computation_time_ns)
{
    auto cert = makeBaseCertificate(bottleneck_bound, wasserstein_bound, numerical_residual,
                                    confidence_bucket, computation_hash, computation_time_ns);
    cert.timestamp_ns = getTimestampNs();
    cert.stability_level =
        computeStabilityLevel(cert.numerical_residual, cert.convergence_iterations);
    return cert;
}

StabilityCertificate CertificateFactory::createPh5Ph6Certificate(
    float bottleneck_bound, float wasserstein_bound, float numerical_residual,
    uint8_t confidence_bucket, uint32_t computation_hash, uint64_t computation_time_ns,
    float highdim_condition_estimate, float effective_rank_estimate,
    uint16_t max_dimension_computed, uint16_t precision_event_count, float compression_ratio,
    float memory_efficiency_score)
{
    auto cert = makeBaseCertificate(bottleneck_bound, wasserstein_bound, numerical_residual,
                                    confidence_bucket, computation_hash, computation_time_ns);
    cert.highdim_condition_estimate = finiteAtLeast(highdim_condition_estimate, 1.0f, 1.0f);
    cert.effective_rank_estimate = finiteNonnegative(effective_rank_estimate, 0.0f);
    cert.max_dimension_computed = max_dimension_computed;
    cert.precision_event_count = precision_event_count;
    cert.compression_ratio = finiteNonnegative(compression_ratio, 0.0f);
    cert.memory_efficiency_score = finiteUnit(memory_efficiency_score, 0.0f);
    if (max_dimension_computed > 4)
        cert.setFlag(StabilityCertificate::FLAG_HIGH_DIMENSIONAL);
    if (precision_event_count > 0)
        cert.setFlag(StabilityCertificate::FLAG_PRECISION_DOWNGRADED);
    cert.timestamp_ns = getTimestampNs();
    cert.stability_level =
        computeStabilityLevel(cert.numerical_residual, cert.convergence_iterations);
    return cert;
}

StabilityCertificate CertificateFactory::createEigenpairCertificate(float numerical_residual,
                                                                    uint8_t convergence_iterations,
                                                                    uint8_t confidence_bucket,
                                                                    uint32_t computation_hash,
                                                                    uint64_t computation_time_ns)
{
    auto cert = makeBaseCertificate(numerical_residual, numerical_residual, numerical_residual,
                                    confidence_bucket, computation_hash, computation_time_ns);
    cert.convergence_iterations = convergence_iterations;
    cert.timestamp_ns = getTimestampNs();
    cert.stability_level =
        computeStabilityLevel(cert.numerical_residual, cert.convergence_iterations);
    return cert;
}

StabilityCertificate CertificateFactory::createStreamingCertificate(float update_error_bound,
                                                                    float stability_metric,
                                                                    uint8_t confidence_bucket,
                                                                    uint32_t computation_hash,
                                                                    uint64_t computation_time_ns)
{
    auto cert = makeBaseCertificate(update_error_bound, stability_metric, update_error_bound,
                                    confidence_bucket, computation_hash, computation_time_ns);
    cert.timestamp_ns = getTimestampNs();
    cert.stability_level =
        computeStabilityLevel(cert.numerical_residual, cert.convergence_iterations);
    return cert;
}

StabilityCertificate CertificateFactory::createApproximateCertificate(float approximation_error,
                                                                      float theoretical_bound,
                                                                      uint8_t confidence_bucket,
                                                                      uint32_t computation_hash,
                                                                      uint64_t computation_time_ns)
{
    auto cert = makeBaseCertificate(approximation_error, theoretical_bound, approximation_error,
                                    confidence_bucket, computation_hash, computation_time_ns);
    cert.setFlag(StabilityCertificate::FLAG_APPROXIMATE);
    cert.timestamp_ns = getTimestampNs();
    cert.stability_level =
        computeStabilityLevel(cert.numerical_residual, cert.convergence_iterations);
    return cert;
}

StabilityCertificate CertificateFactory::createMinimalCertificate(uint32_t computation_hash,
                                                                  uint64_t computation_time_ns)
{
    return createPersistenceCertificate(0.0f, 0.0f, 0.0f, 0, computation_hash, computation_time_ns);
}

uint64_t CertificateFactory::getTimestampNs()
{
    const auto now = std::chrono::high_resolution_clock::now();
    static uint64_t last = 0;
    uint64_t ts = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count());
    if (ts <= last)
    {
        ts = last + 1;
    }
    last = ts;
    return ts;
}

uint8_t CertificateFactory::computeConfidenceBucket(float error, float bound)
{
    if (bound <= 0.0f || !std::isfinite(error) || !std::isfinite(bound))
        return 0;
    const float ratio = std::abs(error) / bound;
    if (ratio < 0.01f)
        return 250;
    if (ratio < 0.10f)
        return 200;
    if (ratio < 0.50f)
        return 150;
    return 100;
}

uint8_t CertificateFactory::computeStabilityLevel(float numerical_residual, uint8_t iterations)
{
    if (numerical_residual < 1e-8f && iterations < 10)
        return 250;
    if (numerical_residual < 1e-6f && iterations < 50)
        return 200;
    if (numerical_residual < 1e-4f && iterations < 100)
        return 150;
    return 100;
}

CertificateValidator::CertificateValidator()
    : rules_()
{}

CertificateValidator::ValidationResult
CertificateValidator::validateForMlTraining(const StabilityCertificate &cert) const
{
    if (!cert.isValid())
        return ValidationResult::REJECT_UNSTABLE;
    if (!meetsMinimumQuality(cert))
        return ValidationResult::REJECT_INSUFFICIENT;
    if (!meetsMlRequirements(cert))
        return ValidationResult::ACCEPT_DEGRADED;
    if (cert.isHighQuality())
        return ValidationResult::ACCEPT_HIGH_QUALITY;
    return ValidationResult::ACCEPT_STANDARD;
}

CertificateValidator::ValidationResult
CertificateValidator::validateForResearch(const StabilityCertificate &cert) const
{
    if (!cert.isValid())
        return ValidationResult::REJECT_UNSTABLE;
    return cert.isAcceptable() ? ValidationResult::ACCEPT_STANDARD
                               : ValidationResult::ACCEPT_DEGRADED;
}

CertificateValidator::ValidationResult
CertificateValidator::validateForDebugging(const StabilityCertificate &cert) const
{
    return cert.isValid() ? ValidationResult::ACCEPT_DEGRADED : ValidationResult::REJECT_UNSTABLE;
}

CertificateValidator::ValidationResult
CertificateValidator::validate(const StabilityCertificate &cert) const
{
    return validateForMlTraining(cert);
}

bool CertificateValidator::meetsMinimumQuality(const StabilityCertificate &cert) const
{
    return static_cast<float>(cert.confidence_bucket) >= rules_.min_confidence_bucket &&
           cert.numerical_residual <= rules_.max_numerical_residual &&
           static_cast<float>(cert.stability_level) >= rules_.min_stability_level &&
           cert.computation_time_ns <= rules_.max_computation_time_ns;
}

bool CertificateValidator::meetsMlRequirements(const StabilityCertificate &cert) const
{
    if (!rules_.allow_approximate_for_ml && cert.isApproximate())
    {
        return false;
    }
    return cert.hasGoodConditioning();
}

CertificateAggregator::CertificateAggregator()
    : certificate_count_(0)
{}

bool CertificateAggregator::addCertificate(const StabilityCertificate &cert)
{
    if (certificate_count_ >= MAX_CERTIFICATES)
        return false;
    certificates_[certificate_count_++] = cert;
    return true;
}

StabilityCertificate CertificateAggregator::getAggregatedCertificate() const
{
    if (certificate_count_ == 0)
    {
        return CertificateFactory::createMinimalCertificate(0, 0);
    }
    float avg_confidence = 0.0f;
    float avg_residual = 0.0f;
    uint64_t max_time = 0;
    uint16_t max_dim = 0;
    for (size_t i = 0; i < certificate_count_; ++i)
    {
        avg_confidence += static_cast<float>(certificates_[i].confidence_bucket);
        avg_residual += certificates_[i].numerical_residual;
        max_time = std::max(max_time, certificates_[i].computation_time_ns);
        max_dim = std::max(max_dim, certificates_[i].max_dimension_computed);
    }
    avg_confidence /= static_cast<float>(certificate_count_);
    avg_residual /= static_cast<float>(certificate_count_);
    return CertificateFactory::createPh5Ph6Certificate(
        avg_residual, avg_residual, avg_residual,
        static_cast<uint8_t>(std::clamp(avg_confidence, 0.0f, 255.0f)),
        certificates_[0].computation_hash, max_time, 1.0f, static_cast<float>(max_dim), max_dim, 0,
        0.0f, 1.0f);
}

CertificateAggregator::AggregationStats CertificateAggregator::getStats() const
{
    AggregationStats stats{};
    stats.total_certificates = certificate_count_;
    stats.average_confidence = computeAverageConfidence();
    stats.average_numerical_residual = computeAverageNumericalResidual();
    stats.worst_numerical_residual = computeWorstNumericalResidual();
    countQualityLevels(stats.high_quality_count, stats.acceptable_count, stats.degraded_count,
                       stats.rejected_count);
    return stats;
}

void CertificateAggregator::clear()
{
    certificate_count_ = 0;
}

float CertificateAggregator::computeAverageConfidence() const
{
    if (certificate_count_ == 0)
        return 0.0f;
    float sum = 0.0f;
    for (size_t i = 0; i < certificate_count_; ++i)
    {
        sum += static_cast<float>(certificates_[i].confidence_bucket);
    }
    return sum / static_cast<float>(certificate_count_);
}

float CertificateAggregator::computeAverageNumericalResidual() const
{
    if (certificate_count_ == 0)
        return 0.0f;
    float sum = 0.0f;
    for (size_t i = 0; i < certificate_count_; ++i)
        sum += certificates_[i].numerical_residual;
    return sum / static_cast<float>(certificate_count_);
}

float CertificateAggregator::computeWorstNumericalResidual() const
{
    float worst = 0.0f;
    for (size_t i = 0; i < certificate_count_; ++i)
        worst = std::max(worst, certificates_[i].numerical_residual);
    return worst;
}

void CertificateAggregator::countQualityLevels(size_t &high, size_t &acceptable, size_t &degraded,
                                               size_t &rejected) const
{
    high = acceptable = degraded = rejected = 0;
    for (size_t i = 0; i < certificate_count_; ++i)
    {
        const auto &cert = certificates_[i];
        if (!cert.isValid())
        {
            ++rejected;
        }
        else if (cert.isHighQuality())
        {
            ++high;
        }
        else if (cert.isAcceptable())
        {
            ++acceptable;
        }
        else
        {
            ++degraded;
        }
    }
}

} // namespace nerve::instrumentation
