#include "nerve/spectral/persistent_laplacian.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace nerve::spectral
{

SpectralAnomalyDetector::SpectralAnomalyDetector(const AnomalyConfig &config)
    : config_(config)
    , adaptive_threshold_(config.anomaly_threshold)
{}

SpectralAnomalyDetector::AnomalyResult
SpectralAnomalyDetector::detectAnomaly(const SpectralDecomposition &current_spectrum)
{
    AnomalyResult result;
    result.is_anomaly = false;
    result.anomaly_score = 0.0;

    if (reference_spectra_.empty())
    {
        result.anomaly_description = "No reference spectra available";
        return result;
    }

    const auto ref_eigenvalue_stats = computeEigenvalueStatistics(reference_spectra_);
    const auto ref_eigenvector_stats = computeEigenvectorStatistics(reference_spectra_);

    double eigenvalue_score = 0.0;
    double eigenvector_score = 0.0;
    if (config_.use_eigenvalue_anomalies)
    {
        eigenvalue_score = computeEigenvalueAnomalyScore(current_spectrum, ref_eigenvalue_stats);
    }
    if (config_.use_eigenvector_anomalies)
    {
        eigenvector_score = computeEigenvectorAnomalyScore(current_spectrum, ref_eigenvector_stats);
    }

    result.anomaly_score = eigenvalue_score + eigenvector_score;
    result.is_anomaly = result.anomaly_score > adaptive_threshold_;

    for (std::size_t i = 0; i < current_spectrum.eigenpairs.size(); ++i)
    {
        const double deviation =
            std::abs(current_spectrum.eigenpairs[i].eigenvalue - ref_eigenvalue_stats[0]);
        if (deviation > adaptive_threshold_ * ref_eigenvalue_stats[1])
        {
            result.anomalous_eigenpairs.push_back(i);
            result.eigenvalue_deviations.push_back(deviation);
        }
    }

    if (result.is_anomaly)
    {
        result.anomaly_description =
            "Anomaly detected with score " + std::to_string(result.anomaly_score);
    }
    else
    {
        result.anomaly_description = "No anomaly detected";
    }
    return result;
}

void SpectralAnomalyDetector::updateReference(
    const std::vector<SpectralDecomposition> &reference_spectra)
{
    reference_spectra_ = reference_spectra;
    if (reference_spectra_.size() > config_.reference_window_size)
    {
        const auto excess = static_cast<std::vector<SpectralDecomposition>::difference_type>(
            reference_spectra_.size() - config_.reference_window_size);
        reference_spectra_.erase(reference_spectra_.begin(), reference_spectra_.begin() + excess);
    }
}

void SpectralAnomalyDetector::resetReference()
{
    reference_spectra_.clear();
    adaptive_threshold_ = config_.anomaly_threshold;
}

void SpectralAnomalyDetector::updateAdaptiveThreshold(
    const std::vector<AnomalyResult> &recent_results)
{
    if (!config_.enable_adaptive_threshold || recent_results.empty())
    {
        return;
    }

    double avg_score = 0.0;
    for (const auto &result : recent_results)
    {
        avg_score += result.anomaly_score;
    }
    avg_score /= static_cast<double>(recent_results.size());

    if (avg_score > config_.anomaly_threshold * 0.5)
    {
        adaptive_threshold_ *= 1.1;
    }
    else if (avg_score < config_.anomaly_threshold * 0.1)
    {
        adaptive_threshold_ *= 0.9;
    }
}

std::vector<double> SpectralAnomalyDetector::computeEigenvalueStatistics(
    const std::vector<SpectralDecomposition> &spectra)
{
    std::vector<double> all_eigenvalues;
    for (const auto &spectrum : spectra)
    {
        for (const auto &pair : spectrum.eigenpairs)
        {
            all_eigenvalues.push_back(pair.eigenvalue);
        }
    }

    if (all_eigenvalues.empty())
    {
        return {0.0, 1.0};
    }

    double sum = 0.0;
    double sum_sq = 0.0;
    for (double value : all_eigenvalues)
    {
        sum += value;
        sum_sq += value * value;
    }

    const double mean = sum / static_cast<double>(all_eigenvalues.size());
    const double variance = (sum_sq / static_cast<double>(all_eigenvalues.size())) - (mean * mean);
    double std_dev = std::sqrt(std::max(0.0, variance));
    if (std_dev < 1e-10)
    {
        std_dev = 1.0;
    }
    return {mean, std_dev};
}

std::vector<double> SpectralAnomalyDetector::computeEigenvectorStatistics(
    const std::vector<SpectralDecomposition> &spectra)
{
    if (spectra.empty())
    {
        return {0.0, 1.0};
    }

    std::vector<double> magnitudes;
    for (const auto &spectrum : spectra)
    {
        for (const auto &pair : spectrum.eigenpairs)
        {
            if (pair.eigenvector.size() == 0)
            {
                continue;
            }

            double norm = 0.0;
            for (double value : pair.eigenvector)
            {
                norm += value * value;
            }
            magnitudes.push_back(std::sqrt(norm));

            double avg_component = 0.0;
            for (double value : pair.eigenvector)
            {
                avg_component += std::abs(value);
            }
            magnitudes.push_back(avg_component / static_cast<double>(pair.eigenvector.size()));
        }
    }

    if (magnitudes.empty())
    {
        return {0.0, 1.0};
    }

    double sum = 0.0;
    for (double value : magnitudes)
    {
        sum += value;
    }
    const double mean = sum / static_cast<double>(magnitudes.size());

    double sq_sum = 0.0;
    for (double value : magnitudes)
    {
        sq_sum += (value - mean) * (value - mean);
    }
    double std_dev = std::sqrt(sq_sum / static_cast<double>(magnitudes.size()));
    if (std_dev < 1e-10)
    {
        std_dev = 1.0;
    }
    return {mean, std_dev};
}

double
SpectralAnomalyDetector::computeEigenvalueAnomalyScore(const SpectralDecomposition &spectrum,
                                                       const std::vector<double> &reference_stats)
{
    if (reference_stats.size() < 2 || spectrum.eigenpairs.empty())
    {
        return 0.0;
    }

    const double mean = reference_stats[0];
    const double std_dev = reference_stats[1];
    double total_deviation = 0.0;

    for (const auto &pair : spectrum.eigenpairs)
    {
        const double z_score = (pair.eigenvalue - mean) / std_dev;
        total_deviation += z_score * z_score;
    }
    return std::sqrt(total_deviation / static_cast<double>(spectrum.eigenpairs.size()));
}

double
SpectralAnomalyDetector::computeEigenvectorAnomalyScore(const SpectralDecomposition &spectrum,
                                                        const std::vector<double> &reference_stats)
{
    if (reference_stats.size() < 2 || spectrum.eigenpairs.empty())
    {
        return 0.0;
    }

    const double ref_mean = reference_stats[0];
    const double ref_std = reference_stats[1];

    double total_score = 0.0;
    int count = 0;
    for (const auto &pair : spectrum.eigenpairs)
    {
        if (pair.eigenvector.size() == 0)
        {
            continue;
        }

        double norm = 0.0;
        double participation = 0.0;
        for (double value : pair.eigenvector)
        {
            const double value_sq = value * value;
            norm += value_sq;
            participation += value_sq * value_sq;
        }

        norm = std::sqrt(norm);
        if (participation > 1e-15)
        {
            participation = 1.0 / participation;
        }
        else
        {
            participation = 0.0;
        }

        const double components = static_cast<double>(pair.eigenvector.size());
        const double normalized_pr = participation / components;
        const double score = std::abs(norm - ref_mean) / ref_std;
        total_score += score * (1.0 + std::abs(normalized_pr - 0.5));
        ++count;
    }
    return count > 0 ? total_score / static_cast<double>(count) : 0.0;
}

} // namespace nerve::spectral
