#include "nerve/encoders/encoders.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

namespace nerve::encoders
{
namespace
{

constexpr double kEpsilon = 1.0e-12;

void requireFinite(double value, const std::string &context)
{
    if (!std::isfinite(value))
    {
        throw std::invalid_argument(context + " contains a non-finite value");
    }
}

double checkedAdd(double lhs, double rhs, const std::string &context)
{
    const double result = lhs + rhs;
    if (!std::isfinite(result))
    {
        throw std::overflow_error(context + " overflowed");
    }
    return result;
}

double checkedSquare(double value, const std::string &context)
{
    const double square = value * value;
    if (!std::isfinite(square))
    {
        throw std::overflow_error(context + " overflowed");
    }
    return square;
}

Size maxRowWidth(const std::vector<std::vector<double>> &data)
{
    Size width = 0;
    for (const auto &row : data)
    {
        width = std::max(width, row.size());
        for (double value : row)
        {
            requireFinite(value, "encoder utility input");
        }
    }
    return width;
}

std::vector<double> tensorDataChecked(const Tensor &tensor, const std::string &context)
{
    const auto values = tensor.data();
    for (double value : values)
    {
        requireFinite(value, context);
    }
    return values;
}

Size argMaxClass(const std::vector<double> &values, const std::string &context)
{
    if (values.empty())
    {
        throw std::invalid_argument(context + " must not be empty");
    }
    for (double value : values)
    {
        requireFinite(value, context);
    }
    return static_cast<Size>(
        std::distance(values.begin(), std::max_element(values.begin(), values.end())));
}

Size labelClass(const std::vector<double> &label)
{
    if (label.size() == 1)
    {
        requireFinite(label[0], "label");
        const double rounded = std::round(label[0]);
        if (rounded >= 0.0 && std::abs(label[0] - rounded) <= kEpsilon)
        {
            return static_cast<Size>(rounded);
        }
    }
    return argMaxClass(label, "label");
}

} // namespace

std::vector<std::vector<double>>
EncoderUtils::normalizeData(const std::vector<std::vector<double>> &data)
{
    const Size width = maxRowWidth(data);
    if (width == 0)
    {
        return data;
    }

    std::vector<double> min_values(width, std::numeric_limits<double>::infinity());
    std::vector<double> max_values(width, -std::numeric_limits<double>::infinity());
    for (const auto &row : data)
    {
        for (Size col = 0; col < row.size(); ++col)
        {
            min_values[col] = std::min(min_values[col], row[col]);
            max_values[col] = std::max(max_values[col], row[col]);
        }
    }

    auto normalized = data;
    for (auto &row : normalized)
    {
        for (Size col = 0; col < row.size(); ++col)
        {
            const double range = max_values[col] - min_values[col];
            if (!std::isfinite(range))
            {
                throw std::overflow_error("normalization range overflowed");
            }
            const double numerator = row[col] - min_values[col];
            if (!std::isfinite(numerator))
            {
                throw std::overflow_error("normalization numerator overflowed");
            }
            row[col] = std::abs(range) <= kEpsilon ? 0.0 : numerator / range;
        }
    }
    return normalized;
}

std::vector<std::vector<double>>
EncoderUtils::standardizeData(const std::vector<std::vector<double>> &data)
{
    const Size width = maxRowWidth(data);
    if (width == 0)
    {
        return data;
    }

    std::vector<Size> counts(width, 0);
    std::vector<double> sums(width, 0.0);
    std::vector<double> sum_squares(width, 0.0);
    for (const auto &row : data)
    {
        for (Size col = 0; col < row.size(); ++col)
        {
            ++counts[col];
            sums[col] = checkedAdd(sums[col], row[col], "standardization sum");
            sum_squares[col] =
                checkedAdd(sum_squares[col], checkedSquare(row[col], "standardization square"),
                           "standardization square sum");
        }
    }

    std::vector<double> means(width, 0.0);
    std::vector<double> inverse_std(width, 0.0);
    for (Size col = 0; col < width; ++col)
    {
        if (counts[col] == 0)
        {
            continue;
        }
        means[col] = sums[col] / static_cast<double>(counts[col]);
        const double variance =
            sum_squares[col] / static_cast<double>(counts[col]) - means[col] * means[col];
        const double stddev = std::sqrt(std::max(0.0, variance));
        inverse_std[col] = stddev <= kEpsilon ? 0.0 : 1.0 / stddev;
    }

    auto standardized = data;
    for (auto &row : standardized)
    {
        for (Size col = 0; col < row.size(); ++col)
        {
            row[col] = (row[col] - means[col]) * inverse_std[col];
        }
    }
    return standardized;
}

std::vector<std::vector<double>>
EncoderUtils::augmentData(const std::vector<std::vector<double>> &data)
{
    maxRowWidth(data);
    std::vector<std::vector<double>> augmented;
    augmented.reserve(data.size() * 2);
    augmented.insert(augmented.end(), data.begin(), data.end());
    const auto perturbed = applyNoise(data, 0.01);
    augmented.insert(augmented.end(), perturbed.begin(), perturbed.end());
    return augmented;
}

std::vector<double> EncoderUtils::computeFeatureStatistics(const Tensor &features)
{
    const auto values = tensorDataChecked(features, "feature tensor");
    if (values.empty())
    {
        throw std::invalid_argument("feature tensor must not be empty");
    }

    const double n = static_cast<double>(values.size());
    double sum = 0.0;
    for (double value : values)
    {
        sum = checkedAdd(sum, value, "feature statistic sum");
    }
    const double mean = sum / n;
    double squared_error = 0.0;
    double l1_norm = 0.0;
    double l2_square = 0.0;
    Size non_zero = 0;
    for (double value : values)
    {
        const double centered = value - mean;
        if (!std::isfinite(centered))
        {
            throw std::overflow_error("feature statistic centered value overflowed");
        }
        squared_error =
            checkedAdd(squared_error, checkedSquare(centered, "feature statistic squared error"),
                       "feature statistic squared error sum");
        l1_norm = checkedAdd(l1_norm, std::abs(value), "feature statistic l1 norm");
        l2_square = checkedAdd(l2_square, checkedSquare(value, "feature statistic l2 square"),
                               "feature statistic l2 square sum");
        non_zero += std::abs(value) > kEpsilon ? 1 : 0;
    }

    const auto [min_it, max_it] = std::minmax_element(values.begin(), values.end());
    return {
        mean,
        squared_error / n,
        *min_it,
        *max_it,
        l1_norm,
        std::sqrt(l2_square),
        static_cast<double>(non_zero) / n,
    };
}

double EncoderUtils::computeFeatureDiversity(const std::vector<Tensor> &features)
{
    if (features.size() < 2)
    {
        return 0.0;
    }

    std::vector<std::vector<double>> rows;
    rows.reserve(features.size());
    for (const auto &tensor : features)
    {
        rows.push_back(tensorDataChecked(tensor, "feature tensor"));
        if (rows.back().empty())
        {
            throw std::invalid_argument("feature tensor must not be empty");
        }
        if (rows.back().size() != rows.front().size())
        {
            throw std::invalid_argument("feature tensors must have equal sizes");
        }
    }

    double distance_sum = 0.0;
    Size pair_count = 0;
    const double scale = std::sqrt(static_cast<double>(rows.front().size()));
    for (Size i = 0; i < rows.size(); ++i)
    {
        for (Size j = i + 1; j < rows.size(); ++j)
        {
            double squared = 0.0;
            for (Size k = 0; k < rows[i].size(); ++k)
            {
                const double delta = rows[i][k] - rows[j][k];
                if (!std::isfinite(delta))
                {
                    throw std::overflow_error("feature diversity delta overflowed");
                }
                squared = checkedAdd(squared, checkedSquare(delta, "feature diversity square"),
                                     "feature diversity square sum");
            }
            distance_sum += std::sqrt(squared) / std::max(scale, kEpsilon);
            ++pair_count;
        }
    }
    return pair_count == 0 ? 0.0 : distance_sum / static_cast<double>(pair_count);
}

std::vector<double> EncoderUtils::computeFeatureCorrelation(const Tensor &features1,
                                                            const Tensor &features2)
{
    const auto left = tensorDataChecked(features1, "left feature tensor");
    const auto right = tensorDataChecked(features2, "right feature tensor");
    if (left.size() != right.size())
    {
        throw std::invalid_argument("feature tensors must have equal element counts");
    }
    if (left.empty())
    {
        throw std::invalid_argument("feature tensors must not be empty");
    }

    const auto &shape = features1.shape();
    const bool use_columns = shape.size() == 2 && features2.shape() == shape;
    const Size rows = use_columns ? shape[0] : left.size();
    const Size cols = use_columns ? shape[1] : 1;
    std::vector<double> correlations(cols, 0.0);

    for (Size col = 0; col < cols; ++col)
    {
        double left_sum = 0.0;
        double right_sum = 0.0;
        for (Size row = 0; row < rows; ++row)
        {
            const Size index = use_columns ? row * cols + col : row;
            left_sum += left[index];
            right_sum += right[index];
        }
        const double left_mean = left_sum / static_cast<double>(rows);
        const double right_mean = right_sum / static_cast<double>(rows);

        double covariance = 0.0;
        double left_var = 0.0;
        double right_var = 0.0;
        for (Size row = 0; row < rows; ++row)
        {
            const Size index = use_columns ? row * cols + col : row;
            const double left_centered = left[index] - left_mean;
            const double right_centered = right[index] - right_mean;
            covariance += left_centered * right_centered;
            left_var += left_centered * left_centered;
            right_var += right_centered * right_centered;
        }

        const double denom = std::sqrt(left_var * right_var);
        correlations[col] = denom <= kEpsilon ? 0.0 : covariance / denom;
    }
    return correlations;
}

double EncoderUtils::evaluateEncoderPerformance(
    const FeatureEncoder &encoder, const std::vector<std::vector<std::vector<double>>> &test_data,
    const std::vector<std::vector<double>> &test_labels)
{
    const auto encoded = encoder.encodeBatch(test_data);
    std::vector<std::vector<double>> predictions;
    predictions.reserve(encoded.size());
    for (const auto &tensor : encoded)
    {
        predictions.push_back(tensorDataChecked(tensor, "encoder prediction"));
    }
    return computeAccuracy(predictions, test_labels);
}

std::vector<double> EncoderUtils::evaluateEncoderRobustness(
    const FeatureEncoder &encoder, const std::vector<std::vector<std::vector<double>>> &test_data,
    double noise_level)
{
    if (!std::isfinite(noise_level) || noise_level < 0.0)
    {
        throw std::invalid_argument("noise level must be finite and non-negative");
    }

    std::vector<double> scores;
    scores.reserve(test_data.size());
    for (const auto &sample : test_data)
    {
        const auto baseline = tensorDataChecked(encoder.encode(sample), "encoder baseline");
        const auto shifted = tensorDataChecked(encoder.encode(applyNoise(sample, noise_level)),
                                               "encoder perturbed output");
        if (baseline.size() != shifted.size())
        {
            throw std::runtime_error("encoder returned inconsistent output sizes");
        }
        double squared_delta = 0.0;
        double baseline_square = 0.0;
        for (Size i = 0; i < baseline.size(); ++i)
        {
            const double delta = baseline[i] - shifted[i];
            squared_delta += delta * delta;
            baseline_square += baseline[i] * baseline[i];
        }
        scores.push_back(std::sqrt(squared_delta) / std::max(1.0, std::sqrt(baseline_square)));
    }
    return scores;
}

std::vector<std::vector<double>>
EncoderUtils::applyNoise(const std::vector<std::vector<double>> &data, double noise_level)
{
    if (!std::isfinite(noise_level) || noise_level < 0.0)
    {
        throw std::invalid_argument("noise level must be finite and non-negative");
    }
    auto perturbed = data;
    for (Size row = 0; row < perturbed.size(); ++row)
    {
        for (Size col = 0; col < perturbed[row].size(); ++col)
        {
            requireFinite(perturbed[row][col], "encoder utility input");
            const double phase =
                static_cast<double>((row + 1) * 73856093ULL ^ (col + 1) * 19349663ULL);
            const double unit = std::sin(phase) * 0.5;
            const double scale = std::max(1.0, std::abs(perturbed[row][col]));
            perturbed[row][col] += noise_level * scale * unit;
        }
    }
    return perturbed;
}

double EncoderUtils::computeAccuracy(const std::vector<std::vector<double>> &predictions,
                                     const std::vector<std::vector<double>> &labels)
{
    if (predictions.size() != labels.size())
    {
        throw std::invalid_argument("prediction and label counts must match");
    }
    if (predictions.empty())
    {
        return 0.0;
    }

    Size correct = 0;
    for (Size i = 0; i < predictions.size(); ++i)
    {
        const Size predicted = argMaxClass(predictions[i], "prediction");
        const Size expected = labelClass(labels[i]);
        correct += predicted == expected ? 1 : 0;
    }
    return static_cast<double>(correct) / static_cast<double>(predictions.size());
}

} // namespace nerve::encoders
