#include "nerve/encoders/encoders.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace nerve::encoders
{
namespace
{

constexpr double kMinBenchmarkSeconds = 1.0e-12;

void requireFinite(double value, const std::string &context)
{
    if (!std::isfinite(value))
    {
        throw std::invalid_argument(context + " contains a non-finite value");
    }
}

std::vector<double> checkedTensorData(const Tensor &tensor, const std::string &context)
{
    const auto values = tensor.data();
    for (double value : values)
    {
        requireFinite(value, context);
    }
    return values;
}

std::string encoderKey(const FeatureEncoder &encoder, Size index)
{
    return encoder.getEncoderType() + "#" + std::to_string(index);
}

double elapsedMilliseconds(std::chrono::steady_clock::time_point start,
                           std::chrono::steady_clock::time_point stop)
{
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

} // namespace

void EncoderUtils::visualizeFeatures(const Tensor &features, const std::string &output_file)
{
    if (output_file.empty())
    {
        throw std::invalid_argument("output file path must not be empty");
    }
    const auto values = checkedTensorData(features, "feature tensor");
    std::ofstream output(output_file);
    if (!output)
    {
        throw std::runtime_error("failed to open feature output file: " + output_file);
    }

    output << "shape";
    for (Size dim : features.shape())
    {
        output << ',' << dim;
    }
    output << '\n' << "index,value\n";
    for (Size i = 0; i < values.size(); ++i)
    {
        output << i << ',' << values[i] << '\n';
    }
}

void EncoderUtils::plotEncoderComparison(
    const std::vector<std::unique_ptr<FeatureEncoder>> &encoders,
    const std::vector<std::vector<std::vector<double>>> &test_data, const std::string &output_file)
{
    if (output_file.empty())
    {
        throw std::invalid_argument("output file path must not be empty");
    }
    std::ofstream output(output_file);
    if (!output)
    {
        throw std::runtime_error("failed to open encoder comparison file: " + output_file);
    }

    output << "encoder,elapsed_ms,samples,mean_abs_feature\n";
    for (Size i = 0; i < encoders.size(); ++i)
    {
        if (!encoders[i])
        {
            throw std::invalid_argument("encoder comparison received a null encoder");
        }
        const auto start = std::chrono::steady_clock::now();
        const auto encoded = encoders[i]->encodeBatch(test_data);
        const double elapsed_ms = elapsedMilliseconds(start, std::chrono::steady_clock::now());
        double abs_sum = 0.0;
        Size count = 0;
        for (const auto &tensor : encoded)
        {
            for (double value : checkedTensorData(tensor, "encoder comparison output"))
            {
                abs_sum += std::abs(value);
                ++count;
            }
        }
        const double mean_abs = count == 0 ? 0.0 : abs_sum / static_cast<double>(count);
        output << encoderKey(*encoders[i], i) << ',' << elapsed_ms << ',' << test_data.size() << ','
               << mean_abs << '\n';
    }
}

void EncoderUtils::profileEncoder(const FeatureEncoder &encoder,
                                  const std::vector<std::vector<std::vector<double>>> &test_data)
{
    const auto start = std::chrono::steady_clock::now();
    const auto encoded = encoder.encodeBatch(test_data);
    const double elapsed_ms = elapsedMilliseconds(start, std::chrono::steady_clock::now());
    Size values = 0;
    for (const auto &tensor : encoded)
    {
        values += checkedTensorData(tensor, "profiled encoder output").size();
    }
    std::clog << "encoder=" << encoder.getEncoderType() << ",samples=" << test_data.size()
              << ",values=" << values << ",elapsed_ms=" << elapsed_ms << '\n';
}

std::map<std::string, double>
EncoderUtils::benchmarkEncoders(const std::vector<std::unique_ptr<FeatureEncoder>> &encoders,
                                const std::vector<std::vector<std::vector<double>>> &test_data)
{
    std::map<std::string, double> throughput;
    for (Size i = 0; i < encoders.size(); ++i)
    {
        if (!encoders[i])
        {
            throw std::invalid_argument("benchmark received a null encoder");
        }
        const auto start = std::chrono::steady_clock::now();
        const auto encoded = encoders[i]->encodeBatch(test_data);
        const double elapsed_ms = elapsedMilliseconds(start, std::chrono::steady_clock::now());
        for (const auto &tensor : encoded)
        {
            checkedTensorData(tensor, "benchmark encoder output");
        }
        const double seconds = std::max(elapsed_ms / 1000.0, kMinBenchmarkSeconds);
        throughput[encoderKey(*encoders[i], i)] = static_cast<double>(test_data.size()) / seconds;
    }
    return throughput;
}

} // namespace nerve::encoders
