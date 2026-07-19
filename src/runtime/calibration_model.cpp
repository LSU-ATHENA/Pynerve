#include "calibration_model_detail.hpp"
#include "nerve/runtime/calibration_model.hpp"

#include <fcntl.h>
#ifndef _WIN32
#include <unistd.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <ranges>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
namespace nerve::runtime
{
namespace
{
constexpr double kMinGateConfidence = 0.80;
constexpr double kMaxGateErrorBound = 0.25;
constexpr std::size_t kMinGateSamples = 20;
struct BucketAggregate
{
    double predicted_time_sum = 0.0;
    double predicted_memory_sum = 0.0;
    double observed_time_sum = 0.0;
    double observed_memory_sum = 0.0;
    double confidence_sum = 0.0;
    std::vector<double> relative_errors;
    std::size_t sample_count = 0;
};
struct CalibrationState
{
    std::mutex mutex;
    std::string store_path = defaultCalibrationStorePath();
    std::unordered_map<std::string, BucketAggregate> buckets;
    bool loaded = false;
};
CalibrationState &calibrationState()
{
    static CalibrationState state;
    return state;
}
std::int64_t unixTimeMsNow()
{
    using namespace std::chrono;
    return static_cast<std::int64_t>(
        duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count());
}
std::string makeBucketId(const PredictionKey &key)
{
    return key.hardware_fingerprint + "|" + key.algorithm + "|" + key.problem_bucket;
}
std::string makeBucketId(const CalibrationSample &sample)
{
    PredictionKey key;
    key.hardware_fingerprint = sample.hardware_fingerprint;
    key.problem_bucket = sample.problem_bucket;
    key.algorithm = sample.algorithm;
    return makeBucketId(key);
}
void ingestSample(BucketAggregate *aggregate, const CalibrationSample &sample)
{
    const double safe_predicted_time = std::max(std::abs(sample.predicted_time_ms), 1e-9);
    const double computed_relative_error =
        std::abs(sample.observed_time_ms - sample.predicted_time_ms) / safe_predicted_time;
    const double relative_error =
        (std::isfinite(sample.relative_error) && sample.relative_error >= 0.0)
            ? sample.relative_error
            : computed_relative_error;
    const double confidence = std::clamp(sample.confidence, 0.0, 1.0);
    aggregate->predicted_time_sum += sample.predicted_time_ms;
    aggregate->predicted_memory_sum += sample.predicted_memory_mb;
    aggregate->observed_time_sum += sample.observed_time_ms;
    aggregate->observed_memory_sum += sample.observed_memory_mb;
    aggregate->confidence_sum += confidence;
    aggregate->relative_errors.push_back(relative_error);
    aggregate->sample_count += 1;
}
errors::ErrorResult<void> ensureLoadedIfNeeded()
{
    auto &state = calibrationState();
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        if (state.loaded)
        {
            return errors::ErrorResult<void>::success();
        }
    }
    return loadCalibrationModel();
}
errors::ErrorResult<void> appendLineAtomic(const std::string &path, const std::string &line)
{
    const std::filesystem::path outputPath(path);
    const std::filesystem::path parent = outputPath.parent_path();
    if (!parent.empty())
    {
        std::error_code ec;
        std::filesystem::create_directories(parent, ec);
        if (ec)
        {
            return errors::ErrorResult<void>::error(errors::ErrorCode::E00_IO_TIMEOUT);
        }
    }
#ifndef _WIN32
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (fd < 0)
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E00_IO_TIMEOUT);
    }
    const std::string line_with_newline = line + '\n';
    const char *buffer = line_with_newline.c_str();
    std::size_t remaining = line_with_newline.size();
    while (remaining > 0)
    {
        const ssize_t written = ::write(fd, buffer, remaining);
        if (written <= 0)
        {
            ::close(fd);
            return errors::ErrorResult<void>::error(errors::ErrorCode::E00_IO_TIMEOUT);
        }
        buffer += written;
        remaining -= static_cast<std::size_t>(written);
    }
    if (::fsync(fd) != 0)
    {
        ::close(fd);
        return errors::ErrorResult<void>::error(errors::ErrorCode::E00_IO_TIMEOUT);
    }
    if (::close(fd) != 0)
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E00_IO_TIMEOUT);
    }
    return errors::ErrorResult<void>::success();
#else
    // Fallback: use C++ standard library file I/O on Windows.
    std::ofstream output(path, std::ios::app);
    if (!output.is_open())
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E00_IO_TIMEOUT);
    }
    output << line << '\n';
    if (output.fail())
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E00_IO_TIMEOUT);
    }
    output.flush();
    return errors::ErrorResult<void>::success();
#endif
}
PredictionWithBounds aggregateToPrediction(const std::string &bucket_id,
                                           const BucketAggregate &aggregate)
{
    PredictionWithBounds prediction;
    if (aggregate.sample_count == 0)
    {
        return prediction;
    }
    prediction.available = true;
    prediction.calibration_bucket_id = bucket_id;
    prediction.sample_count = aggregate.sample_count;
    prediction.predicted_time_ms =
        aggregate.observed_time_sum / static_cast<double>(aggregate.sample_count);
    prediction.predicted_memory_mb =
        aggregate.observed_memory_sum / static_cast<double>(aggregate.sample_count);
    const double raw_confidence =
        aggregate.confidence_sum / static_cast<double>(aggregate.sample_count);
    const double sample_factor = static_cast<double>(aggregate.sample_count) /
                                 static_cast<double>(aggregate.sample_count + 5);
    prediction.confidence = std::clamp(raw_confidence * sample_factor, 0.0, 1.0);
    if (!aggregate.relative_errors.empty())
    {
        std::vector<double> sorted = aggregate.relative_errors;
        std::ranges::sort(sorted);
        const std::size_t index =
            static_cast<std::size_t>(std::floor(0.9 * static_cast<double>(sorted.size() - 1)));
        prediction.error_bound = sorted[index];
    }
    return prediction;
}
} // namespace
std::string defaultCalibrationStorePath()
{
    return ".nerve/calibration/predictor_calibration.v1.jsonl";
}
void setCalibrationStorePathForTesting(const std::string &path)
{
    auto &state = calibrationState();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.store_path = path.empty() ? defaultCalibrationStorePath() : path;
    state.buckets.clear();
    state.loaded = false;
}
errors::ErrorResult<void> loadCalibrationModel()
{
    auto &state = calibrationState();
    std::lock_guard<std::mutex> lock(state.mutex);
    std::ifstream input(state.store_path);
    if (!input.is_open())
    {
        state.buckets.clear();
        state.loaded = true;
        return errors::ErrorResult<void>::success();
    }
    std::unordered_map<std::string, BucketAggregate> loaded_buckets;
    std::string line;
    while (std::getline(input, line))
    {
        const std::string trimmed = detail::trimCopy(line);
        if (trimmed.empty())
        {
            continue;
        }
        CalibrationSample sample;
        if (!detail::parseJsonSample(trimmed, &sample))
        {
            state.buckets.clear();
            state.loaded = false;
            return errors::ErrorResult<void>::error(errors::ErrorCode::E31_SCHEMA_VERSION);
        }
        BucketAggregate &aggregate = loaded_buckets[makeBucketId(sample)];
        ingestSample(&aggregate, sample);
    }
    state.buckets = std::move(loaded_buckets);
    state.loaded = true;
    return errors::ErrorResult<void>::success();
}
errors::ErrorResult<void> recordCalibrationSample(const CalibrationSample &sample)
{
    auto loaded_status = ensureLoadedIfNeeded();
    if (loaded_status.isError())
    {
        return loaded_status;
    }
    CalibrationSample normalized = sample;
    normalized.schema_version = "v1";
    if (normalized.timestamp_unix_ms <= 0)
    {
        normalized.timestamp_unix_ms = unixTimeMsNow();
    }
    if (!detail::isValidCalibrationSample(normalized))
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E50_PH_ABORT);
    }
    std::string store_path;
    {
        auto &state = calibrationState();
        std::lock_guard<std::mutex> lock(state.mutex);
        store_path = state.store_path;
    }
    auto append_status = appendLineAtomic(store_path, detail::serializeSampleJson(normalized));
    if (append_status.isError())
    {
        return append_status;
    }
    auto &state = calibrationState();
    std::lock_guard<std::mutex> lock(state.mutex);
    BucketAggregate &aggregate = state.buckets[makeBucketId(normalized)];
    ingestSample(&aggregate, normalized);
    return errors::ErrorResult<void>::success();
}
PredictionWithBounds queryCalibratedPrediction(const PredictionKey &key)
{
    const auto loaded_status = ensureLoadedIfNeeded();
    if (loaded_status.isError())
    {
        return PredictionWithBounds{};
    }
    const std::string bucket_id = makeBucketId(key);
    auto &state = calibrationState();
    std::lock_guard<std::mutex> lock(state.mutex);
    const auto it = state.buckets.find(bucket_id);
    if (it == state.buckets.end())
    {
        return PredictionWithBounds{};
    }
    return aggregateToPrediction(bucket_id, it->second);
}
SelectionGateDiagnostics evaluatePredictionGate(const PredictionWithBounds &prediction)
{
    SelectionGateDiagnostics diagnostics;
    diagnostics.confidence = prediction.confidence;
    diagnostics.error_bound = prediction.error_bound;
    diagnostics.sample_count = prediction.sample_count;
    diagnostics.calibration_bucket_id = prediction.calibration_bucket_id;
    if (!prediction.available)
    {
        diagnostics.gate_passed = false;
        diagnostics.gate_reason = "missing_prediction_bucket";
        return diagnostics;
    }
    if (prediction.confidence < kMinGateConfidence)
    {
        diagnostics.gate_passed = false;
        diagnostics.gate_reason = "low_confidence";
        return diagnostics;
    }
    if (prediction.error_bound > kMaxGateErrorBound)
    {
        diagnostics.gate_passed = false;
        diagnostics.gate_reason = "high_error_bound";
        return diagnostics;
    }
    if (prediction.sample_count < kMinGateSamples)
    {
        diagnostics.gate_passed = false;
        diagnostics.gate_reason = "insufficient_samples";
        return diagnostics;
    }
    diagnostics.gate_passed = true;
    diagnostics.gate_reason = "passed";
    return diagnostics;
}
} // namespace nerve::runtime
