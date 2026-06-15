#include "calibration_model_detail.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <ranges>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace nerve::runtime::detail
{
namespace
{

std::string escapeJson(const std::string &value)
{
    std::string escaped;
    escaped.reserve(value.size());
    for (const char ch : value)
    {
        if (ch == '\\' || ch == '"')
        {
            escaped.push_back('\\');
        }
        escaped.push_back(ch);
    }
    return escaped;
}

bool extractJsonString(const std::string &line, const std::string &key, std::string *value)
{
    const std::string prefix = "\"" + key + "\":\"";
    const std::size_t start = line.find(prefix);
    if (start == std::string::npos)
    {
        return false;
    }
    std::size_t index = start + prefix.size();
    std::string decoded;
    bool escaped = false;
    while (index < line.size())
    {
        const char ch = line[index++];
        if (escaped)
        {
            switch (ch)
            {
                case '"':
                case '\\':
                case '/':
                    decoded.push_back(ch);
                    break;
                case 'n':
                    decoded.push_back('\n');
                    break;
                case 'r':
                    decoded.push_back('\r');
                    break;
                case 't':
                    decoded.push_back('\t');
                    break;
                default:
                    return false;
            }
            escaped = false;
            continue;
        }
        if (ch == '\\')
        {
            escaped = true;
            continue;
        }
        if (ch == '"')
        {
            *value = std::move(decoded);
            return true;
        }
        decoded.push_back(ch);
    }
    return false;
}

bool extractJsonDouble(const std::string &line, const std::string &key, double *value)
{
    const std::string prefix = "\"" + key + "\":";
    const std::size_t start = line.find(prefix);
    if (start == std::string::npos)
    {
        return false;
    }
    std::size_t index = start + prefix.size();
    std::size_t end = index;
    while (end < line.size() && line[end] != ',' && line[end] != '}')
    {
        ++end;
    }
    const std::string numeric = trimCopy(line.substr(index, end - index));
    if (numeric.empty())
    {
        return false;
    }
    try
    {
        std::size_t parsed = 0;
        *value = std::stod(numeric, &parsed);
        return parsed == numeric.size() && std::isfinite(*value);
    }
    catch (const std::invalid_argument &)
    {
        return false;
    }
    catch (const std::out_of_range &)
    {
        return false;
    }
}

bool extractJsonInt64(const std::string &line, const std::string &key, std::int64_t *value)
{
    const std::string prefix = "\"" + key + "\":";
    const std::size_t start = line.find(prefix);
    if (start == std::string::npos)
    {
        return false;
    }
    std::size_t index = start + prefix.size();
    std::size_t end = index;
    while (end < line.size() && line[end] != ',' && line[end] != '}')
    {
        ++end;
    }
    const std::string numeric = trimCopy(line.substr(index, end - index));
    if (numeric.empty())
    {
        return false;
    }
    try
    {
        std::size_t parsed = 0;
        *value = std::stoll(numeric, &parsed);
        return parsed == numeric.size();
    }
    catch (const std::invalid_argument &)
    {
        return false;
    }
    catch (const std::out_of_range &)
    {
        return false;
    }
}

bool isCleanStringField(const std::string &value)
{
    return !value.empty() && std::ranges::all_of(value, [](char ch) {
        return static_cast<unsigned char>(ch) >= 0x20;
    });
}

} // namespace

std::string trimCopy(const std::string &value)
{
    std::size_t begin = 0;
    while (begin < value.size() && std::isspace(static_cast<unsigned char>(value[begin])) != 0)
    {
        ++begin;
    }
    std::size_t end = value.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1])) != 0)
    {
        --end;
    }
    return value.substr(begin, end - begin);
}

std::string serializeSampleJson(const CalibrationSample &sample)
{
    std::ostringstream json;
    json << std::setprecision(17);
    json << '{' << "\"schema_version\":\"" << escapeJson(sample.schema_version) << "\","
         << "\"hardware_fingerprint\":\"" << escapeJson(sample.hardware_fingerprint) << "\","
         << "\"problem_bucket\":\"" << escapeJson(sample.problem_bucket) << "\","
         << "\"algorithm\":\"" << escapeJson(sample.algorithm) << "\","
         << "\"predicted_time_ms\":" << sample.predicted_time_ms << ','
         << "\"predicted_memory_mb\":" << sample.predicted_memory_mb << ','
         << "\"observed_time_ms\":" << sample.observed_time_ms << ','
         << "\"observed_memory_mb\":" << sample.observed_memory_mb << ','
         << "\"confidence\":" << sample.confidence << ','
         << "\"relative_error\":" << sample.relative_error << ','
         << "\"timestamp_unix_ms\":" << sample.timestamp_unix_ms << '}';
    return json.str();
}

bool isValidCalibrationSample(const CalibrationSample &sample)
{
    if (sample.schema_version != "v1" || !isCleanStringField(sample.hardware_fingerprint) ||
        !isCleanStringField(sample.problem_bucket) || !isCleanStringField(sample.algorithm))
    {
        return false;
    }
    if (sample.timestamp_unix_ms <= 0)
    {
        return false;
    }
    const bool finite_nonnegative =
        std::isfinite(sample.predicted_time_ms) && sample.predicted_time_ms >= 0.0 &&
        std::isfinite(sample.predicted_memory_mb) && sample.predicted_memory_mb >= 0.0 &&
        std::isfinite(sample.observed_time_ms) && sample.observed_time_ms >= 0.0 &&
        std::isfinite(sample.observed_memory_mb) && sample.observed_memory_mb >= 0.0 &&
        std::isfinite(sample.relative_error) && sample.relative_error >= 0.0;
    return finite_nonnegative && std::isfinite(sample.confidence) && sample.confidence >= 0.0 &&
           sample.confidence <= 1.0;
}

bool parseJsonSample(const std::string &line, CalibrationSample *sample)
{
    CalibrationSample parsed;
    if (!extractJsonString(line, "schema_version", &parsed.schema_version) ||
        !extractJsonString(line, "hardware_fingerprint", &parsed.hardware_fingerprint) ||
        !extractJsonString(line, "problem_bucket", &parsed.problem_bucket) ||
        !extractJsonString(line, "algorithm", &parsed.algorithm) ||
        !extractJsonDouble(line, "predicted_time_ms", &parsed.predicted_time_ms) ||
        !extractJsonDouble(line, "predicted_memory_mb", &parsed.predicted_memory_mb) ||
        !extractJsonDouble(line, "observed_time_ms", &parsed.observed_time_ms) ||
        !extractJsonDouble(line, "observed_memory_mb", &parsed.observed_memory_mb) ||
        !extractJsonDouble(line, "confidence", &parsed.confidence) ||
        !extractJsonDouble(line, "relative_error", &parsed.relative_error) ||
        !extractJsonInt64(line, "timestamp_unix_ms", &parsed.timestamp_unix_ms))
    {
        return false;
    }
    if (!isValidCalibrationSample(parsed))
    {
        return false;
    }
    *sample = std::move(parsed);
    return true;
}

} // namespace nerve::runtime::detail
