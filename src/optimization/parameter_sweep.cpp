#include "nerve/optimization/parameter_sweep.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_set>

namespace nerve::optimization
{

std::vector<double> Parameter::generateValues() const
{
    if (!custom_values.empty())
    {
        return custom_values;
    }
    if (num_steps <= 1)
    {
        return {min_value};
    }
    std::vector<double> values;
    values.reserve(num_steps);
    if (scale_type == "log")
    {
        if (min_value <= 0.0 || max_value <= 0.0)
        {
            throw std::invalid_argument("log scale requires positive bounds");
        }
        double log_min = std::log(min_value);
        double log_max = std::log(max_value);
        double step = (log_max - log_min) / static_cast<double>(num_steps - 1);
        for (size_t i = 0; i < num_steps; ++i)
        {
            values.push_back(std::exp(log_min + step * static_cast<double>(i)));
        }
    }
    else
    {
        double step = (max_value - min_value) / static_cast<double>(num_steps - 1);
        for (size_t i = 0; i < num_steps; ++i)
        {
            values.push_back(min_value + step * static_cast<double>(i));
        }
    }
    return values;
}

bool Parameter::isValid() const
{
    return !name.empty() && min_value <= max_value && num_steps > 0;
}

std::vector<uint8_t> ParameterCombination::serialize() const
{
    std::vector<uint8_t> buf;
    uint32_t num_entries = static_cast<uint32_t>(values.size());
    buf.insert(buf.end(), reinterpret_cast<const uint8_t *>(&num_entries),
               reinterpret_cast<const uint8_t *>(&num_entries) + sizeof(num_entries));
    for (const auto &[key, val] : values)
    {
        uint32_t key_len = static_cast<uint32_t>(key.size());
        buf.insert(buf.end(), reinterpret_cast<const uint8_t *>(&key_len),
                   reinterpret_cast<const uint8_t *>(&key_len) + sizeof(key_len));
        buf.insert(buf.end(), key.begin(), key.end());
        buf.insert(buf.end(), reinterpret_cast<const uint8_t *>(&val),
                   reinterpret_cast<const uint8_t *>(&val) + sizeof(val));
    }
    {
        const auto &h = params_hash;
        buf.insert(buf.end(), h.begin(), h.end());
    }
    buf.insert(buf.end(), reinterpret_cast<const uint8_t *>(&combination_id),
               reinterpret_cast<const uint8_t *>(&combination_id) + sizeof(combination_id));
    uint32_t num_tags = static_cast<uint32_t>(tags.size());
    buf.insert(buf.end(), reinterpret_cast<const uint8_t *>(&num_tags),
               reinterpret_cast<const uint8_t *>(&num_tags) + sizeof(num_tags));
    for (const auto &tag : tags)
    {
        uint32_t tag_len = static_cast<uint32_t>(tag.size());
        buf.insert(buf.end(), reinterpret_cast<const uint8_t *>(&tag_len),
                   reinterpret_cast<const uint8_t *>(&tag_len) + sizeof(tag_len));
        buf.insert(buf.end(), tag.begin(), tag.end());
    }
    return buf;
}

bool ParameterCombination::deserialize(const std::vector<uint8_t> &data)
{
    if (data.size() < sizeof(uint32_t))
    {
        return false;
    }
    const uint8_t *ptr = data.data();
    const uint8_t *end = data.data() + data.size();
    uint32_t num_entries;
    std::memcpy(&num_entries, ptr, sizeof(num_entries));
    ptr += sizeof(num_entries);
    values.clear();
    for (uint32_t i = 0; i < num_entries; ++i)
    {
        if (ptr + sizeof(uint32_t) > end)
        {
            return false;
        }
        uint32_t key_len;
        std::memcpy(&key_len, ptr, sizeof(key_len));
        ptr += sizeof(key_len);
        if (ptr + key_len > end)
        {
            return false;
        }
        std::string key(reinterpret_cast<const char *>(ptr), key_len);
        ptr += key_len;
        if (ptr + sizeof(double) > end)
        {
            return false;
        }
        double val;
        std::memcpy(&val, ptr, sizeof(val));
        ptr += sizeof(val);
        values[key] = val;
    }
    if (ptr + params_hash.size() > end)
    {
        return false;
    }
    std::memcpy(params_hash.data(), ptr, params_hash.size());
    ptr += params_hash.size();
    if (ptr + sizeof(combination_id) > end)
    {
        return false;
    }
    std::memcpy(&combination_id, ptr, sizeof(combination_id));
    ptr += sizeof(combination_id);
    if (ptr + sizeof(uint32_t) > end)
    {
        return false;
    }
    uint32_t num_tags;
    std::memcpy(&num_tags, ptr, sizeof(num_tags));
    ptr += sizeof(num_tags);
    tags.clear();
    for (uint32_t i = 0; i < num_tags; ++i)
    {
        if (ptr + sizeof(uint32_t) > end)
        {
            return false;
        }
        uint32_t tag_len;
        std::memcpy(&tag_len, ptr, sizeof(tag_len));
        ptr += sizeof(tag_len);
        if (ptr + tag_len > end)
        {
            return false;
        }
        std::string tag(reinterpret_cast<const char *>(ptr), tag_len);
        ptr += tag_len;
        tags.push_back(tag);
    }
    return true;
}

std::array<uint8_t, 32> ParameterCombination::computeHash() const
{
    auto data = serialize();
    std::array<uint8_t, 32> hash{};
    uint64_t h = 0x243f6a8885a308d3ULL;
    for (size_t i = 0; i < data.size(); ++i)
    {
        h = h ^ (static_cast<uint64_t>(data[i]) << ((i % 8) * 8));
        h = h * 0xc6a4a7935bd1e995ULL;
    }
    for (size_t i = 0; i < 8; ++i)
    {
        hash[i] = static_cast<uint8_t>((h >> (i * 8)) & 0xFF);
    }
    return hash;
}

std::string ParameterCombination::toString() const
{
    std::string s;
    for (const auto &[k, v] : values)
    {
        if (!s.empty())
        {
            s += ", ";
        }
        s += k + "=" + std::to_string(v);
    }
    return s;
}

ParameterSweepEngine::ParameterSweepEngine(const SweepConfig &config)
    : config_(config)
{}

ParameterSweepEngine::~ParameterSweepEngine() = default;

std::vector<EvaluationResult>
ParameterSweepEngine::runSweep(const std::vector<std::vector<float>> &input_data,
                               FeatureComputationFunction compute_function)
{
    sweep_running_ = true;
    std::vector<EvaluationResult> results;
    auto combinations = generateCombinations();
    progress_.total_combinations = combinations.size();
    progress_.completed_combinations = 0;
    progress_.failed_combinations = 0;
    progress_.best_score = -std::numeric_limits<double>::infinity();
    for (const auto &comb : combinations)
    {
        auto result = evaluateCombination(comb, input_data, compute_function);
        results.push_back(result);
        progress_.completed_combinations++;
        if (!result.success)
        {
            progress_.failed_combinations++;
        }
        if (result.evaluation_score > progress_.best_score)
        {
            progress_.best_score = result.evaluation_score;
            progress_.best_combination = comb;
        }
        progress_.completion_percentage = 100.0 *
                                          static_cast<double>(progress_.completed_combinations) /
                                          static_cast<double>(progress_.total_combinations);
        if (shouldStopEarly())
        {
            break;
        }
    }
    progress_.average_score = 0.0;
    if (!results.empty())
    {
        double sum = 0.0;
        for (const auto &r : results)
        {
            sum += r.evaluation_score;
        }
        progress_.average_score = sum / static_cast<double>(results.size());
    }
    sweep_running_ = false;
    return results;
}

std::future<std::vector<EvaluationResult>>
ParameterSweepEngine::runSweepAsync(const std::vector<std::vector<float>> &input_data,
                                    FeatureComputationFunction compute_function)
{
    return std::async(std::launch::async, [this, &input_data, compute_function]() {
        return runSweep(input_data, compute_function);
    });
}

EvaluationResult
ParameterSweepEngine::evaluateCombination(const ParameterCombination &combination,
                                          const std::vector<std::vector<float>> &input_data,
                                          FeatureComputationFunction compute_function)
{
    EvaluationResult result;
    result.combination = combination;
    try
    {
        std::unordered_map<std::string, std::vector<uint8_t>> cache;
        result = compute_function(combination, input_data, cache);
    }
    catch (const std::exception &e)
    {
        result.success = false;
        result.error_message = e.what();
        result.evaluation_score = -std::numeric_limits<double>::infinity();
    }
    return result;
}

std::vector<ParameterCombination> ParameterSweepEngine::generateCombinations()
{
    if (config_.sweep_strategy == "random")
    {
        return generateRandomCombinations(config_.max_total_combinations);
    }
    return generateGridCombinations();
}

std::vector<ParameterCombination> ParameterSweepEngine::generateRandomCombinations(size_t count)
{
    std::vector<ParameterCombination> result;
    if (config_.parameters.empty())
    {
        return result;
    }
    for (size_t c = 0; c < count && c < config_.max_total_combinations; ++c)
    {
        ParameterCombination comb;
        comb.combination_id = c;
        for (const auto &param : config_.parameters)
        {
            auto vals = param.generateValues();
            size_t idx = c % vals.size();
            comb.values[param.name] = vals[idx];
        }
        comb.params_hash = comb.computeHash();
        result.push_back(comb);
    }
    return result;
}

std::vector<ParameterCombination> ParameterSweepEngine::generateAdaptiveCombinations(
    const std::vector<EvaluationResult> &previous_results)
{
    (void)previous_results;
    return generateCombinations();
}

std::vector<ParameterCombination> ParameterSweepEngine::generateGridCombinations()
{
    std::vector<ParameterCombination> result;
    if (config_.parameters.empty())
    {
        return result;
    }
    std::vector<std::vector<double>> param_values;
    param_values.reserve(config_.parameters.size());
    for (const auto &param : config_.parameters)
    {
        param_values.push_back(param.generateValues());
    }
    std::vector<size_t> indices(config_.parameters.size(), 0);
    size_t total = 1;
    for (const auto &vals : param_values)
    {
        total *= vals.size();
    }
    for (size_t c = 0; c < total && c < config_.max_total_combinations; ++c)
    {
        ParameterCombination comb;
        comb.combination_id = c;
        for (size_t i = 0; i < config_.parameters.size(); ++i)
        {
            comb.values[config_.parameters[i].name] = param_values[i][indices[i]];
        }
        comb.params_hash = comb.computeHash();
        result.push_back(comb);
        size_t carry = 1;
        for (size_t i = 0; i < indices.size(); ++i)
        {
            indices[i] += carry;
            carry = indices[i] / param_values[i].size();
            indices[i] %= param_values[i].size();
            if (carry == 0)
            {
                break;
            }
        }
    }
    return result;
}

void ParameterSweepEngine::setMemoizationCache(size_t cache_size)
{
    config_.memoization_cache_size = cache_size;
}

void ParameterSweepEngine::clearMemoizationCache()
{
    std::unique_lock lock(cache_mutex_);
    memoization_cache_.clear();
}

std::unordered_map<std::string, std::vector<uint8_t>>
ParameterSweepEngine::getMemoizationCache() const
{
    std::shared_lock lock(cache_mutex_);
    return memoization_cache_;
}

void ParameterSweepEngine::enableIntermediateReuse(bool enable)
{
    config_.enableIntermediateReuse = enable;
}

std::unordered_map<std::string, std::vector<uint8_t>>
ParameterSweepEngine::getIntermediateCache() const
{
    std::shared_lock lock(intermediate_mutex_);
    return intermediate_cache_;
}

ParameterSweepEngine::SweepProgress ParameterSweepEngine::getProgress() const
{
    std::lock_guard lock(progress_mutex_);
    return progress_;
}

void ParameterSweepEngine::resetProgress()
{
    std::lock_guard lock(progress_mutex_);
    progress_ = SweepProgress{};
}

void ParameterSweepEngine::enableEarlyStopping(bool enable)
{
    config_.enableEarlyStopping = enable;
}

bool ParameterSweepEngine::shouldStopEarly() const
{
    if (!config_.enableEarlyStopping)
    {
        return false;
    }
    return early_stop_triggered_;
}

} // namespace nerve::optimization
