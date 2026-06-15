#include "nerve/feature_access/feature_flags.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace nerve::feature_access
{
namespace
{

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

std::string normalizeBooleanToken(const std::string &value)
{
    std::string normalized = trimCopy(value);
    std::ranges::transform(normalized, normalized.begin(),
                           [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return normalized;
}

std::optional<bool> parseEnabledValue(const std::string &value)
{
    const std::string normalized = normalizeBooleanToken(value);
    if (normalized == "true" || normalized == "1" || normalized == "enabled" ||
        normalized == "on" || normalized == "yes")
    {
        return true;
    }
    if (normalized == "false" || normalized == "0" || normalized == "disabled" ||
        normalized == "off" || normalized == "no")
    {
        return false;
    }
    return std::nullopt;
}

void applyEnvironmentFlag(FeatureFlagManager &manager, const char *name, FeatureFlag flag)
{
    const char *value = std::getenv(name);
    if (value == nullptr)
    {
        return;
    }
    const std::optional<bool> enabled = parseEnabledValue(value);
    if (enabled.has_value())
    {
        manager.setEnabled(flag, *enabled);
    }
}

} // namespace

void FeatureFlagManager::loadFromConfig(const std::string &config_file)
{
    std::ifstream file(config_file);
    if (!file.is_open())
    {
        std::lock_guard<std::mutex> lock(mutex_);
        config_parse_errors_.clear();
        return;
    }

    std::vector<std::pair<FeatureFlag, bool>> updates;
    std::vector<std::string> parse_errors;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(file, line))
    {
        ++line_number;
        const std::string trimmed = trimCopy(line);
        if (trimmed.empty() || trimmed[0] == '#')
        {
            continue;
        }
        const std::size_t separator = trimmed.find('=');
        if (separator == std::string::npos)
        {
            parse_errors.push_back("malformed feature flag assignment at line " +
                                   std::to_string(line_number));
            continue;
        }
        const std::string feature_name = trimCopy(trimmed.substr(0, separator));
        const std::string value = trimCopy(trimmed.substr(separator + 1));
        const std::optional<FeatureFlag> flag = parseFeatureFlag(feature_name);
        if (!flag.has_value())
        {
            parse_errors.push_back("unknown feature flag at line " + std::to_string(line_number) +
                                   ": " + feature_name);
            continue;
        }
        const std::optional<bool> enabled = parseEnabledValue(value);
        if (!enabled.has_value())
        {
            parse_errors.push_back("invalid feature flag value at line " +
                                   std::to_string(line_number) + ": " + value);
            continue;
        }
        updates.emplace_back(*flag, *enabled);
    }

    std::vector<std::pair<FeatureFlag, bool>> changed;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        config_parse_errors_ = std::move(parse_errors);
        for (const auto &[flag, enabled] : updates)
        {
            const bool old_state =
                feature_states_.contains(flag) ? feature_states_[flag] : defaultFeatureState(flag);
            feature_states_[flag] = enabled;
            if (old_state != enabled)
            {
                changed.emplace_back(flag, enabled);
            }
        }
    }
    for (const auto &[flag, enabled] : changed)
    {
        notifyFeatureChange(flag, enabled);
    }
}

void FeatureFlagManager::saveToConfig(const std::string &config_file) const
{
    std::ofstream file(config_file);
    if (!file.is_open())
    {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    file << "# Nerve Feature Flag Configuration\n";
    file << "# Generated automatically - do not edit manually\n\n";
    for (const auto &[flag, info] : feature_info_)
    {
        auto state_it = feature_states_.find(flag);
        const bool enabled =
            state_it != feature_states_.end() ? state_it->second : defaultFeatureState(flag);
        file << info.name << " = " << (enabled ? "enabled" : "disabled") << "\n";
        file << "# " << info.description << "\n\n";
    }
}

void FeatureFlagManager::setFromEnvironment()
{
    applyEnvironmentFlag(*this, "TOPOLOGIB_ENABLE_PH4", FeatureFlag::PH4);
    applyEnvironmentFlag(*this, "TOPOLOGIB_ENABLE_DIFFERENTIABLE_PERSISTENCE",
                         FeatureFlag::DIFFERENTIABLE_PERSISTENCE);
    applyEnvironmentFlag(*this, "TOPOLOGIB_ENABLE_SHEAF_LAPLACIAN_ADVANCED",
                         FeatureFlag::SHEAF_LAPLACIAN_ADVANCED);

    const char *advanced_env = std::getenv("TOPOLOGIB_ENABLE_ALL_EXPERIMENTAL");
    if (advanced_env != nullptr && parseEnabledValue(advanced_env).value_or(false))
    {
        enableAll();
    }
}

} // namespace nerve::feature_access
