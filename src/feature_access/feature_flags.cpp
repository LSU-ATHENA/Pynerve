#include "nerve/feature_access/feature_flags.hpp"

#include <algorithm>
#include <array>
#include <ranges>
#include <utility>

namespace nerve::feature_access
{
namespace
{

constexpr std::array kDefaultFeatureStates{
    std::pair{FeatureFlag::PH4, TOPOLOGIB_FEATURE_FLAG_PH4 != 0},
    std::pair{FeatureFlag::DIFFERENTIABLE_PERSISTENCE,
              TOPOLOGIB_FEATURE_FLAG_DIFFERENTIABLE_PERSISTENCE != 0},
    std::pair{FeatureFlag::SHEAF_LAPLACIAN_ADVANCED,
              TOPOLOGIB_FEATURE_FLAG_SHEAF_LAPLACIAN_ADVANCED != 0},
    std::pair{FeatureFlag::ADVANCED_STREAMING, TOPOLOGIB_FEATURE_FLAG_ADVANCED_STREAMING != 0},
    std::pair{FeatureFlag::RESEARCH_MODE, TOPOLOGIB_FEATURE_FLAG_RESEARCH_MODE != 0},
};

} // namespace

FeatureFlagManager &FeatureFlagManager::instance()
{
    static FeatureFlagManager manager;
    return manager;
}

FeatureFlagManager::FeatureFlagManager()
{
    initializeDefaultFeatures();
}

bool FeatureFlagManager::defaultFeatureState(FeatureFlag flag)
{
    for (const auto &[candidate, enabled] : kDefaultFeatureStates)
    {
        if (candidate == flag)
        {
            return enabled;
        }
    }
    return false;
}

bool FeatureFlagManager::isEnabled(FeatureFlag flag) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = feature_states_.find(flag);
    if (it != feature_states_.end())
    {
        return it->second;
    }
    return defaultFeatureState(flag);
}

void FeatureFlagManager::setEnabled(FeatureFlag flag, bool enabled)
{
    bool state_changed = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = feature_states_.find(flag);
        const bool old_state = it != feature_states_.end() ? it->second : defaultFeatureState(flag);
        feature_states_[flag] = enabled;
        state_changed = old_state != enabled;
    }
    if (state_changed)
    {
        notifyFeatureChange(flag, enabled);
    }
}

void FeatureFlagManager::enable(FeatureFlag flag)
{
    setEnabled(flag, true);
}
void FeatureFlagManager::disable(FeatureFlag flag)
{
    setEnabled(flag, false);
}

void FeatureFlagManager::setEnabledFlags(const std::unordered_set<FeatureFlag> &flags)
{
    std::vector<std::pair<FeatureFlag, bool>> changed;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto &[flag, state] : feature_states_)
        {
            const bool desired_state = flags.contains(flag);
            if (state != desired_state)
            {
                changed.emplace_back(flag, desired_state);
            }
            state = desired_state;
        }
        for (FeatureFlag flag : flags)
        {
            if (feature_states_.contains(flag))
            {
                continue;
            }
            const bool old_state = defaultFeatureState(flag);
            feature_states_[flag] = true;
            if (!old_state)
            {
                changed.emplace_back(flag, true);
            }
        }
    }
    for (const auto &[flag, enabled] : changed)
    {
        notifyFeatureChange(flag, enabled);
    }
}

void FeatureFlagManager::enableAll()
{
    std::vector<FeatureFlag> changed;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto &[flag, state] : feature_states_)
        {
            if (!state)
            {
                changed.push_back(flag);
            }
            state = true;
        }
    }
    for (FeatureFlag flag : changed)
    {
        notifyFeatureChange(flag, true);
    }
}

void FeatureFlagManager::disableAll()
{
    std::vector<FeatureFlag> changed;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto &[flag, state] : feature_states_)
        {
            if (state)
            {
                changed.push_back(flag);
            }
            state = false;
        }
    }
    for (FeatureFlag flag : changed)
    {
        notifyFeatureChange(flag, false);
    }
}

void FeatureFlagManager::resetToDefaults()
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        feature_states_.clear();
        feature_info_.clear();
        feature_callbacks_.clear();
        feature_groups_.clear();
        config_parse_errors_.clear();
        diagnostics_.clear();
    }
    initializeDefaultFeatures();
}

FeatureFlagInfo FeatureFlagManager::getFeatureInfo(FeatureFlag flag) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = feature_info_.find(flag);
    if (it != feature_info_.end())
    {
        return it->second;
    }
    FeatureFlagInfo info;
    info.flag = flag;
    info.name = feature_flag_to_string(flag);
    info.description = "Unregistered feature flag";
    info.category = "unregistered";
    info.enabledByDefault = defaultFeatureState(flag);
    info.requiresRebuild = false;
    return info;
}

std::vector<FeatureFlagInfo> FeatureFlagManager::getAllFeatures() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<FeatureFlagInfo> features;
    features.reserve(feature_info_.size());
    for (const auto &entry : feature_info_)
    {
        features.push_back(entry.second);
    }
    return features;
}

std::vector<FeatureFlagInfo>
FeatureFlagManager::getFeaturesByCategory(const std::string &category) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<FeatureFlagInfo> features;
    for (const auto &entry : feature_info_)
    {
        const auto &info = entry.second;
        if (info.category == category)
        {
            features.push_back(info);
        }
    }
    return features;
}

std::vector<FeatureFlag> FeatureFlagManager::getEnabledFeatures() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<FeatureFlag> enabled;
    for (const auto &[flag, state] : feature_states_)
    {
        if (state)
        {
            enabled.push_back(flag);
        }
    }
    return enabled;
}

void FeatureFlagManager::registerFeature(const FeatureFlagInfo &info)
{
    std::lock_guard<std::mutex> lock(mutex_);
    feature_info_[info.flag] = info;
    if (!feature_states_.contains(info.flag))
    {
        feature_states_[info.flag] = info.enabledByDefault;
    }
    if (!info.category.empty())
    {
        auto &group = feature_groups_[info.category];
        if (std::ranges::find(group, info.flag) == group.end())
        {
            group.push_back(info.flag);
        }
    }
}

void FeatureFlagManager::unregisterFeature(FeatureFlag flag)
{
    std::lock_guard<std::mutex> lock(mutex_);
    feature_info_.erase(flag);
    feature_states_.erase(flag);
    feature_callbacks_.erase(flag);
    for (auto &entry : feature_groups_)
    {
        auto &flags = entry.second;
        flags.erase(std::remove(flags.begin(), flags.end(), flag), flags.end());
    }
}

bool FeatureFlagManager::isFeatureAvailable(FeatureFlag flag) const
{
    FeatureFlagInfo info = getFeatureInfo(flag);
    if (info.availability_check)
    {
        return info.availability_check();
    }
    return true;
}

std::string FeatureFlagManager::getFeatureStatus(FeatureFlag flag) const
{
    const bool enabled = isEnabled(flag);
    const bool available = isFeatureAvailable(flag);
    if (enabled && available)
    {
        return "ENABLED";
    }
    if (enabled && !available)
    {
        return "ENABLED_BUT_MISSING";
    }
    return "DISABLED";
}

std::vector<std::string> FeatureFlagManager::getConfigParseErrors() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return config_parse_errors_;
}
void FeatureFlagManager::clearConfigParseErrors()
{
    std::lock_guard<std::mutex> lock(mutex_);
    config_parse_errors_.clear();
}
std::vector<std::string> FeatureFlagManager::getDiagnostics() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return diagnostics_;
}
void FeatureFlagManager::clearDiagnostics()
{
    std::lock_guard<std::mutex> lock(mutex_);
    diagnostics_.clear();
}

void FeatureFlagManager::setFeatureCallback(FeatureFlag flag, std::function<void(bool)> callback)
{
    std::lock_guard<std::mutex> lock(mutex_);
    feature_callbacks_[flag] = std::move(callback);
}

void FeatureFlagManager::enableFeatureGroup(const std::string &group_name)
{
    std::vector<FeatureFlag> changed;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = feature_groups_.find(group_name);
        if (it == feature_groups_.end())
        {
            return;
        }
        for (FeatureFlag flag : it->second)
        {
            const bool old_state =
                feature_states_.contains(flag) ? feature_states_[flag] : defaultFeatureState(flag);
            feature_states_[flag] = true;
            if (!old_state)
            {
                changed.push_back(flag);
            }
        }
    }
    for (FeatureFlag flag : changed)
    {
        notifyFeatureChange(flag, true);
    }
}

void FeatureFlagManager::disableFeatureGroup(const std::string &group_name)
{
    std::vector<FeatureFlag> changed;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = feature_groups_.find(group_name);
        if (it == feature_groups_.end())
        {
            return;
        }
        for (FeatureFlag flag : it->second)
        {
            const bool old_state =
                feature_states_.contains(flag) ? feature_states_[flag] : defaultFeatureState(flag);
            feature_states_[flag] = false;
            if (old_state)
            {
                changed.push_back(flag);
            }
        }
    }
    for (FeatureFlag flag : changed)
    {
        notifyFeatureChange(flag, false);
    }
}

FeatureFlagGuard::FeatureFlagGuard(FeatureFlag flag, bool scoped_state)
    : flag_(flag)
    , original_state_(FeatureFlagManager::instance().isEnabled(flag))
    , restored_(false)
{
    FeatureFlagManager::instance().setEnabled(flag_, scoped_state);
}

FeatureFlagGuard::~FeatureFlagGuard()
{
    if (!restored_)
    {
        FeatureFlagManager::instance().setEnabled(flag_, original_state_);
    }
}

FeatureFlagGuard::FeatureFlagGuard(FeatureFlagGuard &&other) noexcept
    : flag_(other.flag_)
    , original_state_(other.original_state_)
    , restored_(other.restored_)
{
    other.restored_ = true;
}

FeatureFlagGuard &FeatureFlagGuard::operator=(FeatureFlagGuard &&other) noexcept
{
    if (this != &other)
    {
        if (!restored_)
        {
            FeatureFlagManager::instance().setEnabled(flag_, original_state_);
        }
        flag_ = other.flag_;
        original_state_ = other.original_state_;
        restored_ = other.restored_;
        other.restored_ = true;
    }
    return *this;
}

} // namespace nerve::feature_access
