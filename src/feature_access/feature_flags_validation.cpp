
#include "nerve/feature_access/feature_flags.hpp"

#include <algorithm>
#include <array>
#include <exception>
#include <fstream>
#include <ranges>
#include <sstream>
#include <string_view>
#include <utility>

namespace nerve::feature_access
{
namespace
{

constexpr std::array<std::pair<FeatureFlag, std::string_view>, 16> kFeatureFlagNames{{
    {FeatureFlag::PH4, "PH4"},
    {FeatureFlag::DIFFERENTIABLE_PERSISTENCE, "DIFFERENTIABLE_PERSISTENCE"},
    {FeatureFlag::SHEAF_LAPLACIAN_ADVANCED, "SHEAF_LAPLACIAN_ADVANCED"},
    {FeatureFlag::ADVANCED_STREAMING, "ADVANCED_STREAMING"},
    {FeatureFlag::RESEARCH_MODE, "RESEARCH_MODE"},
    {FeatureFlag::DEBUG_VISUALIZATION, "DEBUG_VISUALIZATION"},
    {FeatureFlag::ADVANCED_PROFILING, "ADVANCED_PROFILING"},
    {FeatureFlag::NEW_SIMPLICIAL_COMPLEX, "NEW_SIMPLICIAL_COMPLEX"},
    {FeatureFlag::ADVANCED_FILTRATION, "ADVANCED_FILTRATION"},
    {FeatureFlag::QUANTUM_TOPOLOGY, "QUANTUM_TOPOLOGY"},
    {FeatureFlag::GPU_ADVANCED_KERNELS, "GPU_ADVANCED_KERNELS"},
    {FeatureFlag::PARALLEL_PERSISTENCE, "PARALLEL_PERSISTENCE"},
    {FeatureFlag::MEMORY_OPTIMIZATION, "MEMORY_OPTIMIZATION"},
    {FeatureFlag::PYTHON_ADVANCED, "PYTHON_ADVANCED"},
    {FeatureFlag::RUST_ADVANCED, "RUST_ADVANCED"},
    {FeatureFlag::JULIA_ADVANCED, "JULIA_ADVANCED"},
}};

} // namespace

bool FeatureFlagManager::validateDependencies() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto &[flag, state] : feature_states_)
    {
        if (state && !checkDependencies(flag))
        {
            return false;
        }
    }
    return true;
}

std::vector<std::string> FeatureFlagManager::getDependencyViolations() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> violations;
    for (const auto &[flag, state] : feature_states_)
    {
        if (!state || checkDependencies(flag))
        {
            continue;
        }
        auto it = feature_info_.find(flag);
        const std::string feature_name =
            it != feature_info_.end() ? it->second.name : feature_flag_to_string(flag);
        violations.push_back("Feature " + feature_name + " has unmet dependencies");
    }
    return violations;
}

std::string FeatureFlagManager::getStatusReport() const
{
    std::vector<std::pair<FeatureFlagInfo, bool>> entries;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        entries.reserve(feature_info_.size());
        for (const auto &[flag, info] : feature_info_)
        {
            auto state_it = feature_states_.find(flag);
            const bool enabled =
                state_it != feature_states_.end() ? state_it->second : defaultFeatureState(flag);
            entries.emplace_back(info, enabled);
        }
    }

    std::ostringstream report;
    report << "Nerve Feature Flag Status Report\n";
    report << "===================================\n\n";

    for (const auto &[info, enabled] : entries)
    {
        report << info.name << ": " << (enabled ? "ENABLED" : "DISABLED") << "\n";
        report << "  Description: " << info.description << "\n";
        report << "  Category: " << info.category << "\n";
        if (!info.dependencies.empty())
        {
            report << "  Dependencies: ";
            for (std::size_t i = 0; i < info.dependencies.size(); ++i)
            {
                if (i > 0)
                {
                    report << ", ";
                }
                report << info.dependencies[i];
            }
            report << "\n";
        }
        if (info.availability_check)
        {
            const bool available = info.availability_check();
            report << "  Available: " << (available ? "YES" : "NO") << "\n";
        }
        report << "\n";
    }
    return report.str();
}

void FeatureFlagManager::initializeDefaultFeatures()
{
    const auto register_default = [this](FeatureFlag flag, const char *name,
                                         const char *description, const char *category) {
        FeatureFlagInfo info;
        info.flag = flag;
        info.name = name;
        info.description = description;
        info.category = category;
        info.enabledByDefault = defaultFeatureState(flag);
        info.requiresRebuild = false;
        registerFeature(info);
    };

    register_default(FeatureFlag::PH4, "PH4", "PH4 production persistent homology algorithms",
                     "persistence");
    register_default(FeatureFlag::DIFFERENTIABLE_PERSISTENCE, "DIFFERENTIABLE_PERSISTENCE",
                     "Differentiable persistence computation for ML integration", "ml");
    register_default(FeatureFlag::SHEAF_LAPLACIAN_ADVANCED, "SHEAF_LAPLACIAN_ADVANCED",
                     "Advanced sheaf Laplacian features for surface analysis", "spectral");
    register_default(FeatureFlag::ADVANCED_STREAMING, "ADVANCED_STREAMING",
                     "Advanced streaming algorithms for real-time topology", "streaming");
    register_default(FeatureFlag::RESEARCH_MODE, "RESEARCH_MODE",
                     "Enable research-specific features and diagnostics", "development");
}

bool FeatureFlagManager::checkDependencies(FeatureFlag flag) const
{
    auto info_it = feature_info_.find(flag);
    if (info_it == feature_info_.end())
    {
        return true;
    }
    const FeatureFlagInfo &info = info_it->second;
    for (const std::string &dependency : info.dependencies)
    {
        const std::optional<FeatureFlag> dep_flag = parseFeatureFlag(dependency);
        if (!dep_flag.has_value())
        {
            return false;
        }
        auto state_it = feature_states_.find(*dep_flag);
        const bool dep_enabled =
            state_it != feature_states_.end() ? state_it->second : defaultFeatureState(*dep_flag);
        if (!dep_enabled)
        {
            return false;
        }
    }
    return true;
}

void FeatureFlagManager::notifyFeatureChange(FeatureFlag flag, bool enabled)
{
    std::function<void(bool)> callback;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = feature_callbacks_.find(flag);
        if (it != feature_callbacks_.end())
        {
            callback = it->second;
        }
    }

    if (!callback)
    {
        return;
    }

    try
    {
        callback(enabled);
    }
    catch (const std::exception &error)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        diagnostics_.push_back("feature callback failed for " + feature_flag_to_string(flag) +
                               ": " + error.what());
    }
    catch (...)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        diagnostics_.push_back("feature callback failed for " + feature_flag_to_string(flag) +
                               ": unknown exception");
    }
}

std::string FeatureFlagManager::feature_flag_to_string(FeatureFlag flag) const
{
    const auto it = std::find_if(kFeatureFlagNames.begin(), kFeatureFlagNames.end(),
                                 [flag](const auto &entry) { return entry.first == flag; });
    if (it == kFeatureFlagNames.end())
    {
        return "UNKNOWN";
    }
    return std::string(it->second);
}

std::optional<FeatureFlag> FeatureFlagManager::parseFeatureFlag(const std::string &name) const
{
    const auto it =
        std::find_if(kFeatureFlagNames.begin(), kFeatureFlagNames.end(),
                     [&name](const auto &entry) { return entry.second == std::string_view(name); });
    if (it == kFeatureFlagNames.end())
    {
        return std::nullopt;
    }
    return it->first;
}

bool FeatureFlagValidator::validateAdvancedFeaturesDisabled()
{
    auto &manager = FeatureFlagManager::instance();
    auto advanced_features = manager.getFeaturesByCategory("advanced");
    for (const auto &info : advanced_features)
    {
        if (manager.isEnabled(info.flag))
        {
            return false;
        }
    }
    return true;
}

bool FeatureFlagValidator::validateFeatureDependencies()
{
    return FeatureFlagManager::instance().validateDependencies();
}

std::vector<std::string> FeatureFlagValidator::getValidationErrors()
{
    std::vector<std::string> errors;
    if (!validateAdvancedFeaturesDisabled())
    {
        errors.push_back("Advanced features should be disabled by default in CI");
    }
    if (!validateFeatureDependencies())
    {
        auto violations = FeatureFlagManager::instance().getDependencyViolations();
        errors.insert(errors.end(), violations.begin(), violations.end());
    }
    return errors;
}

void FeatureFlagValidator::generateValidationReport(const std::string &filename)
{
    std::ofstream file(filename);
    if (!file.is_open())
    {
        return;
    }

    auto &manager = FeatureFlagManager::instance();
    file << "Nerve Feature Flag Validation Report\n";
    file << "====================================\n\n";
    file << "Validation Status: ";
    auto errors = getValidationErrors();
    if (errors.empty())
    {
        file << "PASSED\n\n";
    }
    else
    {
        file << "FAILED\n\n";
        file << "Errors:\n";
        for (const auto &error : errors)
        {
            file << "  - " << error << "\n";
        }
        file << "\n";
    }
    file << manager.getStatusReport();
}

} // namespace nerve::feature_access
