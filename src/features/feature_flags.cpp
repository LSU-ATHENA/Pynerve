
#include "nerve/features/feature_flags.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace nerve::features
{
namespace
{

#ifdef ENABLE_PH5
constexpr bool kPh5BuildEnabled = true;
#else
constexpr bool kPh5BuildEnabled = false;
#endif

#ifdef ENABLE_PH6
constexpr bool kPh6BuildEnabled = true;
#else
constexpr bool kPh6BuildEnabled = false;
#endif

#ifdef ENABLE_ADVANCED_FEATURES
constexpr bool kAdvancedFeaturesBuildEnabled = true;
#else
constexpr bool kAdvancedFeaturesBuildEnabled = false;
#endif

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

std::string lowercaseCopy(std::string value)
{
    std::ranges::transform(value, value.begin(),
                           [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value;
}

bool parseBoolean(const std::string &value)
{
    const std::string normalized = lowercaseCopy(trimCopy(value));
    return normalized == "true" || normalized == "1" || normalized == "enabled" ||
           normalized == "on" || normalized == "yes";
}

size_t featureIndex(FeatureFlags::Feature feature)
{
    const size_t index = static_cast<size_t>(feature);
    if (index >= static_cast<size_t>(FeatureFlags::Feature::COUNT))
    {
        throw std::invalid_argument("feature enum value is out of range");
    }
    return index;
}

void applyEnvironmentFlag(FeatureFlags &flags, const char *name, FeatureFlags::Feature feature)
{
    const char *value = std::getenv(name);
    if (value != nullptr)
    {
        if (parseBoolean(value))
        {
            flags.enableFeature(feature);
        }
        else
        {
            flags.disableFeature(feature);
        }
    }
}

} // namespace

const FeatureFlags::FeatureInfo FeatureFlags::FEATURE_INFO[static_cast<size_t>(Feature::COUNT)] = {
    {"PH5_PERSISTENCE", "PH5 persistence computation with witness complexes", "PH5", {}},
    {"PH6_HIERARCHICAL", "PH6 hierarchical witness sampling", "PH6", {}},
    {"PH5_WITNESS_SAMPLING",
     "PH5 witness sampling optimization",
     "PH5",
     {Feature::PH5_PERSISTENCE}},
    {"PH6_COMPRESSION", "PH6 compression techniques", "PH6", {Feature::PH6_HIERARCHICAL}},
    {"HIGH_DIMENSIONAL_EXTENSIONS", "High-dimensional extensions for PH5/PH6", "Core", {}},
    {"INCREMENTAL_UPDATES",
     "Incremental updates for streaming data",
     "Core",
     {Feature::HIGH_DIMENSIONAL_EXTENSIONS}},
    {"SPECTRAL_INTEGRATION",
     "Spectral pipeline integration",
     "Core",
     {Feature::HIGH_DIMENSIONAL_EXTENSIONS, Feature::INCREMENTAL_UPDATES}},
    {"ADVANCED_ALGORITHMS", "Advanced persistent homology algorithms", "Core", {}},
    {"DEBUG_VISUALIZATION", "Debug visualization tools", "Debug", {}}};
FeatureFlags::FeatureFlags()
{
    for (size_t i = 0; i < static_cast<size_t>(Feature::COUNT); ++i)
    {
        features_[i] = DEFAULT_ENABLED[i];
    }
    loadFromEnvironment();
}
void FeatureFlags::enableFeature(Feature feature)
{
    setFeatureState(feature, true);
}
void FeatureFlags::disableFeature(Feature feature)
{
    setFeatureState(feature, false);
}
bool FeatureFlags::isFeatureEnabled(Feature feature) const
{
    return features_[featureIndex(feature)];
}
void FeatureFlags::enableFeatures(const std::string &feature_list)
{
    auto features = parseFeatureList(feature_list);
    for (auto feature : features)
    {
        enableFeature(feature);
    }
}
void FeatureFlags::disableFeatures(const std::string &feature_list)
{
    auto features = parseFeatureList(feature_list);
    for (auto feature : features)
    {
        disableFeature(feature);
    }
}
void FeatureFlags::setAllFeatures(bool enabled)
{
    for (size_t i = 0; i < static_cast<size_t>(Feature::COUNT); ++i)
    {
        features_[i] = enabled;
    }
}
std::string FeatureFlags::getFeatureName(Feature feature) const
{
    return FEATURE_INFO[featureIndex(feature)].name;
}
std::string FeatureFlags::getFeatureDescription(Feature feature) const
{
    return FEATURE_INFO[featureIndex(feature)].description;
}
std::vector<FeatureFlags::Feature> FeatureFlags::getEnabledFeatures() const
{
    std::vector<Feature> enabled;
    for (size_t i = 0; i < static_cast<size_t>(Feature::COUNT); ++i)
    {
        if (features_[i])
        {
            enabled.push_back(static_cast<Feature>(i));
        }
    }
    return enabled;
}
std::vector<FeatureFlags::Feature> FeatureFlags::getDisabledFeatures() const
{
    std::vector<Feature> disabled;
    for (size_t i = 0; i < static_cast<size_t>(Feature::COUNT); ++i)
    {
        if (!features_[i])
        {
            disabled.push_back(static_cast<Feature>(i));
        }
    }
    return disabled;
}
void FeatureFlags::loadFromEnvironment()
{
    const char *env_features = std::getenv("NERVE_FEATURES");
    if (env_features)
    {
        enableFeatures(std::string(env_features));
    }
    applyEnvironmentFlag(*this, "NERVE_ENABLE_PH5", Feature::PH5_PERSISTENCE);
    applyEnvironmentFlag(*this, "NERVE_ENABLE_PH6", Feature::PH6_HIERARCHICAL);
    const char *advanced_features = std::getenv("NERVE_ENABLE_ADVANCED_FEATURES");
    if (advanced_features != nullptr && parseBoolean(advanced_features))
    {
        enableFeatures("PH5_PERSISTENCE,PH6_HIERARCHICAL,ADVANCED_ALGORITHMS");
    }
}
void FeatureFlags::loadFromConfigFile(const std::string &config_file)
{
    std::ifstream file(config_file);
    if (!file.is_open())
    {
        return;
    }
    std::string line;
    while (std::getline(file, line))
    {
        const std::string trimmed = trimCopy(line);
        if (trimmed.empty() || trimmed[0] == '#')
        {
            continue;
        }
        const std::size_t separator = trimmed.find('=');
        if (separator == std::string::npos)
        {
            continue;
        }
        const std::string key = trimCopy(trimmed.substr(0, separator));
        const std::string value = trimCopy(trimmed.substr(separator + 1));
        if (key == "features")
        {
            enableFeatures(value);
        }
        else if (key == "enable_all")
        {
            setAllFeatures(parseBoolean(value));
        }
        else
        {
            for (size_t i = 0; i < static_cast<size_t>(Feature::COUNT); ++i)
            {
                if (FEATURE_INFO[i].name == key)
                {
                    setFeatureState(static_cast<Feature>(i), parseBoolean(value));
                    break;
                }
            }
        }
    }
}
void FeatureFlags::saveToConfigFile(const std::string &config_file) const
{
    std::ofstream file(config_file);
    if (!file.is_open())
    {
        return;
    }
    auto enabled = getEnabledFeatures();
    if (!enabled.empty())
    {
        file << "features=";
        for (size_t i = 0; i < enabled.size(); ++i)
        {
            if (i > 0)
                file << ",";
            file << getFeatureName(enabled[i]);
        }
        file << "\n";
    }
    file << "\n";
    for (size_t i = 0; i < static_cast<size_t>(Feature::COUNT); ++i)
    {
        file << getFeatureName(static_cast<Feature>(i)) << "="
             << (features_[i] ? "enabled" : "disabled") << "\n";
    }
}
bool FeatureFlags::validateFeatureDependencies() const
{
    std::vector<std::string> errors;
    for (size_t i = 0; i < static_cast<size_t>(Feature::COUNT); ++i)
    {
        if (!features_[i])
            continue;
        const auto &info = FEATURE_INFO[i];
        for (auto dep : info.dependencies)
        {
            if (!features_[static_cast<size_t>(dep)])
            {
                errors.push_back("Feature " + info.name + " requires " + getFeatureName(dep) +
                                 " which is disabled");
            }
        }
    }
    return errors.empty();
}
std::vector<std::string> FeatureFlags::getValidationErrors() const
{
    std::vector<std::string> errors;
    for (size_t i = 0; i < static_cast<size_t>(Feature::COUNT); ++i)
    {
        if (!features_[i])
            continue;
        const auto &info = FEATURE_INFO[i];
        for (auto dep : info.dependencies)
        {
            if (!features_[static_cast<size_t>(dep)])
            {
                errors.push_back("Feature " + info.name + " requires " + getFeatureName(dep) +
                                 " which is disabled");
            }
        }
    }
    return errors;
}
bool FeatureFlags::isBuildTimeEnabled(Feature feature)
{
    if (feature == Feature::PH5_PERSISTENCE)
    {
        return kPh5BuildEnabled;
    }
    if (feature == Feature::PH6_HIERARCHICAL)
    {
        return kPh6BuildEnabled;
    }
    if (feature == Feature::HIGH_DIMENSIONAL_EXTENSIONS)
    {
        return kAdvancedFeaturesBuildEnabled;
    }
    return false;
}
std::vector<FeatureFlags::Feature> FeatureFlags::getBuildTimeEnabledFeatures()
{
    std::vector<Feature> enabled;
    for (size_t i = 0; i < static_cast<size_t>(Feature::COUNT); ++i)
    {
        if (isBuildTimeEnabled(static_cast<Feature>(i)))
        {
            enabled.push_back(static_cast<Feature>(i));
        }
    }
    return enabled;
}
bool FeatureFlags::isPh5Enabled() const
{
    return isFeatureEnabled(Feature::PH5_PERSISTENCE) ||
           isFeatureEnabled(Feature::PH5_WITNESS_SAMPLING);
}
bool FeatureFlags::isPh6Enabled() const
{
    return isFeatureEnabled(Feature::PH6_HIERARCHICAL) ||
           isFeatureEnabled(Feature::PH6_COMPRESSION);
}
bool FeatureFlags::isHighDimensionalEnabled() const
{
    return isFeatureEnabled(Feature::HIGH_DIMENSIONAL_EXTENSIONS);
}
bool FeatureFlags::isFeatureAccessEnabled() const
{
    return isFeatureEnabled(Feature::ADVANCED_ALGORITHMS) ||
           isFeatureEnabled(Feature::DEBUG_VISUALIZATION);
}
void FeatureFlags::setFeatureState(Feature feature, bool enabled)
{
    features_[featureIndex(feature)] = enabled;
}
std::vector<FeatureFlags::Feature>
FeatureFlags::parseFeatureList(const std::string &feature_list) const
{
    std::vector<FeatureFlags::Feature> features;
    std::istringstream iss(feature_list);
    std::string feature_name;
    while (std::getline(iss, feature_name, ','))
    {
        feature_name = trimCopy(feature_name);
        if (feature_name.empty())
        {
            continue;
        }
        for (size_t i = 0; i < static_cast<size_t>(Feature::COUNT); ++i)
        {
            if (FEATURE_INFO[i].name == feature_name)
            {
                features.push_back(static_cast<Feature>(i));
                break;
            }
        }
    }
    return features;
}
RuntimeFeatureChecker::RuntimeFeatureChecker(const FeatureFlags &flags)
    : flags_(flags)
{}
bool RuntimeFeatureChecker::checkFeatureEnabled(FeatureFlags::Feature feature) const
{
    return flags_.isFeatureEnabled(feature);
}
void RuntimeFeatureChecker::requireFeature(FeatureFlags::Feature feature,
                                           const std::string &context) const
{
    if (!checkFeatureEnabled(feature))
    {
        std::string message =
            "required feature '" + flags_.getFeatureName(feature) + "' is not enabled";
        if (!context.empty())
        {
            message += " for " + context;
        }
        throw std::runtime_error(message);
    }
}
void RuntimeFeatureChecker::warnIfDisabled(FeatureFlags::Feature feature,
                                           const std::string &context) const
{
    if (!checkFeatureEnabled(feature))
    {
        usage_log_.emplace_back(feature, context.empty() ? "disabled" : "disabled:" + context);
    }
}
void RuntimeFeatureChecker::logFeatureUsage(FeatureFlags::Feature feature,
                                            const std::string &operation) const
{
    usage_log_.emplace_back(feature, operation);
}
void RuntimeFeatureChecker::logFeaturePerformance(FeatureFlags::Feature feature,
                                                  double duration_ms) const
{
    const std::string operation =
        usage_log_.empty() ? "performance" : usage_log_.back().second + ":performance";
    usage_log_.emplace_back(feature, operation + "_ms=" + std::to_string(duration_ms));
}
} // namespace nerve::features
