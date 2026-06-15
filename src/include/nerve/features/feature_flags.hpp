
#pragma once
#include <string>
#include <utility>
#include <vector>

namespace nerve::features
{
class FeatureFlags
{
public:
    enum class Feature
    {
        PH5_PERSISTENCE = 0,
        PH6_HIERARCHICAL = 1,
        PH5_WITNESS_SAMPLING = 2,
        PH6_COMPRESSION = 3,
        HIGH_DIMENSIONAL_EXTENSIONS = 4,
        INCREMENTAL_UPDATES = 5,
        SPECTRAL_INTEGRATION = 6,
        ADVANCED_ALGORITHMS = 7,
        DEBUG_VISUALIZATION = 8,
        COUNT = 9
    };
    static constexpr bool DEFAULT_ENABLED[static_cast<size_t>(Feature::COUNT)] = {
        true, true, true, true, true, true, true, true, false};
    FeatureFlags();
    ~FeatureFlags() = default;
    void enableFeature(Feature feature);
    void disableFeature(Feature feature);
    bool isFeatureEnabled(Feature feature) const;
    void enableFeatures(const std::string &feature_list);
    void disableFeatures(const std::string &feature_list);
    void setAllFeatures(bool enabled);
    std::string getFeatureName(Feature feature) const;
    std::string getFeatureDescription(Feature feature) const;
    std::vector<Feature> getEnabledFeatures() const;
    std::vector<Feature> getDisabledFeatures() const;
    void loadFromEnvironment();
    void loadFromConfigFile(const std::string &config_file);
    void saveToConfigFile(const std::string &config_file) const;
    bool validateFeatureDependencies() const;
    std::vector<std::string> getValidationErrors() const;
    static bool isBuildTimeEnabled(Feature feature);
    static std::vector<Feature> getBuildTimeEnabledFeatures();
    bool isPh5Enabled() const;
    bool isPh6Enabled() const;
    bool isHighDimensionalEnabled() const;
    bool isFeatureAccessEnabled() const;

private:
    bool features_[static_cast<size_t>(Feature::COUNT)];
    struct FeatureInfo
    {
        std::string name;
        std::string description;
        std::string category;
        std::vector<Feature> dependencies;
    };
    static const FeatureInfo FEATURE_INFO[static_cast<size_t>(Feature::COUNT)];
    void setFeatureState(Feature feature, bool enabled);
    std::vector<Feature> parseFeatureList(const std::string &feature_list) const;
};
class RuntimeFeatureChecker
{
public:
    explicit RuntimeFeatureChecker(const FeatureFlags &flags);
    bool checkFeatureEnabled(FeatureFlags::Feature feature) const;
    void requireFeature(FeatureFlags::Feature feature, const std::string &context = "") const;
    void warnIfDisabled(FeatureFlags::Feature feature, const std::string &context = "") const;
    void logFeatureUsage(FeatureFlags::Feature feature, const std::string &operation) const;
    void logFeaturePerformance(FeatureFlags::Feature feature, double duration_ms) const;
#define REQUIRE_PH5_FEATURE(context)                                                               \
    do                                                                                             \
    {                                                                                              \
        if (!runtime_checker_.checkFeatureEnabled(FeatureFlags::Feature::PH5_PERSISTENCE))         \
        {                                                                                          \
            runtime_checker_.requireFeature(FeatureFlags::Feature::PH5_PERSISTENCE, context);      \
        }                                                                                          \
    } while (0)
#define REQUIRE_PH6_FEATURE(context)                                                               \
    do                                                                                             \
    {                                                                                              \
        if (!runtime_checker_.checkFeatureEnabled(FeatureFlags::Feature::PH6_HIERARCHICAL))        \
        {                                                                                          \
            runtime_checker_.requireFeature(FeatureFlags::Feature::PH6_HIERARCHICAL, context);     \
        }                                                                                          \
    } while (0)
#define REQUIRE_HIGH_DIMENSIONAL_FEATURE(context)                                                  \
    do                                                                                             \
    {                                                                                              \
        if (!runtime_checker_.checkFeatureEnabled(                                                 \
                FeatureFlags::Feature::HIGH_DIMENSIONAL_EXTENSIONS))                               \
        {                                                                                          \
            runtime_checker_.requireFeature(FeatureFlags::Feature::HIGH_DIMENSIONAL_EXTENSIONS,    \
                                            context);                                              \
        }                                                                                          \
    } while (0)
#define REQUIRE_ADVANCED_FEATURE(context)                                                          \
    do                                                                                             \
    {                                                                                              \
        if (!runtime_checker_.checkFeatureEnabled(FeatureFlags::Feature::ADVANCED_ALGORITHMS))     \
        {                                                                                          \
            runtime_checker_.requireFeature(FeatureFlags::Feature::ADVANCED_ALGORITHMS, context);  \
        }                                                                                          \
    } while (0)
#define LOG_FEATURE_USAGE(feature, operation)                                                      \
    do                                                                                             \
    {                                                                                              \
        runtime_checker_.logFeatureUsage(FeatureFlags::Feature::feature, operation);               \
    } while (0)
private:
    const FeatureFlags &flags_;
    mutable std::vector<std::pair<FeatureFlags::Feature, std::string>> usage_log_;
};
} // namespace nerve::features
