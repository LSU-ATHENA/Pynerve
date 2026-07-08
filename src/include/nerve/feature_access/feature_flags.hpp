
#pragma once
#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#define TOPOLOGIB_FEATURE_FLAG_PH4 1
#define TOPOLOGIB_FEATURE_FLAG_DIFFERENTIABLE_PERSISTENCE 1
#define TOPOLOGIB_FEATURE_FLAG_SHEAF_LAPLACIAN_ADVANCED 1
#define TOPOLOGIB_FEATURE_FLAG_ADVANCED_STREAMING 1
#define TOPOLOGIB_FEATURE_FLAG_RESEARCH_MODE 0
#define TOPOLOGIB_FEATURE_ACCESS_API_ENABLED(flag)                                                 \
    nerve::feature_access::FeatureFlagManager::instance().isEnabled(flag)
#define TOPOLOGIB_FEATURE_ACCESS_API_CALL(flag, call)                                              \
    do                                                                                             \
    {                                                                                              \
        if (TOPOLOGIB_FEATURE_ACCESS_API_ENABLED(flag))                                            \
        {                                                                                          \
            call;                                                                                  \
        }                                                                                          \
    } while (0)
#define TOPOLOGIB_FEATURE_ACCESS_API_RETURN(flag, default_return, call)                            \
    (TOPOLOGIB_FEATURE_ACCESS_API_ENABLED(flag) ? (call) : (default_return))

namespace nerve::feature_access
{
enum class FeatureFlag : uint32_t
{
    PH4 = 1,
    DIFFERENTIABLE_PERSISTENCE = 2,
    SHEAF_LAPLACIAN_ADVANCED = 3,
    ADVANCED_STREAMING = 4,
    RESEARCH_MODE = 5,
    DEBUG_VISUALIZATION = 6,
    ADVANCED_PROFILING = 7,
    NEW_SIMPLICIAL_COMPLEX = 8,
    ADVANCED_FILTRATION = 9,
    QUANTUM_TOPOLOGY = 10,
    GPU_ADVANCED_KERNELS = 11,
    PARALLEL_PERSISTENCE = 12,
    MEMORY_OPTIMIZATION = 13,
    PYTHON_ADVANCED = 14,
    RUST_ADVANCED = 15,
    JULIA_ADVANCED = 16
};
struct FeatureFlagInfo
{
    FeatureFlag flag;
    std::string name;
    std::string description;
    std::string category;
    bool enabledByDefault;
    bool requiresRebuild;
    std::vector<std::string> dependencies;
    std::function<bool()> availability_check;
    FeatureFlagInfo()
        : flag(FeatureFlag::DIFFERENTIABLE_PERSISTENCE)
        , enabledByDefault(false)
        , requiresRebuild(false)
    {}
};
class FeatureFlagManager
{
public:
    static FeatureFlagManager &instance();
    bool isEnabled(FeatureFlag flag) const;
    void setEnabled(FeatureFlag flag, bool enabled);
    void enable(FeatureFlag flag);
    void disable(FeatureFlag flag);
    void setEnabledFlags(const std::unordered_set<FeatureFlag> &flags);
    void enableAll();
    void disableAll();
    void resetToDefaults();
    FeatureFlagInfo getFeatureInfo(FeatureFlag flag) const;
    std::vector<FeatureFlagInfo> getAllFeatures() const;
    std::vector<FeatureFlagInfo> getFeaturesByCategory(const std::string &category) const;
    std::vector<FeatureFlag> getEnabledFeatures() const;
    void registerFeature(const FeatureFlagInfo &info);
    void unregisterFeature(FeatureFlag flag);
    void loadFromConfig(const std::string &config_file);
    void saveToConfig(const std::string &config_file) const;
    void setFromEnvironment();
    bool validateDependencies() const;
    std::vector<std::string> getDependencyViolations() const;
    std::string getStatusReport() const;
    bool isFeatureAvailable(FeatureFlag flag) const;
    std::string getFeatureStatus(FeatureFlag flag) const;
    std::vector<std::string> getConfigParseErrors() const;
    void clearConfigParseErrors();
    std::vector<std::string> getDiagnostics() const;
    void clearDiagnostics();
    void setFeatureCallback(FeatureFlag flag, std::function<void(bool)> callback);
    void enableFeatureGroup(const std::string &group_name);
    void disableFeatureGroup(const std::string &group_name);

private:
    FeatureFlagManager();
    ~FeatureFlagManager() = default;
    mutable std::mutex mutex_;
    std::unordered_map<FeatureFlag, bool> feature_states_;
    std::unordered_map<FeatureFlag, FeatureFlagInfo> feature_info_;
    std::unordered_map<FeatureFlag, std::function<void(bool)>> feature_callbacks_;
    std::unordered_map<std::string, std::vector<FeatureFlag>> feature_groups_;
    std::vector<std::string> config_parse_errors_;
    std::vector<std::string> diagnostics_;
    static bool defaultFeatureState(FeatureFlag flag);
    void initializeDefaultFeatures();
    bool checkDependencies(FeatureFlag flag) const;
    void notifyFeatureChange(FeatureFlag flag, bool enabled);
    std::string feature_flag_to_string(FeatureFlag flag) const;
    std::optional<FeatureFlag> parseFeatureFlag(const std::string &name) const;
};
class FeatureFlagGuard
{
public:
    FeatureFlagGuard(FeatureFlag flag, bool scoped_state);
    ~FeatureFlagGuard();
    FeatureFlagGuard(const FeatureFlagGuard &) = delete;
    FeatureFlagGuard &operator=(const FeatureFlagGuard &) = delete;
    FeatureFlagGuard(FeatureFlagGuard &&other) noexcept;
    FeatureFlagGuard &operator=(FeatureFlagGuard &&other) noexcept;

private:
    FeatureFlag flag_;
    bool original_state_;
    bool restored_;
};
template <FeatureFlag Flag>
struct CompileTimeFeatureFlag
{
    static constexpr bool isEnabled() { return false; }
};
template <>
struct CompileTimeFeatureFlag<FeatureFlag::DIFFERENTIABLE_PERSISTENCE>
{
    static constexpr bool isEnabled()
    {
        return TOPOLOGIB_FEATURE_FLAG_DIFFERENTIABLE_PERSISTENCE != 0;
    }
};
template <>
struct CompileTimeFeatureFlag<FeatureFlag::SHEAF_LAPLACIAN_ADVANCED>
{
    static constexpr bool isEnabled()
    {
        return TOPOLOGIB_FEATURE_FLAG_SHEAF_LAPLACIAN_ADVANCED != 0;
    }
};
template <>
struct CompileTimeFeatureFlag<FeatureFlag::ADVANCED_STREAMING>
{
    static constexpr bool isEnabled() { return TOPOLOGIB_FEATURE_FLAG_ADVANCED_STREAMING != 0; }
};
template <>
struct CompileTimeFeatureFlag<FeatureFlag::RESEARCH_MODE>
{
    static constexpr bool isEnabled() { return TOPOLOGIB_FEATURE_FLAG_RESEARCH_MODE != 0; }
};
template <FeatureFlag Flag, typename ReturnType, typename... Args>
class FeatureGatedApi
{
public:
    using FeatureFunction = std::function<ReturnType(Args...)>;
    explicit FeatureGatedApi(FeatureFunction feature_func,
                             const std::string &name = "feature_gated_api");
    ReturnType operator()(Args... args) const;
    void setFeatureFunction(FeatureFunction func);
    bool isEnabled() const;
    const std::string &getName() const;

private:
    FeatureFunction feature_func_;
    std::string name_;
    FeatureFlag flag_;
};
class FeatureFlagValidator
{
public:
    static bool validateAdvancedFeaturesDisabled();
    static bool validateFeatureDependencies();
    static std::vector<std::string> getValidationErrors();
    static void generateValidationReport(const std::string &filename);
};
#define TOPOLOGIB_FEATURE_ACCESS_API_DIFF_PERSISTENCE(return_type, name, ...)                      \
    nerve::feature_access::FeatureGatedApi<                                                        \
        nerve::feature_access::FeatureFlag::DIFFERENTIABLE_PERSISTENCE, return_type,               \
        ##__VA_ARGS__>                                                                             \
        name
#define TOPOLOGIB_FEATURE_ACCESS_API_SHEAF(return_type, name, ...)                                 \
    nerve::feature_access::FeatureGatedApi<                                                        \
        nerve::feature_access::FeatureFlag::SHEAF_LAPLACIAN_ADVANCED, return_type, ##__VA_ARGS__>  \
        name
template <FeatureFlag Flag, typename ReturnType, typename... Args>
FeatureGatedApi<Flag, ReturnType, Args...>::FeatureGatedApi(FeatureFunction feature_func,
                                                            const std::string &name)
    : feature_func_(feature_func)
    , name_(name)
    , flag_(Flag)
{}
template <FeatureFlag Flag, typename ReturnType, typename... Args>
ReturnType FeatureGatedApi<Flag, ReturnType, Args...>::operator()(Args... args) const
{
    if (FeatureFlagManager::instance().isEnabled(flag_))
    {
        return feature_func_(args...);
    }
    throw std::runtime_error("Feature is disabled: " + name_);
}
template <FeatureFlag Flag, typename ReturnType, typename... Args>
void FeatureGatedApi<Flag, ReturnType, Args...>::setFeatureFunction(FeatureFunction func)
{
    feature_func_ = func;
}
template <FeatureFlag Flag, typename ReturnType, typename... Args>
bool FeatureGatedApi<Flag, ReturnType, Args...>::isEnabled() const
{
    return FeatureFlagManager::instance().isEnabled(flag_);
}
template <FeatureFlag Flag, typename ReturnType, typename... Args>
const std::string &FeatureGatedApi<Flag, ReturnType, Args...>::getName() const
{
    return name_;
}
} // namespace nerve::feature_access
