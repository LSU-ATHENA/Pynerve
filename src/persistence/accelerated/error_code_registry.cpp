
#include "nerve/errors/errors.hpp"

#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace nerve::persistence::accelerated
{

struct ErrorInfo
{
    errors::ErrorCode code;
    std::string name;
    std::string description;
    std::string category;
    std::string suggested_action;
};

class ErrorCodeRegistryImpl
{
public:
    static ErrorCodeRegistryImpl &instance()
    {
        static ErrorCodeRegistryImpl registry;
        return registry;
    }

    void registerError(const ErrorInfo &info)
    {
        errors_.insert_or_assign(info.code, info);
        errors_by_category_[info.category].push_back(info.code);
    }

    const ErrorInfo &getErrorInfo(errors::ErrorCode code) const
    {
        auto it = errors_.find(code);
        if (it != errors_.end())
        {
            return it->second;
        }
        static const ErrorInfo unknown_info{errors::ErrorCode::UNKNOWN, "UNKNOWN",
                                            "Unknown accelerated error code", "General",
                                            "Inspect diagnostics and logs"};
        return unknown_info;
    }

    std::vector<errors::ErrorCode> getErrorsByCategory(const std::string &category) const
    {
        auto it = errors_by_category_.find(category);
        if (it == errors_by_category_.end())
        {
            return {};
        }
        return it->second;
    }

    std::vector<std::string> getAllCategories() const
    {
        std::vector<std::string> categories;
        categories.reserve(errors_by_category_.size());
        for (const auto &[category, _] : errors_by_category_)
        {
            categories.push_back(category);
        }
        return categories;
    }

    const std::unordered_map<errors::ErrorCode, ErrorInfo> &getAllErrors() const { return errors_; }

    std::string formatErrorMessage(errors::ErrorCode code, const std::string &context) const
    {
        const ErrorInfo &info = getErrorInfo(code);
        std::ostringstream oss;
        oss << "[" << info.name << "] " << info.description;
        if (!context.empty())
        {
            oss << "\nContext: " << context;
        }
        oss << "\nCategory: " << info.category;
        oss << "\nSuggested action: " << info.suggested_action;
        return oss.str();
    }

    bool isRecoverable(errors::ErrorCode code) const
    {
        const ErrorInfo &info = getErrorInfo(code);
        return info.category == "GPU" || info.category == "Resource";
    }

private:
    ErrorCodeRegistryImpl() { initializeErrorCodes(); }

    void initializeErrorCodes()
    {
        registerError({errors::ErrorCode::E10_GPU_OOM, "GPU_OOM",
                       "GPU out-of-memory condition while executing accelerated stage", "GPU",
                       "Reduce batch size or enable deterministic CPU implementation"});
        registerError({errors::ErrorCode::E11_GPU_LAUNCH_FAIL, "GPU_LAUNCH_FAIL",
                       "GPU kernel launch failed", "GPU",
                       "Check GPU capability and kernel launch parameters"});
        registerError({errors::ErrorCode::E50_PH_ABORT, "PERSISTENCE_ABORT",
                       "Persistence computation aborted", "Algorithm",
                       "Inspect stage diagnostics and retry with deterministic CPU mode"});
        registerError({errors::ErrorCode::E30_DET_MISMATCH, "DETERMINISM_MISMATCH",
                       "Determinism contract violation detected", "Determinism",
                       "Re-run with deterministic options and identical seeds"});
        registerError({errors::ErrorCode::E40_CPU_OVERLOAD, "CPU_OVERLOAD",
                       "CPU resources are saturated", "Resource",
                       "Lower thread count or reduce workload"});
        registerError({errors::ErrorCode::E41_RESOURCE_LIMIT, "RESOURCE_LIMIT",
                       "Resource limit reached in accelerated execution", "Resource",
                       "Reduce problem size or increase limits"});
        registerError({errors::ErrorCode::E60_NUMA_BIND_FAIL, "NUMA_BIND_FAIL",
                       "NUMA binding failed", "NUMA", "Use topology-aware recovery policy"});
        registerError({errors::ErrorCode::E61_NUMA_AFFINITY_FAIL, "NUMA_AFFINITY_FAIL",
                       "NUMA affinity assignment failed", "NUMA",
                       "Retry with relaxed placement policy"});
        registerError({errors::ErrorCode::E62_NUMA_MIGRATION_ERROR, "NUMA_MIGRATION_ERROR",
                       "NUMA page migration failed", "NUMA",
                       "Disable NUMA migration for this run"});
        registerError({errors::ErrorCode::E54_PH4_INVALID_INPUT, "INVALID_INPUT",
                       "Invalid input for accelerated persistence", "Input",
                       "Validate point cloud shape and radius bounds"});
        registerError({errors::ErrorCode::E55_PH4_SPARSE_CONVERGENCE_FAIL,
                       "SPARSE_CONVERGENCE_FAIL", "Sparse reduction failed to converge",
                       "Algorithm", "Switch to exact deterministic reducer path"});
        registerError({errors::ErrorCode::E92_MEMORY_PRESSURE, "MEMORY_PRESSURE",
                       "Runtime memory pressure detected", "Resource",
                       "Use smaller chunks or lower max dimension"});
    }

    std::unordered_map<errors::ErrorCode, ErrorInfo> errors_;
    std::unordered_map<std::string, std::vector<errors::ErrorCode>> errors_by_category_;
};

namespace error_registry
{

void registerError(errors::ErrorCode code, const std::string &name, const std::string &description,
                   const std::string &category, const std::string &suggested_action)
{
    ErrorCodeRegistryImpl::instance().registerError(
        {code, name, description, category, suggested_action});
}

const ErrorInfo &getErrorInfo(errors::ErrorCode code)
{
    return ErrorCodeRegistryImpl::instance().getErrorInfo(code);
}

std::vector<errors::ErrorCode> getErrorsByCategory(const std::string &category)
{
    return ErrorCodeRegistryImpl::instance().getErrorsByCategory(category);
}

std::vector<std::string> getAllCategories()
{
    return ErrorCodeRegistryImpl::instance().getAllCategories();
}

std::string formatErrorMessage(errors::ErrorCode code, const std::string &context)
{
    return ErrorCodeRegistryImpl::instance().formatErrorMessage(code, context);
}

bool isRecoverable(errors::ErrorCode code)
{
    return ErrorCodeRegistryImpl::instance().isRecoverable(code);
}

const std::unordered_map<errors::ErrorCode, ErrorInfo> &getAllErrors()
{
    return ErrorCodeRegistryImpl::instance().getAllErrors();
}

void printErrorSummary()
{
    const auto &all_errors = getAllErrors();
    std::unordered_map<std::string, std::vector<const ErrorInfo *>> by_category;
    for (const auto &[_, info] : all_errors)
    {
        by_category[info.category].push_back(&info);
    }

    std::cout << "Accelerated Error Codes\n";
    std::cout << "========================\n";
    for (const auto &[category, list] : by_category)
    {
        std::cout << category << ":\n";
        for (const auto *info : list)
        {
            std::cout << "  " << info->name << ": " << info->description << '\n';
            std::cout << "    Action: " << info->suggested_action << '\n';
        }
        std::cout << '\n';
    }
}

bool isValidErrorCode(errors::ErrorCode code)
{
    return getAllErrors().find(code) != getAllErrors().end();
}

std::string getErrorCategory(errors::ErrorCode code)
{
    return getErrorInfo(code).category;
}

std::vector<errors::ErrorCode> getRecoverableErrors()
{
    std::vector<errors::ErrorCode> recoverable;
    for (const auto &[code, _] : getAllErrors())
    {
        if (isRecoverable(code))
        {
            recoverable.push_back(code);
        }
    }
    return recoverable;
}

} // namespace error_registry

} // namespace nerve::persistence::accelerated
