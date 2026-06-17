#pragma once
#include "nerve/common/accelerated_types.hpp"
#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/core/rng/determinism_contract.hpp"
#include "nerve/errors/errors.hpp"
#include "nerve/persistence/core/core_types.hpp"

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace nerve::persistence::accelerated::error_registry
{
struct ErrorInfo
{
    std::string name;
    std::string category;
    std::string description;
};
ErrorInfo getErrorInfo(nerve::errors::ErrorCode code);
bool isValidErrorCode(nerve::errors::ErrorCode code);
std::vector<std::string> getAllCategories();
std::vector<ErrorInfo> getErrorsByCategory(const std::string &category);
} // namespace nerve::persistence::accelerated::error_registry

namespace nerve::persistence::accelerated::exception_safety
{
class ExceptionGuard
{
public:
    using CleanupFn = std::function<void()>;
    explicit ExceptionGuard(CleanupFn cleanup)
        : cleanup_(std::move(cleanup))
    {}
    ~ExceptionGuard()
    {
        if (cleanup_)
            cleanup_();
    }
    ExceptionGuard(const ExceptionGuard &) = delete;
    ExceptionGuard &operator=(const ExceptionGuard &) = delete;

private:
    CleanupFn cleanup_;
};
class AtomicCounter
{
public:
    explicit AtomicCounter(int initial)
        : value_(initial)
    {}
    int increment() { return ++value_; }
    int decrement() { return --value_; }
    int get() const { return value_; }

private:
    std::atomic<int> value_;
};
} // namespace nerve::persistence::accelerated::exception_safety

namespace nerve::persistence::accelerated::optimization_recommendations
{
std::vector<std::string> suggestActions(const nerve::common::PerformanceMetrics &metrics);
}

namespace nerve::persistence::accelerated::validation
{
struct ValidationResult
{
    bool isValid = false;
    std::string message;
};
ValidationResult validateApiIntegration(const nerve::core::BufferView<const double> &points,
                                        size_t point_dim, const nerve::common::VRConfig &config);
ValidationResult
validateMathematicalCorrectness(const std::vector<nerve::persistence::Pair> &diagram,
                                const nerve::core::DeterminismContract &contract);
} // namespace nerve::persistence::accelerated::validation

namespace nerve::persistence::accelerated::performance
{
void updateLastMetrics(const nerve::common::PerformanceMetrics &metrics);
nerve::common::PerformanceMetrics getLastMetrics();
} // namespace nerve::persistence::accelerated::performance

namespace nerve::persistence::accelerated::performance_impact
{
double computeRuntimeChange(const nerve::common::PerformanceMetrics &current,
                            const nerve::common::PerformanceMetrics &previous);
}
