#pragma once
#include "nerve/common/accelerated_types.hpp"
#include "nerve/core/buffer_view.hpp"
#include "nerve/core/rng/determinism_contract.hpp"
#include "nerve/errors/errors.hpp"
#include "nerve/persistence/core/core_types.hpp"

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
    explicit ExceptionGuard(CleanupFn cleanup);
    ~ExceptionGuard();
    ExceptionGuard(const ExceptionGuard &) = delete;
    ExceptionGuard &operator=(const ExceptionGuard &) = delete;
};
class AtomicCounter
{
public:
    explicit AtomicCounter(int initial);
    int increment();
    int decrement();
    int get() const;
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
