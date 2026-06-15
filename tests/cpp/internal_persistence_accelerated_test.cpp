
#include "nerve/common/accelerated_types.hpp"
#include "nerve/core/rng/determinism_contract.hpp"
#include "nerve/persistence/accelerated/detail/accelerated_detail.hpp"
#include "nerve/persistence/vr/vr_fast_ops.hpp"

#include <cmath>
#include <limits>
#include <mutex>
#include <string>
#include <vector>

namespace
{

bool check_error_code_lookup()
{
    namespace reg = nerve::persistence::accelerated::error_registry;
    auto info = reg::getErrorInfo(nerve::errors::ErrorCode::E10_GPU_OOM);
    if (info.name.empty())
        return false;
    if (info.category != "GPU")
        return false;
    if (!reg::isValidErrorCode(nerve::errors::ErrorCode::E10_GPU_OOM))
        return false;
    auto categories = reg::getAllCategories();
    if (categories.empty())
        return false;
    auto gpu_errors = reg::getErrorsByCategory("GPU");
    if (gpu_errors.empty())
        return false;
    return true;
}

bool check_exception_safety_guard()
{
    using nerve::persistence::accelerated::exception_safety::ExceptionGuard;
    bool cleaned = false;
    {
        ExceptionGuard guard([&cleaned]() { cleaned = true; });
        (void)guard;
    }
    if (!cleaned)
        return false;
    return true;
}

bool check_atomic_counter()
{
    using nerve::persistence::accelerated::exception_safety::AtomicCounter;
    AtomicCounter counter(0);
    auto v = counter.increment();
    if (v != 1)
        return false;
    v = counter.decrement();
    if (v != 0)
        return false;
    if (counter.get() != 0)
        return false;
    return true;
}

bool check_optimization_advice()
{
    namespace rec = nerve::persistence::accelerated::optimization_recommendations;
    nerve::common::PerformanceMetrics metrics;
    metrics.total_time_ms = 100.0;
    metrics.gpu_time_ms = 20.0;
    metrics.cpu_time_ms = 80.0;
    metrics.gpu_bytes = 1024.0 * 1024.0 * 1024.0;
    metrics.gpu_available = true;
    auto advice = rec::suggestActions(metrics);
    if (advice.empty())
        return false;
    return true;
}

bool check_validation_for_valid_config()
{
    nerve::common::VRConfig config;
    config.max_dim = 2;
    config.max_radius = 1.0;
    std::vector<double> pts = {0.0, 0.0, 1.0, 0.0};
    nerve::core::BufferView<const double> view(pts.data(), pts.size());
    auto result =
        nerve::persistence::accelerated::validation::validateApiIntegration(view, 2, config);
    if (!result.isValid)
        return false;
    return true;
}

bool check_performance_helpers()
{
    namespace perf = nerve::persistence::accelerated::performance;
    namespace impact = nerve::persistence::accelerated::performance_impact;
    nerve::common::PerformanceMetrics m1, m2;
    m1.total_time_ms = 100.0;
    m1.gpu_bytes = 1.0;
    m2.total_time_ms = 80.0;
    m2.gpu_bytes = 0.5;
    double change = impact::computeRuntimeChange(m1, m2);
    if (std::abs(change - (-0.2)) > 1e-12)
        return false;
    perf::updateLastMetrics(m1);
    auto retrieved = perf::getLastMetrics();
    if (retrieved.total_time_ms != 100.0)
        return false;
    return true;
}

bool check_validation_mathematical()
{
    std::vector<nerve::persistence::Pair> diagram = {
        {0.0, 1.0, 0}, {0.5, std::numeric_limits<double>::infinity(), 1}};
    nerve::core::DeterminismContract contract;
    auto result = nerve::persistence::accelerated::validation::validateMathematicalCorrectness(
        diagram, contract);
    if (!result.isValid)
        return false;
    return true;
}

} // namespace

int main()
{
    if (!check_error_code_lookup())
        return 1;
    if (!check_exception_safety_guard())
        return 1;
    if (!check_atomic_counter())
        return 1;
    if (!check_optimization_advice())
        return 1;
    if (!check_validation_for_valid_config())
        return 1;
    if (!check_performance_helpers())
        return 1;
    if (!check_validation_mathematical())
        return 1;
    return 0;
}
