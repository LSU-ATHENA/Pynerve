#include "nerve/common/accelerated_types.hpp"
#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/core/rng/determinism_contract.hpp"
#include "nerve/core_types.hpp"
#include "nerve/errors/errors.hpp"
#include "nerve/persistence/accelerated/accelerated_api.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace
{

using nerve::Dimension;
using nerve::Index;
using nerve::Size;
using nerve::core::BufferView;
using nerve::core::DeterminismContract;
using nerve::core::DeterminismLevel;

bool check_error_registry_singleton()
{
    auto &registry = nerve::errors::ErrorRegistry::instance();
    auto metadata = registry.getMetadata(nerve::errors::ErrorCode::SUCCESS);
    if (metadata.code != nerve::errors::ErrorCode::SUCCESS)
    {
        std::cerr << "error registry SUCCESS code mismatch\n";
        return false;
    }
    return true;
}

bool check_error_code_registry_lookup()
{
    auto &registry = nerve::errors::ErrorRegistry::instance();
    auto metadata = registry.getMetadata(nerve::errors::ErrorCode::E50_PH_ABORT);
    if (metadata.code != nerve::errors::ErrorCode::E50_PH_ABORT)
    {
        std::cerr << "error registry E50_PH_ABORT lookup failed\n";
        return false;
    }
    return true;
}

bool check_error_code_category()
{
    auto &registry = nerve::errors::ErrorRegistry::instance();
    auto cat = registry.getCategory(nerve::errors::ErrorCode::SUCCESS);
    if (cat != nerve::errors::ErrorCategory::SUCCESS)
    {
        std::cerr << "error category for SUCCESS should be SUCCESS\n";
        return false;
    }
    return true;
}

bool check_error_code_severity()
{
    auto &registry = nerve::errors::ErrorRegistry::instance();
    auto sev = registry.getSeverity(nerve::errors::ErrorCode::E50_PH_ABORT);
    (void)sev;
    return true;
}

bool check_error_code_is_transient()
{
    auto &registry = nerve::errors::ErrorRegistry::instance();
    bool trans = registry.isTransient(nerve::errors::ErrorCode::E10_GPU_OOM);
    (void)trans;
    return true;
}

bool check_error_code_is_user_error()
{
    auto &registry = nerve::errors::ErrorRegistry::instance();
    bool user = registry.isUserError(nerve::errors::ErrorCode::E51_PH_INPUT);
    (void)user;
    return true;
}

bool check_error_code_get_all_codes()
{
    auto &registry = nerve::errors::ErrorRegistry::instance();
    auto codes = registry.getAllCodes();
    if (codes.empty())
    {
        std::cerr << "getAllCodes should return at least 1 code\n";
        return false;
    }
    return true;
}

bool check_validate_vr_input_good()
{
    BufferView<const double> view;
    std::vector<double> points = {0.0, 0.0, 1.0, 0.0};
    view = {points.data(), points.size()};
    DeterminismContract contract;
    auto result = nerve::persistence::accelerated::utils::validateVrInput(view, 2, contract);
    if (result.isError())
    {
        std::cerr << "validateVrInput on good input should pass\n";
        return false;
    }
    return true;
}

bool check_validate_vr_input_bad_dim()
{
    std::vector<double> points = {0.0, 0.0, 1.0, 0.0};
    BufferView<const double> view(points.data(), points.size());
    DeterminismContract contract;
    auto result = nerve::persistence::accelerated::utils::validateVrInput(view, 0, contract);
    if (result.isSuccess())
    {
        std::cerr << "validateVrInput with bad dim should fail\n";
        return false;
    }
    return true;
}

bool check_validate_vr_input_non_finite()
{
    std::vector<double> points = {0.0, 0.0, std::numeric_limits<double>::quiet_NaN(), 0.0};
    BufferView<const double> view(points.data(), points.size());
    DeterminismContract contract(DeterminismLevel::STRICT);
    auto result = nerve::persistence::accelerated::utils::validateVrInput(view, 2, contract);
    if (result.isSuccess())
    {
        std::cerr << "validateVrInput with NaN under STRICT should fail\n";
        return false;
    }
    return true;
}

bool check_validate_estimator_radius_good()
{
    try
    {
        nerve::persistence::accelerated::utils::validateEstimatorRadius(1.0);
    }
    catch (...)
    {
        std::cerr << "validateEstimatorRadius(1.0) should not throw\n";
        return false;
    }
    return true;
}

bool check_validate_estimator_radius_bad()
{
    try
    {
        nerve::persistence::accelerated::utils::validateEstimatorRadius(-1.0);
        std::cerr << "validateEstimatorRadius(-1.0) should throw\n";
        return false;
    }
    catch (const std::invalid_argument &)
    {
        return true;
    }
    catch (...)
    {
        std::cerr << "validateEstimatorRadius(-1.0) threw wrong exception\n";
        return false;
    }
}

bool check_estimate_computation_time()
{
    double time = nerve::persistence::accelerated::utils::estimateComputationTime(100, 2, 1.0);
    if (time <= 0.0)
    {
        std::cerr << "computation time should be positive, got " << time << "\n";
        return false;
    }
    return true;
}

bool check_estimate_vr_edge_density()
{
    double density = nerve::persistence::accelerated::utils::estimateVrEdgeDensity(100, 2, 1.0);
    if (density <= 0.0 || density > 1.0)
    {
        std::cerr << "edge density should be in (0,1], got " << density << "\n";
        return false;
    }
    return true;
}

bool check_estimate_memory_usage()
{
    size_t mem = nerve::persistence::accelerated::utils::estimateMemoryUsage(100, 2, 1.0);
    if (mem == 0)
    {
        std::cerr << "memory estimate should be positive\n";
        return false;
    }
    return true;
}

bool check_error_context_default()
{
    nerve::errors::ErrorContext ctx;
    if (ctx.timestampNs != 0)
    {
        std::cerr << "default timestampNs should be 0\n";
        return false;
    }
    if (ctx.durationMs != 0.0)
    {
        std::cerr << "default durationMs should be 0.0\n";
        return false;
    }
    return true;
}

} // namespace

int main()
{
    if (!check_error_registry_singleton())
    {
        std::cerr << "FAIL: error registry singleton\n";
        return 1;
    }
    if (!check_error_code_registry_lookup())
    {
        std::cerr << "FAIL: error code registry lookup\n";
        return 1;
    }
    if (!check_error_code_category())
    {
        std::cerr << "FAIL: error code category\n";
        return 1;
    }
    if (!check_error_code_severity())
    {
        std::cerr << "FAIL: error code severity\n";
        return 1;
    }
    if (!check_error_code_is_transient())
    {
        std::cerr << "FAIL: error code is transient\n";
        return 1;
    }
    if (!check_error_code_is_user_error())
    {
        std::cerr << "FAIL: error code is user error\n";
        return 1;
    }
    if (!check_error_code_get_all_codes())
    {
        std::cerr << "FAIL: error code get all codes\n";
        return 1;
    }
    if (!check_validate_vr_input_good())
    {
        std::cerr << "FAIL: validate VR input good\n";
        return 1;
    }
    if (!check_validate_vr_input_bad_dim())
    {
        std::cerr << "FAIL: validate VR input bad dim\n";
        return 1;
    }
    if (!check_validate_vr_input_non_finite())
    {
        std::cerr << "FAIL: validate VR input non-finite\n";
        return 1;
    }
    if (!check_validate_estimator_radius_good())
    {
        std::cerr << "FAIL: validate estimator radius good\n";
        return 1;
    }
    if (!check_validate_estimator_radius_bad())
    {
        std::cerr << "FAIL: validate estimator radius bad\n";
        return 1;
    }
    if (!check_estimate_computation_time())
    {
        std::cerr << "FAIL: estimate computation time\n";
        return 1;
    }
    if (!check_estimate_vr_edge_density())
    {
        std::cerr << "FAIL: estimate VR edge density\n";
        return 1;
    }
    if (!check_estimate_memory_usage())
    {
        std::cerr << "FAIL: estimate memory usage\n";
        return 1;
    }
    if (!check_error_context_default())
    {
        std::cerr << "FAIL: error context default\n";
        return 1;
    }
    return 0;
}
