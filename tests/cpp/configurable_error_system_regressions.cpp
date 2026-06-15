#include "nerve/errors/configurable_error_system.hpp"

#include <iostream>
#include <string>

namespace
{

bool check_singleton_access()
{
    auto &inst = nerve::errors::ConfigurableErrorSystemBase::instance();
    (void)inst;
    return true;
}

bool check_error_policy_get_set_roundtrip()
{
    nerve::errors::ErrorPolicy policy;
    policy.throw_on_error = false;
    policy.log_all_errors = false;
    policy.enable_stack_trace = true;
    policy.minimum_log_severity = nerve::errors::Severity::Error;

    auto &sys = nerve::errors::ConfigurableErrorSystemBase::instance();
    sys.setPolicy(policy);
    auto retrieved = sys.getPolicy();

    if (retrieved.throw_on_error != false)
    {
        std::cerr << "throw_on_error should be false\n";
        return false;
    }
    if (retrieved.log_all_errors != false)
    {
        std::cerr << "log_all_errors should be false\n";
        return false;
    }
    if (retrieved.enable_stack_trace != true)
    {
        std::cerr << "enable_stack_trace should be true\n";
        return false;
    }
    if (retrieved.minimum_log_severity != nerve::errors::Severity::Error)
    {
        std::cerr << "minimum_log_severity should be Error\n";
        return false;
    }
    return true;
}

bool check_error_code_lookup()
{
    auto &sys = nerve::errors::ConfigurableErrorSystemBase::instance();
    sys.registerErrorCode(nerve::errors::ErrorCode::E10_GPU_OOM, "GPU out of memory");
    auto desc = sys.getErrorDescription(nerve::errors::ErrorCode::E10_GPU_OOM);
    if (desc.find("GPU") == std::string::npos)
    {
        std::cerr << "expected GPU in description, got: " << desc << "\n";
        return false;
    }

    auto unknown = sys.getErrorDescription(static_cast<nerve::errors::ErrorCode>(0xDEAD));
    if (unknown != "Unknown error")
    {
        std::cerr << "unknown code should return 'Unknown error'\n";
        return false;
    }
    return true;
}

bool check_multiple_severity_levels()
{
    int info = static_cast<int>(nerve::errors::Severity::Info);
    int warning = static_cast<int>(nerve::errors::Severity::Warning);
    int error = static_cast<int>(nerve::errors::Severity::Error);
    int critical = static_cast<int>(nerve::errors::Severity::Critical);

    if (info >= warning || warning >= error || error >= critical)
    {
        std::cerr << "severity ordering should be Info < Warning < Error < Critical\n";
        return false;
    }
    return true;
}

} // namespace

int main()
{
    if (!check_singleton_access())
    {
        std::cerr << "FAIL: singleton access\n";
        return 1;
    }
    if (!check_error_policy_get_set_roundtrip())
    {
        std::cerr << "FAIL: error policy get/set roundtrip\n";
        return 1;
    }
    if (!check_error_code_lookup())
    {
        std::cerr << "FAIL: error code lookup\n";
        return 1;
    }
    if (!check_multiple_severity_levels())
    {
        std::cerr << "FAIL: multiple severity levels\n";
        return 1;
    }
    return 0;
}
