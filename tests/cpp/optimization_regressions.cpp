#include "nerve/common/accelerated_types.hpp"
#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/core_types.hpp"
#include "nerve/optimization/component_optimizations.hpp"
#include "nerve/optimization/hardware_optimizations.hpp"
#include "nerve/optimization/parameter_sweep.hpp"
#include "test_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <random>
#include <tuple>
#include <vector>

namespace
{

using nerve::optimization::AcceleratedFeatureCache;
using nerve::optimization::Parameter;
using nerve::optimization::ParameterCombination;
using nerve::optimization::ParameterSweepEngine;
using nerve::optimization::SweepConfig;
using namespace nerve::test;


bool check_parameter_construction()
{
    Parameter param;
    param.name = "max_radius";
    param.min_value = 0.0;
    param.max_value = 10.0;
    param.num_steps = 10;

    if (param.name != "max_radius")
    {
        std::cerr << "parameter name not set\n";
        return false;
    }
    return true;
}

bool check_parameter_generate_values()
{
    Parameter param;
    param.name = "test";
    param.min_value = 0.0;
    param.max_value = 5.0;
    param.num_steps = 6;
    param.scale_type = "linear";

    auto values = param.generateValues();
    if (values.empty())
    {
        std::cerr << "generateValues returned empty\n";
        return false;
    }
    if (values.size() != 6)
    {
        std::cerr << "expected 6 values, got " << values.size() << "\n";
        return false;
    }
    return true;
}

bool check_parameter_is_valid()
{
    Parameter param;
    param.name = "valid";
    param.min_value = 0.0;
    param.max_value = 1.0;
    param.num_steps = 5;

    if (!param.isValid())
    {
        std::cerr << "valid parameter should pass isValid\n";
        return false;
    }

    Parameter invalid;
    invalid.name = "";
    if (invalid.isValid())
    {
        std::cerr << "empty name should be invalid\n";
        return false;
    }
    return true;
}

bool check_parameter_combination()
{
    ParameterCombination comb;
    comb.values["param1"] = 1.0;
    comb.values["param2"] = 2.0;

    auto hash = comb.computeHash();
    (void)hash;

    auto serialized = comb.serialize();
    if (serialized.empty())
    {
        std::cerr << "serialized empty\n";
        return false;
    }
    return true;
}

bool check_sweep_config_default()
{
    SweepConfig config;
    if (config.max_concurrent_evaluations != 4)
    {
        std::cerr << "default max_concurrent_evaluations should be 4\n";
        return false;
    }
    return true;
}

bool check_parameter_sweep_engine_construction()
{
    SweepConfig config;
    Parameter param;
    param.name = "x";
    param.min_value = 0.0;
    param.max_value = 1.0;
    param.num_steps = 3;
    config.parameters.push_back(param);

    ParameterSweepEngine engine(config);
    (void)engine;
    return true;
}

bool check_parameter_sweep_generate_combinations()
{
    SweepConfig config;
    Parameter param;
    param.name = "x";
    param.min_value = 0.0;
    param.max_value = 1.0;
    param.num_steps = 3;
    config.parameters.push_back(param);

    ParameterSweepEngine engine(config);
    auto combinations = engine.generateCombinations();
    if (combinations.empty())
    {
        std::cerr << "generateCombinations should produce results\n";
        return false;
    }
    return true;
}

bool check_param_combination_deserialize()
{
    ParameterCombination comb;
    comb.values["a"] = 1.0;
    comb.values["b"] = 2.0;

    auto serialized = comb.serialize();
    ParameterCombination restored;
    if (!restored.deserialize(serialized))
    {
        std::cerr << "deserialize failed\n";
        return false;
    }
    return true;
}

bool check_hardware_prefetch()
{
    int data = 42;
    nerve::optimization::prefetch<nerve::optimization::PrefetchLevel::L1>(&data);
    nerve::optimization::prefetchWrite<nerve::optimization::PrefetchLevel::L2>(&data);
    return true;
}

} // namespace

int main()
{
    if (!check_parameter_construction())
    {
        std::cerr << "FAIL: Parameter construction\n";
        return 1;
    }
    if (!check_parameter_generate_values())
    {
        std::cerr << "FAIL: Parameter generateValues\n";
        return 1;
    }
    if (!check_parameter_is_valid())
    {
        std::cerr << "FAIL: Parameter isValid\n";
        return 1;
    }
    if (!check_parameter_combination())
    {
        std::cerr << "FAIL: ParameterCombination\n";
        return 1;
    }
    if (!check_sweep_config_default())
    {
        std::cerr << "FAIL: SweepConfig default\n";
        return 1;
    }
    if (!check_parameter_sweep_engine_construction())
    {
        std::cerr << "FAIL: ParameterSweepEngine construction\n";
        return 1;
    }
    if (!check_parameter_sweep_generate_combinations())
    {
        std::cerr << "FAIL: generateCombinations\n";
        return 1;
    }
    if (!check_param_combination_deserialize())
    {
        std::cerr << "FAIL: deserialize\n";
        return 1;
    }
    if (!check_hardware_prefetch())
    {
        std::cerr << "FAIL: hardware prefetch\n";
        return 1;
    }
    return 0;
}
