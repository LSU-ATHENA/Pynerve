#include "nerve/common/accelerated_types.hpp"
#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/core_types.hpp"
#include "nerve/persistence/core/core_types.hpp"
#include "nerve/persistence/streaming/streaming_vr.hpp"

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

using nerve::Size;
using nerve::common::VRConfig;
using nerve::persistence::FastVR;
using nerve::persistence::Pair;
using nerve::persistence::StreamingVR;

bool check_fast_vr_construction()
{
    FastVR vr;
    (void)vr;
    return true;
}

bool check_streaming_vr_construction()
{
    StreamingVR vr(64, 8);
    (void)vr;
    return true;
}

bool check_streaming_vr_construction_defaults()
{
    StreamingVR vr(1024, 64);
    (void)vr;
    return true;
}

bool check_fast_vr_on_triangle()
{
    FastVR vr;
    std::vector<double> points = {0.0, 0.0, 1.0, 0.0, 0.0, 1.0};
    VRConfig config;
    config.max_radius = 2.0;
    config.max_dim = 2;
    config.algorithm = nerve::common::VRAlgorithmSelection::EXACT_STANDARD;

    auto result = vr.computeVRResult(points, 2, 3, config);
    if (result.isError())
    {
        std::cerr << "FastVR on triangle returned error\n";
        return false;
    }

    auto pairs = result.value();
    if (pairs.empty())
    {
        std::cerr << "FastVR on triangle should produce pairs\n";
        return false;
    }

    bool found_h0_ess = false;
    for (const auto &p : pairs)
    {
        if (p.dimension == 0 && p.isInfinite())
            found_h0_ess = true;
        if (!p.isInfinite() && !(p.birth <= p.death + 1e-12))
        {
            std::cerr << "birth<=death invariant violated\n";
            return false;
        }
    }
    if (!found_h0_ess)
    {
        std::cerr << "FastVR on triangle should have H0 essential\n";
        return false;
    }
    return true;
}

bool check_fast_vr_on_two_points()
{
    FastVR vr;
    std::vector<double> points = {0.0, 0.0, 1.0, 0.0};
    VRConfig config;
    config.max_radius = 1.5;
    config.max_dim = 1;
    config.algorithm = nerve::common::VRAlgorithmSelection::EXACT_STANDARD;

    auto result = vr.computeVRResult(points, 2, 2, config);
    if (result.isError())
    {
        std::cerr << "FastVR on two points returned error\n";
        return false;
    }

    auto pairs = result.value();
    if (pairs.empty())
    {
        std::cerr << "FastVR on two points should produce pairs\n";
        return false;
    }

    Size finite_h0 = 0;
    Size essential_h0 = 0;
    for (const auto &p : pairs)
    {
        if (p.dimension == 0)
        {
            if (p.isInfinite())
                ++essential_h0;
            else
                ++finite_h0;
        }
    }
    if (finite_h0 != 1)
    {
        std::cerr << "expected 1 finite H0, got " << finite_h0 << "\n";
        return false;
    }
    if (essential_h0 != 1)
    {
        std::cerr << "expected 1 essential H0, got " << essential_h0 << "\n";
        return false;
    }
    return true;
}

bool check_streaming_vr_on_triangle()
{
    StreamingVR vr(64, 8);
    std::vector<double> points = {0.0, 0.0, 1.0, 0.0, 0.0, 1.0};
    VRConfig config;
    config.max_radius = 2.0;
    config.max_dim = 2;
    config.algorithm = nerve::common::VRAlgorithmSelection::EXACT_STANDARD;

    auto result = vr.computeStreamingResult(points, 2, 3, config);
    if (result.isError())
    {
        std::cerr << "StreamingVR on triangle returned error\n";
        return false;
    }

    auto pairs = result.value();
    if (pairs.empty())
    {
        std::cerr << "StreamingVR on triangle should produce pairs\n";
        return false;
    }

    for (const auto &p : pairs)
    {
        if (!p.isInfinite())
        {
            if (!(p.birth <= p.death + 1e-12))
            {
                std::cerr << "streaming: birth<=death violated: birth=" << p.birth
                          << " death=" << p.death << "\n";
                return false;
            }
            if (p.lifetime() < -1e-12)
            {
                std::cerr << "streaming: negative persistence\n";
                return false;
            }
        }
    }
    return true;
}

bool check_fast_vr_empty()
{
    FastVR vr;
    std::vector<double> points;
    VRConfig config;
    config.max_radius = 1.0;
    config.max_dim = 2;

    auto result = vr.computeVRResult(points, 2, 0, config);
    if (result.isSuccess())
    {
        auto pairs = result.value();
        if (!pairs.empty())
        {
            std::cerr << "empty input should produce empty pairs\n";
            return false;
        }
    }
    return true;
}

bool check_streaming_vr_empty()
{
    StreamingVR vr(64, 8);
    std::vector<double> points;
    VRConfig config;
    config.max_radius = 1.0;
    config.max_dim = 2;

    auto result = vr.computeStreamingResult(points, 2, 0, config);
    if (result.isSuccess())
    {
        auto pairs = result.value();
        if (!pairs.empty())
        {
            std::cerr << "streaming empty input should produce empty pairs\n";
            return false;
        }
    }
    return true;
}

} // namespace

int main()
{
    if (!check_fast_vr_construction())
    {
        std::cerr << "FAIL: FastVR construction\n";
        return 1;
    }
    if (!check_streaming_vr_construction())
    {
        std::cerr << "FAIL: StreamingVR construction\n";
        return 1;
    }
    if (!check_streaming_vr_construction_defaults())
    {
        std::cerr << "FAIL: StreamingVR construction defaults\n";
        return 1;
    }
    if (!check_fast_vr_on_triangle())
    {
        std::cerr << "FAIL: FastVR on triangle\n";
        return 1;
    }
    if (!check_fast_vr_on_two_points())
    {
        std::cerr << "FAIL: FastVR on two points\n";
        return 1;
    }
    if (!check_streaming_vr_on_triangle())
    {
        std::cerr << "FAIL: StreamingVR on triangle\n";
        return 1;
    }
    if (!check_fast_vr_empty())
    {
        std::cerr << "FAIL: FastVR empty\n";
        return 1;
    }
    if (!check_streaming_vr_empty())
    {
        std::cerr << "FAIL: StreamingVR empty\n";
        return 1;
    }
    return 0;
}
