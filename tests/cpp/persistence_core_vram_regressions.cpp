#include "nerve/core_types.hpp"
#include "nerve/persistence/memory/vram_algorithms.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <vector>

namespace
{

using nerve::persistence::vram::Algorithm;
using nerve::persistence::vram::VRAMConfig;

bool check_vram_config_defaults()
{
    VRAMConfig config;
    if (config.total_vram_bytes != 0)
    {
        std::cerr << "default total_vram_bytes expected 0, got " << config.total_vram_bytes << "\n";
        return false;
    }
    if (config.available_vram_bytes != 0)
    {
        std::cerr << "default available_vram_bytes expected 0\n";
        return false;
    }
    if (std::abs(config.safety_fraction - 0.8) > 1e-12)
    {
        std::cerr << "default safety_fraction expected 0.8, got " << config.safety_fraction << "\n";
        return false;
    }
    return true;
}

bool check_safe_bytes_zero()
{
    VRAMConfig config;
    config.total_vram_bytes = 0;
    config.available_vram_bytes = 0;
    if (config.safeBytes() != 0)
    {
        std::cerr << "safeBytes with zero input expected 0\n";
        return false;
    }
    return true;
}

bool check_safe_bytes_with_values()
{
    VRAMConfig config;
    config.total_vram_bytes = 1024ULL * 1024ULL * 1024ULL;
    config.available_vram_bytes = 1024ULL * 1024ULL * 1024ULL;
    config.safety_fraction = 0.8;

    auto safe = config.safeBytes();
    auto expected = static_cast<std::size_t>(0.8 * 1024ULL * 1024ULL * 1024ULL);
    if (safe != expected)
    {
        std::cerr << "safeBytes expected " << expected << ", got " << safe << "\n";
        return false;
    }
    return true;
}

bool check_vram_algorithm_selection_full_gpu()
{
    VRAMConfig config;
    config.total_vram_bytes = 2ULL * 1024ULL * 1024ULL * 1024ULL;
    config.available_vram_bytes = 2ULL * 1024ULL * 1024ULL * 1024ULL;

    auto algo = config.select(1000, 3);
    if (algo != Algorithm::FULL_GPU)
    {
        std::cerr << "expected FULL_GPU for 2GB VRAM\n";
        return false;
    }
    return true;
}

bool check_vram_algorithm_selection_chunked()
{
    VRAMConfig config;
    config.total_vram_bytes = 512ULL * 1024ULL * 1024ULL;
    config.available_vram_bytes = 512ULL * 1024ULL * 1024ULL;

    auto algo = config.select(1000, 3);
    if (algo != Algorithm::CHUNKED)
    {
        std::cerr << "expected CHUNKED for 512MB VRAM\n";
        return false;
    }
    return true;
}

bool check_vram_algorithm_selection_streaming()
{
    VRAMConfig config;
    config.total_vram_bytes = 64ULL * 1024ULL * 1024ULL;
    config.available_vram_bytes = 64ULL * 1024ULL * 1024ULL;

    auto algo = config.select(1000, 3);
    if (algo != Algorithm::STREAMING)
    {
        std::cerr << "expected STREAMING for 64MB VRAM\n";
        return false;
    }
    return true;
}

bool check_vram_algorithm_selection_empty()
{
    VRAMConfig config;
    config.total_vram_bytes = 1024ULL * 1024ULL * 1024ULL;
    config.available_vram_bytes = 1024ULL * 1024ULL * 1024ULL;

    auto algo = config.select(0, 0);
    if (algo != Algorithm::HYBRID)
    {
        std::cerr << "expected HYBRID for empty input\n";
        return false;
    }
    return true;
}

bool check_select_algorithm_small()
{
    auto algo = nerve::persistence::vram::selectAlgorithm(100, 3, 256ULL * 1024ULL * 1024ULL);
    if (algo != Algorithm::STREAMING)
    {
        std::cerr << "selectAlgorithm expected STREAMING for 256MB\n";
        return false;
    }
    return true;
}

bool check_vram_config_safe_bytes_zero_safety()
{
    VRAMConfig config;
    config.total_vram_bytes = 1024ULL * 1024ULL * 1024ULL;
    config.available_vram_bytes = 1024ULL * 1024ULL * 1024ULL;
    config.safety_fraction = 0.0;

    if (config.safeBytes() != 0)
    {
        std::cerr << "safeBytes with zero safety fraction expected 0\n";
        return false;
    }
    return true;
}

bool check_vram_config_safe_bytes_overflow_safety()
{
    VRAMConfig config;
    config.total_vram_bytes = 1024ULL * 1024ULL * 1024ULL;
    config.available_vram_bytes = 1024ULL * 1024ULL * 1024ULL;
    config.safety_fraction = 2.0;

    auto safe = config.safeBytes();
    if (safe != config.available_vram_bytes)
    {
        std::cerr << "safeBytes with safety_fraction>=1.0 should cap at available\n";
        return false;
    }
    return true;
}

} // namespace

int main()
{
    if (!check_vram_config_defaults())
    {
        std::cerr << "FAIL: VRAM config defaults\n";
        return 1;
    }
    if (!check_safe_bytes_zero())
    {
        std::cerr << "FAIL: safe bytes zero\n";
        return 1;
    }
    if (!check_safe_bytes_with_values())
    {
        std::cerr << "FAIL: safe bytes with values\n";
        return 1;
    }
    if (!check_vram_algorithm_selection_full_gpu())
    {
        std::cerr << "FAIL: VRAM algorithm selection full GPU\n";
        return 1;
    }
    if (!check_vram_algorithm_selection_chunked())
    {
        std::cerr << "FAIL: VRAM algorithm selection chunked\n";
        return 1;
    }
    if (!check_vram_algorithm_selection_streaming())
    {
        std::cerr << "FAIL: VRAM algorithm selection streaming\n";
        return 1;
    }
    if (!check_vram_algorithm_selection_empty())
    {
        std::cerr << "FAIL: VRAM algorithm selection empty\n";
        return 1;
    }
    if (!check_select_algorithm_small())
    {
        std::cerr << "FAIL: select algorithm small\n";
        return 1;
    }
    if (!check_vram_config_safe_bytes_zero_safety())
    {
        std::cerr << "FAIL: VRAM config safe bytes zero safety\n";
        return 1;
    }
    if (!check_vram_config_safe_bytes_overflow_safety())
    {
        std::cerr << "FAIL: VRAM config safe bytes overflow safety\n";
        return 1;
    }
    return 0;
}
