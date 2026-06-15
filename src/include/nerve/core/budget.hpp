
#pragma once

#include <cstddef>

namespace nerve
{

struct PersistenceBudget
{
    std::size_t memory_limit_mb = 1024;
    std::size_t time_limit_ms = 1000;
    bool strict_budget_enforcement = true;
};

} // namespace nerve
