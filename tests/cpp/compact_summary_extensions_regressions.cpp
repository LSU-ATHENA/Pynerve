#include "nerve/core/detail/compact_summary_extensions.hpp"

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <vector>

namespace
{

bool check_highdim_compact_summary_construction_defaults()
{
    nerve::core::HighDimCompactSummary summary;
    if (summary.getAllHighdimBettiNumbers().size() != 0)
    {
        std::cerr << "default summary should have no Betti numbers\n";
        return false;
    }
    if (summary.getHighdimBettiTop8().size() != 0)
    {
        std::cerr << "default summary should have empty top 8\n";
        return false;
    }
    if (summary.getMemoryEfficiency() != 0.0)
    {
        std::cerr << "default memory efficiency should be 0\n";
        return false;
    }
    return true;
}

bool check_add_pair_and_retrieve_by_dimension()
{
    nerve::core::HighDimCompactSummary summary;
    summary.setHighdimBettiNumber(0, 100);
    summary.setHighdimBettiNumber(1, 50);
    summary.setHighdimBettiNumber(2, 25);

    if (summary.getHighdimBettiNumber(0) != 100)
    {
        std::cerr << "dim 0 betti should be 100\n";
        return false;
    }
    if (summary.getHighdimBettiNumber(1) != 50)
    {
        std::cerr << "dim 1 betti should be 50\n";
        return false;
    }
    if (summary.getHighdimBettiNumber(2) != 25)
    {
        std::cerr << "dim 2 betti should be 25\n";
        return false;
    }
    if (summary.getHighdimBettiNumber(5) != 0)
    {
        std::cerr << "unset dim should return 0\n";
        return false;
    }
    if (summary.getTotalHighdimBettiNumber() != 175)
    {
        std::cerr << "total betti should be 175\n";
        return false;
    }
    return true;
}

bool check_top_pairs_filtering_produces_valid_results()
{
    nerve::core::HighDimCompactSummary summary;
    summary.setHighdimBettiNumber(0, 10);
    summary.setHighdimBettiNumber(1, 100);
    summary.setHighdimBettiNumber(2, 50);
    summary.setHighdimBettiNumber(3, 200);
    summary.setHighdimBettiNumber(4, 5);

    const auto &top8 = summary.getHighdimBettiTop8();
    if (top8.size() > 8)
    {
        std::cerr << "top8 should have at most 8 entries\n";
        return false;
    }
    if (top8.empty())
    {
        std::cerr << "top8 should not be empty with 5 betti numbers\n";
        return false;
    }
    if (top8[0] != 200)
    {
        std::cerr << "largest betti should be first, expected 200 got " << top8[0] << "\n";
        return false;
    }
    return true;
}

bool check_ph5_compact_summary_construction()
{
    nerve::core::PH5CompactSummary ph5;
    (void)ph5;
    return true;
}

bool check_ph6_compact_summary_construction()
{
    nerve::core::PH6CompactSummary ph6;
    (void)ph6;
    return true;
}

} // namespace

int main()
{
    if (!check_highdim_compact_summary_construction_defaults())
    {
        std::cerr << "FAIL: highdim compact summary construction defaults\n";
        return 1;
    }
    if (!check_add_pair_and_retrieve_by_dimension())
    {
        std::cerr << "FAIL: add pair and retrieve by dimension\n";
        return 1;
    }
    if (!check_top_pairs_filtering_produces_valid_results())
    {
        std::cerr << "FAIL: top pairs filtering produces valid results\n";
        return 1;
    }
    if (!check_ph5_compact_summary_construction())
    {
        std::cerr << "FAIL: ph5 compact summary construction\n";
        return 1;
    }
    if (!check_ph6_compact_summary_construction())
    {
        std::cerr << "FAIL: ph6 compact summary construction\n";
        return 1;
    }
    return 0;
}
