
#include "nerve/persistence/kernels/ph4_ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

namespace
{

using nerve::Dimension;
using nerve::Field;
using nerve::persistence::CompactSummary;
using nerve::persistence::Pair;
using nerve::persistence::StabilityCertificate;

std::mt19937_64 make_rng()
{
    std::mt19937_64 rng(42);
    return rng;
}

bool check_stability_certificate_default()
{
    StabilityCertificate cert;
    if (std::abs(cert.getStabilityConstant() - 0.0) > 1e-10)
        return false;
    if (std::abs(cert.getNumericalResidual() - 0.0) > 1e-10)
        return false;
    if (cert.isExact())
        return false;
    if (!cert.isValid())
        return false;
    return true;
}

bool check_stability_certificate_valid()
{
    StabilityCertificate cert(1.0, 0.5, true);
    if (std::abs(cert.getStabilityConstant() - 1.0) > 1e-10)
        return false;
    if (std::abs(cert.getNumericalResidual() - 0.5) > 1e-10)
        return false;
    if (!cert.isExact())
        return false;
    if (!cert.isValid())
        return false;
    return true;
}

bool check_stability_certificate_invalid_negative()
{
    StabilityCertificate cert(-1.0, 0.5, false);
    if (cert.isValid())
        return false;
    return true;
}

bool check_stability_certificate_validate_valid()
{
    StabilityCertificate cert(1.0, 0.5, true);
    cert.validateCertificate();
    return true;
}

bool check_compact_summary_default()
{
    CompactSummary summary;
    if (summary.getTotalPairs() != 0)
        return false;
    if (std::abs(summary.getTotalPersistence() - 0.0) > 1e-10)
        return false;
    if (summary.getMemorySavedBytes() != 0)
        return false;
    if (std::abs(summary.getCompressionRatio() - 1.0) > 1e-10)
        return false;
    auto top = summary.getTopPairs(10);
    if (!top.empty())
        return false;
    return true;
}

bool check_compact_summary_add_pairs()
{
    CompactSummary summary;
    summary.addPair({0.0, 1.0, 0});
    summary.addPair({0.5, 2.0, 1});
    summary.addPair({0.0, 3.0, 0});
    if (summary.getTotalPairs() != 3)
        return false;
    if (std::abs(summary.getTotalPersistence() - (1.0 + 1.5 + 3.0)) > 1e-10)
        return false;
    auto top1 = summary.getTopPairs(1);
    if (top1.size() != 1)
        return false;
    if (std::abs(top1[0].lifetime() - 3.0) > 1e-10)
        return false;
    return true;
}

bool check_compact_summary_top_pairs()
{
    CompactSummary summary;
    summary.addPair({0.0, 1.0, 0});
    summary.addPair({0.0, 5.0, 0});
    summary.addPair({0.0, 2.0, 0});
    auto top2 = summary.getTopPairs(2);
    if (top2.size() != 2)
        return false;
    if (top2[0].lifetime() < top2[1].lifetime())
        return false;
    return true;
}

bool check_compact_summary_compression_ratio()
{
    CompactSummary summary;
    for (int i = 0; i < 100; ++i)
    {
        summary.addPair({static_cast<double>(i), static_cast<double>(i + 1), 0});
    }
    double ratio = summary.getCompressionRatio();
    if (ratio <= 0.0 || ratio > 1.0)
        return false;
    return true;
}

bool check_compact_summary_large_threshold()
{
    CompactSummary summary;
    for (int i = 0; i < 2000; ++i)
    {
        summary.addPair({static_cast<double>(i), static_cast<double>(i + 1), 0});
    }
    if (summary.getTotalPairs() != 2000)
        return false;
    if (summary.getMemorySavedBytes() == 0)
        return false;
    return true;
}

bool check_compact_summary_mixed_dimensions()
{
    CompactSummary summary;
    summary.addPair({0.0, 1.0, 0});
    summary.addPair({0.0, 2.0, 1});
    summary.addPair({1.0, 3.0, 0});
    summary.addPair({0.5, 1.5, 2});
    if (summary.getTotalPairs() != 4)
        return false;
    auto top = summary.getTopPairs(4);
    if (top.size() != 4)
        return false;
    return true;
}

bool check_compact_summary_zero_persistence_pairs()
{
    CompactSummary summary;
    summary.addPair({0.0, 0.0, 0});
    if (std::abs(summary.getTotalPersistence() - 0.0) > 1e-10)
        return false;
    return true;
}

} // namespace

int main()
{
    if (!check_stability_certificate_default())
    {
        std::cerr << "FAIL: stability certificate default\n";
        return 1;
    }
    if (!check_stability_certificate_valid())
    {
        std::cerr << "FAIL: stability certificate valid\n";
        return 1;
    }
    if (!check_stability_certificate_invalid_negative())
    {
        std::cerr << "FAIL: stability certificate invalid negative\n";
        return 1;
    }
    if (!check_stability_certificate_validate_valid())
    {
        std::cerr << "FAIL: stability certificate validate valid\n";
        return 1;
    }
    if (!check_compact_summary_default())
    {
        std::cerr << "FAIL: compact summary default\n";
        return 1;
    }
    if (!check_compact_summary_add_pairs())
    {
        std::cerr << "FAIL: compact summary add pairs\n";
        return 1;
    }
    if (!check_compact_summary_top_pairs())
    {
        std::cerr << "FAIL: compact summary top pairs\n";
        return 1;
    }
    if (!check_compact_summary_compression_ratio())
    {
        std::cerr << "FAIL: compact summary compression ratio\n";
        return 1;
    }
    if (!check_compact_summary_large_threshold())
    {
        std::cerr << "FAIL: compact summary large threshold\n";
        return 1;
    }
    if (!check_compact_summary_mixed_dimensions())
    {
        std::cerr << "FAIL: compact summary mixed dimensions\n";
        return 1;
    }
    if (!check_compact_summary_zero_persistence_pairs())
    {
        std::cerr << "FAIL: compact summary zero persistence pairs\n";
        return 1;
    }
    return 0;
}
