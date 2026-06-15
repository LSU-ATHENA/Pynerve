#include "nerve/persistence/kernels/ph4_ops.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace nerve::persistence
{

StabilityCertificate::StabilityCertificate(double stability_constant, double numerical_residual,
                                           bool is_exact)
    : stability_constant_(stability_constant)
    , numerical_residual_(numerical_residual)
    , is_exact_(is_exact)
    , is_valid_(std::isfinite(stability_constant) && std::isfinite(numerical_residual) &&
                stability_constant >= 0.0 && numerical_residual >= 0.0)
{}

void StabilityCertificate::validateCertificate() const
{
    if (!is_valid_)
    {
        throw std::runtime_error("invalid stability certificate");
    }
    if (numerical_residual_ > stability_constant_)
    {
        throw std::runtime_error("numerical residual exceeds stability constant");
    }
}

CompactSummary::CompactSummary()
    : top_pairs_()
    , total_pairs_(0)
    , total_persistence_(0.0)
    , memory_saved_bytes_(0)
{}

void CompactSummary::addPair(const Pair &pair)
{
    total_pairs_ += 1;
    total_persistence_ += pair.lifetime();
    top_pairs_.push_back(pair);
    compressPairs();
}

std::vector<Pair> CompactSummary::getTopPairs(size_t k) const
{
    std::vector<Pair> ranked = top_pairs_;
    const size_t keep = std::min(k, ranked.size());
    std::partial_sort(
        ranked.begin(), ranked.begin() + static_cast<std::ptrdiff_t>(keep), ranked.end(),
        [](const Pair &lhs, const Pair &rhs) { return lhs.lifetime() > rhs.lifetime(); });
    ranked.resize(keep);
    return ranked;
}

double CompactSummary::getCompressionRatio() const
{
    if (total_pairs_ == 0)
    {
        return 1.0;
    }
    return static_cast<double>(top_pairs_.size()) / static_cast<double>(total_pairs_);
}

void CompactSummary::compressPairs()
{
    constexpr size_t kMaxStoredPairs = 1000;
    if (top_pairs_.size() <= kMaxStoredPairs)
    {
        return;
    }
    std::partial_sort(
        top_pairs_.begin(), top_pairs_.begin() + static_cast<std::ptrdiff_t>(kMaxStoredPairs),
        top_pairs_.end(),
        [](const Pair &lhs, const Pair &rhs) { return lhs.lifetime() > rhs.lifetime(); });
    top_pairs_.erase(top_pairs_.begin() + static_cast<std::ptrdiff_t>(kMaxStoredPairs),
                     top_pairs_.end());
    memory_saved_bytes_ = (total_pairs_ - kMaxStoredPairs) * sizeof(Pair);
}

} // namespace nerve::persistence
