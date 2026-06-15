#include "nerve/feature_access/feature_flags.hpp"
#include "nerve/features/feature_flags.hpp"

#include <cstddef>
#include <cstdlib>
#include <iostream>

namespace
{
constexpr bool defaultRuntimeFeatureEnabled(nerve::features::FeatureFlags::Feature feature)
{
    return nerve::features::FeatureFlags::DEFAULT_ENABLED[static_cast<size_t>(feature)];
}

static_assert(
    defaultRuntimeFeatureEnabled(nerve::features::FeatureFlags::Feature::PH5_PERSISTENCE));
static_assert(
    defaultRuntimeFeatureEnabled(nerve::features::FeatureFlags::Feature::PH6_HIERARCHICAL));
static_assert(
    defaultRuntimeFeatureEnabled(nerve::features::FeatureFlags::Feature::PH5_WITNESS_SAMPLING));
static_assert(
    defaultRuntimeFeatureEnabled(nerve::features::FeatureFlags::Feature::PH6_COMPRESSION));
static_assert(defaultRuntimeFeatureEnabled(
    nerve::features::FeatureFlags::Feature::HIGH_DIMENSIONAL_EXTENSIONS));
static_assert(
    defaultRuntimeFeatureEnabled(nerve::features::FeatureFlags::Feature::INCREMENTAL_UPDATES));
static_assert(
    defaultRuntimeFeatureEnabled(nerve::features::FeatureFlags::Feature::SPECTRAL_INTEGRATION));
static_assert(
    defaultRuntimeFeatureEnabled(nerve::features::FeatureFlags::Feature::ADVANCED_ALGORITHMS));
static_assert(
    !defaultRuntimeFeatureEnabled(nerve::features::FeatureFlags::Feature::DEBUG_VISUALIZATION));

static_assert(TOPOLOGIB_FEATURE_FLAG_PH4 == 1);
static_assert(TOPOLOGIB_FEATURE_FLAG_DIFFERENTIABLE_PERSISTENCE == 1);
static_assert(TOPOLOGIB_FEATURE_FLAG_SHEAF_LAPLACIAN_ADVANCED == 1);
static_assert(TOPOLOGIB_FEATURE_FLAG_ADVANCED_STREAMING == 1);
static_assert(TOPOLOGIB_FEATURE_FLAG_RESEARCH_MODE == 0);
} // namespace

int main()
{
#if __cplusplus < 202002L
    std::cerr << "Nerve test requirements\n";
    return 1;
#endif

#if defined(NERVE_EXPECT_MPI_REQUIRED) && NERVE_EXPECT_MPI_REQUIRED
    std::cerr << "MPI must remain optional for base builds\n";
    return 1;
#endif

    return EXIT_SUCCESS;
}
