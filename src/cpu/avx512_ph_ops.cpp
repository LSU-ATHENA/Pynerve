#include "nerve/cpu/simd.hpp"
#include "nerve/persistence/vr/vr_fast_ops.hpp"
#include "nerve/persistence/vr/vr_fast_simd_ops.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <sstream>
#include <thread>
#include <vector>

#if defined(__i386__) || defined(__x86_64__)
#include <cpuid.h>
#endif

namespace nerve::cpu::simd
{

namespace
{

constexpr bool kIsX86 =
#if defined(__i386__) || defined(__x86_64__)
    true;
#else
    false;
#endif

[[nodiscard]] bool queryCpuid(const unsigned leaf, const unsigned subleaf, unsigned *eax,
                              unsigned *ebx, unsigned *ecx, unsigned *edx)
{
#if defined(__i386__) || defined(__x86_64__)
    return __get_cpuid_count(leaf, subleaf, eax, ebx, ecx, edx) != 0;
#else
    (void)leaf;
    (void)subleaf;
    (void)eax;
    (void)ebx;
    (void)ecx;
    (void)edx;
    return false;
#endif
}

[[nodiscard]] std::uint64_t readXcr0()
{
#if defined(__i386__) || defined(__x86_64__)
    std::uint32_t eax = 0;
    std::uint32_t edx = 0;
    __asm__ volatile(".byte 0x0f, 0x01, 0xd0" : "=a"(eax), "=d"(edx) : "c"(0));
    return (static_cast<std::uint64_t>(edx) << 32U) | eax;
#else
    return 0U;
#endif
}

[[nodiscard]] bool osSupportsAvxState()
{
    const std::uint64_t xcr0 = readXcr0();
    return (xcr0 & 0x6U) == 0x6U;
}

[[nodiscard]] bool osSupportsAvx512State()
{
    const std::uint64_t xcr0 = readXcr0();
    return (xcr0 & 0xE6U) == 0xE6U;
}

[[nodiscard]] bool hasFeatureBitLeaf1Ecx(const unsigned bit)
{
    unsigned eax = 0;
    unsigned ebx = 0;
    unsigned ecx = 0;
    unsigned edx = 0;
    if (!queryCpuid(1U, 0U, &eax, &ebx, &ecx, &edx))
    {
        return false;
    }
    return (ecx & (1U << bit)) != 0U;
}

[[nodiscard]] bool hasFeatureBitLeaf7Ebx(const unsigned bit)
{
    unsigned eax = 0;
    unsigned ebx = 0;
    unsigned ecx = 0;
    unsigned edx = 0;
    if (!queryCpuid(7U, 0U, &eax, &ebx, &ecx, &edx))
    {
        return false;
    }
    return (ebx & (1U << bit)) != 0U;
}

[[nodiscard]] bool osHasXsaveAndAvxEnabled()
{
    if (!hasFeatureBitLeaf1Ecx(27U) || !hasFeatureBitLeaf1Ecx(28U))
    {
        return false;
    }
    return osSupportsAvxState();
}

[[nodiscard]] nerve::persistence::PersistenceDiagram
toDiagram(const std::vector<nerve::persistence::Pair> &pairs)
{
    nerve::persistence::PersistenceDiagram diagram;
    diagram.pairs.reserve(pairs.size());
    for (const auto &pair : pairs)
    {
        diagram.pairs.emplace_back(pair.birth, pair.death, pair.dimension);
    }
    return diagram;
}

} // namespace

bool CPUFeatureDetector::hasAVX512F()
{
    if (!kIsX86 || !osHasXsaveAndAvxEnabled())
    {
        return false;
    }
    return hasFeatureBitLeaf7Ebx(16U) && osSupportsAvx512State();
}

bool CPUFeatureDetector::hasAVX512VL()
{
    return hasAVX512F() && hasFeatureBitLeaf7Ebx(31U);
}

bool CPUFeatureDetector::hasAVX512BW()
{
    return hasAVX512F() && hasFeatureBitLeaf7Ebx(30U);
}

bool CPUFeatureDetector::hasAVX2()
{
    if (!kIsX86 || !osHasXsaveAndAvxEnabled())
    {
        return false;
    }
    return hasFeatureBitLeaf7Ebx(5U);
}

bool CPUFeatureDetector::hasFMA()
{
    if (!kIsX86 || !osHasXsaveAndAvxEnabled())
    {
        return false;
    }
    return hasFeatureBitLeaf1Ecx(12U);
}

int CPUFeatureDetector::getMaxSIMDWidth()
{
    if (hasAVX512F())
    {
        return 512;
    }
    if (hasAVX2())
    {
        return 256;
    }
    return 128;
}

std::string CPUFeatureDetector::getCPUModel()
{
    if (!kIsX86)
    {
        return "non-x86";
    }

#if defined(__i386__) || defined(__x86_64__)
    const unsigned max_ext = __get_cpuid_max(0x80000000U, nullptr);
    if (max_ext < 0x80000004U)
    {
        return "x86";
    }

    std::array<unsigned, 12> regs{};
    for (unsigned leaf = 0x80000002U; leaf <= 0x80000004U; ++leaf)
    {
        unsigned eax = 0;
        unsigned ebx = 0;
        unsigned ecx = 0;
        unsigned edx = 0;
        if (!queryCpuid(leaf, 0U, &eax, &ebx, &ecx, &edx))
        {
            return "x86";
        }
        const std::size_t base = static_cast<std::size_t>(leaf - 0x80000002U) * 4U;
        regs[base + 0U] = eax;
        regs[base + 1U] = ebx;
        regs[base + 2U] = ecx;
        regs[base + 3U] = edx;
    }

    std::array<char, 49> brand{};
    std::memcpy(brand.data(), regs.data(), 48U);
    brand[48] = '\0';

    std::string model(brand.data());
    const auto first = model.find_first_not_of(' ');
    if (first == std::string::npos)
    {
        return "x86";
    }
    model.erase(0U, first);
    const auto last = model.find_last_not_of(' ');
    model.erase(last + 1U);
    return model;
#else
    return "x86";
#endif
}

int CPUFeatureDetector::getNumCores()
{
    const unsigned threads = std::thread::hardware_concurrency();
    return threads == 0U ? 1 : static_cast<int>(threads);
}

int CPUFeatureDetector::getNumThreads()
{
    return getNumCores();
}

bool hasAVX512F()
{
    return CPUFeatureDetector::hasAVX512F();
}

bool hasAVX2()
{
    return CPUFeatureDetector::hasAVX2();
}

int getMaxSIMDWidth()
{
    return CPUFeatureDetector::getMaxSIMDWidth();
}

std::string getCPUModel()
{
    return CPUFeatureDetector::getCPUModel();
}

int getNumCores()
{
    return CPUFeatureDetector::getNumCores();
}

int getNumThreads()
{
    return CPUFeatureDetector::getNumThreads();
}

nerve::persistence::PersistenceDiagram computeCPUOptimized(std::span<const double> points,
                                                           const size_t n_points,
                                                           const size_t point_dim,
                                                           const double max_distance)
{
    nerve::persistence::PersistenceDiagram diagram;

    if (n_points == 0 || point_dim == 0 || points.empty())
    {
        return diagram;
    }

    if (n_points > std::numeric_limits<size_t>::max() / point_dim || !std::isfinite(max_distance))
    {
        return diagram;
    }

    const size_t expected_values = n_points * point_dim;
    if (points.size() < expected_values)
    {
        return diagram;
    }
    for (size_t i = 0; i < expected_values; ++i)
    {
        if (!std::isfinite(points[i]))
        {
            return diagram;
        }
    }

    const double radius =
        (max_distance > 0.0) ? max_distance : std::numeric_limits<double>::infinity();

    persistence::VRConfig config{};
    config.max_dim = 2;
    config.max_radius = radius;
    config.num_threads = static_cast<Size>(std::max(1, getNumThreads()));
    config.algorithm = persistence::VRAlgorithmSelection::AUTO;

    const bool supports_simd = hasAVX512F() || hasAVX2();
    if (supports_simd && n_points <= 1024U && point_dim <= 16U)
    {
        config.algorithm = persistence::VRAlgorithmSelection::FAST_SIMD;
    }

    core::BufferView<const double> view(points.data(), expected_values);
    std::vector<persistence::Pair> pairs;
    if (config.algorithm == persistence::VRAlgorithmSelection::FAST_SIMD)
    {
        pairs = persistence::computeVrPersistenceFastSimd(view, point_dim, config);
    }
    else
    {
        pairs = persistence::computeVrPersistenceFast(view, point_dim, config);
    }
    return toDiagram(pairs);
}

std::string getCPUOptimizationReport()
{
    std::ostringstream report;
    report << "CPU optimization report\n";
    report << "  model: " << getCPUModel() << '\n';
    report << "  simd_width_bits: " << getMaxSIMDWidth() << '\n';
    report << "  avx512f: " << (hasAVX512F() ? "yes" : "no") << '\n';
    report << "  avx2: " << (hasAVX2() ? "yes" : "no") << '\n';
    report << "  cores: " << getNumCores() << '\n';
    report << "  threads: " << getNumThreads() << '\n';
    return report.str();
}

} // namespace nerve::cpu::simd
