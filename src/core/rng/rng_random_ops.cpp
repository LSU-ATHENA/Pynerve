#include "nerve/core/rng/random.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numbers>
#include <numeric>
#include <ranges>
#include <stdexcept>

namespace nerve::core
{

constinit static const uint64_t SM_CONST_0 = 0x9E3779B97F4A7C15ULL;
constinit static const uint64_t SM_CONST_1 = 0xBF58476D1CE4E5B9ULL;
constinit static const uint64_t SM_CONST_2 = 0x94D049BB133111EBULL;
constexpr uint64_t CLUSTER_SEED_OFFSET = 1000ULL;

namespace
{

[[nodiscard]] size_t checkedProduct(size_t lhs, size_t rhs, const char *context)
{
    if (rhs != 0 && lhs > std::numeric_limits<size_t>::max() / rhs)
    {
        throw std::length_error(context);
    }
    return lhs * rhs;
}

[[nodiscard]] constexpr uint64_t mixSeed(uint64_t x) noexcept
{
    x += SM_CONST_0;
    x = (x ^ (x >> 30)) * SM_CONST_1;
    x = (x ^ (x >> 27)) * SM_CONST_2;
    return x ^ (x >> 31);
}

[[nodiscard]] uint64_t checkedSeedOffset(uint64_t base, uint64_t offset)
{
    if (base > std::numeric_limits<uint64_t>::max() - offset)
    {
        return mixSeed(base ^ offset);
    }
    return base + offset;
}

} // namespace

PRNGKey::PRNGKey() noexcept
    : seed_{makeUnseededSeed()}
    , counter_{0}
{}

uint64_t PRNGKey::makeUnseededSeed() noexcept
{
    try
    {
        std::random_device rd;
        const uint64_t hi = static_cast<uint64_t>(rd()) << 32;
        const uint64_t lo = static_cast<uint64_t>(rd());
        return hi ^ lo;
    }
    catch (const std::exception &e)
    {
        fprintf(stderr, "error: %s\n", e.what());
        const auto now = static_cast<uint64_t>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count());
        const auto addr = reinterpret_cast<uintptr_t>(&now);
        return splitmix64(now ^ static_cast<uint64_t>(addr));
    }
}

std::vector<PRNGKey> PRNGKey::split(int n) const
{
    if (n < 0)
    {
        throw std::invalid_argument("Split count cannot be negative");
    }
    std::vector<PRNGKey> keys;
    keys.reserve(n);

    uint64_t base = seed_;
    for (int i = 0; i < n; ++i)
    {
        uint64_t new_seed = splitmix64(base + static_cast<uint64_t>(i));
        keys.emplace_back(new_seed, 0);
    }

    return keys;
}

std::vector<double> PRNGKey::normal(size_t n) const
{
    std::vector<double> result;
    result.reserve(n);

    constexpr double pi = std::numbers::pi;

    for (size_t i = 0; i < n; i += 2)
    {
        uint64_t u1_uint = advance();
        uint64_t u2_uint = advance();

        constexpr double max_val = static_cast<double>(std::numeric_limits<uint64_t>::max());
        double u1 = static_cast<double>(u1_uint) / max_val;
        double u2 = static_cast<double>(u2_uint) / max_val;

        u1 = std::max(u1, 1e-10);

        double radius = std::sqrt(-2.0 * std::log(u1));
        double theta = 2.0 * pi * u2;

        result.emplace_back(radius * std::cos(theta));
        if (i + 1 < n)
        {
            result.emplace_back(radius * std::sin(theta));
        }
    }

    return result;
}

std::vector<double> PRNGKey::uniform(size_t n, double low, double high) const
{
    if (!std::isfinite(low) || !std::isfinite(high) || !(low < high))
    {
        throw std::invalid_argument("Uniform range must be finite and ordered");
    }
    std::vector<double> result;
    result.reserve(n);

    double range = high - low;

    for (size_t i = 0; i < n; ++i)
    {
        uint64_t u = advance();
        double val = static_cast<double>(u) / static_cast<double>(UINT64_MAX);
        result.push_back(low + val * range);
    }

    return result;
}

std::vector<int64_t> PRNGKey::randint(size_t n, int64_t low, int64_t high) const
{
    if (low >= high)
    {
        throw std::invalid_argument("randint requires low < high");
    }
    std::vector<int64_t> result;
    result.reserve(n);

    uint64_t range = static_cast<uint64_t>(high) - static_cast<uint64_t>(low);

    for (size_t i = 0; i < n; ++i)
    {
        uint64_t u = advance();
        int64_t val = static_cast<int64_t>(static_cast<uint64_t>(low) + (u % range));
        result.push_back(val);
    }

    return result;
}

std::vector<int64_t> PRNGKey::permutation(size_t n) const
{
    std::vector<int64_t> result(n);
    std::iota(result.begin(), result.end(), 0);

    if (n == 0)
    {
        return result;
    }
    for (size_t i = n - 1; i > 0; --i)
    {
        uint64_t u = advance();
        size_t j = static_cast<size_t>(u % (i + 1));
        std::swap(result[i], result[j]);
    }

    return result;
}

namespace datasets
{

std::vector<std::vector<double>> sphere(size_t n_samples, double radius, double noise, int dim,
                                        const PRNGKey &key)
{
    if (dim < 0 || !std::isfinite(radius) || !std::isfinite(noise) || noise < 0.0)
    {
        throw std::invalid_argument("Invalid sphere dataset parameters");
    }
    std::vector<std::vector<double>> points;
    points.reserve(n_samples);

    const size_t embedding_dim = static_cast<size_t>(dim) + 2;
    auto normals = key.normal(checkedProduct(n_samples, embedding_dim, "Sphere sample overflow"));

    for (size_t i = 0; i < n_samples; ++i)
    {
        std::vector<double> point;
        point.reserve(dim + 2);

        double norm_sq = 0.0;
        for (size_t j = 0; j < embedding_dim; ++j)
        {
            double val = normals[i * embedding_dim + j];
            point.push_back(val);
            norm_sq += val * val;
        }

        double norm = std::sqrt(norm_sq);
        if (norm == 0.0)
        {
            norm = 1.0;
        }
        for (auto &p : point)
        {
            p = radius * p / norm;
        }

        if (noise > 0.0)
        {
            PRNGKey noise_key(checkedSeedOffset(key.seed(), static_cast<uint64_t>(i)));
            auto noise_vals = noise_key.normal(point.size());
            for (size_t j = 0; j < point.size(); ++j)
            {
                point[j] += noise * noise_vals[j];
            }
        }

        points.push_back(std::move(point));
    }

    return points;
}

std::vector<std::vector<double>> torus(size_t n_samples, double major_radius, double minor_radius,
                                       double noise, const PRNGKey &key)
{
    if (!std::isfinite(major_radius) || !std::isfinite(minor_radius) || !std::isfinite(noise) ||
        minor_radius < 0.0 || noise < 0.0)
    {
        throw std::invalid_argument("Invalid torus dataset parameters");
    }
    std::vector<std::vector<double>> points;
    points.reserve(n_samples);

    constexpr double two_pi = 2.0 * std::numbers::pi;

    auto u_vals = key.uniform(n_samples, 0.0, two_pi);
    auto v_vals = key.uniform(n_samples, 0.0, two_pi);

    for (size_t i = 0; i < n_samples; ++i)
    {
        double u = u_vals[i];
        double v = v_vals[i];

        double x = (major_radius + minor_radius * std::cos(v)) * std::cos(u);
        double y = (major_radius + minor_radius * std::cos(v)) * std::sin(u);
        double z = minor_radius * std::sin(v);

        if (noise > 0.0)
        {
            PRNGKey noise_key(checkedSeedOffset(key.seed(), static_cast<uint64_t>(i)));
            auto noise_vals = noise_key.normal(3);
            x += noise * noise_vals[0];
            y += noise * noise_vals[1];
            z += noise * noise_vals[2];
        }

        points.push_back({x, y, z});
    }

    return points;
}

std::vector<std::vector<double>> swiss_roll(size_t n_samples, double noise, const PRNGKey &key)
{
    if (!std::isfinite(noise) || noise < 0.0)
    {
        throw std::invalid_argument("Swiss roll noise must be finite and non-negative");
    }
    std::vector<std::vector<double>> points;
    points.reserve(n_samples);

    constexpr double pi = std::numbers::pi;
    auto t_vals = key.uniform(n_samples, 1.5 * pi, 4.5 * pi);
    auto height_vals = key.uniform(n_samples, 0.0, 30.0);

    for (size_t i = 0; i < n_samples; ++i)
    {
        double t = t_vals[i];
        double height = height_vals[i];

        double x = t * std::cos(t);
        double y = height;
        double z = t * std::sin(t);

        if (noise > 0.0)
        {
            PRNGKey noise_key(checkedSeedOffset(key.seed(), static_cast<uint64_t>(i)));
            auto noise_vals = noise_key.normal(3);
            x += noise * noise_vals[0];
            y += noise * noise_vals[1];
            z += noise * noise_vals[2];
        }

        points.push_back({x, y, z});
    }

    return points;
}

std::vector<std::vector<double>> cube(size_t n_samples, double noise, const PRNGKey &key)
{
    if (!std::isfinite(noise) || noise < 0.0)
    {
        throw std::invalid_argument("Cube noise must be finite and non-negative");
    }
    std::vector<std::vector<double>> points;
    points.reserve(n_samples);

    auto coords = key.uniform(checkedProduct(n_samples, 3, "Cube sample overflow"), -1.0, 1.0);

    for (size_t i = 0; i < n_samples; ++i)
    {
        double x = coords[i * 3 + 0];
        double y = coords[i * 3 + 1];
        double z = coords[i * 3 + 2];

        if (noise > 0.0)
        {
            PRNGKey noise_key(checkedSeedOffset(key.seed(), static_cast<uint64_t>(i)));
            auto noise_vals = noise_key.normal(3);
            x += noise * noise_vals[0];
            y += noise * noise_vals[1];
            z += noise * noise_vals[2];
        }

        points.push_back({x, y, z});
    }

    return points;
}

std::vector<std::vector<double>> clusters(size_t n_samples, int n_clusters, double cluster_std,
                                          const PRNGKey &key)
{
    if (n_clusters <= 0 || !std::isfinite(cluster_std) || cluster_std < 0.0)
    {
        throw std::invalid_argument("Invalid cluster dataset parameters");
    }
    std::vector<std::vector<double>> points;
    points.reserve(n_samples);

    auto centers = key.uniform(
        checkedProduct(static_cast<size_t>(n_clusters), 2, "Cluster center count overflow"), -10.0,
        10.0);

    size_t per_cluster = n_samples / n_clusters;
    size_t remainder = n_samples % n_clusters;

    for (int c = 0; c < n_clusters; ++c)
    {
        double cx = centers[c * 2 + 0];
        double cy = centers[c * 2 + 1];

        size_t n = per_cluster + (c < static_cast<int>(remainder) ? 1 : 0);

        for (size_t i = 0; i < n; ++i)
        {
            const uint64_t cluster_offset = static_cast<uint64_t>(c) * CLUSTER_SEED_OFFSET;
            PRNGKey pt_key(checkedSeedOffset(checkedSeedOffset(key.seed(), cluster_offset),
                                             static_cast<uint64_t>(i)));
            auto noise = pt_key.normal(2);

            double x = cx + cluster_std * noise[0];
            double y = cy + cluster_std * noise[1];

            points.push_back({x, y});
        }
    }

    auto perm = key.permutation(points.size());
    std::vector<std::vector<double>> shuffled;
    shuffled.reserve(points.size());

    std::ranges::transform(perm, std::back_inserter(shuffled), [&points](int64_t idx) {
        return std::move(points[static_cast<size_t>(idx)]);
    });

    return shuffled;
}

} // namespace datasets

namespace ranges
{

void add_noise(std::span<std::vector<double>> points, double stddev, const PRNGKey &key)
{
    if (!std::isfinite(stddev) || stddev < 0.0)
    {
        throw std::invalid_argument("Noise standard deviation must be finite and non-negative");
    }
    for (size_t i = 0; i < points.size(); ++i)
    {
        auto &point = points[i];
        PRNGKey noise_key(checkedSeedOffset(key.seed(), static_cast<uint64_t>(i)));
        auto noise_vals = noise_key.normal(point.size());

        std::ranges::transform(point, noise_vals, point.begin(),
                               [stddev](double p, double n) { return p + stddev * n; });
    }
}

void normalize_to_sphere(std::span<std::vector<double>> points, double radius)
{
    for (auto &&point : points)
    {
        double norm_sq = std::transform_reduce(point.begin(), point.end(), 0.0, std::plus<>(),
                                               [](double x) { return x * x; });

        double norm = std::sqrt(norm_sq);
        if (norm > 0.0)
        {
            std::ranges::for_each(point, [radius, norm](double &p) { p = radius * p / norm; });
        }
    }
}

} // namespace ranges

} // namespace nerve::core
