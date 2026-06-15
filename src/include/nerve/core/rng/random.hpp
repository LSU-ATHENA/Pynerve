#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <compare>
#include <concepts>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <numbers>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

namespace nerve::core
{

class PRNGKey
{
public:
    explicit constexpr PRNGKey(uint64_t seed) noexcept
        : seed_{seed}
        , counter_{0}
    {}

    constexpr PRNGKey(uint64_t seed, uint64_t counter) noexcept
        : seed_{seed}
        , counter_{counter}
    {}

    PRNGKey() noexcept;

    constexpr auto operator<=>(const PRNGKey &other) const = default;

    constexpr bool operator==(const PRNGKey &other) const = default;

    PRNGKey(const PRNGKey &other) = default;

    PRNGKey(PRNGKey &&other) noexcept = default;

    [[nodiscard]] std::vector<PRNGKey> split(int n = 2) const;

    [[nodiscard]] constexpr uint64_t seed() const noexcept { return seed_; }

    [[nodiscard]] constexpr uint64_t counter() const noexcept { return counter_; }

    [[nodiscard]] std::vector<double> normal(size_t n) const;

    [[nodiscard]] std::vector<double> uniform(size_t n, double low = 0.0, double high = 1.0) const;

    [[nodiscard]] std::vector<int64_t> randint(size_t n, int64_t low, int64_t high) const;

    [[nodiscard]] std::vector<int64_t> permutation(size_t n) const;

    template <typename T>
        requires std::copyable<T>
    [[nodiscard]] std::vector<T> choice(std::span<const T> population, size_t n,
                                        bool replace = true) const;

private:
    uint64_t seed_;
    mutable uint64_t counter_;

    [[nodiscard]] static uint64_t makeUnseededSeed() noexcept;

    [[nodiscard]] static constexpr uint64_t splitmix64(uint64_t x) noexcept
    {
        x += 0x9E3779B97F4A7C15ULL;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
        return x ^ (x >> 31);
    }

    [[nodiscard]] uint64_t advance() const noexcept
    {
        uint64_t z = seed_ + counter_++;
        return splitmix64(z);
    }
};

template <typename T>
concept Numeric = std::integral<T> || std::floating_point<T>;

namespace datasets
{

[[nodiscard]] std::vector<std::vector<double>> sphere(size_t n_samples, double radius, double noise,
                                                      int dim, const PRNGKey &key);

[[nodiscard]] std::vector<std::vector<double>>
torus(size_t n_samples, double major_radius, double minor_radius, double noise, const PRNGKey &key);

[[nodiscard]] std::vector<std::vector<double>> swiss_roll(size_t n_samples, double noise,
                                                          const PRNGKey &key);

[[nodiscard]] std::vector<std::vector<double>> cube(size_t n_samples, double noise,
                                                    const PRNGKey &key);

[[nodiscard]] std::vector<std::vector<double>> clusters(size_t n_samples, int n_clusters,
                                                        double cluster_std, const PRNGKey &key);

template <Numeric T, size_t Dim>
[[nodiscard]] std::vector<std::array<T, Dim>>
generate_grid(std::array<T, Dim> min, std::array<T, Dim> max, std::array<size_t, Dim> resolution,
              const PRNGKey &key);

} // namespace datasets

namespace ranges
{

void add_noise(std::span<std::vector<double>> points, double stddev, const PRNGKey &key);

void normalize_to_sphere(std::span<std::vector<double>> points, double radius);

template <typename Pred>
    requires std::predicate<Pred, const std::vector<double> &>
[[nodiscard]] std::vector<std::vector<double>>
filter_points(std::span<const std::vector<double>> points, Pred &&predicate);

} // namespace ranges

template <typename T>
    requires std::copyable<T>
std::vector<T> PRNGKey::choice(std::span<const T> population, size_t n, bool replace) const
{
    if (population.empty())
    {
        if (n == 0)
        {
            return {};
        }
        throw std::invalid_argument("Cannot sample from an empty population");
    }
    if (!replace && n > population.size())
    {
        throw std::invalid_argument("Sample size exceeds population size without replacement");
    }

    std::vector<T> result;
    result.reserve(n);
    if (replace)
    {
        for (size_t i = 0; i < n; ++i)
        {
            const size_t idx = static_cast<size_t>(advance() % population.size());
            result.push_back(population[idx]);
        }
        return result;
    }

    const auto perm = permutation(population.size());
    for (size_t i = 0; i < n; ++i)
    {
        result.push_back(population[static_cast<size_t>(perm[i])]);
    }
    return result;
}

namespace datasets
{

template <Numeric T, size_t Dim>
std::vector<std::array<T, Dim>> generate_grid(std::array<T, Dim> min, std::array<T, Dim> max,
                                              std::array<size_t, Dim> resolution, const PRNGKey &)
{
    size_t total_points = 1;
    for (size_t axis = 0; axis < Dim; ++axis)
    {
        if (resolution[axis] == 0 || min[axis] > max[axis])
        {
            throw std::invalid_argument("Invalid grid bounds or resolution");
        }
        if (total_points > std::numeric_limits<size_t>::max() / resolution[axis])
        {
            throw std::length_error("Grid resolution overflow");
        }
        total_points *= resolution[axis];
    }

    std::vector<std::array<T, Dim>> grid;
    grid.reserve(total_points);
    for (size_t linear = 0; linear < total_points; ++linear)
    {
        size_t index = linear;
        std::array<T, Dim> point{};
        for (size_t axis = 0; axis < Dim; ++axis)
        {
            const size_t coord = index % resolution[axis];
            index /= resolution[axis];
            if (resolution[axis] == 1)
            {
                point[axis] = min[axis];
                continue;
            }
            const double t = static_cast<double>(coord) / static_cast<double>(resolution[axis] - 1);
            const double value =
                static_cast<double>(min[axis]) +
                t * (static_cast<double>(max[axis]) - static_cast<double>(min[axis]));
            point[axis] = static_cast<T>(value);
        }
        grid.push_back(point);
    }
    return grid;
}

} // namespace datasets

namespace ranges
{

template <typename Pred>
    requires std::predicate<Pred, const std::vector<double> &>
std::vector<std::vector<double>> filter_points(std::span<const std::vector<double>> points,
                                               Pred &&predicate)
{
    std::vector<std::vector<double>> filtered;
    for (const auto &point : points)
    {
        if (std::invoke(predicate, point))
        {
            filtered.push_back(point);
        }
    }
    return filtered;
}

} // namespace ranges

} // namespace nerve::core
