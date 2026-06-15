#pragma once

#if defined(__CUDACC__) && defined(__GNUC__) && __GNUC__ < 13

#include <tuple>

#define NERVE_DEFAULT_COMPARE(Type, ...)                                                           \
    bool operator==(const Type &other) const                                                       \
    {                                                                                              \
        return std::tie(__VA_ARGS__) == std::tie(other.__VA_ARGS__);                               \
    }                                                                                              \
    bool operator<(const Type &other) const                                                        \
    {                                                                                              \
        return std::tie(__VA_ARGS__) < std::tie(other.__VA_ARGS__);                                \
    }

#define NERVE_DEFAULT_COMPARE_MEMBERS(...)                                                         \
    bool operator==(const auto &other) const                                                       \
    {                                                                                              \
        return std::tie(__VA_ARGS__) == std::tie(other.__VA_ARGS__);                               \
    }                                                                                              \
    bool operator<(const auto &other) const                                                        \
    {                                                                                              \
        return std::tie(__VA_ARGS__) < std::tie(other.__VA_ARGS__);                                \
    }

#else

#include <compare>

#define NERVE_DEFAULT_COMPARE(Type, ...)                                                           \
    auto operator<=>(const Type &) const = default;                                                \
    bool operator==(const Type &) const = default;

#define NERVE_DEFAULT_COMPARE_MEMBERS(...)                                                         \
    auto operator<=>(const auto &) const = default;                                                \
    bool operator==(const auto &) const = default;

#endif
