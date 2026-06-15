
#pragma once
#include "ownership_policy.hpp"

#include <type_traits>
namespace nerve::core::ownership_utils
{
template <typename T>
struct is_view_type : std::false_type
{};
template <typename T>
struct is_view_type<BufferView<T>> : std::true_type
{};
template <>
struct is_view_type<PointView> : std::true_type
{};
template <typename T>
struct is_owned_type : std::false_type
{};
template <typename T>
struct is_owned_type<OwnedBuffer<T>> : std::true_type
{};
template <>
struct is_owned_type<PointBuffer> : std::true_type
{};
template <typename T>
constexpr bool is_hot_path_compatible = is_view_type<T>::value;
template <typename T>
constexpr bool is_zero_copy_capable = is_view_type<T>::value || is_owned_type<T>::value;
#define ASSERT_VIEW_TYPE(T)                                                                        \
    static_assert(is_view_type<T>::value, "API must accept non-owning view type for hot paths")
#define ASSERT_NOT_VECTOR(T)                                                                       \
    static_assert(!std::is_same_v<T, std::vector<typename T::value_type>>,                         \
                  "API should use BufferView instead of std::vector")
#define ASSERT_OWNERSHIP_AWARE(T)                                                                  \
    static_assert(is_zero_copy_capable<T>, "API should use ownership-aware types")
template <typename InputType, typename OutputType>
struct validate_ownership_transfer
{
    static_assert(is_view_type<InputType>::value || is_owned_type<InputType>::value,
                  "Input must be ownership-aware");
    static_assert(is_view_type<OutputType>::value || is_owned_type<OutputType>::value,
                  "Output must be ownership-aware");
    static constexpr bool value = true;
};
template <typename ParamType>
constexpr bool validateHotPathApi()
{
    return is_view_type<ParamType>::value;
}
template <typename T>
constexpr bool shouldBeZeroCopy()
{
    return is_view_type<T>::value;
}
} // namespace nerve::core::ownership_utils
