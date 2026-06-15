#include "nerve/errors/errors.hpp"

#include <string>
#include <vector>

namespace nerve::errors::tda
{
namespace
{

std::size_t absoluteDifference(std::size_t lhs, std::size_t rhs)
{
    return lhs > rhs ? lhs - rhs : rhs - lhs;
}

} // namespace

} // namespace nerve::errors::tda
