#include "nerve/persistence/vr/vr_distance_tiled_ops.hpp"

#include <cassert>
#include <limits>
#include <vector>

namespace
{

void assertTwoPointMatrix(const std::vector<double> &matrix)
{
    assert(matrix.size() == 4);
    assert(matrix[1] == 5.0);
    assert(matrix[2] == 5.0);
}

void assertRejectedByAll(const std::vector<double> &points)
{
    assert(nerve::persistence::computeDistanceMatrixTiled(points, 2, 2, 1).empty());
    assert(nerve::persistence::computeDistanceMatrixNumaAware(points, 2, 2, 1).empty());
    assert(nerve::persistence::computeDistanceMatrixHierarchical(points, 2, 2).empty());
}

} // namespace

int main()
{
    const std::vector<double> points{0.0, 0.0, 3.0, 4.0};
    assertTwoPointMatrix(nerve::persistence::computeDistanceMatrixTiled(points, 2, 2, 1));
    assertTwoPointMatrix(nerve::persistence::computeDistanceMatrixNumaAware(points, 2, 2, 1));
    assertTwoPointMatrix(nerve::persistence::computeDistanceMatrixHierarchical(points, 2, 2));

    const std::vector<double> trailing_value{0.0, 0.0, 3.0, 4.0, 9.0};
    assertRejectedByAll(trailing_value);

    const std::vector<double> nonfinite{0.0, 0.0, std::numeric_limits<double>::quiet_NaN(), 4.0};
    assertRejectedByAll(nonfinite);

    const std::vector<double> overflow_prone{0.0, 0.0, std::numeric_limits<double>::max(), 4.0};
    assertRejectedByAll(overflow_prone);

    return 0;
}
