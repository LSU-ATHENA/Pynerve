#pragma once

#include "nerve/errors/errors.hpp"
#include "nerve/formats/packed_boundary_matrix.hpp"
#include "nerve/types.hpp"

#include <cstddef>

namespace nerve::formats::gpu
{

errors::ErrorResult<GpuScanResult> launchPackedScan(const GpuPackedLayout &layout,
                                                    void *stream_handle, int device_id);

} // namespace nerve::formats::gpu
