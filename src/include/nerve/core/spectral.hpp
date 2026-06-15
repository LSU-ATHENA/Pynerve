#pragma once

#ifndef HAS_EIGEN
#define HAS_EIGEN 0
#endif
#if HAS_EIGEN && __has_include(<Eigen/Sparse>) && __has_include(<Eigen/Dense>)
#include "nerve/spectral/persistent_laplacian.hpp"
#endif
#include "nerve/spectral/symmetric_eigendecomposition.hpp"
