/**
 * @file nerve_torch_bindings.cpp
 * @brief PyBind11 bindings for nerve::torch ATen-based classes.
 */

#include "nerve/errors/errors.hpp"
#include "nerve/torch/boundary_matrix.hpp"
#include "nerve/torch/filtration.hpp"
#include "nerve/torch/mapper.hpp"
#include "nerve/torch/ml_operations.hpp"
#include "nerve/torch/persistence_diagram.hpp"
#include "nerve/torch/simplex_tree.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace py = pybind11;
using namespace nerve::torch;

namespace nerve::torch
{
at::Tensor vr_build(const at::Tensor &points, double max_radius);
at::Tensor vr_build_with_metric(const at::Tensor &points, double max_radius,
                                const std::string &metric);
at::Tensor vr_persistence(const at::Tensor &distance_matrix, int64_t max_dim);
at::Tensor ph_compute(const at::Tensor &filtration, int64_t max_dim);
at::Tensor ph_vr(const at::Tensor &points, int64_t max_dim, double max_radius);
at::Tensor ph_grad(const at::Tensor &filtration, int64_t max_dim);
double diagram_wasserstein(const at::Tensor &input, const at::Tensor &other, double p);
double diagram_bottleneck(const at::Tensor &input, const at::Tensor &other);
at::Tensor diagram_landscape(const at::Tensor &input, int64_t num_samples);
at::Tensor diagram_betti(const at::Tensor &input, int64_t dim);
} // namespace nerve::torch

// clang-format off
#include "detail/nerve_torch_bindings_classes.inl"
#include "detail/nerve_torch_bindings_persistence_core.inl"
#include "detail/nerve_torch_bindings_persistence_runtime.inl"
#include "detail/nerve_torch_bindings_functional_api.inl"
#include "detail/nerve_torch_bindings_exceptions.inl"
// clang-format on

PYBIND11_MODULE(nerve_torch_internal, m)
{
    m.doc() = "Nerve PyTorch-native bindings with ATen tensor interop";
    register_exception_translators(m);
    bind_persistence_diagram(m);
    bind_simplex_tree(m);
    bind_functional_api(m);
    m.attr("__version__") = "0.1.0";
    m.attr("DEFAULT_PI_RESOLUTION") = 64;
    m.attr("DEFAULT_PI_SIGMA") = 0.1;
    m.attr("DEFAULT_LANDSCAPE_K") = 5;
    m.attr("DEFAULT_LANDSCAPE_SAMPLES") = 100;
}
