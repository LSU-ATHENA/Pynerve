// PyTorch Native Operator Registration
// Registers Nerve operations with torch::library for seamless PyTorch
// integration

#include <torch/library.h>
#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <map>
#include <string>
#include <vector>

// Forward declarations for existing Nerve functions
namespace nerve::torch
{

// Forward Declarations - Implemented in Other Files

// VR operations (vietoris_rips_torch.cpp)
at::Tensor vr_build(const at::Tensor &points, double max_radius);
at::Tensor vr_fast(const at::Tensor &points, double max_radius, const std::string &algorithm);
at::Tensor vr_persistence(const at::Tensor &distance_matrix, int64_t max_dim);

// Diagram operations (diagram_operations_torch.cpp)
double diagram_wasserstein(const at::Tensor &input, const at::Tensor &other, double p);
double diagram_bottleneck(const at::Tensor &input, const at::Tensor &other);
at::Tensor diagram_landscape(const at::Tensor &input, int64_t num_samples);
at::Tensor diagram_betti(const at::Tensor &input, int64_t dim);

// Filtration operations
at::Tensor filtration_distance_matrix(const at::Tensor &points, const std::string &metric);
at::Tensor filtration_alpha(const at::Tensor &points);
at::Tensor filtration_witness(const at::Tensor &points, const at::Tensor &landmarks);

// PH operations (autograd_torch.cpp)
at::Tensor ph_compute(const at::Tensor &filtration, int64_t max_dim);
at::Tensor ph_grad(const at::Tensor &filtration, int64_t max_dim);
at::Tensor differentiable_persistence(const at::Tensor &filtration_values);
at::Tensor ph_vr(const at::Tensor &points, int64_t max_dim, double max_radius);
at::Tensor ph_witness(const at::Tensor &landmarks, const at::Tensor &witnesses, int64_t max_dim,
                      double max_radius);
at::Tensor ph_alpha(const at::Tensor &points, int64_t max_dim);
at::Tensor ph_persistence(const at::Tensor &boundary_matrix, const at::Tensor &filtration_values,
                          int64_t max_dim);
double ph_wasserstein(const at::Tensor &d1, const at::Tensor &d2, double p, double q);
double ph_bottleneck(const at::Tensor &d1, const at::Tensor &d2);
at::Tensor ph_image(const at::Tensor &diagram, int64_t height, int64_t width, double sigma);
at::Tensor ml_persistence_image(const at::Tensor &diagram, int64_t resolution_birth,
                                int64_t resolution_death, double sigma, double birth_min,
                                double birth_max, double death_min, double death_max,
                                const std::string &weight_fn);

at::Tensor vietoris_rips(const at::Tensor &points, double max_edge_length);
at::Tensor vietoris_rips_fast(const at::Tensor &points, double max_edge_length,
                              const std::string &algorithm);
double wasserstein_distance(const at::Tensor &diagram1, const at::Tensor &diagram2, double p);
double bottleneck_distance(const at::Tensor &diagram1, const at::Tensor &diagram2);

void validate_filtration_points(const at::Tensor &points, const char *name)
{
    TORCH_CHECK(points.defined(), name, " must be defined");
    TORCH_CHECK(points.dim() == 2, name, " must be a 2D tensor [N, D]");
    TORCH_CHECK(points.size(0) > 0, name, " must contain at least one point");
    TORCH_CHECK(points.size(1) > 0, name, " must contain at least one coordinate dimension");
    TORCH_CHECK(points.is_floating_point(), name, " must be a floating-point tensor");
    TORCH_CHECK(at::isfinite(points).all().item<bool>(), name,
                " must contain only finite coordinates");
}

void xor_sorted(std::vector<int64_t> &lhs, const std::vector<int64_t> &rhs)
{
    std::vector<int64_t> result;
    result.reserve(lhs.size() + rhs.size());
    std::set_symmetric_difference(lhs.begin(), lhs.end(), rhs.begin(), rhs.end(),
                                  std::back_inserter(result));
    lhs = std::move(result);
}

at::Tensor filtration_distance_matrix(const at::Tensor &points, const std::string &metric)
{
    validate_filtration_points(points, "points");
    at::Tensor work = points.to(at::kDouble);
    at::Tensor diff = work.unsqueeze(1) - work.unsqueeze(0);
    if (metric == "euclidean" || metric == "l2")
    {
        at::Tensor squared = diff * diff;
        return at::sqrt(at::sum(squared, -1)).to(points.device());
    }
    if (metric == "manhattan" || metric == "l1")
    {
        return at::sum(at::abs(diff), -1).to(points.device());
    }
    if (metric == "chebyshev" || metric == "linf")
    {
        return std::get<0>(at::max(at::abs(diff), -1)).to(points.device());
    }
    TORCH_CHECK(false, "unsupported filtration distance metric: ", metric);
    return at::Tensor();
}

at::Tensor filtration_alpha(const at::Tensor &points)
{
    validate_filtration_points(points, "points");
    at::Tensor work = points.to(at::kDouble);
    at::Tensor distances = at::cdist(work, work) * 0.5;
    return distances.to(points.device());
}

at::Tensor filtration_witness(const at::Tensor &points, const at::Tensor &landmarks)
{
    validate_filtration_points(points, "points");
    validate_filtration_points(landmarks, "landmarks");
    TORCH_CHECK(points.size(1) == landmarks.size(1), "points and landmarks dimensions differ");
    at::Tensor work_points = points.to(at::kDouble);
    at::Tensor work_landmarks = landmarks.to(at::kDouble).to(points.device());
    at::Tensor diff = work_landmarks.unsqueeze(1) - work_points.unsqueeze(0);
    at::Tensor squared = diff * diff;
    return at::sqrt(at::sum(squared, -1)).to(points.device());
}

at::Tensor ph_vr(const at::Tensor &points, int64_t max_dim, double max_radius)
{
    at::Tensor distance_matrix = vr_build(points, max_radius);
    return vr_persistence(distance_matrix, max_dim);
}

at::Tensor ph_witness(const at::Tensor &landmarks, const at::Tensor &witnesses, int64_t max_dim,
                      double max_radius)
{
    validate_filtration_points(landmarks, "landmarks");
    validate_filtration_points(witnesses, "witnesses");
    TORCH_CHECK(max_dim >= 0, "max_dim must be non-negative");
    TORCH_CHECK(max_radius > 0.0 && !std::isnan(max_radius), "max_radius must be positive");
    TORCH_CHECK(landmarks.size(1) == witnesses.size(1),
                "landmarks and witnesses dimensions differ");

    at::Tensor landmarks_cpu = landmarks.contiguous().cpu().to(at::kDouble);
    at::Tensor witnesses_cpu = witnesses.contiguous().cpu().to(at::kDouble);
    const int64_t n_landmarks = landmarks_cpu.size(0);
    at::Tensor distances = at::cdist(witnesses_cpu, landmarks_cpu).contiguous();
    auto dist = distances.accessor<double, 2>();

    at::Tensor matrix =
        at::full({n_landmarks, n_landmarks}, std::numeric_limits<double>::infinity(),
                 at::TensorOptions().dtype(at::kDouble));
    auto matrix_acc = matrix.accessor<double, 2>();
    for (int64_t i = 0; i < n_landmarks; ++i)
    {
        matrix_acc[i][i] = 0.0;
    }
    for (int64_t i = 0; i < n_landmarks; ++i)
    {
        for (int64_t j = i + 1; j < n_landmarks; ++j)
        {
            double best = std::numeric_limits<double>::infinity();
            for (int64_t w = 0; w < witnesses_cpu.size(0); ++w)
            {
                best = std::min(best, std::max(dist[w][i], dist[w][j]));
            }
            if (best <= max_radius)
            {
                matrix_acc[i][j] = best;
                matrix_acc[j][i] = best;
            }
        }
    }
    return vr_persistence(matrix.to(landmarks.device()), max_dim);
}

at::Tensor ph_alpha(const at::Tensor &points, int64_t max_dim)
{
    TORCH_CHECK(max_dim >= 0, "max_dim must be non-negative");
    at::Tensor distance_matrix = filtration_alpha(points);
    return vr_persistence(distance_matrix, max_dim);
}

at::Tensor ph_persistence(const at::Tensor &boundary_matrix, const at::Tensor &filtration_values,
                          int64_t max_dim)
{
    TORCH_CHECK(max_dim >= 0, "max_dim must be non-negative");
    TORCH_CHECK(boundary_matrix.defined() && boundary_matrix.dim() == 2,
                "boundary_matrix must be a 2D tensor");
    TORCH_CHECK(boundary_matrix.is_floating_point(),
                "boundary_matrix must be a floating-point tensor");
    TORCH_CHECK(filtration_values.defined() && filtration_values.dim() == 1,
                "filtration_values must be a 1D tensor");
    TORCH_CHECK(filtration_values.is_floating_point(),
                "filtration_values must be a floating-point tensor");

    at::Tensor matrix_cpu = boundary_matrix.contiguous().cpu().to(at::kDouble);
    at::Tensor filtration_cpu = filtration_values.contiguous().cpu().to(at::kDouble);
    TORCH_CHECK(at::isfinite(matrix_cpu).all().item<bool>(),
                "boundary_matrix must contain only finite values");
    TORCH_CHECK(at::isfinite(filtration_cpu).all().item<bool>(),
                "filtration_values must contain only finite values");

    const int64_t rows = matrix_cpu.size(0);
    const int64_t cols = matrix_cpu.size(1);
    TORCH_CHECK(filtration_cpu.size(0) >= std::max(rows, cols),
                "filtration_values must cover every row and column");

    auto matrix = matrix_cpu.accessor<double, 2>();
    std::vector<std::vector<int64_t>> columns(static_cast<size_t>(cols));
    for (int64_t col = 0; col < cols; ++col)
    {
        auto &column = columns[static_cast<size_t>(col)];
        for (int64_t row = 0; row < rows; ++row)
        {
            if (std::abs(matrix[row][col]) > 1e-12)
            {
                column.push_back(row);
            }
        }
        std::sort(column.begin(), column.end());
    }

    std::map<int64_t, int64_t> pivot_to_col;
    std::vector<std::pair<int64_t, int64_t>> pair_indices;
    for (int64_t col = 0; col < cols; ++col)
    {
        auto &column = columns[static_cast<size_t>(col)];
        while (!column.empty())
        {
            const int64_t pivot = column.back();
            const auto basis = pivot_to_col.find(pivot);
            if (basis == pivot_to_col.end())
            {
                pivot_to_col.emplace(pivot, col);
                pair_indices.emplace_back(pivot, col);
                break;
            }
            xor_sorted(column, columns[static_cast<size_t>(basis->second)]);
        }
    }

    at::Tensor diagram = at::empty({static_cast<int64_t>(pair_indices.size()), 2},
                                   at::TensorOptions().dtype(at::kDouble));
    auto filt = filtration_cpu.accessor<double, 1>();
    auto out = diagram.accessor<double, 2>();
    for (int64_t i = 0; i < static_cast<int64_t>(pair_indices.size()); ++i)
    {
        const auto [birth, death] = pair_indices[static_cast<size_t>(i)];
        out[i][0] = filt[birth];
        out[i][1] = filt[death];
    }
    return diagram.to(boundary_matrix.device()).to(filtration_values.scalar_type());
}

double ph_wasserstein(const at::Tensor &d1, const at::Tensor &d2, double p, double q)
{
    (void)q;
    return diagram_wasserstein(d1, d2, p);
}

double ph_bottleneck(const at::Tensor &d1, const at::Tensor &d2)
{
    return diagram_bottleneck(d1, d2);
}

at::Tensor ph_image(const at::Tensor &diagram, int64_t height, int64_t width, double sigma)
{
    return ml_persistence_image(diagram, width, height, sigma, 0.0, 0.0, 0.0, 0.0, "persistence");
}

} // namespace nerve::torch

// TORCH_LIBRARY Registration

TORCH_LIBRARY(pynerve, m)
{
    // VR Operations (vr_*)
    m.def("vr_build(Tensor input, float max_radius) -> Tensor", &nerve::torch::vr_build);
    m.def("vr_fast(Tensor input, float max_radius, str algorithm) -> Tensor",
          &nerve::torch::vr_fast);
    m.def("vr_persistence(Tensor input, int max_dim) -> Tensor", &nerve::torch::vr_persistence);

    // Diagram Operations (diagram_*)
    m.def("diagram_wasserstein(Tensor input, Tensor other, float p) -> float",
          &nerve::torch::diagram_wasserstein);
    m.def("diagram_bottleneck(Tensor input, Tensor other) -> float",
          &nerve::torch::diagram_bottleneck);
    m.def("diagram_landscape(Tensor input, int num_samples) -> Tensor",
          &nerve::torch::diagram_landscape);
    m.def("diagram_betti(Tensor input, int dim) -> Tensor", &nerve::torch::diagram_betti);

    // Filtration Operations (filtration_*)
    m.def("filtration_distance_matrix(Tensor input, str metric) -> Tensor",
          &nerve::torch::filtration_distance_matrix);
    m.def("filtration_alpha(Tensor input) -> Tensor", &nerve::torch::filtration_alpha);
    m.def("filtration_witness(Tensor input, Tensor landmarks) -> Tensor",
          &nerve::torch::filtration_witness);

    // Persistence Homology (ph_*)
    m.def("ph_compute(Tensor filtration, int max_dim) -> Tensor", &nerve::torch::ph_compute);
    m.def("ph_grad(Tensor filtration, int max_dim) -> Tensor", &nerve::torch::ph_grad);

    // Full pipeline operations (used by Python API)
    m.def("ph_vr(Tensor points, int max_dim, float max_radius) -> Tensor", &nerve::torch::ph_vr);
    // clang-format off
  m.def("ph_witness(Tensor landmarks, Tensor witnesses, int max_dim, float max_radius) -> Tensor", &nerve::torch::ph_witness);
    // clang-format on
    m.def("ph_alpha(Tensor points, int max_dim) -> Tensor", &nerve::torch::ph_alpha);
    // clang-format off
  m.def("ph_persistence(Tensor boundary_matrix, Tensor filtration_values, int max_dim) -> Tensor", &nerve::torch::ph_persistence);
    // clang-format on

    // Diagram distance operations
    m.def("ph_wasserstein(Tensor d1, Tensor d2, float p, float q) -> float",
          &nerve::torch::ph_wasserstein);
    m.def("ph_bottleneck(Tensor d1, Tensor d2) -> float", &nerve::torch::ph_bottleneck);

    // Persistence image
    m.def("ph_image(Tensor diagram, int height, int width, float sigma) -> Tensor",
          &nerve::torch::ph_image);
}

// Implementation Notes
// After registration, operations are available as:
//
// Available via torch.ops.pynerve:
//   - Python: torch.ops.pynerve.vr_build(points, max_radius)
//   - Python: torch.ops.pynerve.diagram_wasserstein(d1, d2, p=2.0)
//   - Python: torch.ops.pynerve.ph_grad(filtration, max_dim=2)
//
// For autograd support:
//   - Python: torch.ops.pynerve.ph_grad(filtration, max_dim=2)
//   - This enables gradient flow through topology operations
