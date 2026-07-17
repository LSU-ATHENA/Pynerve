#include "nerve/core/buffer.hpp"
#include "nerve/determinism.hpp"
#include "nerve/persistence/kernels/ph5_ph6_ops.hpp"
#include "nerve/persistence/utils/api.hpp"

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <cstdlib>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
namespace py = pybind11;
namespace
{
using PH5PH6 = nerve::persistence::PH5PH6Engine<std::vector<double>>;

struct ParsedPoints
{
    py::array_t<double, py::array::c_style | py::array::forcecast> array;
    const double *data = nullptr;
    nerve::Size rows = 0;
    nerve::Size cols = 0;
};

nerve::Size checked_product(nerve::Size lhs, nerve::Size rhs, const char *name)
{
    if (rhs != 0 && lhs > std::numeric_limits<nerve::Size>::max() / rhs)
    {
        throw std::invalid_argument(std::string(name) + " shape overflows size_t");
    }
    return lhs * rhs;
}

void validate_point_values(const py::buffer_info &info)
{
    const auto *data = static_cast<const double *>(info.ptr);
    for (py::ssize_t i = 0; i < info.size; ++i)
    {
        if (!std::isfinite(data[i]))
        {
            throw std::invalid_argument("points contain NaN or infinite values");
        }
    }
}

void validate_point_buffer(const py::buffer_info &info, nerve::Size rows, nerve::Size cols)
{
    const nerve::Size expected = checked_product(rows, cols, "points");
    if (static_cast<nerve::Size>(info.size) != expected)
    {
        throw std::invalid_argument("points size must equal rows * dimension");
    }
    validate_point_values(info);
}

void validate_options(const nerve::persistence::PersistenceOptions &options)
{
    if (options.max_radius != std::numeric_limits<double>::infinity() &&
        (!std::isfinite(options.max_radius) || options.max_radius < 0.0))
    {
        throw std::invalid_argument("max_radius must be finite and non-negative (inf allowed)");
    }
    if (!std::isfinite(options.error_tolerance) || options.error_tolerance < 0.0)
    {
        throw std::invalid_argument("error_tolerance must be finite and non-negative");
    }
}

void validate_ph5_config(const PH5PH6::Config &config)
{
    if (!std::isfinite(config.numerical_tolerance) || config.numerical_tolerance < 0.0)
    {
        throw std::invalid_argument(
            "PH5PH6Config numerical_tolerance must be finite and non-negative");
    }
    if (config.max_iterations == 0)
    {
        throw std::invalid_argument("PH5PH6Config max_iterations must be positive");
    }
}

void validate_max_dimension(size_t max_dim)
{
    if (max_dim == 0)
    {
        throw std::invalid_argument("max_dimension must be positive");
    }
}

void throw_ph5_failure(const PH5PH6 &engine, const char *default_value)
{
    std::string message = engine.getLastError();
    if (message.empty() || message == "No errors recorded")
    {
        message = default_value;
    }
    throw std::runtime_error(message);
}

ParsedPoints parse_points(const py::handle &points_obj)
{
    if (py::isinstance<py::array_t<double>>(points_obj))
    {
        py::array_t<double, py::array::c_style | py::array::forcecast> arr =
            py::cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(points_obj);
        py::buffer_info info = arr.request();
        if (info.ndim != 2)
        {
            throw std::invalid_argument("points must be a 2D array");
        }
        const auto rows = static_cast<nerve::Size>(info.shape[0]);
        const auto cols = static_cast<nerve::Size>(info.shape[1]);
        if (rows == 0 || cols == 0)
        {
            throw std::invalid_argument("points cannot be empty");
        }
        validate_point_buffer(info, rows, cols);
        ParsedPoints parsed;
        parsed.array = arr;
        parsed.data = static_cast<const double *>(info.ptr);
        parsed.rows = rows;
        parsed.cols = cols;
        return parsed;
    }
    py::array raw = py::array::ensure(points_obj);
    if (!raw)
    {
        throw std::invalid_argument("points must be a 2D array-like object");
    }
    if (raw.ndim() != 2)
    {
        throw std::invalid_argument("points must be a 2D array");
    }
    py::array_t<double, py::array::c_style | py::array::forcecast> points(raw);
    py::buffer_info info = points.request();
    const auto rows = static_cast<nerve::Size>(info.shape[0]);
    const auto cols = static_cast<nerve::Size>(info.shape[1]);
    if (rows == 0 || cols == 0)
    {
        throw std::invalid_argument("points cannot be empty");
    }
    validate_point_buffer(info, rows, cols);
    ParsedPoints parsed;
    parsed.array = std::move(points);
    parsed.data = static_cast<const double *>(info.ptr);
    parsed.rows = rows;
    parsed.cols = cols;
    return parsed;
}

PH5PH6::PointContainer to_ph5_points(const py::handle &points_obj)
{
    ParsedPoints parsed = parse_points(points_obj);
    PH5PH6::PointContainer points;
    points.reserve(parsed.rows);
    for (size_t i = 0; i < parsed.rows; ++i)
    {
        std::vector<double> point;
        point.reserve(parsed.cols);
        for (size_t j = 0; j < parsed.cols; ++j)
        {
            point.push_back(parsed.data[i * parsed.cols + j]);
        }
        points.push_back(std::move(point));
    }
    return points;
}

py::list to_python_pairs(const PH5PH6::ResultType &result)
{
    py::list pairs;
    for (const auto &pair : result)
    {
        py::list simplex;
        for (size_t idx : pair.first)
        {
            simplex.append(idx);
        }
        pairs.append(py::make_tuple(simplex, pair.second));
    }
    return pairs;
}

using Ph5PairMethod = std::optional<PH5PH6::ResultType> (PH5PH6::*)(const PH5PH6::PointContainer &,
                                                                    size_t);
using Ph5BoolMethod = bool (PH5PH6::*)(const PH5PH6::PointContainer &, size_t);

py::list run_ph5_pair_method(PH5PH6 &engine, const py::handle &points_obj, size_t max_dim,
                             Ph5PairMethod method)
{
    validate_max_dimension(max_dim);
    auto result = (engine.*method)(to_ph5_points(points_obj), max_dim);
    if (!result)
    {
        throw_ph5_failure(engine, "PH5/PH6 persistence computation failed");
    }
    return to_python_pairs(*result);
}

bool run_ph5_bool_method(PH5PH6 &engine, const py::handle &points_obj, size_t max_dim,
                         Ph5BoolMethod method)
{
    validate_max_dimension(max_dim);
    return (engine.*method)(to_ph5_points(points_obj), max_dim);
}

bool run_ph5_stability(PH5PH6 &engine, const py::handle &points_obj, size_t max_dim,
                       size_t num_runs)
{
    validate_max_dimension(max_dim);
    if (num_runs == 0)
    {
        throw std::invalid_argument("num_runs must be positive");
    }
    return engine.runStabilityTest(to_ph5_points(points_obj), max_dim, num_runs);
}

std::vector<nerve::persistence::PersistenceEvent> parse_events(const py::iterable &events)
{
    std::vector<nerve::persistence::PersistenceEvent> parsed;
    for (const auto &item : events)
    {
        py::tuple event = py::cast<py::tuple>(item);
        if (event.size() != 2)
        {
            throw std::invalid_argument("each event must be a tuple: (type, simplex)");
        }
        nerve::persistence::PersistenceEvent translated;
        const std::string event_type = py::cast<std::string>(event[0]);
        if (event_type == "add")
        {
            translated.type = nerve::persistence::PersistenceEvent::Type::ADD_SIMPLEX;
        }
        else if (event_type == "remove")
        {
            translated.type = nerve::persistence::PersistenceEvent::Type::REMOVE_SIMPLEX;
        }
        else
        {
            throw std::invalid_argument("event type must be 'add' or 'remove'");
        }
        translated.simplex = nerve::algebra::Simplex(py::cast<std::vector<nerve::Index>>(event[1]));
        parsed.push_back(std::move(translated));
    }
    return parsed;
}
py::dict to_python_result(const nerve::persistence::PersistenceResult &result)
{
    py::list pairs;
    for (const auto &pair : result.pairs)
    {
        pairs.append(py::make_tuple(pair.birth, pair.death, pair.dimension));
    }
    py::dict diagnostics;
    diagnostics["elapsed_ms"] = result.diagnostics.elapsed_ms;
    diagnostics["operations"] = result.diagnostics.operations;
    diagnostics["pairs"] = result.diagnostics.pairs;
    diagnostics["memory_bytes"] = result.diagnostics.memory_bytes;
    diagnostics["backend"] = static_cast<int>(result.diagnostics.backend);
    diagnostics["mode"] = static_cast<int>(result.diagnostics.mode);
    diagnostics["approximation_applied"] = result.diagnostics.approximation_applied;
    diagnostics["approximation_tolerance"] = result.diagnostics.approximation_tolerance;
    py::dict out;
    out["pairs"] = pairs;
    out["betti_numbers"] = result.betti_numbers;
    out["diagnostics"] = diagnostics;
    return out;
}
template <typename Fn>
py::dict run_point_cloud_entrypoint(const py::handle &points_obj,
                                    const nerve::persistence::PersistenceOptions &options, Fn &&fn)
{
    validate_options(options);
    ParsedPoints points = parse_points(points_obj);
    nerve::core::BufferView<const double> view(points.data, points.rows * points.cols);
    auto result = fn(view, points.cols, options);
    if (result.is_error())
    {
        throw std::runtime_error(result.error().message);
    }
    return to_python_result(result.value());
}
} // namespace
PYBIND11_MODULE(pynerve_internal, m)
{
    m.doc() = "Nerve persistence bindings";
#ifndef _WIN32
    // CUBLAS_WORKSPACE_CONFIG is a CUDA environment variable;
    // setenv is POSIX-only — skip on Windows (no CUDA in wheel builds).
    setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8", 1);
#endif

    py::enum_<nerve::persistence::PersistenceMode>(m, "PersistenceMode")
        .value("EXACT", nerve::persistence::PersistenceMode::EXACT)
        .value("APPROX", nerve::persistence::PersistenceMode::APPROX)
        .export_values();
    py::enum_<nerve::persistence::PersistenceBackend>(m, "PersistenceBackend")
        .value("CPU_EXACT", nerve::persistence::PersistenceBackend::CPU_EXACT)
        .value("CPU_ADAPTIVE_ACCELERATION",
               nerve::persistence::PersistenceBackend::CPU_ADAPTIVE_ACCELERATION)
        .value("CUDA_HYBRID", nerve::persistence::PersistenceBackend::CUDA_HYBRID)
        .export_values();
    py::class_<nerve::persistence::PersistenceOptions>(m, "PersistenceOptions")
        .def(py::init<>())
        .def_readwrite("mode", &nerve::persistence::PersistenceOptions::mode)
        .def_readwrite("backend", &nerve::persistence::PersistenceOptions::backend)
        .def_readwrite("max_dim", &nerve::persistence::PersistenceOptions::max_dim)
        .def_readwrite("max_radius", &nerve::persistence::PersistenceOptions::max_radius)
        .def_readwrite("threads", &nerve::persistence::PersistenceOptions::threads)
        .def_readwrite("error_tolerance", &nerve::persistence::PersistenceOptions::error_tolerance);
    m.def(
        "compute_persistence",
        [](const py::object &points, const nerve::persistence::PersistenceOptions &options) {
            return run_point_cloud_entrypoint(points, options, nerve::persistence::compute);
        },
        py::arg("points"), py::arg("options") = nerve::persistence::PersistenceOptions{});
    m.def(
        "compute_persistence_ph4",
        [](const py::object &points, const nerve::persistence::PersistenceOptions &options) {
            return run_point_cloud_entrypoint(points, options,
                                              nerve::persistence::computePersistencePh4);
        },
        py::arg("points"), py::arg("options") = nerve::persistence::PersistenceOptions{});
    m.def(
        "compute_persistence_ph5",
        [](const py::object &points, const nerve::persistence::PersistenceOptions &options) {
            return run_point_cloud_entrypoint(points, options,
                                              nerve::persistence::computePersistencePh5);
        },
        py::arg("points"), py::arg("options") = nerve::persistence::PersistenceOptions{});
    m.def(
        "compute_persistence_ph6",
        [](const py::object &points, const nerve::persistence::PersistenceOptions &options) {
            return run_point_cloud_entrypoint(points, options,
                                              nerve::persistence::computePersistencePh6);
        },
        py::arg("points"), py::arg("options") = nerve::persistence::PersistenceOptions{});
    m.def(
        "compute_persistence_cohomology",
        [](const py::object &points, const nerve::persistence::PersistenceOptions &options) {
            return run_point_cloud_entrypoint(points, options,
                                              nerve::persistence::computePersistenceCohomology);
        },
        py::arg("points"), py::arg("options") = nerve::persistence::PersistenceOptions{});
    m.def(
        "update_persistence",
        [](const py::iterable &events, const nerve::persistence::PersistenceOptions &options) {
            validate_options(options);
            auto translated = parse_events(events);
            auto result = nerve::persistence::updatePersistence(translated, options);
            if (result.is_error())
            {
                throw std::runtime_error(result.error().message);
            }
            return to_python_result(result.value());
        },
        py::arg("events"), py::arg("options") = nerve::persistence::PersistenceOptions{});
    py::class_<PH5PH6::Config>(m, "PH5PH6Config")
        .def(py::init<>())
        .def_readwrite("numerical_tolerance", &PH5PH6::Config::numerical_tolerance)
        .def_readwrite("max_iterations", &PH5PH6::Config::max_iterations)
        .def_readwrite("validate_results", &PH5PH6::Config::validate_results)
        .def_readwrite("require_bitwise_reproducibility",
                       &PH5PH6::Config::require_bitwise_reproducibility)
        .def_readwrite("enable_checksum_validation", &PH5PH6::Config::enable_checksum_validation)
        .def_readwrite("computation_id", &PH5PH6::Config::computation_id);
    py::class_<PH5PH6::ComputationMetrics>(m, "PH5PH6Metrics")
        .def_readonly("computation_time_ms", &PH5PH6::ComputationMetrics::computation_time_ms)
        .def_readonly("peak_memory_bytes", &PH5PH6::ComputationMetrics::peak_memory_bytes)
        .def_readonly("original_simplices", &PH5PH6::ComputationMetrics::original_simplices)
        .def_readonly("final_simplices", &PH5PH6::ComputationMetrics::final_simplices)
        .def_readonly("compression_ratio", &PH5PH6::ComputationMetrics::compression_ratio)
        .def_readonly("quality_score", &PH5PH6::ComputationMetrics::quality_score)
        .def_readonly("passed_stability_checks",
                      &PH5PH6::ComputationMetrics::passed_stability_checks)
        .def_readonly("numerical_errors", &PH5PH6::ComputationMetrics::numerical_errors)
        .def_readonly("checksum_validation_passed",
                      &PH5PH6::ComputationMetrics::checksum_validation_passed);
    py::class_<PH5PH6>(m, "PH5PH6Engine")
        .def(py::init([](const PH5PH6::Config &config) {
                 validate_ph5_config(config);
                 return PH5PH6(config);
             }),
             py::arg("config") = PH5PH6::Config{})
        .def(
            "compute_persistence_cohomology",
            [](PH5PH6 &self, const py::object &points_obj, size_t max_dim) {
                return run_ph5_pair_method(self, points_obj, max_dim,
                                           &PH5PH6::computePersistenceCohomology);
            },
            py::arg("points"), py::arg("max_dimension"))
        .def(
            "compute_persistence_compressed_witness",
            [](PH5PH6 &self, const py::object &points_obj, size_t max_dim) {
                return run_ph5_pair_method(self, points_obj, max_dim,
                                           &PH5PH6::computePersistenceCompressedWitness);
            },
            py::arg("points"), py::arg("max_dimension"))
        .def(
            "compute_persistence_block_sparse",
            [](PH5PH6 &self, const py::object &points_obj, size_t max_dim) {
                return run_ph5_pair_method(self, points_obj, max_dim,
                                           &PH5PH6::computePersistenceBlockSparse);
            },
            py::arg("points"), py::arg("max_dimension"))
        .def(
            "compute_persistence_hybrid",
            [](PH5PH6 &self, const py::object &points_obj, size_t max_dim) {
                return run_ph5_pair_method(self, points_obj, max_dim,
                                           &PH5PH6::computePersistenceHybrid);
            },
            py::arg("points"), py::arg("max_dimension"))
        .def("get_computation_metrics", &PH5PH6::getComputationMetrics)
        .def("has_errors", &PH5PH6::hasErrors)
        .def("get_last_error", &PH5PH6::getLastError)
        .def(
            "validate_deterministic_result",
            [](PH5PH6 &self, const py::object &points_obj, size_t max_dim) {
                return run_ph5_bool_method(self, points_obj, max_dim,
                                           &PH5PH6::validateDeterministicResult);
            },
            py::arg("points"), py::arg("max_dimension"))
        .def(
            "run_stability_test",
            [](PH5PH6 &self, const py::object &points_obj, size_t max_dim, size_t num_runs) {
                return run_ph5_stability(self, points_obj, max_dim, num_runs);
            },
            py::arg("points"), py::arg("max_dimension"), py::arg("num_runs") = 10);
    m.def("determinism_seed", &nerve::determinism::seed,
          "Set the thread-local RNG seed for deterministic reproducibility");
}
