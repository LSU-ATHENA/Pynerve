#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>

#include "detail/numpy_arrays.hpp"
#include "nerve/algorithms/distance.hpp"
#include "nerve/algorithms/mapper.hpp"
#include "nerve/nn/diagram_conv.hpp"

namespace py = pybind11;
using namespace nerve::algorithms;
using namespace nerve::nn;
using nerve::python_bindings::copy_to_array;
using nerve::python_bindings::to_span;

namespace {

template <typename T>
using CArray = py::array_t<T, py::array::c_style | py::array::forcecast>;

template <typename T>
struct ValidatedArray {
  CArray<T> array;
  std::span<const T> values;
};

size_t checked_product(size_t lhs, size_t rhs, const char *name) {
  if (rhs != 0 && lhs > std::numeric_limits<size_t>::max() / rhs) {
    throw std::invalid_argument(std::string(name) + " shape overflows size_t");
  }
  return lhs * rhs;
}

template <typename T>
void validate_finite_values(const py::buffer_info &info, const char *name) {
  const T *data = static_cast<const T *>(info.ptr);
  for (py::ssize_t i = 0; i < info.size; ++i) {
    if (!std::isfinite(static_cast<double>(data[i]))) {
      throw std::invalid_argument(std::string(name) +
                                  " must contain only finite values");
    }
  }
}

template <typename T>
ValidatedArray<T> validate_point_array(CArray<T> array, size_t rows,
                                       size_t dim, const char *name) {
  if (rows == 0) {
    throw std::invalid_argument(std::string(name) + " row count must be positive");
  }
  if (dim == 0) {
    throw std::invalid_argument(std::string(name) + " dimension must be positive");
  }
  const size_t expected = checked_product(rows, dim, name);
  py::buffer_info info = array.request();
  if (info.ndim != 1 && info.ndim != 2) {
    throw std::invalid_argument(std::string(name) +
                                " must be a rank-1 flat or rank-2 array");
  }
  if (static_cast<size_t>(info.size) != expected) {
    throw std::invalid_argument(std::string(name) +
                                " size must equal rows * dim");
  }
  if (info.ndim == 2 &&
      (static_cast<size_t>(info.shape[0]) != rows ||
       static_cast<size_t>(info.shape[1]) != dim)) {
    throw std::invalid_argument(std::string(name) +
                                " shape must match rows and dim");
  }
  validate_finite_values<T>(info, name);
  return {std::move(array),
          std::span<const T>(static_cast<const T *>(info.ptr), expected)};
}

template <typename T>
ValidatedArray<T> validate_exact_array(CArray<T> array, size_t expected,
                                       const char *name) {
  py::buffer_info info = array.request();
  if (static_cast<size_t>(info.size) != expected) {
    throw std::invalid_argument(std::string(name) +
                                " size does not match expected layout");
  }
  validate_finite_values<T>(info, name);
  return {std::move(array),
          std::span<const T>(static_cast<const T *>(info.ptr), expected)};
}

template <typename T>
ValidatedArray<T> validate_array_values(CArray<T> array, const char *name) {
  py::buffer_info info = array.request();
  validate_finite_values<T>(info, name);
  return {std::move(array),
          std::span<const T>(static_cast<const T *>(info.ptr),
                             static_cast<size_t>(info.size))};
}

}  // namespace

template <typename T = float>
void bind_distance_computer(py::module &m, const char *name) {
  using DC = DistanceMatrixComputer<T>;
  using Config = typename DC::Config;

  py::enum_<typename Config::Metric>(
      m, (std::string("DistanceMetric") + name).c_str())
      .value("EUCLIDEAN", Config::Metric::EUCLIDEAN)
      .value("MANHATTAN", Config::Metric::MANHATTAN)
      .value("COSINE", Config::Metric::COSINE)
      .value("CHEBYSHEV", Config::Metric::CHEBYSHEV);

  py::class_<Config>(m, (std::string("DistanceConfig") + name).c_str())
      .def(py::init<>())
      .def_readwrite("metric", &Config::metric)
      .def_readwrite("use_simd", &Config::use_simd)
      .def_readwrite("use_openmp", &Config::use_openmp)
      .def_readwrite("block_size", &Config::block_size);

  py::class_<DC>(m, name)
      .def(py::init<Config>(), py::arg("config") = Config())
      .def(
          "compute",
          [](DC &self, CArray<T> points, size_t n_points, size_t dim) {
            auto parsed = validate_point_array(std::move(points), n_points, dim, "points");
            auto result = self.compute(parsed.values, n_points, dim);
            return copy_to_array(result, {n_points, n_points});
          },
          py::arg("points"), py::arg("n_points"), py::arg("dim"),
          "Compute full distance matrix")
      .def(
          "compute_pairwise",
          [](DC &self, CArray<T> set_a, size_t n_a, CArray<T> set_b,
             size_t n_b, size_t dim) {
            auto parsed_a = validate_point_array(std::move(set_a), n_a, dim, "set_a");
            auto parsed_b = validate_point_array(std::move(set_b), n_b, dim, "set_b");
            auto result = self.compute_pairwise(parsed_a.values, n_a,
                                                parsed_b.values, n_b, dim);
            return copy_to_array(result, {n_a, n_b});
          },
          py::arg("set_a"), py::arg("n_a"), py::arg("set_b"), py::arg("n_b"),
          py::arg("dim"));
}

template <typename T = float>
void bind_knn_computer(py::module &m, const char *name) {
  using KNN = KNNComputer<T>;
  using Config = typename KNN::Config;
  using Result = typename KNN::Result;

  py::enum_<typename Config::Algorithm>(
      m, (std::string("KNNAlgorithm") + name).c_str())
      .value("BRUTE_FORCE", Config::Algorithm::BRUTE_FORCE);

  py::enum_<typename Config::Metric>(m,
                                     (std::string("KNNMetric") + name).c_str())
      .value("EUCLIDEAN", Config::Metric::EUCLIDEAN)
      .value("MANHATTAN", Config::Metric::MANHATTAN)
      .value("COSINE", Config::Metric::COSINE);

  py::class_<Config>(m, (std::string("KNNConfig") + name).c_str())
      .def(py::init<>())
      .def_readwrite("k", &Config::k)
      .def_readwrite("algorithm", &Config::algorithm)
      .def_readwrite("metric", &Config::metric)
      .def_readwrite("use_openmp", &Config::use_openmp);

  py::class_<Result>(m, (std::string("KNNResult") + name).c_str())
      .def_readonly("distances", &Result::distances)
      .def_readonly("indices", &Result::indices)
      .def_readonly("n_points", &Result::n_points)
      .def_readonly("k", &Result::k)
      .def("to_numpy", [](const Result &r) {
        return py::make_tuple(copy_to_array(r.distances, {r.n_points, r.k}),
                              copy_to_array(r.indices, {r.n_points, r.k}));
      });

  py::class_<KNN>(m, name)
      .def(py::init<Config>(), py::arg("config") = Config())
      .def(
          "compute",
          [](KNN &self, CArray<T> points, size_t n_points, size_t dim) {
            auto parsed = validate_point_array(std::move(points), n_points, dim, "points");
            return self.compute(parsed.values, n_points, dim);
          },
          py::arg("points"), py::arg("n_points"), py::arg("dim"),
          "Compute k-NN for all points")
      .def(
          "compute_query",
          [](KNN &self, CArray<T> reference, size_t n_ref,
             CArray<T> queries, size_t n_query, size_t dim) {
            auto parsed_ref =
                validate_point_array(std::move(reference), n_ref, dim, "reference");
            auto parsed_queries =
                validate_point_array(std::move(queries), n_query, dim, "queries");
            return self.compute_query(parsed_ref.values, n_ref,
                                      parsed_queries.values, n_query, dim);
          },
          py::arg("reference"), py::arg("n_ref"), py::arg("queries"),
          py::arg("n_query"), py::arg("dim"));
}

template <typename T = float>
void bind_mapper(py::module &m, const char *name) {
  using MA = MapperAlgorithm<T>;
  using Config = typename MA::Config;
  using Result = typename MA::Result;
  using Graph = MapperGraph<T>;
  using Node = MapperNode<T>;
  using Edge = MapperEdge<T>;

  py::class_<Node>(m, (std::string("MapperNode") + name).c_str())
      .def_readwrite("id", &Node::id)
      .def_readwrite("point_indices", &Node::point_indices)
      .def_readwrite("centroid", &Node::centroid)
      .def("size", &Node::size);

  py::class_<Edge>(m, (std::string("MapperEdge") + name).c_str())
      .def_readwrite("source", &Edge::source)
      .def_readwrite("target", &Edge::target)
      .def_readwrite("weight", &Edge::weight)
      .def_readwrite("overlap_size", &Edge::overlap_size);

  py::class_<Graph>(m, (std::string("MapperGraph") + name).c_str())
      .def_readwrite("nodes", &Graph::nodes)
      .def_readwrite("edges", &Graph::edges)
      .def("n_nodes", &Graph::n_nodes)
      .def("n_edges", &Graph::n_edges)
      .def("build_adjacency", &Graph::build_adjacency);

  py::class_<Result>(m, (std::string("MapperResult") + name).c_str())
      .def_readwrite("graph", &Result::graph)
      .def_readwrite("filter_values", &Result::filter_values)
      .def_readwrite("cover_sets", &Result::cover_sets)
      .def_readwrite("metadata", &Result::metadata);

  py::class_<Config>(m, (std::string("MapperConfig") + name).c_str())
      .def(py::init<>())
      .def_readwrite("cover_resolution", &Config::cover_resolution)
      .def_readwrite("cover_overlap", &Config::cover_overlap)
      .def_readwrite("return_graph", &Config::return_graph)
      .def_readwrite("compute_meta", &Config::compute_meta);

  py::class_<MA>(m, name)
      .def(py::init<Config>(), py::arg("config") = Config())
      .def(
          "compute",
          [](MA &self, CArray<T> point_cloud, size_t n_points,
             size_t dim) {
            auto parsed =
                validate_point_array(std::move(point_cloud), n_points, dim, "point_cloud");
            return self.compute(parsed.values, n_points, dim);
          },
          py::arg("point_cloud"), py::arg("n_points"), py::arg("dim"),
          "Run mapper algorithm on point cloud");

  m.def(
      "connected_components",
      [](const Graph &graph) { return connected_components(graph); },
      py::arg("graph"), "Find connected components of mapper graph");
}

template <typename T = float>
void bind_diagram_conv(py::module &m, const char *name) {
  using DC = DiagramConv1D<T>;
  using Config = typename DC::Config;

  py::class_<Config>(m, (std::string("DiagramConvConfig") + name).c_str())
      .def(py::init<>())
      .def_readwrite("in_channels", &Config::in_channels)
      .def_readwrite("out_channels", &Config::out_channels)
      .def_readwrite("kernel_size", &Config::kernel_size)
      .def_readwrite("stride", &Config::stride)
      .def_readwrite("use_persistence_weighting",
                     &Config::use_persistence_weighting);

  py::class_<DC>(m, name)
      .def(py::init<Config>(), py::arg("config") = Config())
      .def(
          "forward",
          [](DC &self, CArray<T> diagram, CArray<T> features,
             size_t batch_size, size_t n_pairs) {
            if (batch_size == 0) {
              throw std::invalid_argument("batch_size must be positive");
            }
            const size_t diagram_size =
                checked_product(checked_product(batch_size, n_pairs, "diagram"), 3, "diagram");
            auto parsed_diagram =
                validate_exact_array(std::move(diagram), diagram_size, "diagram");
            auto parsed_features =
                validate_array_values(std::move(features), "features");
            auto result = self.forward(parsed_diagram.values, parsed_features.values,
                                       batch_size, n_pairs);
            return copy_to_array(
                result,
                {batch_size,
                 static_cast<size_t>(self.output_size(n_pairs) / batch_size)});
          },
          py::arg("diagram"), py::arg("features"), py::arg("batch_size"),
          py::arg("n_pairs"));
}

PYBIND11_MODULE(algorithms_bindings, m) {
  m.doc() = "Nerve Algorithms: High-performance C++ implementations";

  m.attr("__version__") = "0.1.0";

  bind_distance_computer<float>(m, "DistanceMatrixComputerF");
  bind_distance_computer<double>(m, "DistanceMatrixComputerD");
  bind_knn_computer<float>(m, "KNNComputerF");
  bind_knn_computer<double>(m, "KNNComputerD");

  bind_mapper<float>(m, "MapperAlgorithmF");
  bind_mapper<double>(m, "MapperAlgorithmD");

  bind_diagram_conv<float>(m, "DiagramConv1DF");
  bind_diagram_conv<double>(m, "DiagramConv1DD");

  m.def(
      "pairwise_distances",
      [](CArray<float> points, size_t n, size_t dim) {
        DistanceMatrixComputer<float> computer;
        auto parsed = validate_point_array(std::move(points), n, dim, "points");
        auto result = computer.compute(parsed.values, n, dim);
        return copy_to_array(result, {n, n});
      },
      py::arg("points"), py::arg("n_points"), py::arg("dim"),
      "Compute pairwise distance matrix (convenience function)");

  m.def(
      "knn",
      [](CArray<float> points, size_t n, size_t dim, size_t k) {
        KNNComputer<float>::Config config;
        config.k = k;
        KNNComputer<float> computer(config);
        auto parsed = validate_point_array(std::move(points), n, dim, "points");
        auto result = computer.compute(parsed.values, n, dim);
        return py::make_tuple(copy_to_array(result.distances, {n, result.k}),
                              copy_to_array(result.indices, {n, result.k}));
      },
      py::arg("points"), py::arg("n_points"), py::arg("dim"), py::arg("k") = 5,
      "Compute k-nearest neighbors (convenience function)");

  m.attr("HAS_OPENMP") =
#ifdef NERVE_USE_OPENMP
      true;
#else
      false;
#endif

  m.attr("HAS_SIMD") =
#ifdef NERVE_USE_SIMD
      true;
#else
      false;
#endif

  m.attr("HAS_CUDA") =
#ifdef NERVE_USE_CUDA
      true;
#else
      false;
#endif
}
