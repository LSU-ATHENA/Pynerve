#pragma once

#include <algorithm>
#include <span>
#include <stdexcept>
#include <vector>

#include <pybind11/numpy.h>

namespace nerve::python_bindings {

namespace py = pybind11;

template <typename T> std::span<const T> to_span(py::array_t<T> array) {
  py::buffer_info info = array.request();
  return std::span<const T>(static_cast<const T *>(info.ptr), info.size);
}

template <typename T>
py::array_t<T> copy_to_array(const T *data, size_t size,
                             std::initializer_list<size_t> shape) {
  std::vector<py::ssize_t> py_shape;
  py_shape.reserve(shape.size());
  size_t expected_size = 1;
  for (size_t extent : shape) {
    py_shape.push_back(static_cast<py::ssize_t>(extent));
    expected_size *= extent;
  }
  if (size != expected_size) {
    throw std::runtime_error("array shape does not match source data");
  }

  py::array_t<T> out(py_shape);
  if (size > 0) {
    std::copy_n(data, size, out.mutable_data());
  }
  return out;
}

template <typename T>
py::array_t<T> copy_to_array(const std::vector<T> &values,
                             std::initializer_list<size_t> shape) {
  return copy_to_array(values.data(), values.size(), shape);
}

} // namespace nerve::python_bindings
