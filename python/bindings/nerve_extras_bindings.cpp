// Nerve extras bindings -- NPY I/O, thread affinity, SIMD NN ops

#include "memory/safe_memory_pool.hpp"
#include "nerve/algorithms/distance_c.h"
#include "nerve/algorithms/persistence_vectorization.hpp"
#include "nerve/anomaly/topology_drift.hpp"
#include "nerve/batching/micro_batching.hpp"
#include "nerve/cache/feature_cache.hpp"
#include "nerve/compression/model_aware_compression.hpp"
#include "nerve/core/thread_affinity.hpp"
#include "nerve/error/error_handling.hpp"
#include "nerve/filtration/simd_filtration.hpp"
#include "nerve/io/npy_io.hpp"
#include "nerve/nn/simd_nn.hpp"
#include "nerve/serialization/ph5_ph6_schema_serializer.hpp"
#include "nerve/serialization/serialization_manager.hpp"
#include "nerve/sheaf/gpu_sheaf.hpp"
#include "nerve/spectral/laplacian.hpp"
#include "nerve/spectral/persistent_laplacian.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace
{

// Helper: convert a C++ NpyArray to a numpy ndarray.
py::array_t<double> npy_array_to_numpy(const nerve::io::NpyArray &arr)
{
    if (arr.header.dtype != nerve::io::NpyDataType::Float64)
    {
        throw std::runtime_error("Only Float64 NPY arrays can be converted to numpy arrays; "
                                 "use .data for raw bytes");
    }
    std::vector<py::ssize_t> shape(arr.header.shape.begin(), arr.header.shape.end());
    auto result = py::array_t<double>(shape);
    std::memcpy(result.mutable_data(), arr.data.data(), arr.data.size());
    return result;
}

// Helper: convert a numpy ndarray to a C++ NpyArray.
nerve::io::NpyArray
numpy_to_npy_array(py::array_t<double, py::array::c_style | py::array::forcecast> arr)
{
    py::buffer_info info = arr.request();
    std::vector<nerve::Size> shape;
    for (py::ssize_t i = 0; i < info.ndim; ++i)
        shape.push_back(static_cast<nerve::Size>(info.shape[i]));

    nerve::io::NpyArray result;
    result.header.dtype = nerve::io::NpyDataType::Float64;
    result.header.shape = shape;
    result.header.fortran_order = false;
    auto data_size = static_cast<size_t>(info.size) * sizeof(double);
    result.data.resize(data_size);
    std::memcpy(result.data.data(), info.ptr, data_size);
    return result;
}

} // anonymous namespace

PYBIND11_MODULE(nerve_extras, m)
{
    m.doc() = "Nerve extras: NPY I/O, thread affinity, SIMD NN operations";

    // io/npy_io -- NPY file format I/O
    {
        auto npy = m.def_submodule("npy", "NPY file format I/O (loadNpy / saveNpy)");

        py::enum_<nerve::io::NpyDataType>(npy, "NpyDataType")
            .value("FLOAT32", nerve::io::NpyDataType::Float32)
            .value("FLOAT64", nerve::io::NpyDataType::Float64)
            .value("INT32", nerve::io::NpyDataType::Int32)
            .value("INT64", nerve::io::NpyDataType::Int64)
            .value("UINT32", nerve::io::NpyDataType::Uint32)
            .value("UINT64", nerve::io::NpyDataType::Uint64)
            .export_values();

        py::class_<nerve::io::NpyHeader>(npy, "NpyHeader")
            .def(py::init<>())
            .def_readwrite("dtype", &nerve::io::NpyHeader::dtype)
            .def_readwrite("shape", &nerve::io::NpyHeader::shape)
            .def_readwrite("fortran_order", &nerve::io::NpyHeader::fortran_order)
            .def("__repr__", [](const nerve::io::NpyHeader &h) {
                std::string repr = "NpyHeader(dtype=" + nerve::io::npyDtypeToString(h.dtype);
                repr += ", shape=(";
                for (size_t i = 0; i < h.shape.size(); ++i)
                {
                    if (i > 0)
                        repr += ", ";
                    repr += std::to_string(h.shape[i]);
                }
                repr += "), fortran_order=";
                repr += (h.fortran_order ? "True" : "False");
                repr += ")";
                return repr;
            });

        py::class_<nerve::io::NpyArray>(npy, "NpyArray")
            .def(py::init<>())
            .def_readonly("header", &nerve::io::NpyArray::header)
            .def_property_readonly(
                "data",
                [](const nerve::io::NpyArray &a) {
                    return py::bytes(reinterpret_cast<const char *>(a.data.data()), a.data.size());
                })
            .def("to_numpy", &npy_array_to_numpy, "Convert a Float64 NPY array to a numpy ndarray")
            .def("__repr__", [](const nerve::io::NpyArray &a) {
                return "NpyArray(shape=" + std::to_string(a.data.size()) + " bytes)";
            });

        npy.def("load_npy", &nerve::io::loadNpy, py::arg("path"),
                "Load a .npy file from disk, returns an NpyArray");
        npy.def("save_npy", &nerve::io::saveNpy, py::arg("path"), py::arg("array"),
                "Save an NpyArray to a .npy file on disk");
        npy.def("load_npy_from_memory", &nerve::io::loadNpyFromMemory, py::arg("buffer"),
                "Load an NpyArray from raw bytes in memory");
        npy.def("save_npy_to_memory", &nerve::io::saveNpyToMemory, py::arg("array"),
                "Save an NpyArray to raw bytes in memory");

        // Convenience: load .npy directly to numpy array
        npy.def(
            "load_npy_to_numpy",
            [](const std::string &path) { return npy_array_to_numpy(nerve::io::loadNpy(path)); },
            py::arg("path"), "Load a .npy file and convert to a numpy ndarray (Float64 only)");
        npy.def(
            "save_numpy_to_npy",
            [](const std::string &path,
               py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
                nerve::io::saveNpy(path, numpy_to_npy_array(arr));
            },
            py::arg("path"), py::arg("array"), "Save a numpy ndarray to a .npy file");

        npy.def("dtype_to_string", &nerve::io::npyDtypeToString, py::arg("dtype"),
                "Convert an NpyDataType to its NPY format string (e.g. '<f8')");
        npy.def("dtype_from_string", &nerve::io::npyDtypeFromString, py::arg("s"),
                "Parse an NPY format string to an NpyDataType");
        npy.def("dtype_size", &nerve::io::npyDtypeSize, py::arg("dtype"),
                "Return the byte size of an NpyDataType");
    }

    // core/thread_affinity -- CPU topology and thread pinning
    {
        auto thr =
            m.def_submodule("threading", "CPU topology detection and thread-pinning utilities");

        py::class_<nerve::core::CpuTopology>(thr, "CpuTopology")
            .def(py::init<>())
            .def_readonly("num_packages", &nerve::core::CpuTopology::num_packages)
            .def_readonly("num_cores", &nerve::core::CpuTopology::num_cores)
            .def_readonly("num_threads", &nerve::core::CpuTopology::num_threads)
            .def_readonly("numa_nodes", &nerve::core::CpuTopology::numa_nodes)
            .def("package_of", &nerve::core::CpuTopology::packageOf, py::arg("cpu_id"),
                 "Return the package (socket) ID for a logical CPU")
            .def("numa_node_of", &nerve::core::CpuTopology::numaNodeOf, py::arg("cpu_id"),
                 "Return the NUMA node for a logical CPU")
            .def("core_of", &nerve::core::CpuTopology::coreOf, py::arg("cpu_id"),
                 "Return the physical core for a logical CPU")
            .def("same_core_as", &nerve::core::CpuTopology::sameCoreAs, py::arg("a"), py::arg("b"),
                 "Return True if two logical CPUs share a physical core")
            .def("same_numa_as", &nerve::core::CpuTopology::sameNumaAs, py::arg("a"), py::arg("b"),
                 "Return True if two logical CPUs share a NUMA node")
            .def("__repr__", [](const nerve::core::CpuTopology &t) {
                return "CpuTopology(packages=" + std::to_string(t.num_packages) +
                       ", cores=" + std::to_string(t.num_cores) +
                       ", threads=" + std::to_string(t.num_threads) +
                       ", numa_nodes=" + std::to_string(t.numa_nodes) + ")";
            });

        thr.def("detect_cpu_topology", &nerve::core::detectCpuTopology,
                "Detect the system CPU topology");

        thr.def("pin_current_thread_to_core", &nerve::core::pinCurrentThreadToCore,
                py::arg("cpu_id"), "Pin the calling thread to a specific CPU core");
        thr.def("pin_current_thread_to_package", &nerve::core::pinCurrentThreadToPackage,
                py::arg("package_id"), "Pin the calling thread to a specific CPU package/socket");
        thr.def("pin_current_thread_to_numa_node", &nerve::core::pinCurrentThreadToNumaNode,
                py::arg("numa_node"), "Pin the calling thread to a specific NUMA node");

        thr.def("get_current_cpu", &nerve::core::getCurrentCpu,
                "Return the logical CPU ID of the calling thread");
        thr.def("get_current_numa_node", &nerve::core::getCurrentNumaNode,
                "Return the NUMA node of the calling thread");
    }

    // nn/simd_nn -- SIMD-accelerated neural network operations
    {
        auto simd = m.def_submodule("simd_nn", "SIMD-accelerated neural network primitives");

        simd.def(
            "relu",
            [](py::array_t<double, py::array::c_style | py::array::forcecast> data) {
                py::buffer_info info = data.request();
                nerve::nn::simdReLU(static_cast<double *>(info.ptr),
                                    static_cast<nerve::Size>(info.size));
                return data;
            },
            py::arg("data"), "Apply ReLU in-place to a numpy array of doubles, return the array");

        simd.def(
            "sigmoid",
            [](py::array_t<double, py::array::c_style | py::array::forcecast> data) {
                py::buffer_info info = data.request();
                nerve::nn::simdSigmoid(static_cast<double *>(info.ptr),
                                       static_cast<nerve::Size>(info.size));
                return data;
            },
            py::arg("data"),
            "Apply sigmoid in-place to a numpy array of doubles, return the array");

        simd.def(
            "tanh",
            [](py::array_t<double, py::array::c_style | py::array::forcecast> data) {
                py::buffer_info info = data.request();
                nerve::nn::simdTanh(static_cast<double *>(info.ptr),
                                    static_cast<nerve::Size>(info.size));
                return data;
            },
            py::arg("data"), "Apply tanh in-place to a numpy array of doubles, return the array");

        simd.def(
            "batch_norm",
            [](py::array_t<double, py::array::c_style | py::array::forcecast> data, double mean,
               double std_inv) {
                py::buffer_info info = data.request();
                nerve::nn::simdBatchNorm(static_cast<double *>(info.ptr),
                                         static_cast<nerve::Size>(info.size), mean, std_inv);
                return data;
            },
            py::arg("data"), py::arg("mean"), py::arg("std_inv"),
            "Apply batch normalization in-place: (x - mean) * std_inv");

        simd.def(
            "softmax",
            [](py::array_t<double, py::array::c_style | py::array::forcecast> data) {
                py::buffer_info info = data.request();
                nerve::nn::simdSoftmax(static_cast<double *>(info.ptr),
                                       static_cast<nerve::Size>(info.size));
                return data;
            },
            py::arg("data"),
            "Apply softmax in-place to a 1D numpy array of doubles, return the array");
    }

    // batching/micro_batching -- micro-batch processing
    {
        auto batch = m.def_submodule("batching", "Micro-batch processing utilities");

        py::class_<nerve::batching::BatchConfig>(batch, "BatchConfig")
            .def(py::init<>())
            .def_readwrite("max_batch_size", &nerve::batching::BatchConfig::max_batch_size)
            .def_readwrite("max_wait_time_ms", &nerve::batching::BatchConfig::max_wait_time_ms)
            .def_readwrite("min_batch_size", &nerve::batching::BatchConfig::min_batch_size)
            .def_readwrite("num_batch_threads", &nerve::batching::BatchConfig::num_batch_threads)
            .def_readwrite("max_queue_size", &nerve::batching::BatchConfig::max_queue_size)
            .def_readwrite("enable_zero_copy", &nerve::batching::BatchConfig::enable_zero_copy);

        py::class_<nerve::batching::MicroBatchProcessor>(batch, "MicroBatchProcessor")
            .def(py::init<nerve::Size, nerve::Size>(), py::arg("batch_size") = 32,
                 py::arg("max_pending") = 128)
            .def("submit", &nerve::batching::MicroBatchProcessor::submit, py::arg("input"),
                 "Submit a data vector for processing")
            .def("flush_batch", &nerve::batching::MicroBatchProcessor::flushBatch,
                 "Flush the current pending batch")
            .def_property_readonly("processed", &nerve::batching::MicroBatchProcessor::processed,
                                   "Number of items processed")
            .def_property_readonly("batches_completed",
                                   &nerve::batching::MicroBatchProcessor::batchesCompleted,
                                   "Number of batches completed")
            .def_property_readonly("pending", &nerve::batching::MicroBatchProcessor::pending,
                                   "Number of pending items")
            .def("set_batch_size", &nerve::batching::MicroBatchProcessor::setBatchSize,
                 py::arg("batch_size"), "Set the batch size");
    }

    // cache/feature_cache -- feature caching with LRU eviction
    {
        auto cache = m.def_submodule("cache", "Feature caching with LRU eviction");

        py::class_<nerve::cache::FeatureCache::CacheConfig>(cache, "CacheConfig")
            .def(py::init<>())
            .def_readwrite("max_entries", &nerve::cache::FeatureCache::CacheConfig::max_entries)
            .def_readwrite("feature_dim", &nerve::cache::FeatureCache::CacheConfig::feature_dim)
            .def_readwrite("ttl_ns", &nerve::cache::FeatureCache::CacheConfig::ttl_ns)
            .def_readwrite("eviction_threshold",
                           &nerve::cache::FeatureCache::CacheConfig::eviction_threshold)
            .def_readwrite("enable_lru", &nerve::cache::FeatureCache::CacheConfig::enable_lru)
            .def_readwrite("enable_shared_memory",
                           &nerve::cache::FeatureCache::CacheConfig::enable_shared_memory)
            .def_readwrite("shm_name", &nerve::cache::FeatureCache::CacheConfig::shm_name);

        py::class_<nerve::cache::FeatureCache::FeatureEntry>(cache, "FeatureEntry")
            .def(py::init<>())
            .def_readonly("timestamp_ns", &nerve::cache::FeatureCache::FeatureEntry::timestamp_ns)
            .def_readonly("symbol_id", &nerve::cache::FeatureCache::FeatureEntry::symbol_id)
            .def_readonly("features", &nerve::cache::FeatureCache::FeatureEntry::features)
            .def_readonly("access_count", &nerve::cache::FeatureCache::FeatureEntry::access_count)
            .def_readonly("is_valid", &nerve::cache::FeatureCache::FeatureEntry::isValid);

        py::class_<nerve::cache::FeatureCache::CacheStats>(cache, "CacheStats")
            .def(py::init<>())
            .def_readonly("total_entries", &nerve::cache::FeatureCache::CacheStats::total_entries)
            .def_readonly("memory_usage_bytes",
                          &nerve::cache::FeatureCache::CacheStats::memory_usage_bytes)
            .def_readonly("hit_rate", &nerve::cache::FeatureCache::CacheStats::hit_rate)
            .def_readonly("miss_rate", &nerve::cache::FeatureCache::CacheStats::miss_rate)
            .def_readonly("evictions", &nerve::cache::FeatureCache::CacheStats::evictions)
            .def_readonly("accesses", &nerve::cache::FeatureCache::CacheStats::accesses);

        py::class_<nerve::cache::FeatureCache>(cache, "FeatureCache")
            .def(py::init<const nerve::cache::FeatureCache::CacheConfig &>(), py::arg("config"))
            .def("put_feature", &nerve::cache::FeatureCache::putFeature, py::arg("symbol_id"),
                 py::arg("timestamp_ns"), py::arg("features"),
                 "Store a feature vector for a symbol at a timestamp")
            .def(
                "get_feature",
                [](nerve::cache::FeatureCache &self, int64_t symbol_id, int64_t timestamp_ns) {
                    std::vector<float> features;
                    bool found = self.getFeature(symbol_id, timestamp_ns, features);
                    return py::make_tuple(found, features);
                },
                py::arg("symbol_id"), py::arg("timestamp_ns"),
                "Retrieve a feature vector; returns (found, features)")
            .def(
                "get_latest_feature",
                [](nerve::cache::FeatureCache &self, int64_t symbol_id) {
                    std::vector<float> features;
                    int64_t timestamp_ns = 0;
                    bool found = self.getLatestFeature(symbol_id, features, timestamp_ns);
                    return py::make_tuple(found, features, timestamp_ns);
                },
                py::arg("symbol_id"),
                "Retrieve the latest feature for a symbol; returns (found, features, timestamp_ns)")
            .def("put_features_batch", &nerve::cache::FeatureCache::putFeaturesBatch,
                 py::arg("symbol_ids"), py::arg("timestamps"), py::arg("features"),
                 "Store multiple feature vectors")
            .def(
                "get_features_batch",
                [](nerve::cache::FeatureCache &self, const std::vector<int64_t> &symbol_ids,
                   const std::vector<int64_t> &timestamps) {
                    std::vector<std::vector<float>> features;
                    std::vector<bool> found;
                    self.getFeaturesBatch(symbol_ids, timestamps, features, found);
                    return py::make_tuple(found, features);
                },
                py::arg("symbol_ids"), py::arg("timestamps"), "Retrieve multiple feature vectors")
            .def("evict_expired", &nerve::cache::FeatureCache::evictExpired,
                 "Evict expired entries")
            .def("evict_lru", &nerve::cache::FeatureCache::evictLru, py::arg("target_size"),
                 "Evict LRU entries down to target_size")
            .def("clear_symbol", &nerve::cache::FeatureCache::clearSymbol, py::arg("symbol_id"),
                 "Clear all entries for a symbol")
            .def("clear_all", &nerve::cache::FeatureCache::clearAll, "Clear all cached entries")
            .def("get_stats", &nerve::cache::FeatureCache::getStats, "Get cache statistics")
            .def("reset_stats", &nerve::cache::FeatureCache::resetStats, "Reset cache statistics");

        py::class_<nerve::cache::RingBuffer>(cache, "RingBuffer")
            .def(py::init<size_t>(), py::arg("capacity"),
                 "Create a ring buffer with the given capacity")
            .def("push", &nerve::cache::RingBuffer::push, py::arg("entry"),
                 "Push an entry; returns False if full")
            .def("size", &nerve::cache::RingBuffer::size, "Current number of entries")
            .def("capacity", &nerve::cache::RingBuffer::capacity, "Maximum capacity")
            .def("empty", &nerve::cache::RingBuffer::empty, "Check if empty")
            .def("full", &nerve::cache::RingBuffer::full, "Check if full")
            .def("clear", &nerve::cache::RingBuffer::clear, "Clear all entries");

        py::class_<nerve::cache::CacheManager>(cache, "CacheManager")
            .def_static("instance", &nerve::cache::CacheManager::instance,
                        py::return_value_policy::reference, "Get the global CacheManager singleton")
            .def("register_cache", &nerve::cache::CacheManager::registerCache, py::arg("name"),
                 py::arg("cache"), "Register a named cache")
            .def("get_cache", &nerve::cache::CacheManager::getCache, py::arg("name"),
                 "Get a registered cache by name")
            .def("cleanup_all_caches", &nerve::cache::CacheManager::cleanupAllCaches,
                 "Clear all registered caches")
            .def("get_cache_names", &nerve::cache::CacheManager::getCacheNames,
                 "Get names of all registered caches");
    }

    // filtration/simd_filtration -- SIMD-accelerated filtration ops
    {
        auto filt = m.def_submodule("filtration", "SIMD-accelerated filtration operations");

        filt.def(
            "batch_filter_values",
            [](py::array_t<double, py::array::c_style | py::array::forcecast> values,
               nerve::Size start_dim, nerve::Size end_dim, double threshold) {
                py::buffer_info info = values.request();
                nerve::filtration::simdBatchFilterValues(static_cast<double *>(info.ptr),
                                                         static_cast<nerve::Size>(info.size),
                                                         start_dim, end_dim, threshold);
                return values;
            },
            py::arg("values"), py::arg("start_dim"), py::arg("end_dim"), py::arg("threshold"),
            "Filter values below threshold to zero (SIMD-accelerated, in-place)");

        filt.def(
            "sort_pairs_by_birth",
            [](py::array_t<double, py::array::c_style | py::array::forcecast> pairs) {
                py::buffer_info info = pairs.request();
                if (info.ndim != 2 || info.shape[1] != 2)
                {
                    throw std::invalid_argument(
                        "pairs must be a 2D array with shape (N, 2): [birth, death]");
                }
                auto n = static_cast<size_t>(info.shape[0]);
                auto *data = static_cast<double *>(info.ptr);
                // Build Pair objects from birth/death columns, call C++ sort, write back
                std::vector<nerve::Pair> vec;
                vec.reserve(n);
                for (size_t i = 0; i < n; ++i)
                    vec.emplace_back(data[i * 2], data[i * 2 + 1], 0, -1, -1);
                nerve::filtration::simdSortPairsByBirth(vec.data(), static_cast<nerve::Size>(n));
                for (size_t i = 0; i < n; ++i)
                {
                    data[i * 2] = vec[i].birth;
                    data[i * 2 + 1] = vec[i].death;
                }
                return pairs;
            },
            py::arg("pairs"),
            "Sort persistence pairs in-place by birth time (accepts Nx2 numpy array)");
    }

    // anomaly/topology_drift -- topological anomaly detection
    {
        auto anomaly = m.def_submodule("anomaly", "Topological anomaly and drift detection");

        py::class_<nerve::anomaly::PersistenceProfile>(anomaly, "PersistenceProfile")
            .def(py::init<>())
            .def_readwrite("num_bins", &nerve::anomaly::PersistenceProfile::num_bins)
            .def_readwrite("total_pairs", &nerve::anomaly::PersistenceProfile::total_pairs)
            .def_readwrite("birth_mean", &nerve::anomaly::PersistenceProfile::birth_mean)
            .def_readwrite("death_mean", &nerve::anomaly::PersistenceProfile::death_mean)
            .def_readwrite("persistence_mean",
                           &nerve::anomaly::PersistenceProfile::persistence_mean)
            .def_readwrite("birth_variance", &nerve::anomaly::PersistenceProfile::birth_variance)
            .def_readwrite("death_variance", &nerve::anomaly::PersistenceProfile::death_variance)
            .def_readwrite("persistence_variance",
                           &nerve::anomaly::PersistenceProfile::persistence_variance)
            .def_readwrite("birth_histogram", &nerve::anomaly::PersistenceProfile::birth_histogram)
            .def_readwrite("death_histogram", &nerve::anomaly::PersistenceProfile::death_histogram)
            .def_readwrite("persistence_histogram",
                           &nerve::anomaly::PersistenceProfile::persistence_histogram);

        // BettiChangeDetector
        py::class_<nerve::anomaly::BettiChangeDetector::ChangeConfig>(anomaly, "ChangeConfig")
            .def(py::init<>())
            .def_readwrite("significance_level",
                           &nerve::anomaly::BettiChangeDetector::ChangeConfig::significance_level)
            .def_readwrite("min_window_size",
                           &nerve::anomaly::BettiChangeDetector::ChangeConfig::min_window_size)
            .def_readwrite("max_window_size",
                           &nerve::anomaly::BettiChangeDetector::ChangeConfig::max_window_size)
            .def_readwrite(
                "enable_trend_detection",
                &nerve::anomaly::BettiChangeDetector::ChangeConfig::enable_trend_detection)
            .def_readwrite(
                "min_change_threshold",
                &nerve::anomaly::BettiChangeDetector::ChangeConfig::min_change_threshold);

        py::class_<nerve::anomaly::BettiChangeDetector::ChangePoint>(anomaly, "ChangePoint")
            .def(py::init<>())
            .def_readwrite("timestamp_ns",
                           &nerve::anomaly::BettiChangeDetector::ChangePoint::timestamp_ns)
            .def_readwrite("window_index",
                           &nerve::anomaly::BettiChangeDetector::ChangePoint::window_index)
            .def_readwrite("betti_before",
                           &nerve::anomaly::BettiChangeDetector::ChangePoint::betti_before)
            .def_readwrite("betti_after",
                           &nerve::anomaly::BettiChangeDetector::ChangePoint::betti_after)
            .def_readwrite("change_magnitude",
                           &nerve::anomaly::BettiChangeDetector::ChangePoint::change_magnitude)
            .def_readwrite("p_value", &nerve::anomaly::BettiChangeDetector::ChangePoint::p_value)
            .def_readwrite("change_type",
                           &nerve::anomaly::BettiChangeDetector::ChangePoint::change_type);

        py::class_<nerve::anomaly::BettiChangeDetector::TrendInfo>(anomaly, "TrendInfo")
            .def(py::init<>())
            .def_readwrite("is_trending",
                           &nerve::anomaly::BettiChangeDetector::TrendInfo::is_trending)
            .def_readwrite("trend_slope",
                           &nerve::anomaly::BettiChangeDetector::TrendInfo::trend_slope)
            .def_readwrite("trend_strength",
                           &nerve::anomaly::BettiChangeDetector::TrendInfo::trend_strength)
            .def_readwrite("trend_direction",
                           &nerve::anomaly::BettiChangeDetector::TrendInfo::trend_direction);

        py::class_<nerve::anomaly::BettiChangeDetector>(anomaly, "BettiChangeDetector")
            .def(py::init<const nerve::anomaly::BettiChangeDetector::ChangeConfig &>(),
                 py::arg("config"))
            .def("detect_changes", &nerve::anomaly::BettiChangeDetector::detectChanges,
                 py::arg("betti_sequences"), py::arg("timestamps"),
                 "Detect Betti number change points across sequences")
            .def("detect_single_change", &nerve::anomaly::BettiChangeDetector::detectSingleChange,
                 py::arg("betti_before"), py::arg("betti_after"), py::arg("timestamp_ns"),
                 "Detect a single Betti change between two windows")
            .def(
                "update_and_detect",
                [](nerve::anomaly::BettiChangeDetector &self, const std::vector<double> &new_betti,
                   int64_t timestamp_ns) {
                    nerve::anomaly::BettiChangeDetector::ChangePoint cp;
                    bool detected = self.updateAndDetect(new_betti, timestamp_ns, cp);
                    return py::make_tuple(detected, cp);
                },
                py::arg("new_betti"), py::arg("timestamp_ns"),
                "Update and detect; returns (detected, change_point)")
            .def("compute_statistical_distance",
                 &nerve::anomaly::BettiChangeDetector::computeStatisticalDistance, py::arg("seq1"),
                 py::arg("seq2"), "Compute statistical distance between two sequences")
            .def("compute_p_value", &nerve::anomaly::BettiChangeDetector::computePValue,
                 py::arg("seq1"), py::arg("seq2"), "Compute p-value for two sequences")
            .def("analyze_trend", &nerve::anomaly::BettiChangeDetector::analyzeTrend,
                 py::arg("betti_sequence"), "Analyze trend in a Betti sequence, returns TrendInfo");

        // LifetimeDriftDetector
        py::class_<nerve::anomaly::LifetimeDriftDetector::DriftConfig>(anomaly, "DriftConfig")
            .def(py::init<>())
            .def_readwrite("drift_threshold",
                           &nerve::anomaly::LifetimeDriftDetector::DriftConfig::drift_threshold)
            .def_readwrite(
                "reference_window_size",
                &nerve::anomaly::LifetimeDriftDetector::DriftConfig::reference_window_size)
            .def_readwrite(
                "detection_window_size",
                &nerve::anomaly::LifetimeDriftDetector::DriftConfig::detection_window_size)
            .def_readwrite(
                "enable_adaptive_threshold",
                &nerve::anomaly::LifetimeDriftDetector::DriftConfig::enable_adaptive_threshold)
            .def_readwrite("distance_metric",
                           &nerve::anomaly::LifetimeDriftDetector::DriftConfig::distance_metric)
            .def_readwrite("wasserstein_order",
                           &nerve::anomaly::LifetimeDriftDetector::DriftConfig::wasserstein_order);

        py::class_<nerve::anomaly::LifetimeDriftDetector::DriftPoint>(anomaly, "DriftPoint")
            .def(py::init<>())
            .def_readwrite("timestamp_ns",
                           &nerve::anomaly::LifetimeDriftDetector::DriftPoint::timestamp_ns)
            .def_readwrite("drift_score",
                           &nerve::anomaly::LifetimeDriftDetector::DriftPoint::drift_score)
            .def_readwrite("p_value", &nerve::anomaly::LifetimeDriftDetector::DriftPoint::p_value)
            .def_readwrite("reference_lifetimes",
                           &nerve::anomaly::LifetimeDriftDetector::DriftPoint::reference_lifetimes)
            .def_readwrite("current_lifetimes",
                           &nerve::anomaly::LifetimeDriftDetector::DriftPoint::current_lifetimes)
            .def_readwrite("distance_value",
                           &nerve::anomaly::LifetimeDriftDetector::DriftPoint::distance_value)
            .def_readwrite("is_drift_detected",
                           &nerve::anomaly::LifetimeDriftDetector::DriftPoint::is_drift_detected);

        py::class_<nerve::anomaly::LifetimeDriftDetector>(anomaly, "LifetimeDriftDetector")
            .def(py::init<const nerve::anomaly::LifetimeDriftDetector::DriftConfig &>(),
                 py::arg("config"))
            .def("detect_drift", &nerve::anomaly::LifetimeDriftDetector::detectDrift,
                 py::arg("lifetime_sequences"), py::arg("timestamps"),
                 "Detect drift across lifetime sequences")
            .def("detect_single_drift", &nerve::anomaly::LifetimeDriftDetector::detectSingleDrift,
                 py::arg("reference_lifetimes"), py::arg("current_lifetimes"),
                 py::arg("timestamp_ns"),
                 "Detect a single drift event between two lifetime distributions")
            .def(
                "update_and_detect",
                [](nerve::anomaly::LifetimeDriftDetector &self,
                   const std::vector<float> &new_lifetimes, int64_t timestamp_ns) {
                    nerve::anomaly::LifetimeDriftDetector::DriftPoint dp;
                    bool detected = self.updateAndDetect(new_lifetimes, timestamp_ns, dp);
                    return py::make_tuple(detected, dp);
                },
                py::arg("new_lifetimes"), py::arg("timestamp_ns"),
                "Update and detect drift; returns (detected, drift_point)")
            .def("compute_sliced_wasserstein_distance",
                 &nerve::anomaly::LifetimeDriftDetector::computeSlicedWassersteinDistance,
                 py::arg("dist1"), py::arg("dist2"), py::arg("num_projections") = 100,
                 "Compute sliced Wasserstein distance between two distributions")
            .def("compute_emd_distance", &nerve::anomaly::LifetimeDriftDetector::computeEmdDistance,
                 py::arg("dist1"), py::arg("dist2"),
                 "Compute Earth Mover's Distance between two distributions")
            .def("compute_kl_divergence",
                 &nerve::anomaly::LifetimeDriftDetector::computeKlDivergence, py::arg("dist1"),
                 py::arg("dist2"), "Compute KL divergence between two distributions")
            .def("update_reference_distribution",
                 &nerve::anomaly::LifetimeDriftDetector::updateReferenceDistribution,
                 py::arg("new_reference"), "Update the reference distribution")
            .def("reset_reference", &nerve::anomaly::LifetimeDriftDetector::resetReference,
                 "Reset the reference distribution");

        // MarketAnomalyDetector
        py::class_<nerve::anomaly::MarketAnomalyDetector::MarketConfig>(anomaly, "MarketConfig")
            .def(py::init<>())
            .def_readwrite(
                "price_change_threshold",
                &nerve::anomaly::MarketAnomalyDetector::MarketConfig::price_change_threshold)
            .def_readwrite(
                "volume_spike_threshold",
                &nerve::anomaly::MarketAnomalyDetector::MarketConfig::volume_spike_threshold)
            .def_readwrite(
                "topology_anomaly_threshold",
                &nerve::anomaly::MarketAnomalyDetector::MarketConfig::topology_anomaly_threshold)
            .def_readwrite("lookback_window",
                           &nerve::anomaly::MarketAnomalyDetector::MarketConfig::lookback_window)
            .def_readwrite(
                "enable_volume_analysis",
                &nerve::anomaly::MarketAnomalyDetector::MarketConfig::enable_volume_analysis)
            .def_readwrite(
                "enable_price_analysis",
                &nerve::anomaly::MarketAnomalyDetector::MarketConfig::enable_price_analysis)
            .def_readwrite(
                "enable_topology_analysis",
                &nerve::anomaly::MarketAnomalyDetector::MarketConfig::enable_topology_analysis);

        py::class_<nerve::anomaly::MarketAnomalyDetector::AnomalyEvent>(anomaly, "AnomalyEvent")
            .def(py::init<>())
            .def_readwrite("timestamp_ns",
                           &nerve::anomaly::MarketAnomalyDetector::AnomalyEvent::timestamp_ns)
            .def_readwrite("anomaly_type",
                           &nerve::anomaly::MarketAnomalyDetector::AnomalyEvent::anomaly_type)
            .def_readwrite("anomaly_score",
                           &nerve::anomaly::MarketAnomalyDetector::AnomalyEvent::anomaly_score)
            .def_readwrite("p_value", &nerve::anomaly::MarketAnomalyDetector::AnomalyEvent::p_value)
            .def_readwrite("description",
                           &nerve::anomaly::MarketAnomalyDetector::AnomalyEvent::description)
            .def_readwrite(
                "contributing_factors",
                &nerve::anomaly::MarketAnomalyDetector::AnomalyEvent::contributing_factors)
            .def_readwrite("is_critical",
                           &nerve::anomaly::MarketAnomalyDetector::AnomalyEvent::is_critical);

        py::class_<nerve::anomaly::MarketAnomalyDetector>(anomaly, "MarketAnomalyDetector")
            .def(py::init<const nerve::anomaly::MarketAnomalyDetector::MarketConfig &>(),
                 py::arg("config"))
            .def("detect_anomalies", &nerve::anomaly::MarketAnomalyDetector::detectAnomalies,
                 py::arg("timestamps"), py::arg("prices"), py::arg("volumes"),
                 py::arg("topological_features"), "Detect anomalies across market data sequences")
            .def("detect_single_anomaly",
                 &nerve::anomaly::MarketAnomalyDetector::detectSingleAnomaly,
                 py::arg("timestamp_ns"), py::arg("price"), py::arg("volume"),
                 py::arg("topological_features"), "Detect a single anomaly event")
            .def(
                "update_and_detect",
                [](nerve::anomaly::MarketAnomalyDetector &self, int64_t timestamp_ns, double price,
                   float volume, const std::vector<float> &topological_features) {
                    nerve::anomaly::MarketAnomalyDetector::AnomalyEvent ae;
                    bool detected =
                        self.updateAndDetect(timestamp_ns, price, volume, topological_features, ae);
                    return py::make_tuple(detected, ae);
                },
                py::arg("timestamp_ns"), py::arg("price"), py::arg("volume"),
                py::arg("topological_features"),
                "Update and detect anomaly; returns (detected, anomaly_event)")
            .def("detect_price_anomaly", &nerve::anomaly::MarketAnomalyDetector::detectPriceAnomaly,
                 py::arg("current_price"), py::arg("price_history"), "Detect a price anomaly")
            .def("detect_volume_anomaly",
                 &nerve::anomaly::MarketAnomalyDetector::detectVolumeAnomaly,
                 py::arg("current_volume"), py::arg("volume_history"), "Detect a volume anomaly")
            .def("detect_topology_anomaly",
                 &nerve::anomaly::MarketAnomalyDetector::detectTopologyAnomaly,
                 py::arg("current_features"), py::arg("feature_history"),
                 "Detect a topological anomaly")
            .def("detect_combined_anomaly",
                 &nerve::anomaly::MarketAnomalyDetector::detectCombinedAnomaly,
                 py::arg("factor_scores"), py::arg("factor_names"),
                 "Detect a combined anomaly from multiple factors")
            .def("update_normal_behavior",
                 &nerve::anomaly::MarketAnomalyDetector::updateNormalBehavior, py::arg("prices"),
                 py::arg("volumes"), py::arg("topological_features"),
                 "Update the baseline normal behavior model")
            .def("reset_normal_behavior",
                 &nerve::anomaly::MarketAnomalyDetector::resetNormalBehavior,
                 "Reset the baseline normal behavior model");

        // OnlinePValueCalculator
        py::class_<nerve::anomaly::OnlinePValueCalculator::PValueConfig>(anomaly, "PValueConfig")
            .def(py::init<>())
            .def_readwrite(
                "significance_level",
                &nerve::anomaly::OnlinePValueCalculator::PValueConfig::significance_level)
            .def_readwrite("min_samples",
                           &nerve::anomaly::OnlinePValueCalculator::PValueConfig::min_samples)
            .def_readwrite(
                "enable_fdr_control",
                &nerve::anomaly::OnlinePValueCalculator::PValueConfig::enable_fdr_control)
            .def_readwrite("fdr_rate",
                           &nerve::anomaly::OnlinePValueCalculator::PValueConfig::fdr_rate);

        py::class_<nerve::anomaly::OnlinePValueCalculator>(anomaly, "OnlinePValueCalculator")
            .def(py::init<const nerve::anomaly::OnlinePValueCalculator::PValueConfig &>(),
                 py::arg("config"))
            .def("compute_p_value", &nerve::anomaly::OnlinePValueCalculator::computePValue,
                 py::arg("test_statistic"), py::arg("null_distribution"),
                 "Compute a p-value against the null distribution")
            .def("compute_empirical_p_value",
                 &nerve::anomaly::OnlinePValueCalculator::computeEmpiricalPValue,
                 py::arg("test_statistic"), py::arg("sample_distribution"),
                 "Compute an empirical p-value from a sample distribution")
            .def("bonferroni_correction",
                 &nerve::anomaly::OnlinePValueCalculator::bonferroniCorrection, py::arg("p_values"),
                 "Apply Bonferroni correction to multiple p-values")
            .def("benjamini_hochberg_fdr",
                 &nerve::anomaly::OnlinePValueCalculator::benjaminiHochbergFdr, py::arg("p_values"),
                 "Apply Benjamini-Hochberg FDR correction")
            .def("update_null_distribution",
                 &nerve::anomaly::OnlinePValueCalculator::updateNullDistribution,
                 py::arg("new_sample"), "Update the null distribution with a new sample")
            .def("update_sample_distribution",
                 &nerve::anomaly::OnlinePValueCalculator::updateSampleDistribution,
                 py::arg("new_sample"), "Update the sample distribution with a new sample")
            .def("is_significant", &nerve::anomaly::OnlinePValueCalculator::isSignificant,
                 py::arg("p_value"), "Check if a p-value is significant")
            .def("multiple_testing_significance",
                 &nerve::anomaly::OnlinePValueCalculator::multipleTestingSignificance,
                 py::arg("p_values"), "Test multiple p-values for significance")
            .def("compute_effect_size", &nerve::anomaly::OnlinePValueCalculator::computeEffectSize,
                 py::arg("sample_mean"), py::arg("null_mean"), py::arg("sample_std"),
                 py::arg("null_std"), "Compute effect size between sample and null")
            .def("compute_confidence_interval",
                 &nerve::anomaly::OnlinePValueCalculator::computeConfidenceInterval,
                 py::arg("samples"), py::arg("confidence_level") = 0.95,
                 "Compute a confidence interval for a set of samples");

        // RegimeChangeDetector
        py::class_<nerve::anomaly::RegimeChangeDetector::RegimeConfig>(anomaly, "RegimeConfig")
            .def(py::init<>())
            .def_readwrite(
                "regime_change_threshold",
                &nerve::anomaly::RegimeChangeDetector::RegimeConfig::regime_change_threshold)
            .def_readwrite("min_regime_duration",
                           &nerve::anomaly::RegimeChangeDetector::RegimeConfig::min_regime_duration)
            .def_readwrite(
                "enable_hmm_detection",
                &nerve::anomaly::RegimeChangeDetector::RegimeConfig::enable_hmm_detection)
            .def_readwrite("num_regimes",
                           &nerve::anomaly::RegimeChangeDetector::RegimeConfig::num_regimes)
            .def_readwrite(
                "transition_probability",
                &nerve::anomaly::RegimeChangeDetector::RegimeConfig::transition_probability);

        py::class_<nerve::anomaly::RegimeChangeDetector::Regime>(anomaly, "Regime")
            .def(py::init<>())
            .def_readwrite("regime_id", &nerve::anomaly::RegimeChangeDetector::Regime::regime_id)
            .def_readwrite("characteristic_features",
                           &nerve::anomaly::RegimeChangeDetector::Regime::characteristic_features)
            .def_readwrite("stability_score",
                           &nerve::anomaly::RegimeChangeDetector::Regime::stability_score)
            .def_readwrite("start_timestamp_ns",
                           &nerve::anomaly::RegimeChangeDetector::Regime::start_timestamp_ns)
            .def_readwrite("end_timestamp_ns",
                           &nerve::anomaly::RegimeChangeDetector::Regime::end_timestamp_ns)
            .def_readwrite("duration_points",
                           &nerve::anomaly::RegimeChangeDetector::Regime::duration_points)
            .def_readwrite("description",
                           &nerve::anomaly::RegimeChangeDetector::Regime::description);

        py::class_<nerve::anomaly::RegimeChangeDetector::RegimeChange>(anomaly, "RegimeChange")
            .def(py::init<>())
            .def_readwrite("timestamp_ns",
                           &nerve::anomaly::RegimeChangeDetector::RegimeChange::timestamp_ns)
            .def_readwrite("from_regime_id",
                           &nerve::anomaly::RegimeChangeDetector::RegimeChange::from_regime_id)
            .def_readwrite("to_regime_id",
                           &nerve::anomaly::RegimeChangeDetector::RegimeChange::to_regime_id)
            .def_readwrite("change_confidence",
                           &nerve::anomaly::RegimeChangeDetector::RegimeChange::change_confidence)
            .def_readwrite("transition_features",
                           &nerve::anomaly::RegimeChangeDetector::RegimeChange::transition_features)
            .def_readwrite("change_description",
                           &nerve::anomaly::RegimeChangeDetector::RegimeChange::change_description);

        py::class_<nerve::anomaly::RegimeChangeDetector::HMMModel>(anomaly, "HMMModel")
            .def(py::init<>())
            .def_readwrite("emission_means",
                           &nerve::anomaly::RegimeChangeDetector::HMMModel::emission_means)
            .def_readwrite("emission_covariances",
                           &nerve::anomaly::RegimeChangeDetector::HMMModel::emission_covariances)
            .def_readwrite("transition_matrix",
                           &nerve::anomaly::RegimeChangeDetector::HMMModel::transition_matrix)
            .def_readwrite("initial_probabilities",
                           &nerve::anomaly::RegimeChangeDetector::HMMModel::initial_probabilities);

        py::class_<nerve::anomaly::RegimeChangeDetector>(anomaly, "RegimeChangeDetector")
            .def(py::init<const nerve::anomaly::RegimeChangeDetector::RegimeConfig &>(),
                 py::arg("config"))
            .def("detect_regimes", &nerve::anomaly::RegimeChangeDetector::detectRegimes,
                 py::arg("topological_features"), py::arg("timestamps"),
                 "Detect distinct regimes in topological feature sequences")
            .def("detect_regime_changes",
                 &nerve::anomaly::RegimeChangeDetector::detectRegimeChanges,
                 py::arg("topological_features"), py::arg("timestamps"),
                 "Detect regime change points")
            .def(
                "update_and_detect",
                [](nerve::anomaly::RegimeChangeDetector &self,
                   const std::vector<float> &new_features, int64_t timestamp_ns) {
                    nerve::anomaly::RegimeChangeDetector::RegimeChange rc;
                    bool detected = self.updateAndDetect(new_features, timestamp_ns, rc);
                    return py::make_tuple(detected, rc);
                },
                py::arg("new_features"), py::arg("timestamp_ns"),
                "Update and detect regime change; returns (detected, regime_change)")
            .def("extract_regime_features",
                 &nerve::anomaly::RegimeChangeDetector::extractRegimeFeatures,
                 py::arg("feature_window"),
                 "Extract characteristic features from a window of topological features")
            .def("characterize_regime", &nerve::anomaly::RegimeChangeDetector::characterizeRegime,
                 py::arg("features"), "Characterize a regime based on its features")
            .def("train_hmm", &nerve::anomaly::RegimeChangeDetector::trainHmm, py::arg("features"),
                 py::arg("regime_labels"), "Train a Hidden Markov Model on labeled regime data")
            .def("predict_regimes_hmm", &nerve::anomaly::RegimeChangeDetector::predictRegimesHmm,
                 py::arg("model"), py::arg("features"),
                 "Predict regimes using a trained HMM model");

        // AnomalyDetectionManager (singleton)
        py::class_<nerve::anomaly::AnomalyDetectionManager::AnomalyReport>(anomaly, "AnomalyReport")
            .def(py::init<>())
            .def_readwrite("betti_changes",
                           &nerve::anomaly::AnomalyDetectionManager::AnomalyReport::betti_changes)
            .def_readwrite("drift_points",
                           &nerve::anomaly::AnomalyDetectionManager::AnomalyReport::drift_points)
            .def_readwrite(
                "market_anomalies",
                &nerve::anomaly::AnomalyDetectionManager::AnomalyReport::market_anomalies)
            .def_readwrite("regime_changes",
                           &nerve::anomaly::AnomalyDetectionManager::AnomalyReport::regime_changes)
            .def_readwrite(
                "overall_anomaly_score",
                &nerve::anomaly::AnomalyDetectionManager::AnomalyReport::overall_anomaly_score)
            .def_readwrite("summary_report",
                           &nerve::anomaly::AnomalyDetectionManager::AnomalyReport::summary_report);

        py::class_<nerve::anomaly::AnomalyDetectionManager>(anomaly, "AnomalyDetectionManager")
            .def_static("instance", &nerve::anomaly::AnomalyDetectionManager::instance,
                        py::return_value_policy::reference,
                        "Get the global AnomalyDetectionManager singleton")
            .def("set_betti_detector_config",
                 &nerve::anomaly::AnomalyDetectionManager::setBettiDetectorConfig,
                 py::arg("config"), "Set the BettiChangeDetector configuration")
            .def("set_drift_detector_config",
                 &nerve::anomaly::AnomalyDetectionManager::setDriftDetectorConfig,
                 py::arg("config"), "Set the LifetimeDriftDetector configuration")
            .def("set_market_detector_config",
                 &nerve::anomaly::AnomalyDetectionManager::setMarketDetectorConfig,
                 py::arg("config"), "Set the MarketAnomalyDetector configuration")
            .def("set_pvalue_config", &nerve::anomaly::AnomalyDetectionManager::setPvalueConfig,
                 py::arg("config"), "Set the OnlinePValueCalculator configuration")
            .def("set_regime_config", &nerve::anomaly::AnomalyDetectionManager::setRegimeConfig,
                 py::arg("config"), "Set the RegimeChangeDetector configuration")
            .def("get_betti_detector", &nerve::anomaly::AnomalyDetectionManager::getBettiDetector,
                 "Get the BettiChangeDetector instance")
            .def("get_drift_detector", &nerve::anomaly::AnomalyDetectionManager::getDriftDetector,
                 "Get the LifetimeDriftDetector instance")
            .def("get_market_detector", &nerve::anomaly::AnomalyDetectionManager::getMarketDetector,
                 "Get the MarketAnomalyDetector instance")
            .def("get_pvalue_calculator",
                 &nerve::anomaly::AnomalyDetectionManager::getPvalueCalculator,
                 "Get the OnlinePValueCalculator instance")
            .def("get_regime_detector", &nerve::anomaly::AnomalyDetectionManager::getRegimeDetector,
                 "Get the RegimeChangeDetector instance")
            .def("detect_all_anomalies",
                 &nerve::anomaly::AnomalyDetectionManager::detectAllAnomalies,
                 py::arg("timestamps"), py::arg("prices"), py::arg("volumes"),
                 py::arg("betti_sequences"), py::arg("lifetime_sequences"),
                 py::arg("topological_features"),
                 "Run all detectors and return a combined AnomalyReport")
            .def("generate_alerts", &nerve::anomaly::AnomalyDetectionManager::generateAlerts,
                 py::arg("report"), "Generate alert strings from an AnomalyReport")
            .def("send_alerts", &nerve::anomaly::AnomalyDetectionManager::sendAlerts,
                 py::arg("alerts"), "Send alert strings");
    }

    // error/error_handling -- circuit breaker, retry, observability
    {
        auto err =
            m.def_submodule("error", "Error handling: circuit breaker, retry, observability");

        py::enum_<nerve::error::ErrorCode>(err, "ErrorCode")
            .value("SUCCESS", nerve::error::ErrorCode::SUCCESS)
            .value("IO_TIMEOUT", nerve::error::ErrorCode::IO_TIMEOUT)
            .value("IO_READ_ERROR", nerve::error::ErrorCode::IO_READ_ERROR)
            .value("IO_WRITE_ERROR", nerve::error::ErrorCode::IO_WRITE_ERROR)
            .value("GPU_OOM", nerve::error::ErrorCode::GPU_OOM)
            .value("GPU_TIMEOUT", nerve::error::ErrorCode::GPU_TIMEOUT)
            .value("GPU_KERNEL_ERROR", nerve::error::ErrorCode::GPU_KERNEL_ERROR)
            .value("NUM_NAN", nerve::error::ErrorCode::NUM_NAN)
            .value("NUM_INF", nerve::error::ErrorCode::NUM_INF)
            .value("NUM_CONVERGENCE_FAILURE", nerve::error::ErrorCode::NUM_CONVERGENCE_FAILURE)
            .value("DET_MISMATCH", nerve::error::ErrorCode::DET_MISMATCH)
            .value("DET_SEED_MISMATCH", nerve::error::ErrorCode::DET_SEED_MISMATCH)
            .value("DET_THREAD_ORDER_MISMATCH", nerve::error::ErrorCode::DET_THREAD_ORDER_MISMATCH)
            .value("CPU_OVERLOAD", nerve::error::ErrorCode::CPU_OVERLOAD)
            .value("CPU_QUEUE_SATURATION", nerve::error::ErrorCode::CPU_QUEUE_SATURATION)
            .value("CPU_AFFINITY_ERROR", nerve::error::ErrorCode::CPU_AFFINITY_ERROR)
            .value("PH_ABORT", nerve::error::ErrorCode::PH_ABORT)
            .value("PH_TIME_BUDGET_EXCEEDED", nerve::error::ErrorCode::PH_TIME_BUDGET_EXCEEDED)
            .value("PH_WINDOW_TOO_LARGE", nerve::error::ErrorCode::PH_WINDOW_TOO_LARGE)
            .export_values();

        py::class_<nerve::error::CallContract>(err, "CallContract")
            .def(py::init<>())
            .def_readwrite("time_budget_ms", &nerve::error::CallContract::time_budget_ms)
            .def_readwrite("strict_time_enforcement",
                           &nerve::error::CallContract::strict_time_enforcement)
            .def_readwrite("operation_name", &nerve::error::CallContract::operation_name)
            .def_readwrite("params_hash", &nerve::error::CallContract::params_hash)
            .def_readwrite("window_start_ns", &nerve::error::CallContract::window_start_ns)
            .def_readwrite("window_end_ns", &nerve::error::CallContract::window_end_ns);

        py::class_<nerve::error::CircuitBreakerConfig>(err, "CircuitBreakerConfig")
            .def(py::init<>())
            .def_readwrite("max_consecutive_failures",
                           &nerve::error::CircuitBreakerConfig::max_consecutive_failures)
            .def_readwrite("cooldown_ms", &nerve::error::CircuitBreakerConfig::cooldown_ms)
            .def_readwrite("enable_automatic_recovery",
                           &nerve::error::CircuitBreakerConfig::enable_automatic_recovery)
            .def_readwrite("recovery_sample_ratio",
                           &nerve::error::CircuitBreakerConfig::recovery_sample_ratio);

        py::class_<nerve::error::ErrorEvent>(err, "ErrorEvent")
            .def(py::init<>())
            .def_readwrite("error_code", &nerve::error::ErrorEvent::error_code)
            .def_readwrite("operation_name", &nerve::error::ErrorEvent::operation_name)
            .def_readwrite("params_hash", &nerve::error::ErrorEvent::params_hash)
            .def_readwrite("window_start_ns", &nerve::error::ErrorEvent::window_start_ns)
            .def_readwrite("window_end_ns", &nerve::error::ErrorEvent::window_end_ns)
            .def_readwrite("duration_ms", &nerve::error::ErrorEvent::duration_ms)
            .def_readwrite("error_message", &nerve::error::ErrorEvent::error_message)
            .def_readwrite("recovery_reason", &nerve::error::ErrorEvent::recovery_reason)
            .def_readwrite("metadata", &nerve::error::ErrorEvent::metadata)
            .def("to_json", &nerve::error::ErrorEvent::toJson, "Serialize to JSON")
            .def("to_structured_log", &nerve::error::ErrorEvent::toStructuredLog,
                 "Serialize to structured log format");

        py::enum_<nerve::error::CircuitBreaker::State>(err, "CircuitBreakerState")
            .value("CLOSED", nerve::error::CircuitBreaker::State::CLOSED)
            .value("OPEN", nerve::error::CircuitBreaker::State::OPEN)
            .value("HALF_OPEN", nerve::error::CircuitBreaker::State::HALF_OPEN)
            .export_values();

        py::class_<nerve::error::CircuitBreaker::CircuitBreakerStats>(err, "CircuitBreakerStats")
            .def(py::init<>())
            .def_readwrite("total_operations",
                           &nerve::error::CircuitBreaker::CircuitBreakerStats::total_operations)
            .def_readwrite(
                "successful_operations",
                &nerve::error::CircuitBreaker::CircuitBreakerStats::successful_operations)
            .def_readwrite("failed_operations",
                           &nerve::error::CircuitBreaker::CircuitBreakerStats::failed_operations)
            .def_readwrite("consecutive_failures",
                           &nerve::error::CircuitBreaker::CircuitBreakerStats::consecutive_failures)
            .def_readwrite("current_state",
                           &nerve::error::CircuitBreaker::CircuitBreakerStats::current_state)
            .def_readwrite("failing_operations",
                           &nerve::error::CircuitBreaker::CircuitBreakerStats::failing_operations);

        py::class_<nerve::error::CircuitBreaker>(err, "CircuitBreaker")
            .def(py::init<const nerve::error::CircuitBreakerConfig &>(), py::arg("config"))
            .def("should_allow_operation", &nerve::error::CircuitBreaker::shouldAllowOperation,
                 py::arg("operation_name"),
                 "Check if an operation should be allowed through the circuit")
            .def("record_success", &nerve::error::CircuitBreaker::recordSuccess,
                 py::arg("operation_name"), "Record a successful operation")
            .def("record_failure", &nerve::error::CircuitBreaker::recordFailure,
                 py::arg("operation_name"), py::arg("error_code"), "Record a failed operation")
            .def("get_state", &nerve::error::CircuitBreaker::getState,
                 "Get the current circuit breaker state")
            .def("is_tripped", &nerve::error::CircuitBreaker::isTripped,
                 "Check if the circuit is open (tripped)")
            .def("reset", &nerve::error::CircuitBreaker::reset,
                 "Reset the circuit breaker to closed state")
            .def("force_open", &nerve::error::CircuitBreaker::forceOpen,
                 "Force the circuit breaker into open state")
            .def("force_close", &nerve::error::CircuitBreaker::forceClose,
                 "Force the circuit breaker into closed state")
            .def("get_stats", &nerve::error::CircuitBreaker::getStats,
                 "Get circuit breaker statistics")
            .def("reset_stats", &nerve::error::CircuitBreaker::resetStats, "Reset all statistics");

        py::class_<nerve::error::RetryBackoffManager::RetryConfig>(err, "RetryConfig")
            .def(py::init<>())
            .def_readwrite("max_retries",
                           &nerve::error::RetryBackoffManager::RetryConfig::max_retries)
            .def_readwrite("backoff_ms",
                           &nerve::error::RetryBackoffManager::RetryConfig::backoff_ms)
            .def_readwrite(
                "enable_exponential_backoff",
                &nerve::error::RetryBackoffManager::RetryConfig::enable_exponential_backoff)
            .def_readwrite("enable_jitter",
                           &nerve::error::RetryBackoffManager::RetryConfig::enable_jitter)
            .def_readwrite("jitter_factor",
                           &nerve::error::RetryBackoffManager::RetryConfig::jitter_factor)
            .def_readwrite("transient_errors",
                           &nerve::error::RetryBackoffManager::RetryConfig::transient_errors);

        py::class_<nerve::error::RetryBackoffManager::RetryStats>(err, "RetryStats")
            .def(py::init<>())
            .def_readwrite("total_retries",
                           &nerve::error::RetryBackoffManager::RetryStats::total_retries)
            .def_readwrite("successful_retries",
                           &nerve::error::RetryBackoffManager::RetryStats::successful_retries)
            .def_readwrite("failed_retries",
                           &nerve::error::RetryBackoffManager::RetryStats::failed_retries)
            .def_readwrite("error_counts",
                           &nerve::error::RetryBackoffManager::RetryStats::error_counts)
            .def_readwrite("operation_retry_counts",
                           &nerve::error::RetryBackoffManager::RetryStats::operation_retry_counts)
            .def_readwrite("average_backoff_ms",
                           &nerve::error::RetryBackoffManager::RetryStats::average_backoff_ms);

        py::class_<nerve::error::RetryBackoffManager>(err, "RetryBackoffManager")
            .def(py::init<const nerve::error::RetryBackoffManager::RetryConfig &>(),
                 py::arg("config"))
            .def("calculate_backoff", &nerve::error::RetryBackoffManager::calculateBackoff,
                 py::arg("retry_attempt"), py::arg("operation_name"),
                 "Calculate the backoff time for a retry attempt")
            .def("is_transient_error", &nerve::error::RetryBackoffManager::isTransientError,
                 py::arg("error_code"), "Check if an error code is considered transient")
            .def("is_permanent_error", &nerve::error::RetryBackoffManager::isPermanentError,
                 py::arg("error_code"), "Check if an error code is considered permanent")
            .def("get_stats", &nerve::error::RetryBackoffManager::getStats, "Get retry statistics")
            .def("reset_stats", &nerve::error::RetryBackoffManager::resetStats,
                 "Reset all statistics");

        py::class_<nerve::error::ErrorObservability::ObservabilityConfig>(err,
                                                                          "ObservabilityConfig")
            .def(py::init<>())
            .def_readwrite(
                "enable_structured_logging",
                &nerve::error::ErrorObservability::ObservabilityConfig::enable_structured_logging)
            .def_readwrite(
                "enable_metric_increment",
                &nerve::error::ErrorObservability::ObservabilityConfig::enable_metric_increment)
            .def_readwrite("log_format",
                           &nerve::error::ErrorObservability::ObservabilityConfig::log_format)
            .def_readwrite("metric_prefix",
                           &nerve::error::ErrorObservability::ObservabilityConfig::metric_prefix)
            .def_readwrite("enable_performance_correlation",
                           &nerve::error::ErrorObservability::ObservabilityConfig::
                               enable_performance_correlation);

        py::class_<nerve::error::ErrorObservability::ObservabilityStats>(err, "ObservabilityStats")
            .def(py::init<>())
            .def_readwrite(
                "total_error_events",
                &nerve::error::ErrorObservability::ObservabilityStats::total_error_events)
            .def_readwrite("error_code_counts",
                           &nerve::error::ErrorObservability::ObservabilityStats::error_code_counts)
            .def_readwrite(
                "operation_error_counts",
                &nerve::error::ErrorObservability::ObservabilityStats::operation_error_counts)
            .def_readwrite(
                "duration_histograms",
                &nerve::error::ErrorObservability::ObservabilityStats::duration_histograms)
            .def_readwrite(
                "performance_correlations",
                &nerve::error::ErrorObservability::ObservabilityStats::performance_correlations);

        py::class_<nerve::error::ErrorObservability>(err, "ErrorObservability")
            .def(py::init<const nerve::error::ErrorObservability::ObservabilityConfig &>(),
                 py::arg("config"))
            .def("log_error_event",
                 py::overload_cast<const nerve::error::ErrorEvent &>(
                     &nerve::error::ErrorObservability::logErrorEvent),
                 py::arg("event"), "Log an error event")
            .def("log_error_event_simple",
                 py::overload_cast<const std::string &, nerve::error::ErrorCode,
                                   const nerve::error::CallContract &, const std::string &>(
                     &nerve::error::ErrorObservability::logErrorEvent),
                 py::arg("operation_name"), py::arg("error_code"), py::arg("contract"),
                 py::arg("error_message") = "", "Log an error event from components")
            .def("increment_metric", &nerve::error::ErrorObservability::incrementMetric,
                 py::arg("metric_name"),
                 py::arg("labels") = std::unordered_map<std::string, std::string>(),
                 "Increment a named metric")
            .def("update_error_histogram", &nerve::error::ErrorObservability::updateErrorHistogram,
                 py::arg("operation_name"), py::arg("error_code"), py::arg("duration_ms"),
                 "Update the error histogram for an operation")
            .def("correlate_performance_with_errors",
                 &nerve::error::ErrorObservability::correlatePerformanceWithErrors,
                 py::arg("operation_name"), py::arg("performance_metric"), py::arg("error_code"),
                 "Correlate a performance metric with errors")
            .def("export_error_report", &nerve::error::ErrorObservability::exportErrorReport,
                 py::arg("time_range") = "1h", "Export an error report as a JSON string")
            .def("export_metrics", &nerve::error::ErrorObservability::exportMetrics,
                 py::arg("format") = "prometheus",
                 "Export metrics in the given format (e.g. 'prometheus')")
            .def("get_stats", &nerve::error::ErrorObservability::getStats,
                 "Get observability statistics")
            .def("reset_stats", &nerve::error::ErrorObservability::resetStats,
                 "Reset all statistics");

        py::class_<nerve::error::ErrorHandlingManager::ErrorHandlingStats>(err,
                                                                           "ErrorHandlingStats")
            .def(py::init<>())
            .def_readwrite(
                "circuit_breaker_stats",
                &nerve::error::ErrorHandlingManager::ErrorHandlingStats::circuit_breaker_stats)
            .def_readwrite("retry_stats",
                           &nerve::error::ErrorHandlingManager::ErrorHandlingStats::retry_stats)
            .def_readwrite(
                "observability_stats",
                &nerve::error::ErrorHandlingManager::ErrorHandlingStats::observability_stats);

        py::class_<nerve::error::ErrorHandlingManager>(err, "ErrorHandlingManager")
            .def_static("instance", &nerve::error::ErrorHandlingManager::instance,
                        py::return_value_policy::reference,
                        "Get the global ErrorHandlingManager singleton")
            .def("set_circuit_breaker_config",
                 &nerve::error::ErrorHandlingManager::setCircuitBreakerConfig, py::arg("config"),
                 "Set the circuit breaker configuration")
            .def("set_retry_config", &nerve::error::ErrorHandlingManager::setRetryConfig,
                 py::arg("config"), "Set the retry configuration")
            .def("set_observability_config",
                 &nerve::error::ErrorHandlingManager::setObservabilityConfig, py::arg("config"),
                 "Set the observability configuration")
            .def("get_circuit_breaker", &nerve::error::ErrorHandlingManager::getCircuitBreaker,
                 "Get the CircuitBreaker instance")
            .def("get_retry_manager", &nerve::error::ErrorHandlingManager::getRetryManager,
                 "Get the RetryBackoffManager instance")
            .def("get_observability", &nerve::error::ErrorHandlingManager::getObservability,
                 "Get the ErrorObservability instance")
            .def("log_error", &nerve::error::ErrorHandlingManager::logError,
                 py::arg("operation_name"), py::arg("error_code"), py::arg("contract"),
                 py::arg("error_message") = "", "Log an error through the observability system")
            .def("get_stats", &nerve::error::ErrorHandlingManager::getStats,
                 "Get combined error handling statistics")
            .def("reset_stats", &nerve::error::ErrorHandlingManager::resetStats,
                 "Reset all error handling statistics")
            .def("is_healthy", &nerve::error::ErrorHandlingManager::isHealthy,
                 "Check if the error handling system is healthy")
            .def("get_health_issues", &nerve::error::ErrorHandlingManager::getHealthIssues,
                 "Get a list of health issue descriptions");
    }

    // memory/safe_memory_pool -- low-level memory management
    {
        auto mem =
            m.def_submodule("memory", "Low-level memory management: pools, allocators, tracking");

        py::enum_<nerve::memory::SizeClass>(mem, "SizeClass")
            .value("TINY16", nerve::memory::SizeClass::Tiny16)
            .value("TINY32", nerve::memory::SizeClass::Tiny32)
            .value("TINY64", nerve::memory::SizeClass::Tiny64)
            .value("TINY128", nerve::memory::SizeClass::Tiny128)
            .value("TINY256", nerve::memory::SizeClass::Tiny256)
            .value("SMALL512", nerve::memory::SizeClass::Small512)
            .value("SMALL1024", nerve::memory::SizeClass::Small1024)
            .value("SMALL2048", nerve::memory::SizeClass::Small2048)
            .value("SMALL4096", nerve::memory::SizeClass::Small4096)
            .value("MEDIUM8192", nerve::memory::SizeClass::Medium8192)
            .value("MEDIUM16384", nerve::memory::SizeClass::Medium16384)
            .value("MEDIUM32768", nerve::memory::SizeClass::Medium32768)
            .value("LARGE65536", nerve::memory::SizeClass::Large65536)
            .value("LARGE131072", nerve::memory::SizeClass::Large131072)
            .value("HUGE262144", nerve::memory::SizeClass::Huge262144)
            .export_values();

        // GlobalPagePool (singleton)
        py::class_<nerve::memory::GlobalPagePool,
                   std::unique_ptr<nerve::memory::GlobalPagePool, py::nodelete>>(mem,
                                                                                 "GlobalPagePool")
            .def_static("instance", &nerve::memory::GlobalPagePool::instance,
                        py::return_value_policy::reference, "Get the global page pool singleton")
            .def("allocate_page", &nerve::memory::GlobalPagePool::allocatePage,
                 "Allocate a huge page from the OS")
            .def("deallocate_page", &nerve::memory::GlobalPagePool::deallocatePage, py::arg("page"),
                 "Deallocate a page back to the OS")
            .def("pages_allocated", &nerve::memory::GlobalPagePool::pagesAllocated,
                 "Number of pages currently allocated")
            .def("hugetlb_pages_allocated", &nerve::memory::GlobalPagePool::hugetlbPagesAllocated,
                 "Number of huge TLB pages allocated")
            .def_property_readonly("page_size", &nerve::memory::GlobalPagePool::pageSize,
                                   "Size of each page in bytes");

        // RawArrayPool
        py::class_<nerve::memory::RawArrayPool>(mem, "RawArrayPool")
            .def(py::init<nerve::Size, bool>(), py::arg("initial_bytes") = 64 * 1024 * 1024,
                 py::arg("use_hugepages") = false,
                 "Create a RawArrayPool with the given initial size")
            .def("allocate", &nerve::memory::RawArrayPool::allocate, py::arg("bytes"),
                 "Allocate memory from the pool")
            .def("deallocate", &nerve::memory::RawArrayPool::deallocate, py::arg("ptr"),
                 py::arg("bytes"), "Deallocate memory back to the pool")
            .def_property_readonly("total_allocated", &nerve::memory::RawArrayPool::totalAllocated,
                                   "Total bytes allocated")
            .def_property_readonly("peak_utilization",
                                   &nerve::memory::RawArrayPool::peakUtilization, "Peak bytes used")
            .def("reset", &nerve::memory::RawArrayPool::reset,
                 "Reset the pool, freeing all allocations");

        // NumaAwareAllocator
        py::class_<nerve::memory::NumaAwareAllocator>(mem, "NumaAwareAllocator")
            .def(py::init<int>(), py::arg("preferred_node") = -1,
                 "Create a NUMA-aware allocator, optionally pinned to a node")
            .def("allocate", &nerve::memory::NumaAwareAllocator::allocate, py::arg("bytes"),
                 py::arg("alignment") = 64,
                 "Allocate aligned memory, preferring the configured NUMA node")
            .def("deallocate", &nerve::memory::NumaAwareAllocator::deallocate, py::arg("ptr"),
                 py::arg("bytes"), "Deallocate memory allocated by this allocator")
            .def_property("preferred_node", &nerve::memory::NumaAwareAllocator::getPreferredNode,
                          &nerve::memory::NumaAwareAllocator::setPreferredNode,
                          "Get or set the preferred NUMA node");

        // SizeClassAllocator
        py::class_<nerve::memory::SizeClassAllocator>(mem, "SizeClassAllocator")
            .def(py::init<>(), "Create a size-class allocator")
            .def("allocate", &nerve::memory::SizeClassAllocator::allocate, py::arg("bytes"),
                 "Allocate memory from the best-fit size class")
            .def("deallocate", &nerve::memory::SizeClassAllocator::deallocate, py::arg("ptr"),
                 py::arg("bytes"), "Deallocate memory back to its size class pool")
            .def("total_allocated", &nerve::memory::SizeClassAllocator::totalAllocated,
                 "Total bytes allocated across all size classes")
            .def("get_size_class", &nerve::memory::SizeClassAllocator::getSizeClass,
                 py::arg("bytes"), "Get the size class for a given byte count");

        // Global tracking free functions
        mem.def("get_global_allocation_count", &nerve::memory::getGlobalAllocationCount,
                "Total number of tracked allocations");
        mem.def("get_global_deallocation_count", &nerve::memory::getGlobalDeallocationCount,
                "Total number of tracked deallocations");
        mem.def("get_global_current_bytes", &nerve::memory::getGlobalCurrentBytes,
                "Currently allocated bytes");
        mem.def("get_global_peak_bytes", &nerve::memory::getGlobalPeakBytes,
                "Peak allocated bytes");
        mem.def("track_alloc_event", &nerve::memory::trackAllocEvent, py::arg("bytes"),
                "Record an allocation event");
        mem.def("track_free_event", &nerve::memory::trackFreeEvent, py::arg("bytes"),
                "Record a deallocation event");
        mem.def("reset_global_memory_stats", &nerve::memory::resetGlobalMemoryStats,
                "Reset all global memory tracking statistics");
        mem.def("get_slab_allocator_diagnostic_count",
                &nerve::memory::getSlabAllocatorDiagnosticCount,
                "Get slab allocator diagnostic count");
        mem.def("estimate_memory_overhead", &nerve::memory::estimateMemoryOverhead,
                py::arg("object_count"), py::arg("object_size"), py::arg("slab_capacity") = 256,
                "Estimate memory overhead for a given number of objects");
    }

    // spectral/laplacian -- spectral Laplacian and Dirac operators
    {
        auto spec = m.def_submodule("spectral",
                                    "Spectral analysis: Laplacian, Dirac, persistent Laplacian");

        py::class_<nerve::spectral::LaplacianConfig>(spec, "LaplacianConfig")
            .def(py::init<>())
            .def_readwrite("enable_gpu", &nerve::spectral::LaplacianConfig::enable_gpu)
            .def_readwrite("threshold", &nerve::spectral::LaplacianConfig::threshold)
            .def_readwrite("prefer_tiled_kernels",
                           &nerve::spectral::LaplacianConfig::prefer_tiled_kernels)
            .def_readwrite("max_gpu_memory_mb",
                           &nerve::spectral::LaplacianConfig::max_gpu_memory_mb);

        py::class_<nerve::spectral::SpectralConfig>(spec, "SpectralConfig")
            .def(py::init<>())
            .def_readwrite("num_eigenpairs", &nerve::spectral::SpectralConfig::num_eigenpairs)
            .def_readwrite("convergence_tolerance",
                           &nerve::spectral::SpectralConfig::convergence_tolerance)
            .def_readwrite("max_iterations", &nerve::spectral::SpectralConfig::max_iterations)
            .def_readwrite("spectral_shift", &nerve::spectral::SpectralConfig::spectral_shift);

        // Laplacian
        py::class_<nerve::spectral::Laplacian>(spec, "Laplacian")
            .def(py::init<>(), "Create an empty Laplacian")

            .def("size", &nerve::spectral::Laplacian::size,
                 "Return the size of the Laplacian matrix")
            .def("max_dimension", &nerve::spectral::Laplacian::maxDimension,
                 "Return the maximum simplex dimension")
            .def("get_laplacian", &nerve::spectral::Laplacian::getLaplacian, py::arg("dimension"),
                 "Return the full Laplacian matrix for a given dimension")
            .def("get_up_laplacian", &nerve::spectral::Laplacian::getUpLaplacian,
                 py::arg("dimension"), "Return the up Laplacian for a given dimension")
            .def("get_down_laplacian", &nerve::spectral::Laplacian::getDownLaplacian,
                 py::arg("dimension"), "Return the down Laplacian for a given dimension")
            .def("get_hodge_laplacian", &nerve::spectral::Laplacian::getHodgeLaplacian,
                 py::arg("dimension"), "Return the Hodge Laplacian for a given dimension")
            .def("eigenvalues", &nerve::spectral::Laplacian::eigenvalues, py::arg("dimension"),
                 py::arg("k") = 0, "Compute eigenvalues for a given dimension")
            .def("eigenvectors", &nerve::spectral::Laplacian::eigenvectors, py::arg("dimension"),
                 py::arg("k") = 0, "Compute eigenvectors for a given dimension")
            .def("spectrum", &nerve::spectral::Laplacian::spectrum, py::arg("dimension"),
                 "Return the full spectrum for a given dimension")
            .def("compute_embedding", &nerve::spectral::Laplacian::computeEmbedding,
                 py::arg("dimension"), py::arg("target_dim") = 2,
                 "Compute a spectral embedding for a given dimension")
            .def("compute_diffusion_map", &nerve::spectral::Laplacian::computeDiffusionMap,
                 py::arg("target_dim") = 2, "Compute a diffusion map embedding")
            .def("heat_kernel", &nerve::spectral::Laplacian::heatKernel, py::arg("dimension"),
                 py::arg("t"), "Compute the heat kernel for a given dimension and time")
            .def("heat_flow", &nerve::spectral::Laplacian::heatFlow, py::arg("initial"),
                 py::arg("dimension"), py::arg("t"), "Compute the heat flow of an initial state")
            .def("compute_spectral_gap", &nerve::spectral::Laplacian::computeSpectralGap,
                 py::arg("dimension"), "Compute the spectral gap for a given dimension")
            .def("compute_cheeger_constants", &nerve::spectral::Laplacian::computeCheegerConstants,
                 "Compute Cheeger constants for all dimensions")
            .def("compute_morse_index", &nerve::spectral::Laplacian::computeMorseIndex,
                 py::arg("dimension"), "Compute the Morse index for a given dimension");

        // DiracOperator
        py::class_<nerve::spectral::DiracOperator>(spec, "DiracOperator")
            .def(py::init<>(), "Create an empty Dirac operator")
            .def("get_dirac", &nerve::spectral::DiracOperator::getDirac, "Return the Dirac matrix")
            .def("get_dirac_squared", &nerve::spectral::DiracOperator::getDiracSquared,
                 "Return the squared Dirac matrix")
            .def("eigenvalues", &nerve::spectral::DiracOperator::eigenvalues, py::arg("k") = 0,
                 "Compute eigenvalues of the Dirac operator")
            .def("eigenvectors", &nerve::spectral::DiracOperator::eigenvectors, py::arg("k") = 0,
                 "Compute eigenvectors of the Dirac operator")
            .def("get_spinor_laplacian", &nerve::spectral::DiracOperator::getSpinorLaplacian,
                 "Return the spinor Laplacian")
            .def("get_chirality_operator", &nerve::spectral::DiracOperator::getChiralityOperator,
                 "Return the chirality operator")
            .def("compute_atiyah_singer_index",
                 &nerve::spectral::DiracOperator::computeAtiyahSingerIndex,
                 "Compute the Atiyah-Singer index")
            .def("compute_analytical_index",
                 &nerve::spectral::DiracOperator::computeAnalyticalIndex,
                 "Compute the analytical index")
            .def("compute_topological_index",
                 &nerve::spectral::DiracOperator::computeTopologicalIndex,
                 "Compute the topological index");
    }

    // sheaf/gpu_sheaf -- sheaf theory operations
    {
        auto sheaf = m.def_submodule(
            "sheaf", "Sheaf theory: engine, hardware detection, parallel/morphism ops");

        // SheafConfig
        py::class_<nerve::sheaf::SheafConfig>(sheaf, "SheafConfig")
            .def(py::init<>())
            .def_readwrite("num_stalks", &nerve::sheaf::SheafConfig::num_stalks)
            .def_readwrite("stalk_dimension", &nerve::sheaf::SheafConfig::stalk_dimension)
            .def_readwrite("use_parallel", &nerve::sheaf::SheafConfig::use_parallel)
            .def_readwrite("use_simd", &nerve::sheaf::SheafConfig::use_simd)
            .def_readwrite("num_threads", &nerve::sheaf::SheafConfig::num_threads)
            .def_readwrite("gpu_batch_size", &nerve::sheaf::SheafConfig::gpu_batch_size)
            .def_readwrite("use_memory_pool", &nerve::sheaf::SheafConfig::use_memory_pool)
            .def_readwrite("memory_pool_size", &nerve::sheaf::SheafConfig::memory_pool_size);

        // SheafResult
        py::class_<nerve::sheaf::SheafResult>(sheaf, "SheafResult")
            .def(py::init<>())
            .def_readwrite("cohomology", &nerve::sheaf::SheafResult::cohomology)
            .def_readwrite("computation_time_ms", &nerve::sheaf::SheafResult::computation_time_ms)
            .def_readwrite("success", &nerve::sheaf::SheafResult::success);

        // SheafHardwareInfo
        py::class_<nerve::sheaf::SheafHardwareInfo>(sheaf, "SheafHardwareInfo")
            .def(py::init<>())
            .def_readwrite("has_gpu", &nerve::sheaf::SheafHardwareInfo::has_gpu)
            .def_readwrite("has_avx512", &nerve::sheaf::SheafHardwareInfo::has_avx512)
            .def_readwrite("has_avx2", &nerve::sheaf::SheafHardwareInfo::has_avx2)
            .def_readwrite("num_cores", &nerve::sheaf::SheafHardwareInfo::num_cores)
            .def("__repr__", [](const nerve::sheaf::SheafHardwareInfo &info) {
                return "SheafHardwareInfo(gpu=" + std::to_string(info.has_gpu) +
                       ", avx512=" + std::to_string(info.has_avx512) +
                       ", avx2=" + std::to_string(info.has_avx2) +
                       ", cores=" + std::to_string(info.num_cores) + ")";
            });

        sheaf.def("detect_sheaf_hardware", &nerve::sheaf::detectSheafHardware,
                  "Detect available hardware acceleration for sheaf operations");

        // Point struct
        py::class_<nerve::sheaf::Point>(sheaf, "Point")
            .def(py::init<>())
            .def_readwrite("x", &nerve::sheaf::Point::x)
            .def_readwrite("y", &nerve::sheaf::Point::y)
            .def_readwrite("z", &nerve::sheaf::Point::z)
            .def("__repr__", [](const nerve::sheaf::Point &p) {
                return "Point(" + std::to_string(p.x) + ", " + std::to_string(p.y) + ", " +
                       std::to_string(p.z) + ")";
            });

        // SheafEngine
        py::class_<nerve::sheaf::SheafEngine>(sheaf, "SheafEngine")
            .def(py::init<const nerve::sheaf::SheafConfig &>(),
                 py::arg("config") = nerve::sheaf::SheafConfig(),
                 "Create a SheafEngine with the given configuration")
            .def("build_sheaf", &nerve::sheaf::SheafEngine::buildSheaf, py::arg("stalk_positions"),
                 py::arg("stalk_dimensions"), "Build a sheaf from stalk positions and dimensions")
            .def("compute_cohomology", &nerve::sheaf::SheafEngine::computeCohomology,
                 py::arg("cocycle"), "Compute sheaf cohomology from a cocycle")
            .def(
                "apply_morphism",
                [](nerve::sheaf::SheafEngine &self, int from_stalk, int to_stalk,
                   const std::vector<float> &input) {
                    std::vector<float> output;
                    self.applyMorphism(from_stalk, to_stalk, input, output);
                    return output;
                },
                py::arg("from_stalk"), py::arg("to_stalk"), py::arg("input"),
                "Apply a morphism from one stalk to another, returns output");

        // Parallel sub-module
        {
            auto para =
                sheaf.def_submodule("parallel", "Parallel sheaf construction and SIMD stalk ops");

            py::class_<nerve::sheaf::parallel::StalkData>(para, "StalkData")
                .def(py::init<int, int>(), py::arg("stalk_id") = 0, py::arg("stalk_dim") = 0,
                     "Create stalk data with given ID and dimension")
                .def_readwrite("id", &nerve::sheaf::parallel::StalkData::id)
                .def_readwrite("dimension", &nerve::sheaf::parallel::StalkData::dimension)
                .def_readwrite("data", &nerve::sheaf::parallel::StalkData::data);

            py::class_<nerve::sheaf::parallel::ParallelSheafBuilder::SheafConfig>(
                para, "ParallelSheafConfig")
                .def(py::init<>())
                .def_readwrite(
                    "num_stalks",
                    &nerve::sheaf::parallel::ParallelSheafBuilder::SheafConfig::num_stalks)
                .def_readwrite(
                    "stalk_dimension",
                    &nerve::sheaf::parallel::ParallelSheafBuilder::SheafConfig::stalk_dimension)
                .def_readwrite("use_simd",
                               &nerve::sheaf::parallel::ParallelSheafBuilder::SheafConfig::use_simd)
                .def_readwrite(
                    "num_threads",
                    &nerve::sheaf::parallel::ParallelSheafBuilder::SheafConfig::num_threads);

            py::class_<nerve::sheaf::parallel::ParallelSheafBuilder>(para, "ParallelSheafBuilder")
                .def(py::init<const nerve::sheaf::parallel::ParallelSheafBuilder::SheafConfig &>(),
                     py::arg("config"), "Create a parallel sheaf builder")
                .def("build", &nerve::sheaf::parallel::ParallelSheafBuilder::build,
                     "Build the sheaf in parallel")
                .def("get_stalks", &nerve::sheaf::parallel::ParallelSheafBuilder::getStalks,
                     "Get the constructed stalk data");

            py::class_<nerve::sheaf::parallel::SIMDStalkOperations>(para, "SIMDStalkOperations")
                .def_static("add_stalks", &nerve::sheaf::parallel::SIMDStalkOperations::addStalks,
                            py::arg("a"), py::arg("b"), py::arg("result"),
                            "Add two stalks element-wise")
                .def_static("scale_stalk", &nerve::sheaf::parallel::SIMDStalkOperations::scaleStalk,
                            py::arg("stalk"), py::arg("scalar"), py::arg("result"),
                            "Scale a stalk by a scalar")
                .def_static("dot_product", &nerve::sheaf::parallel::SIMDStalkOperations::dotProduct,
                            py::arg("a"), py::arg("b"), "Compute dot product of two stalks")
                .def_static("normalize_stalk",
                            &nerve::sheaf::parallel::SIMDStalkOperations::normalizeStalk,
                            py::arg("stalk"), "Normalize a stalk to unit length");

            py::class_<nerve::sheaf::parallel::StalkSpatialHash>(para, "StalkSpatialHash")
                .def(py::init<float>(), py::arg("cell_size") = 1.0f,
                     "Create a spatial hash for stalk positions")
                .def("insert_stalk", &nerve::sheaf::parallel::StalkSpatialHash::insertStalk,
                     py::arg("stalk_id"), py::arg("position"),
                     "Insert a stalk position into the hash")
                .def("get_nearby_stalks",
                     &nerve::sheaf::parallel::StalkSpatialHash::getNearbyStalks,
                     py::arg("position"), "Get stalk IDs near a position");

            py::class_<nerve::sheaf::parallel::SheafParallelBenchmark>(para,
                                                                       "SheafParallelBenchmark")
                .def(py::init<>())
                .def_readwrite("sequential_time_ms",
                               &nerve::sheaf::parallel::SheafParallelBenchmark::sequential_time_ms)
                .def_readwrite("parallel_time_ms",
                               &nerve::sheaf::parallel::SheafParallelBenchmark::parallel_time_ms)
                .def_readwrite("simd_time_ms",
                               &nerve::sheaf::parallel::SheafParallelBenchmark::simd_time_ms)
                .def_readwrite("speedup_parallel",
                               &nerve::sheaf::parallel::SheafParallelBenchmark::speedup_parallel)
                .def_readwrite("speedup_simd",
                               &nerve::sheaf::parallel::SheafParallelBenchmark::speedup_simd)
                .def_readwrite("num_stalks",
                               &nerve::sheaf::parallel::SheafParallelBenchmark::num_stalks)
                .def_readwrite("stalk_dim",
                               &nerve::sheaf::parallel::SheafParallelBenchmark::stalk_dim)
                .def_readwrite("num_threads",
                               &nerve::sheaf::parallel::SheafParallelBenchmark::num_threads);

            para.def("benchmark_parallel_sheaf", &nerve::sheaf::parallel::benchmarkParallelSheaf,
                     py::arg("num_stalks"), py::arg("stalk_dim"), py::arg("num_threads"),
                     "Benchmark parallel vs sequential sheaf construction");
        }

        // GPU sub-module
        {
            auto gpu = sheaf.def_submodule("gpu", "GPU-accelerated sheaf operations");

            py::class_<nerve::sheaf::gpu::SheafGPUBenchmark>(gpu, "SheafGPUBenchmark")
                .def(py::init<>())
                .def_readwrite("cpu_time_ms", &nerve::sheaf::gpu::SheafGPUBenchmark::cpu_time_ms)
                .def_readwrite("gpu_time_ms", &nerve::sheaf::gpu::SheafGPUBenchmark::gpu_time_ms)
                .def_readwrite("speedup", &nerve::sheaf::gpu::SheafGPUBenchmark::speedup)
                .def_readwrite("num_stalks", &nerve::sheaf::gpu::SheafGPUBenchmark::num_stalks)
                .def_readwrite("stalk_dim", &nerve::sheaf::gpu::SheafGPUBenchmark::stalk_dim);
        }

        // Morphism sub-module
        {
            auto morph = sheaf.def_submodule("morphism", "Morphism operations and optimization");

            py::class_<nerve::sheaf::morphism::SparseMorphism>(morph, "SparseMorphism")
                .def(py::init<>())
                .def_readwrite("from_dim", &nerve::sheaf::morphism::SparseMorphism::from_dim)
                .def_readwrite("to_dim", &nerve::sheaf::morphism::SparseMorphism::to_dim)
                .def_readwrite("row_ptr", &nerve::sheaf::morphism::SparseMorphism::row_ptr)
                .def_readwrite("col_idx", &nerve::sheaf::morphism::SparseMorphism::col_idx)
                .def_readwrite("values", &nerve::sheaf::morphism::SparseMorphism::values)
                .def(
                    "apply",
                    [](const nerve::sheaf::morphism::SparseMorphism &self,
                       const std::vector<float> &input) {
                        std::vector<float> output;
                        self.apply(input, output);
                        return output;
                    },
                    py::arg("input"), "Apply the morphism to an input vector")
                .def(
                    "apply_simd",
                    [](const nerve::sheaf::morphism::SparseMorphism &self,
                       const std::vector<float> &input) {
                        std::vector<float> output(input.size());
                        self.applySIMD(input.data(), output.data());
                        return output;
                    },
                    py::arg("input"), "Apply the morphism using SIMD instructions");

            py::class_<nerve::sheaf::morphism::MorphismMemoryPool>(morph, "MorphismMemoryPool")
                .def(py::init<size_t>(), py::arg("initial_size") = 1024ULL * 1024ULL,
                     "Create a morphism memory pool")
                .def("allocate_morphism",
                     &nerve::sheaf::morphism::MorphismMemoryPool::allocateMorphism, py::arg("nnz"),
                     "Allocate memory for a morphism with the given number of non-zeros")
                .def("reset", &nerve::sheaf::morphism::MorphismMemoryPool::reset,
                     "Reset the pool, freeing all allocations");

            py::class_<nerve::sheaf::morphism::BatchedMorphismComputer>(morph,
                                                                        "BatchedMorphismComputer")
                .def(py::init<int>(), py::arg("cache_block_size") = 256,
                     "Create a batched morphism computer")
                .def("add_morphism", &nerve::sheaf::morphism::BatchedMorphismComputer::addMorphism,
                     py::arg("from_stalk"), py::arg("to_stalk"), py::arg("morphism"),
                     "Register a morphism between two stalks")
                .def(
                    "compute_batch",
                    [](nerve::sheaf::morphism::BatchedMorphismComputer &self,
                       const std::vector<int> &stalk_order,
                       const std::vector<std::vector<float>> &stalk_data) {
                        std::vector<std::vector<float>> output_data;
                        self.computeBatch(stalk_order, stalk_data, output_data);
                        return output_data;
                    },
                    py::arg("stalk_order"), py::arg("stalk_data"),
                    "Compute a batch of morphism applications");

            py::class_<nerve::sheaf::morphism::MorphismCompositionOptimizer>(
                morph, "MorphismCompositionOptimizer")
                .def(py::init<>(), "Create a morphism composition optimizer")
                .def("register_chain",
                     &nerve::sheaf::morphism::MorphismCompositionOptimizer::registerChain,
                     py::arg("stalk_chain"), "Register a stalk chain for composition optimization")
                .def("add_morphism",
                     &nerve::sheaf::morphism::MorphismCompositionOptimizer::addMorphism,
                     py::arg("from"), py::arg("to"), py::arg("morphism"),
                     "Add a morphism to the optimizer")
                .def("get_composed",
                     &nerve::sheaf::morphism::MorphismCompositionOptimizer::getComposed,
                     py::arg("from"), py::arg("to"),
                     "Get the composed morphism between two stalks");

            py::class_<nerve::sheaf::morphism::AsyncMorphismQueue>(morph, "AsyncMorphismQueue")
                .def(py::init<>(), "Create an async morphism queue")
                .def("start", &nerve::sheaf::morphism::AsyncMorphismQueue::start,
                     py::arg("num_workers") = 2, "Start worker threads")
                .def("stop", &nerve::sheaf::morphism::AsyncMorphismQueue::stop,
                     "Stop worker threads");
        }
    }

    // compression -- header-only compression primitives
    {
        auto comp = m.def_submodule("compression", "Compression: PCA, quantize, config");

        py::class_<nerve::compression::CompressionConfig>(comp, "CompressionConfig")
            .def(py::init<>())
            .def_readwrite("compression_method",
                           &nerve::compression::CompressionConfig::compression_method)
            .def_readwrite("pca_components", &nerve::compression::CompressionConfig::pca_components)
            .def_readwrite("pca_variance_retained",
                           &nerve::compression::CompressionConfig::pca_variance_retained)
            .def_readwrite("enable_gpu_acceleration",
                           &nerve::compression::CompressionConfig::enable_gpu_acceleration)
            .def_readwrite("target_compression_ratio",
                           &nerve::compression::CompressionConfig::target_compression_ratio)
            .def_readwrite("quality_threshold",
                           &nerve::compression::CompressionConfig::quality_threshold);

        py::class_<nerve::compression::CompressionResult>(comp, "CompressionResult")
            .def(py::init<>())
            .def_readwrite("original_size", &nerve::compression::CompressionResult::original_size)
            .def_readwrite("compressed_size",
                           &nerve::compression::CompressionResult::compressed_size)
            .def_readwrite("compression_ratio",
                           &nerve::compression::CompressionResult::compression_ratio)
            .def_readwrite("quality_score", &nerve::compression::CompressionResult::quality_score)
            .def_readwrite("compression_time_ms",
                           &nerve::compression::CompressionResult::compression_time_ms)
            .def_readwrite("compressed_data",
                           &nerve::compression::CompressionResult::compressed_data)
            .def("__repr__", [](const nerve::compression::CompressionResult &r) {
                return "CompressionResult(ratio=" + std::to_string(r.compression_ratio) +
                       ", quality=" + std::to_string(r.quality_score) +
                       ", time=" + std::to_string(r.compression_time_ms) + "ms)";
            });

        py::class_<nerve::compression::PCACompression>(comp, "PCACompression")
            .def(py::init<const nerve::compression::CompressionConfig &>(),
                 py::arg("config") = nerve::compression::CompressionConfig(),
                 "Create a PCACompression with the given configuration")
            .def("train", &nerve::compression::PCACompression::train, py::arg("data"),
                 "Train the PCA model on a data set")
            .def("compress", &nerve::compression::PCACompression::compress, py::arg("data"),
                 "Compress a data vector; returns CompressionResult")
            .def("decompress", &nerve::compression::PCACompression::decompress, py::arg("data"),
                 "Decompress a data vector")
            .def("compute_quality_score", &nerve::compression::PCACompression::computeQualityScore,
                 py::arg("original"), py::arg("decompressed"),
                 "Compute a quality score comparing original and decompressed data");
    }

    // algorithms -- distance, knn, persistence vectorization
    {
        auto algo =
            m.def_submodule("algorithms", "Distance metrics, KNN, persistence vectorization");

        // C extern wrapper: pairwise_distances_f32
        algo.def(
            "pairwise_distances_f32",
            [](py::array_t<float, py::array::c_style | py::array::forcecast> points) {
                py::buffer_info info = points.request();
                if (info.ndim != 2)
                    throw std::invalid_argument("points must be a 2D array (n, dim)");
                size_t n = static_cast<size_t>(info.shape[0]);
                size_t dim = static_cast<size_t>(info.shape[1]);
                std::vector<float> output(n * n, 0.0f);
                nerve_status_t status = nerve_pairwise_distances_f32_status(
                    static_cast<const float *>(info.ptr), n, dim, output.data());
                if (status != NERVE_STATUS_SUCCESS)
                    throw std::runtime_error("pairwise_distances_f32 failed");
                py::array_t<float> result(
                    {static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(n)});
                std::memcpy(result.mutable_data(), output.data(), n * n * sizeof(float));
                return result;
            },
            py::arg("points"), "Compute pairwise Euclidean distance matrix (float32)");

        // C extern wrapper: pairwise_distances_f64
        algo.def(
            "pairwise_distances_f64",
            [](py::array_t<double, py::array::c_style | py::array::forcecast> points) {
                py::buffer_info info = points.request();
                if (info.ndim != 2)
                    throw std::invalid_argument("points must be a 2D array (n, dim)");
                size_t n = static_cast<size_t>(info.shape[0]);
                size_t dim = static_cast<size_t>(info.shape[1]);
                std::vector<double> output(n * n, 0.0);
                nerve_status_t status = nerve_pairwise_distances_f64_status(
                    static_cast<const double *>(info.ptr), n, dim, output.data());
                if (status != NERVE_STATUS_SUCCESS)
                    throw std::runtime_error("pairwise_distances_f64 failed");
                py::array_t<double> result(
                    {static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(n)});
                std::memcpy(result.mutable_data(), output.data(), n * n * sizeof(double));
                return result;
            },
            py::arg("points"), "Compute pairwise Euclidean distance matrix (float64)");

        // C extern wrapper: knn_f32
        algo.def(
            "knn_f32",
            [](py::array_t<float, py::array::c_style | py::array::forcecast> points, size_t k) {
                py::buffer_info info = points.request();
                if (info.ndim != 2)
                    throw std::invalid_argument("points must be a 2D array (n, dim)");
                size_t n = static_cast<size_t>(info.shape[0]);
                size_t dim = static_cast<size_t>(info.shape[1]);
                std::vector<float> distances(n * k, 0.0f);
                std::vector<size_t> indices(n * k, 0);
                nerve_status_t status =
                    nerve_knn_f32_status(static_cast<const float *>(info.ptr), n, dim, k,
                                         distances.data(), indices.data());
                if (status != NERVE_STATUS_SUCCESS)
                    throw std::runtime_error("knn_f32 failed");
                py::array_t<float> dist_arr(
                    {static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(k)});
                py::array_t<size_t> idx_arr(
                    {static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(k)});
                std::memcpy(dist_arr.mutable_data(), distances.data(), n * k * sizeof(float));
                std::memcpy(idx_arr.mutable_data(), indices.data(), n * k * sizeof(size_t));
                return py::make_tuple(dist_arr, idx_arr);
            },
            py::arg("points"), py::arg("k"),
            "Compute k-nearest neighbors (float32). Returns (distances, indices)");

        // C extern wrapper: knn_f64
        algo.def(
            "knn_f64",
            [](py::array_t<double, py::array::c_style | py::array::forcecast> points, size_t k) {
                py::buffer_info info = points.request();
                if (info.ndim != 2)
                    throw std::invalid_argument("points must be a 2D array (n, dim)");
                size_t n = static_cast<size_t>(info.shape[0]);
                size_t dim = static_cast<size_t>(info.shape[1]);
                std::vector<double> distances(n * k, 0.0);
                std::vector<size_t> indices(n * k, 0);
                nerve_status_t status =
                    nerve_knn_f64_status(static_cast<const double *>(info.ptr), n, dim, k,
                                         distances.data(), indices.data());
                if (status != NERVE_STATUS_SUCCESS)
                    throw std::runtime_error("knn_f64 failed");
                py::array_t<double> dist_arr(
                    {static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(k)});
                py::array_t<size_t> idx_arr(
                    {static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(k)});
                std::memcpy(dist_arr.mutable_data(), distances.data(), n * k * sizeof(double));
                std::memcpy(idx_arr.mutable_data(), indices.data(), n * k * sizeof(size_t));
                return py::make_tuple(dist_arr, idx_arr);
            },
            py::arg("points"), py::arg("k"),
            "Compute k-nearest neighbors (float64). Returns (distances, indices)");

        // Persistence vectorization structs
        py::class_<nerve::algorithms::PersistenceLandscape>(algo, "PersistenceLandscape")
            .def(py::init<>())
            .def_readwrite("landscape_levels",
                           &nerve::algorithms::PersistenceLandscape::landscape_levels)
            .def_readwrite("x_min", &nerve::algorithms::PersistenceLandscape::x_min)
            .def_readwrite("x_max", &nerve::algorithms::PersistenceLandscape::x_max)
            .def_readwrite("num_levels", &nerve::algorithms::PersistenceLandscape::num_levels)
            .def("__repr__", [](const nerve::algorithms::PersistenceLandscape &l) {
                return "PersistenceLandscape(levels=" + std::to_string(l.num_levels) + ", x=[" +
                       std::to_string(l.x_min) + ", " + std::to_string(l.x_max) + "])";
            });

        py::class_<nerve::algorithms::PersistenceImage>(algo, "PersistenceImage")
            .def(py::init<>())
            .def_readwrite("image", &nerve::algorithms::PersistenceImage::image)
            .def_readwrite("resolution", &nerve::algorithms::PersistenceImage::resolution)
            .def_readwrite("sigma", &nerve::algorithms::PersistenceImage::sigma)
            .def_readwrite("birth_min", &nerve::algorithms::PersistenceImage::birth_min)
            .def_readwrite("birth_max", &nerve::algorithms::PersistenceImage::birth_max)
            .def_readwrite("persistence_min", &nerve::algorithms::PersistenceImage::persistence_min)
            .def_readwrite("persistence_max", &nerve::algorithms::PersistenceImage::persistence_max)
            .def("__repr__", [](const nerve::algorithms::PersistenceImage &img) {
                return "PersistenceImage(res=" + std::to_string(img.resolution) +
                       ", sigma=" + std::to_string(img.sigma) + ")";
            });

        // Persistence vectorization functions (explicitly instantiated for float/double)
        algo.def(
            "compute_landscape_f32",
            [](py::array_t<float, py::array::c_style | py::array::forcecast> diagram,
               int num_levels, double resolution) {
                py::buffer_info info = diagram.request();
                if (info.ndim != 2 || info.shape[1] != 2)
                    throw std::invalid_argument("diagram must be shape (N, 2)");
                size_t n = static_cast<size_t>(info.shape[0]);
                std::span<const float> span(static_cast<const float *>(info.ptr), n * 2);
                return nerve::algorithms::compute_landscape<float>(span, n, num_levels, resolution);
            },
            py::arg("diagram"), py::arg("num_levels") = 5, py::arg("resolution") = 0.01,
            "Compute persistence landscape from a float32 diagram (Nx2 array)");

        algo.def(
            "compute_landscape_f64",
            [](py::array_t<double, py::array::c_style | py::array::forcecast> diagram,
               int num_levels, double resolution) {
                py::buffer_info info = diagram.request();
                if (info.ndim != 2 || info.shape[1] != 2)
                    throw std::invalid_argument("diagram must be shape (N, 2)");
                size_t n = static_cast<size_t>(info.shape[0]);
                std::span<const double> span(static_cast<const double *>(info.ptr), n * 2);
                return nerve::algorithms::compute_landscape<double>(span, n, num_levels,
                                                                    resolution);
            },
            py::arg("diagram"), py::arg("num_levels") = 5, py::arg("resolution") = 0.01,
            "Compute persistence landscape from a float64 diagram (Nx2 array)");

        algo.def(
            "compute_persistence_image_f32",
            [](py::array_t<float, py::array::c_style | py::array::forcecast> diagram,
               int resolution, double sigma) {
                py::buffer_info info = diagram.request();
                if (info.ndim != 2 || info.shape[1] != 2)
                    throw std::invalid_argument("diagram must be shape (N, 2)");
                size_t n = static_cast<size_t>(info.shape[0]);
                std::span<const float> span(static_cast<const float *>(info.ptr), n * 2);
                return nerve::algorithms::compute_persistence_image<float>(span, n, resolution,
                                                                           sigma);
            },
            py::arg("diagram"), py::arg("resolution") = 64, py::arg("sigma") = 0.1,
            "Compute persistence image from a float32 diagram (Nx2 array)");

        algo.def(
            "compute_persistence_image_f64",
            [](py::array_t<double, py::array::c_style | py::array::forcecast> diagram,
               int resolution, double sigma) {
                py::buffer_info info = diagram.request();
                if (info.ndim != 2 || info.shape[1] != 2)
                    throw std::invalid_argument("diagram must be shape (N, 2)");
                size_t n = static_cast<size_t>(info.shape[0]);
                std::span<const double> span(static_cast<const double *>(info.ptr), n * 2);
                return nerve::algorithms::compute_persistence_image<double>(span, n, resolution,
                                                                            sigma);
            },
            py::arg("diagram"), py::arg("resolution") = 64, py::arg("sigma") = 0.1,
            "Compute persistence image from a float64 diagram (Nx2 array)");

        algo.def(
            "compute_betti_curve_f32",
            [](py::array_t<float, py::array::c_style | py::array::forcecast> diagram, int max_dim) {
                py::buffer_info info = diagram.request();
                if (info.ndim != 2 || info.shape[1] != 2)
                    throw std::invalid_argument("diagram must be shape (N, 2)");
                size_t n = static_cast<size_t>(info.shape[0]);
                std::span<const float> span(static_cast<const float *>(info.ptr), n * 2);
                return nerve::algorithms::compute_betti_curve<float>(span, n, max_dim);
            },
            py::arg("diagram"), py::arg("max_dim") = -1,
            "Compute Betti curve from a float32 diagram (Nx2 array)");

        algo.def(
            "compute_betti_curve_f64",
            [](py::array_t<double, py::array::c_style | py::array::forcecast> diagram,
               int max_dim) {
                py::buffer_info info = diagram.request();
                if (info.ndim != 2 || info.shape[1] != 2)
                    throw std::invalid_argument("diagram must be shape (N, 2)");
                size_t n = static_cast<size_t>(info.shape[0]);
                std::span<const double> span(static_cast<const double *>(info.ptr), n * 2);
                return nerve::algorithms::compute_betti_curve<double>(span, n, max_dim);
            },
            py::arg("diagram"), py::arg("max_dim") = -1,
            "Compute Betti curve from a float64 diagram (Nx2 array)");
    }

    // serialization -- schema versioning, format negotiation, PH5/PH6 artifact serialization
    {
        auto ser = m.def_submodule(
            "serialization", "Schema versioning, serialization formats, PH5/PH6 artifact support");

        // SchemaVersion
        py::class_<nerve::serialization::SchemaVersion>(ser, "SchemaVersion")
            .def(py::init<uint32_t, uint32_t, uint32_t>(), py::arg("major") = 1,
                 py::arg("minor") = 0, py::arg("patch") = 0,
                 "Create a schema version (major.minor.patch)")
            .def_readwrite("major", &nerve::serialization::SchemaVersion::major)
            .def_readwrite("minor", &nerve::serialization::SchemaVersion::minor)
            .def_readwrite("patch", &nerve::serialization::SchemaVersion::patch)
            .def("to_string", &nerve::serialization::SchemaVersion::toString,
                 "Format as 'major.minor.patch' string")
            .def("is_compatible_with", &nerve::serialization::SchemaVersion::isCompatibleWith,
                 py::arg("other"), "Check if this version is compatible with another")
            .def("__eq__",
                 [](const nerve::serialization::SchemaVersion &self,
                    const nerve::serialization::SchemaVersion &other) { return self == other; },
                 py::arg("other"))
            .def("__lt__", [](const nerve::serialization::SchemaVersion &self, const nerve::serialization::SchemaVersion &other) { return self < other; }, py::arg("other"))
            .def("__le__", [](const nerve::serialization::SchemaVersion &self, const nerve::serialization::SchemaVersion &other) { return self <= other; }, py::arg("other"))
            .def("__ge__", [](const nerve::serialization::SchemaVersion &self, const nerve::serialization::SchemaVersion &other) { return self >= other; }, py::arg("other"))
            .def_static("from_string", &nerve::serialization::SchemaVersion::fromString,
                        py::arg("version_str"),
                        "Parse a 'major.minor.patch' string into a SchemaVersion")
            .def("__repr__", [](const nerve::serialization::SchemaVersion &v) {
                return "SchemaVersion(" + v.toString() + ")";
            });

        // SchemaMetadata
        py::class_<nerve::serialization::SchemaMetadata>(ser, "SchemaMetadata")
            .def(py::init<>())
            .def_readwrite("version", &nerve::serialization::SchemaMetadata::version)
            .def_readwrite("min_compatible_version",
                           &nerve::serialization::SchemaMetadata::minCompatibleVersion)
            .def_readwrite("max_compatible_version",
                           &nerve::serialization::SchemaMetadata::maxCompatibleVersion)
            .def_readwrite("schema_name", &nerve::serialization::SchemaMetadata::schema_name)
            .def_readwrite("description", &nerve::serialization::SchemaMetadata::description)
            .def_readwrite("custom_fields", &nerve::serialization::SchemaMetadata::custom_fields)
            .def("is_version_compatible",
                 &nerve::serialization::SchemaMetadata::isVersionCompatible,
                 py::arg("check_version"), "Check if a version is compatible with this schema")
            .def("to_string", &nerve::serialization::SchemaMetadata::toString,
                 "Serialize metadata to string")
            .def("to_map", &nerve::serialization::SchemaMetadata::toMap,
                 "Convert metadata to a string map");

        // SerializationFormat enum
        py::enum_<nerve::serialization::SerializationFormat>(ser, "SerializationFormat")
            .value("FLATBUFFERS", nerve::serialization::SerializationFormat::FLATBUFFERS)
            .value("ARROW", nerve::serialization::SerializationFormat::ARROW)
            .value("JSON", nerve::serialization::SerializationFormat::JSON)
            .value("BINARY", nerve::serialization::SerializationFormat::BINARY)
            .value("PROTOBUF", nerve::serialization::SerializationFormat::PROTOBUF)
            .export_values();

        // SerializationContext
        py::class_<nerve::serialization::SerializationContext>(ser, "SerializationContext")
            .def(py::init<nerve::serialization::SerializationFormat,
                          nerve::serialization::SchemaVersion>(),
                 py::arg("format") = nerve::serialization::SerializationFormat::FLATBUFFERS,
                 py::arg("version") = nerve::serialization::SchemaVersion(1, 0, 0),
                 "Create a serialization context")
            .def_readwrite("format", &nerve::serialization::SerializationContext::format)
            .def_readwrite("schema_version",
                           &nerve::serialization::SerializationContext::schemaVersion)
            .def_readwrite("options", &nerve::serialization::SerializationContext::options);

        // SerializationErrorCode enum
        py::enum_<nerve::serialization::errors::SerializationErrorCode>(ser,
                                                                        "SerializationErrorCode")
            .value("SUCCESS", nerve::serialization::errors::SerializationErrorCode::SUCCESS)
            .value(
                "INCOMPATIBLE_SCHEMA_VERSION",
                nerve::serialization::errors::SerializationErrorCode::INCOMPATIBLE_SCHEMA_VERSION)
            .value("UNSUPPORTED_SCHEMA_VERSION",
                   nerve::serialization::errors::SerializationErrorCode::UNSUPPORTED_SCHEMA_VERSION)
            .value("SCHEMA_VERSION_NEGOTIATION_FAILED",
                   nerve::serialization::errors::SerializationErrorCode::
                       SCHEMA_VERSION_NEGOTIATION_FAILED)
            .value("SCHEMA_NOT_FOUND",
                   nerve::serialization::errors::SerializationErrorCode::SCHEMA_NOT_FOUND)
            .value("UNSUPPORTED_SERIALIZATION_FORMAT",
                   nerve::serialization::errors::SerializationErrorCode::
                       UNSUPPORTED_SERIALIZATION_FORMAT)
            .value(
                "SERIALIZATION_FORMAT_MISMATCH",
                nerve::serialization::errors::SerializationErrorCode::SERIALIZATION_FORMAT_MISMATCH)
            .value("INVALID_SERIALIZATION_DATA",
                   nerve::serialization::errors::SerializationErrorCode::INVALID_SERIALIZATION_DATA)
            .value("DESERIALIZATION_FAILED",
                   nerve::serialization::errors::SerializationErrorCode::DESERIALIZATION_FAILED)
            .value("METADATA_MISSING",
                   nerve::serialization::errors::SerializationErrorCode::METADATA_MISSING)
            .export_values();

        // VersionNegotiationResult
        py::class_<nerve::serialization::VersionNegotiationResult>(ser, "VersionNegotiationResult")
            .def(py::init<>())
            .def_readwrite("success", &nerve::serialization::VersionNegotiationResult::success)
            .def_readwrite("negotiated_version",
                           &nerve::serialization::VersionNegotiationResult::negotiated_version)
            .def_readwrite("error_message",
                           &nerve::serialization::VersionNegotiationResult::error_message)
            .def_readwrite("requires_conversion",
                           &nerve::serialization::VersionNegotiationResult::requiresConversion)
            .def_readwrite("conversion_strategy",
                           &nerve::serialization::VersionNegotiationResult::conversion_strategy)
            .def_static("success_result",
                        &nerve::serialization::VersionNegotiationResult::successResult,
                        py::arg("version"), py::arg("needs_conversion") = false,
                        py::arg("strategy") = "", "Create a success result")
            .def_static("error_result",
                        &nerve::serialization::VersionNegotiationResult::errorResult,
                        py::arg("error"), "Create an error result")
            .def("__repr__", [](const nerve::serialization::VersionNegotiationResult &r) {
                if (r.success)
                    return "VersionNegotiationResult(success, v" + r.negotiated_version.toString() +
                           ")";
                return "VersionNegotiationResult(error: " + r.error_message + ")";
            });

        // VersionNegotiator
        py::class_<nerve::serialization::VersionNegotiator>(ser, "VersionNegotiator")
            .def(py::init<>())
            .def("register_schema", &nerve::serialization::VersionNegotiator::registerSchema,
                 py::arg("metadata"), "Register a schema with its metadata")
            .def("unregister_schema", &nerve::serialization::VersionNegotiator::unregisterSchema,
                 py::arg("schema_name"), "Unregister a schema by name")
            .def("negotiate_version",
                 py::overload_cast<const std::string &, const nerve::serialization::SchemaVersion &,
                                   const std::vector<nerve::serialization::SchemaVersion> &>(
                     &nerve::serialization::VersionNegotiator::negotiateVersion),
                 py::arg("schema_name"), py::arg("requested_version"),
                 py::arg("supported_versions"), "Negotiate a version against supported versions")
            .def(
                "negotiate_version_simple",
                py::overload_cast<const std::string &, const nerve::serialization::SchemaVersion &>(
                    &nerve::serialization::VersionNegotiator::negotiateVersion),
                py::arg("schema_name"), py::arg("requested_version"),
                "Negotiate a version against registered schema")
            .def("get_schema_metadata", &nerve::serialization::VersionNegotiator::getSchemaMetadata,
                 py::arg("schema_name"), "Get metadata for a registered schema")
            .def("get_registered_schemas",
                 &nerve::serialization::VersionNegotiator::getRegisteredSchemas,
                 "Get names of all registered schemas")
            .def("get_supported_versions",
                 &nerve::serialization::VersionNegotiator::getSupportedVersions,
                 py::arg("schema_name"), "Get supported versions for a schema")
            .def("is_version_supported",
                 &nerve::serialization::VersionNegotiator::isVersionSupported,
                 py::arg("schema_name"), py::arg("version"), "Check if a version is supported")
            .def("are_versions_compatible",
                 &nerve::serialization::VersionNegotiator::areVersionsCompatible,
                 py::arg("schema_name"), py::arg("version1"), py::arg("version2"),
                 "Check if two versions are compatible");

        // FlatBuffersSerializer
        py::class_<nerve::serialization::FlatBuffersSerializer>(ser, "FlatBuffersSerializer")
            .def(py::init<>())
            .def(
                "serialize",
                [](nerve::serialization::FlatBuffersSerializer &self, py::bytes data,
                   const nerve::serialization::SerializationContext &ctx) {
                    std::string raw = data;
                    auto result = self.serialize(raw.data(), raw.size(), ctx);
                    if (result.isError())
                        throw std::runtime_error("FlatBuffers serialization failed: " +
                                                 result.error().message);
                    auto &vec = result.value();
                    return py::bytes(reinterpret_cast<const char *>(vec.data()), vec.size());
                },
                py::arg("data"), py::arg("context"), "Serialize raw bytes using FlatBuffers")
            .def(
                "deserialize",
                [](nerve::serialization::FlatBuffersSerializer &self, py::bytes data,
                   const nerve::serialization::SerializationContext &ctx) {
                    std::string raw = data;
                    std::vector<uint8_t> buf(raw.begin(), raw.end());
                    auto result = self.deserialize(buf, ctx);
                    if (result.isError())
                        throw std::runtime_error("FlatBuffers deserialization failed: " +
                                                 result.error().message);
                    auto &vec = result.value();
                    return py::bytes(reinterpret_cast<const char *>(vec.data()), vec.size());
                },
                py::arg("data"), py::arg("context"), "Deserialize to raw bytes using FlatBuffers")
            .def("get_format_name", &nerve::serialization::FlatBuffersSerializer::getFormatName,
                 "Get the format name")
            .def("get_format", &nerve::serialization::FlatBuffersSerializer::getFormat,
                 "Get the SerializationFormat");

        // ArrowSerializer
        py::class_<nerve::serialization::ArrowSerializer>(ser, "ArrowSerializer")
            .def(py::init<>())
            .def(
                "serialize",
                [](nerve::serialization::ArrowSerializer &self, py::bytes data,
                   const nerve::serialization::SerializationContext &ctx) {
                    std::string raw = data;
                    auto result = self.serialize(raw.data(), raw.size(), ctx);
                    if (result.isError())
                        throw std::runtime_error("Arrow serialization failed: " +
                                                 result.error().message);
                    auto &vec = result.value();
                    return py::bytes(reinterpret_cast<const char *>(vec.data()), vec.size());
                },
                py::arg("data"), py::arg("context"), "Serialize raw bytes using Apache Arrow")
            .def(
                "deserialize",
                [](nerve::serialization::ArrowSerializer &self, py::bytes data,
                   const nerve::serialization::SerializationContext &ctx) {
                    std::string raw = data;
                    std::vector<uint8_t> buf(raw.begin(), raw.end());
                    auto result = self.deserialize(buf, ctx);
                    if (result.isError())
                        throw std::runtime_error("Arrow deserialization failed: " +
                                                 result.error().message);
                    auto &vec = result.value();
                    return py::bytes(reinterpret_cast<const char *>(vec.data()), vec.size());
                },
                py::arg("data"), py::arg("context"), "Deserialize to raw bytes using Apache Arrow")
            .def("get_format_name", &nerve::serialization::ArrowSerializer::getFormatName,
                 "Get the format name")
            .def("get_format", &nerve::serialization::ArrowSerializer::getFormat,
                 "Get the SerializationFormat");

        // SerializationManager (singleton)
        py::class_<nerve::serialization::SerializationManager>(ser, "SerializationManager")
            .def_static("instance", &nerve::serialization::SerializationManager::instance,
                        py::return_value_policy::reference,
                        "Get the global SerializationManager singleton")
            .def(
                "register_serializer",
                [](nerve::serialization::SerializationManager &self,
                   std::unique_ptr<nerve::serialization::Serializer> serializer) {
                    self.registerSerializer(std::move(serializer));
                },
                py::arg("serializer"), "Register a serializer")
            .def("unregister_serializer",
                 &nerve::serialization::SerializationManager::unregisterSerializer,
                 py::arg("format"), "Unregister a serializer by format")
            .def(
                "serialize",
                [](nerve::serialization::SerializationManager &self, const std::string &schema_name,
                   py::bytes data, const nerve::serialization::SerializationContext &context) {
                    std::string raw = data;
                    auto result = self.serialize(schema_name, raw.data(), raw.size(), context);
                    if (result.isError())
                        throw std::runtime_error("Serialization failed: " + result.error().message);
                    auto &vec = result.value();
                    return py::bytes(reinterpret_cast<const char *>(vec.data()), vec.size());
                },
                py::arg("schema_name"), py::arg("data"), py::arg("context"),
                "Serialize data through the manager")
            .def(
                "deserialize",
                [](nerve::serialization::SerializationManager &self, const std::string &schema_name,
                   py::bytes data, const nerve::serialization::SerializationContext &context) {
                    std::string raw = data;
                    std::vector<uint8_t> buf(raw.begin(), raw.end());
                    auto result = self.deserialize(schema_name, buf, context);
                    if (result.isError())
                        throw std::runtime_error("Deserialization failed: " +
                                                 result.error().message);
                    auto &vec = result.value();
                    return py::bytes(reinterpret_cast<const char *>(vec.data()), vec.size());
                },
                py::arg("schema_name"), py::arg("data"), py::arg("context"),
                "Deserialize data through the manager")
            .def(
                "load_from_file",
                [](nerve::serialization::SerializationManager &self, const std::string &path,
                   const std::string &schema_name,
                   nerve::serialization::SerializationFormat format) {
                    auto result = self.loadFromFile(path, schema_name, format);
                    if (result.isError())
                        throw std::runtime_error("Load from file failed: " +
                                                 result.error().message);
                    auto &vec = result.value();
                    return py::bytes(reinterpret_cast<const char *>(vec.data()), vec.size());
                },
                py::arg("path"), py::arg("schema_name"), py::arg("format"),
                "Load serialized data from a file")
            .def(
                "save_to_file",
                [](nerve::serialization::SerializationManager &self, const std::string &path,
                   const std::string &schema_name, py::bytes data,
                   nerve::serialization::SerializationFormat format) {
                    std::string raw = data;
                    auto result =
                        self.saveToFile(path, schema_name, raw.data(), raw.size(), format);
                    if (result.isError())
                        throw std::runtime_error("Save to file failed: " + result.error().message);
                    return result.value();
                },
                py::arg("path"), py::arg("schema_name"), py::arg("data"), py::arg("format"),
                "Save serialized data to a file")
            .def("negotiate_version", &nerve::serialization::SerializationManager::negotiateVersion,
                 py::arg("schema_name"), py::arg("requested_version"), py::arg("format"),
                 "Negotiate a schema version")
            .def("register_schema", &nerve::serialization::SerializationManager::registerSchema,
                 py::arg("metadata"), "Register a schema with the manager")
            .def("get_schema_metadata",
                 &nerve::serialization::SerializationManager::getSchemaMetadata,
                 py::arg("schema_name"), "Get metadata for a schema")
            .def("get_supported_formats",
                 &nerve::serialization::SerializationManager::getSupportedFormats,
                 "Get list of supported serialization formats")
            .def("is_format_supported",
                 &nerve::serialization::SerializationManager::isFormatSupported, py::arg("format"),
                 "Check if a format is supported");

        // Free utility functions
        ser.def("serialization_format_to_string",
                &nerve::serialization::serialization_format_to_string, py::arg("format"),
                "Convert a SerializationFormat to its string representation");
        ser.def("string_to_serialization_format",
                &nerve::serialization::stringToSerializationFormat, py::arg("format_str"),
                "Parse a string into a SerializationFormat");
        ser.def("is_schema_version_valid", &nerve::serialization::isSchemaVersionValid,
                py::arg("version"), "Check if a SchemaVersion is valid");
        ser.def("is_version_compatible", &nerve::serialization::isVersionCompatible,
                py::arg("current"), py::arg("target"), "Check if two versions are compatible");
        ser.def(
            "test_round_trip_compatibility",
            [](const std::string &schema_name, py::bytes data,
               const nerve::serialization::SchemaVersion &version,
               nerve::serialization::SerializationFormat format) {
                std::string raw = data;
                auto result = nerve::serialization::testRoundTripCompatibility(
                    schema_name, raw.data(), raw.size(), version, format);
                if (result.isError())
                    throw std::runtime_error("Round-trip test failed: " + result.error().message);
                return result.value();
            },
            py::arg("schema_name"), py::arg("data"), py::arg("version"), py::arg("format"),
            "Test round-trip serialization compatibility");
        ser.def(
            "migrate_schema",
            [](py::bytes data, const nerve::serialization::SchemaVersion &from_version,
               const nerve::serialization::SchemaVersion &to_version,
               const std::string &schema_name) {
                std::string raw = data;
                std::vector<uint8_t> buf(raw.begin(), raw.end());
                auto result =
                    nerve::serialization::migrateSchema(buf, from_version, to_version, schema_name);
                if (result.isError())
                    throw std::runtime_error("Schema migration failed: " + result.error().message);
                auto &vec = result.value();
                return py::bytes(reinterpret_cast<const char *>(vec.data()), vec.size());
            },
            py::arg("data"), py::arg("from_version"), py::arg("to_version"), py::arg("schema_name"),
            "Migrate serialized data between schema versions");

        // PH5PH6ArtifactMetadata
        py::class_<nerve::serialization::PH5PH6ArtifactMetadata>(ser, "PH5PH6ArtifactMetadata")
            .def(py::init<>())
            .def_readwrite("schema_version",
                           &nerve::serialization::PH5PH6ArtifactMetadata::schema_version)
            .def_readwrite("artifact_type",
                           &nerve::serialization::PH5PH6ArtifactMetadata::artifact_type)
            .def_readwrite("algorithm_variant",
                           &nerve::serialization::PH5PH6ArtifactMetadata::algorithm_variant)
            .def_readwrite("has_highdim_extension",
                           &nerve::serialization::PH5PH6ArtifactMetadata::has_highdim_extension)
            .def_readwrite("max_supported_dimension",
                           &nerve::serialization::PH5PH6ArtifactMetadata::max_supported_dimension)
            .def_readwrite("extension_fields",
                           &nerve::serialization::PH5PH6ArtifactMetadata::extension_fields)
            .def_readwrite("min_compatible_version",
                           &nerve::serialization::PH5PH6ArtifactMetadata::min_compatible_version)
            .def_readwrite("deprecated_fields",
                           &nerve::serialization::PH5PH6ArtifactMetadata::deprecated_fields)
            .def_readwrite("new_fields", &nerve::serialization::PH5PH6ArtifactMetadata::new_fields)
            .def("serialize_metadata",
                 &nerve::serialization::PH5PH6ArtifactMetadata::serializeMetadata,
                 "Serialize metadata to bytes")
            .def(
                "deserialize_metadata",
                [](nerve::serialization::PH5PH6ArtifactMetadata &self, py::bytes data) {
                    std::string raw = data;
                    std::vector<uint8_t> buf(raw.begin(), raw.end());
                    return self.deserializeMetadata(buf);
                },
                py::arg("data"), "Deserialize metadata from bytes")
            .def("to_string", &nerve::serialization::PH5PH6ArtifactMetadata::toString,
                 "Serialize metadata to string");

        // PH5PH6SchemaSerializer
        py::class_<nerve::serialization::PH5PH6SchemaSerializer>(ser, "PH5PH6SchemaSerializer")
            .def(py::init<>())
            .def(
                "serialize",
                [](nerve::serialization::PH5PH6SchemaSerializer &self, py::bytes data,
                   const nerve::serialization::SerializationContext &ctx) {
                    std::string raw = data;
                    auto result = self.serialize(raw.data(), raw.size(), ctx);
                    if (result.isError())
                        throw std::runtime_error("PH5PH6 serialization failed: " +
                                                 result.error().message);
                    auto &vec = result.value();
                    return py::bytes(reinterpret_cast<const char *>(vec.data()), vec.size());
                },
                py::arg("data"), py::arg("context"), "Serialize data using PH5PH6 schema")
            .def(
                "deserialize",
                [](nerve::serialization::PH5PH6SchemaSerializer &self, py::bytes data,
                   const nerve::serialization::SerializationContext &ctx) {
                    std::string raw = data;
                    std::vector<uint8_t> buf(raw.begin(), raw.end());
                    auto result = self.deserialize(buf, ctx);
                    if (result.isError())
                        throw std::runtime_error("PH5PH6 deserialization failed: " +
                                                 result.error().message);
                    auto &vec = result.value();
                    return py::bytes(reinterpret_cast<const char *>(vec.data()), vec.size());
                },
                py::arg("data"), py::arg("context"), "Deserialize data using PH5PH6 schema")
            .def(
                "serialize_ph5_artifact",
                [](nerve::serialization::PH5PH6SchemaSerializer &self, py::bytes data,
                   const nerve::serialization::PH5PH6ArtifactMetadata &meta,
                   const nerve::serialization::SerializationContext &ctx) {
                    std::string raw = data;
                    auto result = self.serializePh5Artifact(raw.data(), raw.size(), meta, ctx);
                    if (result.isError())
                        throw std::runtime_error("PH5 artifact serialization failed: " +
                                                 result.error().message);
                    auto &vec = result.value();
                    return py::bytes(reinterpret_cast<const char *>(vec.data()), vec.size());
                },
                py::arg("data"), py::arg("artifact_metadata"), py::arg("context"),
                "Serialize a PH5 artifact")
            .def(
                "serialize_ph6_artifact",
                [](nerve::serialization::PH5PH6SchemaSerializer &self, py::bytes data,
                   const nerve::serialization::PH5PH6ArtifactMetadata &meta,
                   const nerve::serialization::SerializationContext &ctx) {
                    std::string raw = data;
                    auto result = self.serializePh6Artifact(raw.data(), raw.size(), meta, ctx);
                    if (result.isError())
                        throw std::runtime_error("PH6 artifact serialization failed: " +
                                                 result.error().message);
                    auto &vec = result.value();
                    return py::bytes(reinterpret_cast<const char *>(vec.data()), vec.size());
                },
                py::arg("data"), py::arg("artifact_metadata"), py::arg("context"),
                "Serialize a PH6 artifact")
            .def(
                "serialize_compact_summary",
                [](nerve::serialization::PH5PH6SchemaSerializer &self,
                   const nerve::summary::CompactSummary &summary,
                   const nerve::serialization::PH5PH6ArtifactMetadata &meta,
                   const nerve::serialization::SerializationContext &ctx) {
                    auto result = self.serializeCompactSummary(summary, meta, ctx);
                    if (result.isError())
                        throw std::runtime_error("CompactSummary serialization failed: " +
                                                 result.error().message);
                    auto &vec = result.value();
                    return py::bytes(reinterpret_cast<const char *>(vec.data()), vec.size());
                },
                py::arg("summary"), py::arg("artifact_metadata"), py::arg("context"),
                "Serialize a CompactSummary as a PH5/PH6 artifact")
            .def(
                "deserialize_compact_summary",
                [](nerve::serialization::PH5PH6SchemaSerializer &self, py::bytes data,
                   const nerve::serialization::SerializationContext &ctx) {
                    std::string raw = data;
                    std::vector<uint8_t> buf(raw.begin(), raw.end());
                    auto result = self.deserializeCompactSummary(buf, ctx);
                    if (result.isError())
                        throw std::runtime_error("CompactSummary deserialization failed: " +
                                                 result.error().message);
                    auto &[summary, meta] = result.value();
                    return py::make_tuple(summary, meta);
                },
                py::arg("data"), py::arg("context"), "Deserialize a CompactSummary from bytes")
            .def("is_version_compatible",
                 &nerve::serialization::PH5PH6SchemaSerializer::isVersionCompatible,
                 py::arg("version"), "Check if a schema version is compatible")
            .def("get_min_compatible_version",
                 &nerve::serialization::PH5PH6SchemaSerializer::getMinCompatibleVersion,
                 "Get the minimum compatible schema version")
            .def("get_schema_version",
                 &nerve::serialization::PH5PH6SchemaSerializer::getSchemaVersion,
                 "Get the schema version")
            .def("get_format_name", &nerve::serialization::PH5PH6SchemaSerializer::getFormatName,
                 "Get the format name");

        // PH5PH6SchemaRegistry
        py::class_<nerve::serialization::PH5PH6SchemaRegistry>(ser, "PH5PH6SchemaRegistry")
            .def(py::init<>())
            .def("register_artifact_type",
                 &nerve::serialization::PH5PH6SchemaRegistry::registerArtifactType,
                 py::arg("artifact_type"), py::arg("metadata"),
                 "Register an artifact type with metadata")
            .def("unregister_artifact_type",
                 &nerve::serialization::PH5PH6SchemaRegistry::unregisterArtifactType,
                 py::arg("artifact_type"), "Unregister an artifact type")
            .def("negotiate_version", &nerve::serialization::PH5PH6SchemaRegistry::negotiateVersion,
                 py::arg("artifact_type"), py::arg("requested_version"),
                 "Negotiate schema version for an artifact type")
            .def("get_artifact_metadata",
                 &nerve::serialization::PH5PH6SchemaRegistry::getArtifactMetadata,
                 py::arg("artifact_type"), "Get metadata for an artifact type")
            .def("get_registered_artifact_types",
                 &nerve::serialization::PH5PH6SchemaRegistry::getRegisteredArtifactTypes,
                 "Get all registered artifact type names")
            .def("is_artifact_supported",
                 &nerve::serialization::PH5PH6SchemaRegistry::isArtifactSupported,
                 py::arg("artifact_type"), py::arg("version"),
                 "Check if an artifact type supports a schema version")
            .def("create_serializer", &nerve::serialization::PH5PH6SchemaRegistry::createSerializer,
                 "Create a PH5PH6SchemaSerializer instance")
            .def("create_compatible_serializer",
                 &nerve::serialization::PH5PH6SchemaRegistry::createCompatibleSerializer,
                 "Create a compatible Serializer instance");

        // PH5PH6SchemaMigrator
        py::class_<nerve::serialization::PH5PH6SchemaMigrator>(ser, "PH5PH6SchemaMigrator")
            .def_static(
                "migrate_to_extended",
                [](py::bytes data, const nerve::serialization::SerializationContext &ctx) {
                    std::string raw = data;
                    std::vector<uint8_t> buf(raw.begin(), raw.end());
                    auto result =
                        nerve::serialization::PH5PH6SchemaMigrator::migrateToExtended(buf, ctx);
                    if (!result.success)
                        throw std::runtime_error("Migration failed: " + result.error_message);
                    auto migrated =
                        py::bytes(reinterpret_cast<const char *>(result.migrated_data.data()),
                                  result.migrated_data.size());
                    return py::make_tuple(migrated, result.new_metadata);
                },
                py::arg("data"), py::arg("context"),
                "Migrate data to extended format (returns (migrated_data, metadata))")
            .def_static(
                "migrate_with_extension",
                [](py::bytes data,
                   const nerve::serialization::PH5PH6ArtifactMetadata &extension_metadata,
                   const nerve::serialization::SerializationContext &ctx) {
                    std::string raw = data;
                    std::vector<uint8_t> buf(raw.begin(), raw.end());
                    auto result = nerve::serialization::PH5PH6SchemaMigrator::migrateWithExtension(
                        buf, extension_metadata, ctx);
                    if (!result.success)
                        throw std::runtime_error("Migration failed: " + result.error_message);
                    auto migrated =
                        py::bytes(reinterpret_cast<const char *>(result.migrated_data.data()),
                                  result.migrated_data.size());
                    return py::make_tuple(migrated, result.new_metadata);
                },
                py::arg("data"), py::arg("extension_metadata"), py::arg("context"),
                "Migrate data with extension (returns (migrated_data, metadata))")
            .def_static(
                "validate_data",
                [](py::bytes data) {
                    std::string raw = data;
                    std::vector<uint8_t> buf(raw.begin(), raw.end());
                    return nerve::serialization::PH5PH6SchemaMigrator::validateData(buf);
                },
                py::arg("data"), "Validate PH5/PH6 data integrity")
            .def_static(
                "validate_extended_data",
                [](py::bytes data,
                   const nerve::serialization::PH5PH6ArtifactMetadata &expected_metadata) {
                    std::string raw = data;
                    std::vector<uint8_t> buf(raw.begin(), raw.end());
                    return nerve::serialization::PH5PH6SchemaMigrator::validateExtendedData(
                        buf, expected_metadata);
                },
                py::arg("data"), py::arg("expected_metadata"),
                "Validate extended PH5/PH6 data integrity");
    }
}
