void bind_functional_api(py::module &m)
{
    m.def("vr_persistence_forward", &vr_persistence_forward_py,
          "Compute Vietoris-Rips persistence. Returns (diagrams, mask, num_pairs).",
          py::arg("points"), py::arg("max_dim") = 1,
          py::arg("max_radius") = std::numeric_limits<double>::infinity(),
          py::arg("metric") = "euclidean",
          py::arg("pi_resolution") = 0, // 0 = don't compute
          py::arg("pi_sigma") = 0.1);

    m.def("vr_persistence_backward", &vr_persistence_backward_impl,
          "Compute gradients for VR persistence.", py::arg("grad_diagrams"), py::arg("points"),
          py::arg("birth_idx"), py::arg("death_idx"));

    // Persistence from distance matrix
    m.def(
        "persistence_from_matrix",
        [](const at::Tensor &distance_matrix, int64_t max_dim) {
            using namespace nerve::torch;

            // Direct persistence computation from pre-computed distance matrix
            at::Tensor diagram = vr_persistence(distance_matrix, max_dim);

            // Wrap in standard format
            int64_t n_pairs = diagram.size(0);
            at::Tensor dims = at::zeros(
                {n_pairs}, at::TensorOptions().dtype(at::kLong).device(distance_matrix.device()));
            at::Tensor mask = at::ones(
                {n_pairs}, at::TensorOptions().dtype(at::kBool).device(distance_matrix.device()));
            at::Tensor num_pairs = at::zeros({max_dim + 1}, at::TensorOptions().dtype(at::kLong));
            num_pairs.index_put_({0}, n_pairs);
            num_pairs = num_pairs.to(distance_matrix.device());

            at::Tensor diagram_3d = at::cat({diagram, dims.unsqueeze(1)}, 1);

            return py::make_tuple(diagram_3d, mask, num_pairs);
        },
        "Compute persistence from pre-computed distance matrix. Returns "
        "(diagrams, mask, num_pairs).",
        py::arg("distance_matrix"), py::arg("max_dim") = 1);

    // Diagram Distances

    // Wasserstein distance between two diagrams
    m.def(
        "wasserstein_distance",
        [](const at::Tensor &diagram1, const at::Tensor &diagram2, double p) -> at::Tensor {
            using namespace nerve::torch;

            return at::tensor(diagram_wasserstein(diagram1, diagram2, p));
        },
        "Compute Wasserstein distance between two persistence diagrams", py::arg("diagram1"),
        py::arg("diagram2"), py::arg("p") = 2.0);

    // Bottleneck distance between two diagrams
    m.def(
        "bottleneck_distance",
        [](const at::Tensor &diagram1, const at::Tensor &diagram2) -> at::Tensor {
            using namespace nerve::torch;

            return at::tensor(diagram_bottleneck(diagram1, diagram2));
        },
        "Compute Bottleneck distance between two persistence diagrams", py::arg("diagram1"),
        py::arg("diagram2"));

    // ML Representations

    // Persistence image
    m.def("ml_persistence_image", &ml_persistence_image, "Convert persistence diagram to image",
          py::arg("diagram"), py::arg("resolution_birth") = 20, py::arg("resolution_death") = 20,
          py::arg("sigma") = 0.5, py::arg("birth_min") = 0.0, py::arg("birth_max") = 0.0,
          py::arg("death_min") = 0.0, py::arg("death_max") = 0.0,
          py::arg("weight_fn") = "persistence");

    // Persistence landscape
    m.def("ml_persistence_landscape", &ml_persistence_landscape,
          "Convert persistence diagram to landscape", py::arg("diagram"), py::arg("k") = 5,
          py::arg("num_samples") = 100, py::arg("x_min") = 0.0, py::arg("x_max") = 0.0);

    // Persistence silhouette
    m.def("ml_persistence_silhouette", &ml_persistence_silhouette,
          "Convert persistence diagram to silhouette", py::arg("diagram"),
          py::arg("num_samples") = 100, py::arg("x_min") = 0.0, py::arg("x_max") = 0.0,
          py::arg("weight_fn") = "persistence");

    // Heat kernel signature
    m.def(
        "ml_heat_kernel_signature",
        [](const at::Tensor &diagram, int64_t num_samples, double sigma,
           const std::optional<at::Tensor> &t_values) -> at::Tensor {
            if (t_values.has_value())
            {
                return ml_heat_kernel_signature(diagram, num_samples, sigma, t_values.value());
            }
            at::Tensor default_t = at::logspace(
                -2, 0, 10, 10.0, at::TensorOptions().dtype(at::kDouble).device(at::kCPU));
            return ml_heat_kernel_signature(diagram, num_samples, sigma, default_t);
        },
        "Compute heat kernel signature", py::arg("diagram"), py::arg("num_samples") = 100,
        py::arg("sigma") = 0.1, py::arg("t_values") = py::none());

    // Birth-death curve
    m.def("ml_birth_death_curve", &ml_birth_death_curve, "Compute birth-death histogram",
          py::arg("diagram"), py::arg("num_bins") = 50, py::arg("statistic") = "count");

    // ML Statistics

    // Total persistence
    m.def("ml_total_persistence", &ml_total_persistence, "Compute total persistence",
          py::arg("diagram"), py::arg("dim") = -1, py::arg("p") = 1.0);

    // Mean persistence
    m.def("ml_mean_persistence", &ml_mean_persistence, "Compute mean persistence",
          py::arg("diagram"), py::arg("dim") = -1);

    // Max persistence
    m.def("ml_max_persistence", &ml_max_persistence, "Compute maximum persistence",
          py::arg("diagram"), py::arg("dim") = -1);

    // Persistence variance
    m.def("ml_persistence_variance", &ml_persistence_variance, "Compute persistence variance",
          py::arg("diagram"), py::arg("dim") = -1);

    // Persistence entropy
    m.def("ml_persistence_entropy", &ml_persistence_entropy, "Compute persistence entropy",
          py::arg("diagram"), py::arg("dim") = -1, py::arg("base") = std::exp(1.0));

    // Number of features
    m.def("ml_number_of_features", &ml_number_of_features, "Count number of features",
          py::arg("diagram"), py::arg("dim") = -1, py::arg("min_persistence") = 0.0);

    // Betti curve
    m.def("ml_betti_curve", &ml_betti_curve, "Compute Betti curve", py::arg("diagram"),
          py::arg("num_samples") = 100, py::arg("dim") = -1);

    // Amplitude
    m.def("ml_amplitude", &ml_amplitude, "Compute diagram amplitude", py::arg("diagram"),
          py::arg("metric") = "persistence", py::arg("p") = 1.0, py::arg("dim") = -1);

    // Extract all features
    m.def("ml_extract_features", &ml_extract_features, "Extract all statistics as feature vector",
          py::arg("diagram"), py::arg("dims") = std::vector<int64_t>{0, 1});

    // ML Kernels

    // Gaussian kernel
    m.def("ml_gaussian_kernel", &ml_gaussian_kernel, "Compute Gaussian kernel between diagrams",
          py::arg("d1"), py::arg("d2"), py::arg("sigma") = 0.5,
          py::arg("distance_metric") = "euclidean");

    // Scale-space kernel
    m.def("ml_persistence_scale_space_kernel", &ml_persistence_scale_space_kernel,
          "Compute persistence scale-space kernel", py::arg("d1"), py::arg("d2"),
          py::arg("sigma") = 0.5, py::arg("weight") = 0.5);

    // Sliced Wasserstein kernel
    m.def("ml_sliced_wasserstein_kernel", &ml_sliced_wasserstein_kernel,
          "Compute sliced Wasserstein kernel", py::arg("d1"), py::arg("d2"),
          py::arg("num_slices") = 10, py::arg("sigma") = 0.5);

    // Fisher kernel
    m.def("ml_persistence_fisher_kernel", &ml_persistence_fisher_kernel,
          "Compute persistence Fisher kernel", py::arg("d1"), py::arg("d2"),
          py::arg("sigma") = 0.5);

    // Linear kernel
    m.def("ml_linear_kernel", &ml_linear_kernel, "Compute linear kernel via silhouette",
          py::arg("d1"), py::arg("d2"), py::arg("num_samples") = 100);

    // Compute kernel matrix
    m.def("ml_compute_kernel_matrix", &ml_compute_kernel_matrix, "Compute full kernel matrix",
          py::arg("diagrams"), py::arg("kernel") = "gaussian", py::arg("sigma") = 0.5,
          py::arg("num_slices") = 10);

    // Normalize kernel matrix
    m.def("ml_normalize_kernel_matrix", &ml_normalize_kernel_matrix, "Normalize kernel matrix",
          py::arg("K"));

    // Center kernel matrix
    m.def("ml_center_kernel_matrix", &ml_center_kernel_matrix, "Center kernel matrix",
          py::arg("K"));

    // Mapper Algorithm

    // MapperConfig struct
    py::class_<MapperConfig>(m, "MapperConfig")
        .def(py::init<>())
        .def_readwrite("filter_function", &MapperConfig::filter_function)
        .def_readwrite("pca_components", &MapperConfig::pca_components)
        .def_readwrite("cover_type", &MapperConfig::cover_type)
        .def_readwrite("cover_resolution", &MapperConfig::cover_resolution)
        .def_readwrite("cover_overlap", &MapperConfig::cover_overlap)
        .def_readwrite("clusterer", &MapperConfig::clusterer)
        .def_readwrite("dbscan_eps", &MapperConfig::dbscan_eps)
        .def_readwrite("dbscan_min_samples", &MapperConfig::dbscan_min_samples)
        .def_readwrite("kmeans_k", &MapperConfig::kmeans_k);

    // Enums
    py::enum_<CoverType>(m, "CoverType")
        .value("GRID", CoverType::GRID)
        .value("BALL", CoverType::BALL)
        .value("INTERVAL", CoverType::INTERVAL);

    py::enum_<ClustererType>(m, "ClustererType")
        .value("DBSCAN", ClustererType::DBSCAN)
        .value("SINGLE_LINKAGE", ClustererType::SINGLE_LINKAGE)
        .value("KMEANS", ClustererType::KMEANS)
        .value("CONNECTED", ClustererType::CONNECTED);

    // MapperNode
    py::class_<MapperNode>(m, "MapperNode")
        .def_readonly("id", &MapperNode::id)
        .def_readonly("point_indices", &MapperNode::point_indices)
        .def_readonly("centroid", &MapperNode::centroid)
        .def_readonly("filter_centroid", &MapperNode::filter_centroid)
        .def_readonly("cover_index", &MapperNode::cover_index);

    // MapperEdge
    py::class_<MapperEdge>(m, "MapperEdge")
        .def_readonly("source", &MapperEdge::source)
        .def_readonly("target", &MapperEdge::target)
        .def_readonly("weight", &MapperEdge::weight);

    // MapperGraph
    py::class_<MapperGraph>(m, "MapperGraph")
        .def_readonly("nodes", &MapperGraph::nodes)
        .def_readonly("edges", &MapperGraph::edges)
        .def("to_edge_list", &MapperGraph::to_edge_list)
        .def("to_adjacency_matrix", &MapperGraph::to_adjacency_matrix);

    // Mapper class
    py::class_<Mapper>(m, "Mapper")
        .def(py::init<const MapperConfig &>(), py::arg("config") = MapperConfig{})
        .def("fit_transform", &Mapper::fit_transform)
        .def("get_last_filter_values", &Mapper::get_last_filter_values);

    // Convenience function
    m.def("quick_mapper", &quick_mapper, "Quick mapper with default settings",
          py::arg("point_cloud"), py::arg("filter") = "pca_2d", py::arg("resolution") = 10,
          py::arg("overlap") = 0.25);

    // Filter functions
    m.def("filter_pca", &filter_pca, "PCA filter function for mapper", py::arg("point_cloud"),
          py::arg("n_components") = 2);

    m.def("filter_eccentricity", &filter_eccentricity, "Eccentricity filter function",
          py::arg("point_cloud"));
}
