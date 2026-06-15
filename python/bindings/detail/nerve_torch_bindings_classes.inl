void bind_persistence_diagram(py::module& m) {
    py::class_<PersistenceDiagram>(m, "PersistenceDiagram")
        .def(py::init<>())
        .def(py::init<int64_t>(), py::arg("capacity"))
        .def(py::init<at::Tensor, at::Tensor, at::Tensor, at::Tensor>(),
             py::arg("diagram"), py::arg("dimensions"),
             py::arg("birth_idx"), py::arg("death_idx"))

        // Accessors
        .def("diagram", &PersistenceDiagram::diagram,
             "Get full diagram tensor [N, 2] with [birth, death] values")
        .def("births", &PersistenceDiagram::births,
             "Get birth times [N]")
        .def("deaths", &PersistenceDiagram::deaths,
             "Get death times [N]")
        .def("dimensions", &PersistenceDiagram::dimensions,
             "Get dimensions [N]")
        .def("persistence_lengths", &PersistenceDiagram::persistence_lengths,
             "Get persistence (death - birth) [N]")
        .def("num_pairs", &PersistenceDiagram::num_pairs,
             "Number of persistence pairs")
        .def("max_dimension", &PersistenceDiagram::max_dimension,
             "Maximum homology dimension")

        // Filtering
        .def("get_dimension", &PersistenceDiagram::get_dimension,
             "Get diagram for specific dimension", py::arg("dim"))
        .def("get_finite_points", &PersistenceDiagram::get_finite_points,
             "Get finite points only (death != inf)")
        .def("get_infinite_points", &PersistenceDiagram::get_infinite_points,
             "Get infinite points only")
        .def("threshold", &PersistenceDiagram::threshold,
             "Filter by persistence threshold", py::arg("min_persistence"))
        .def("filter_by_dimension", &PersistenceDiagram::filter_by_dimension,
             "Filter by dimension", py::arg("dim"))

        // Operations
        .def("append", &PersistenceDiagram::append,
             "Concatenate with another diagram", py::arg("other"))
        .def("sort_by_persistence", &PersistenceDiagram::sort_by_persistence,
             "Sort by persistence (descending)")
        .def("normalize", &PersistenceDiagram::normalize,
             "Normalize birth/death to [0, 1]")

        // Statistics
        .def("total_persistence", &PersistenceDiagram::total_persistence,
             "Total persistence (sum of lengths)")
        .def("persistence_variance", &PersistenceDiagram::persistence_variance,
             "Sum of squared persistences")
        .def("mean_persistence_by_dimension", &PersistenceDiagram::mean_persistence_by_dimension,
             "Mean persistence per dimension")
        .def("betti_number", &PersistenceDiagram::betti_number,
             "Betti number at threshold", py::arg("threshold"), py::arg("dim") = -1)
        .def("euler_characteristic", &PersistenceDiagram::euler_characteristic,
             "Euler characteristic at threshold", py::arg("threshold"))
        .def("persistence_entropy", &PersistenceDiagram::persistence_entropy,
             "Persistence entropy")
        .def("landscape_norm", &PersistenceDiagram::landscape_norm,
             "Persistence landscape norm", py::arg("k"))

        // ML Representations
        .def("to_persistence_image", &PersistenceDiagram::to_persistence_image,
             "Convert to persistence image",
             py::arg("resolution") = 64,
             py::arg("sigma") = 0.1,
             py::arg("birth_min") = 0.0,
             py::arg("birth_max") = -1.0,
             py::arg("death_min") = 0.0,
             py::arg("death_max") = -1.0)
        .def("to_persistence_landscape", &PersistenceDiagram::to_persistence_landscape,
             "Convert to persistence landscape",
             py::arg("k") = 5,
             py::arg("num_samples") = 100,
             py::arg("min_val") = 0.0,
             py::arg("max_val") = -1.0)
        .def("to_betti_curve", &PersistenceDiagram::to_betti_curve,
             "Convert to Betti curve",
             py::arg("num_samples") = 100,
             py::arg("min_val") = 0.0,
             py::arg("max_val") = -1.0)
        .def("to_vector", &PersistenceDiagram::to_vector,
             "Vectorized representation for ML",
             py::arg("bins_per_dim") = 10)

        // Distances
        .def("wasserstein_distance", &PersistenceDiagram::wasserstein_distance,
             "Wasserstein distance to another diagram",
             py::arg("other"), py::arg("p") = 2.0)
        .def("bottleneck_distance", &PersistenceDiagram::bottleneck_distance,
             "Bottleneck distance to another diagram",
             py::arg("other"))
        .def("persistence_kernel", &PersistenceDiagram::persistence_kernel,
             "Persistence kernel with another diagram",
             py::arg("other"), py::arg("sigma") = 1.0)

        // Device Support
        .def("to", &PersistenceDiagram::to,
             "Move to device", py::arg("device"))
        .def("device", &PersistenceDiagram::device,
             "Get current device")
        .def("is_cuda", &PersistenceDiagram::is_cuda,
             "Check if on CUDA")

        // Batching
        .def_static("batch", &PersistenceDiagram::batch,
             "Stack diagrams for batch processing",
             py::arg("diagrams"))
        .def("get_batch_item", &PersistenceDiagram::get_batch_item,
             "Get item from batch", py::arg("idx"))
        .def("is_batched", &PersistenceDiagram::is_batched,
             "Check if batched")
        .def("batch_size", &PersistenceDiagram::batch_size,
             "Batch size (1 if not batched)")

        // Serialization
        .def("state_dict", &PersistenceDiagram::state_dict,
             "Export to dictionary of tensors")
        .def("load_state_dict", &PersistenceDiagram::load_state_dict,
             "Load from dictionary", py::arg("state_dict"))

        // Differentiability
        .def("requires_grad", &PersistenceDiagram::requires_grad,
             "Check if supports gradients")
        .def("set_requires_grad", &PersistenceDiagram::set_requires_grad,
             "Enable/disable gradients", py::arg("requires_grad") = true)
        .def("backward", &PersistenceDiagram::backward,
             "Backward pass (if in computational graph)")
        .def("births_grad", &PersistenceDiagram::births_grad,
             "Get gradient w.r.t. birth times")
        .def("deaths_grad", &PersistenceDiagram::deaths_grad,
             "Get gradient w.r.t. death times")

        .def("__repr__", [](const PersistenceDiagram& d) {
            return "PersistenceDiagram(num_pairs=" + std::to_string(d.num_pairs()) +
                   ", max_dim=" + std::to_string(d.max_dimension()) + ")";
        });
}

// SimplexTree bindings

void bind_simplex_tree(py::module& m) {
    py::class_<SimplexTree>(m, "SimplexTree")
        .def(py::init<>())
        .def(py::init<at::Tensor, double>(),
             py::arg("points"), py::arg("max_radius") = 0.0)

        // Insertion
        .def("insert", &SimplexTree::insert,
             "Insert a simplex with filtration value",
             py::arg("vertices"), py::arg("filtration") = 0.0)
        .def("insert_batch", &SimplexTree::insert_batch,
             "Insert batch of simplices",
             py::arg("simplices"), py::arg("filtration_values"))
        .def("remove", &SimplexTree::remove,
             "Remove a simplex and its cofaces",
             py::arg("vertices"))
        .def("set_filtration", &SimplexTree::set_filtration,
             "Set filtration value for a simplex",
             py::arg("vertices"), py::arg("value"))

        // Queries
        .def("contains", &SimplexTree::contains,
             "Check if simplex exists",
             py::arg("vertices"))
        .def("find", &SimplexTree::find,
             "Find node index for a simplex (-1 if not found)",
             py::arg("vertices"))
        .def("get_vertices", &SimplexTree::get_vertices,
             "Get vertices of a simplex from node index",
             py::arg("node_idx"))
        .def("get_cofaces", &SimplexTree::get_cofaces,
             "Get all cofaces of a simplex",
             py::arg("vertices"), py::arg("max_dim") = -1)
        .def("get_faces", &SimplexTree::get_faces,
             "Get all faces of a simplex",
             py::arg("vertices"))
        .def("get_simplices_by_dimension", &SimplexTree::get_simplices_by_dimension,
             "Get simplices in a specific dimension",
             py::arg("dim"))
        .def("get_filtration", &SimplexTree::get_filtration,
             "Get filtration value of a simplex",
             py::arg("vertices"))

        // VR Complex Construction
        .def("build_vr", &SimplexTree::build_vr,
             "Build Vietoris-Rips complex",
             py::arg("points"), py::arg("max_radius"), py::arg("max_dim"))
        .def("build_witness", &SimplexTree::build_witness,
             "Build witness complex",
             py::arg("points"), py::arg("landmarks"),
             py::arg("max_radius"), py::arg("max_dim"))

        // Accessors
        .def("num_simplices", &SimplexTree::num_simplices,
             "Number of simplices")
        .def("max_dimension", &SimplexTree::max_dimension,
             "Maximum dimension")
        .def("filtration_values", &SimplexTree::filtration_values,
             "Get all filtration values as tensor")
        .def("root_idx", &SimplexTree::root_idx,
             "Get root index")

        // Device Support
        .def("to", &SimplexTree::to,
             "Move to device", py::arg("device"))
        .def("device", &SimplexTree::device,
             "Get current device")
        .def("is_cuda", &SimplexTree::is_cuda,
             "Check if on CUDA")

        // Conversion
        .def("to_boundary_matrix", &SimplexTree::to_boundary_matrix,
             "Convert to boundary matrix for given dimension",
             py::arg("dim"))
        .def("get_sorted_simplices", &SimplexTree::get_sorted_simplices,
             "Get filtration-sorted simplex list")
        .def("to_tensor", &SimplexTree::to_tensor,
             "Export to dense tensor representation")

        // Internal Access
        .def("vertex_indices", &SimplexTree::vertex_indices)
        .def("parent_pointers", &SimplexTree::parent_pointers)
        .def("first_child", &SimplexTree::first_child)
        .def("next_sibling", &SimplexTree::next_sibling)

        .def("__repr__", [](const SimplexTree& st) {
            return "SimplexTree(num_simplices=" + std::to_string(st.num_simplices()) +
                   ", max_dim=" + std::to_string(st.max_dimension()) + ")";
        });
}
