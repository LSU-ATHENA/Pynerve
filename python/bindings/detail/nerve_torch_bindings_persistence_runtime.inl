// Implementation: VR Persistence with batched support

namespace
{

bool isSupportedTorchVrMetric(const std::string &metric)
{
    return metric == "euclidean" || metric == "manhattan" || metric == "chebyshev" ||
           metric == "cosine";
}

struct VRPersistenceResult
{
    at::Tensor diagrams;           // [batch, max_pairs, 3] (birth, death, dim)
    at::Tensor mask;               // [batch, max_pairs] bool
    at::Tensor num_pairs;          // [batch, max_dim+1] int
    at::Tensor persistence_images; // Optional [batch, max_dim+1, res, res]
    std::vector<ExtendedPersistenceResult> persistence_info; // For gradients
};

VRPersistenceResult vr_persistence_forward_impl(const at::Tensor &points, int64_t max_dim,
                                                double max_radius, const std::string &metric,
                                                int64_t pi_resolution, double pi_sigma)
{
    using namespace nerve::torch;
    TORCH_CHECK(max_dim >= 0, "max_dim must be non-negative");
    TORCH_CHECK(max_radius > 0.0 && !std::isnan(max_radius),
                "max_radius must be positive and not NaN");
    TORCH_CHECK(isSupportedTorchVrMetric(metric), "Unsupported metric: ", metric);
    TORCH_CHECK(pi_resolution >= 0, "pi_resolution must be non-negative");
    if (pi_resolution > 0)
    {
        TORCH_CHECK(std::isfinite(pi_sigma) && pi_sigma > 0.0,
                    "pi_sigma must be finite and positive when pi_resolution is "
                    "positive");
        TORCH_CHECK(false, "vr_persistence_forward does not return persistence "
                           "images; call persistence_image on returned diagrams");
    }

    // Handle device - use input device if possible
    at::Tensor points_proc = points.contiguous();
    bool was_cuda = points.is_cuda();

    // Default execution path processes on CPU; results are moved back to
    // the caller device after computation.
    if (was_cuda)
    {
        points_proc = points_proc.cpu();
    }

    // Handle batch dimension
    bool was_batched = true;
    if (points_proc.dim() == 2)
    {
        points_proc = points_proc.unsqueeze(0);
        was_batched = false;
    }

    TORCH_CHECK(points_proc.dim() == 3, "Expected 2D or 3D input tensor");
    TORCH_CHECK(points_proc.is_floating_point(), "points must be floating point");
    TORCH_CHECK(points_proc.size(0) > 0, "points must contain at least one batch item");
    TORCH_CHECK(points_proc.size(1) > 0, "points must contain at least one point");
    TORCH_CHECK(points_proc.size(2) > 0, "points must contain at least one coordinate dimension");
    TORCH_CHECK(at::isfinite(points_proc).all().item<bool>(),
                "points must contain only finite coordinates");

    const int64_t batch_size = points_proc.size(0);
    const int64_t n_points = points_proc.size(1);

    // Process each batch item
    std::vector<at::Tensor> batch_diagrams;
    std::vector<at::Tensor> batch_masks;
    std::vector<at::Tensor> batch_num_pairs;
    std::vector<ExtendedPersistenceResult> batch_persistence_info;

    for (int64_t b = 0; b < batch_size; ++b)
    {
        at::Tensor batch_points = points_proc[b]; // [n_points, dim]

        // Build distance matrix
        at::Tensor dist_matrix = vr_build_with_metric(batch_points, max_radius, metric);

        // Compute extended persistence with higher dimensions
        auto persistence = compute_extended_persistence(dist_matrix, max_dim, max_radius);
        batch_persistence_info.push_back(persistence);

        // Convert to tensor format
        const auto n_pairs = static_cast<int64_t>(persistence.pairs.size());
        at::Tensor diagram =
            at::empty({n_pairs, 3}, at::TensorOptions().dtype(at::kDouble).device(at::kCPU));
        auto diag_accessor = diagram.accessor<double, 2>();

        // Count pairs per dimension
        at::Tensor num_pairs =
            at::zeros({max_dim + 1}, at::TensorOptions().dtype(at::kLong).device(at::kCPU));
        auto num_pairs_accessor = num_pairs.accessor<int64_t, 1>();

        for (int64_t pair_index = 0; pair_index < n_pairs; ++pair_index)
        {
            const auto [birth, death, dim] = persistence.pairs[static_cast<size_t>(pair_index)];
            diag_accessor[pair_index][0] = birth;
            diag_accessor[pair_index][1] = death;
            diag_accessor[pair_index][2] = static_cast<double>(dim);
            if (dim <= max_dim)
            {
                num_pairs_accessor[dim]++;
            }
        }

        // Create mask (all valid pairs)
        at::Tensor mask =
            at::ones({n_pairs}, at::TensorOptions().dtype(at::kBool).device(at::kCPU));

        batch_diagrams.push_back(diagram);
        batch_masks.push_back(mask);
        batch_num_pairs.push_back(num_pairs);
    }

    // Pad and stack batch diagrams
    int64_t max_pairs = 0;
    for (const auto &d : batch_diagrams)
    {
        max_pairs = std::max(max_pairs, d.size(0));
    }

    // Pad each diagram to max_pairs
    std::vector<at::Tensor> padded_diagrams;
    std::vector<at::Tensor> padded_masks;
    at::TensorOptions float_opts = points.options().dtype(at::kFloat);
    at::TensorOptions bool_opts = at::TensorOptions().dtype(at::kBool).device(points.device());

    for (size_t i = 0; i < batch_diagrams.size(); ++i)
    {
        int64_t n_pairs = batch_diagrams[i].size(0);
        if (n_pairs < max_pairs)
        {
            // Pad diagrams with zeros
            at::Tensor padding =
                at::zeros({max_pairs - n_pairs, 3}, float_opts.dtype(at::kDouble).device(at::kCPU));
            at::Tensor padded_diag = at::cat({batch_diagrams[i], padding}, 0);
            padded_diagrams.push_back(padded_diag);

            // Pad mask with false
            at::Tensor mask_padding = at::zeros(
                {max_pairs - n_pairs}, at::TensorOptions().dtype(at::kBool).device(at::kCPU));
            at::Tensor padded_mask = at::cat({batch_masks[i], mask_padding}, 0);
            padded_masks.push_back(padded_mask);
        }
        else
        {
            padded_diagrams.push_back(batch_diagrams[i]);
            padded_masks.push_back(batch_masks[i]);
        }
    }

    VRPersistenceResult result;
    result.diagrams = at::stack(padded_diagrams, 0);  // [batch, max_pairs, 3]
    result.mask = at::stack(padded_masks, 0);         // [batch, max_pairs]
    result.num_pairs = at::stack(batch_num_pairs, 0); // [batch, max_dim+1]
    result.persistence_info = batch_persistence_info;
    save_persistence_info(points.data_ptr(), batch_persistence_info);

    // Move to original device and dtype
    result.diagrams = result.diagrams.to(points.device()).to(points.dtype());
    result.mask = result.mask.to(points.device());
    result.num_pairs = result.num_pairs.to(points.device());

    if (!was_batched)
    {
        // Remove batch dimension for single input
        result.diagrams = result.diagrams.squeeze(0);
        result.mask = result.mask.squeeze(0);
        result.num_pairs = result.num_pairs.squeeze(0);
    }

    return result;
}

at::Tensor vr_persistence_backward_impl(const at::Tensor &grad_diagrams, const at::Tensor &points,
                                        const at::Tensor &birth_idx, const at::Tensor &death_idx)
{
    const at::Tensor points_cpu = points.contiguous().cpu().to(at::kDouble);
    const at::Tensor grad_cpu = grad_diagrams.contiguous().cpu().to(at::kDouble);
    (void)birth_idx;
    (void)death_idx;

    if (points_cpu.dim() == 2)
    {
        const at::Tensor grad_pairs = (grad_cpu.dim() == 3) ? grad_cpu[0] : grad_cpu;
        ExtendedPersistenceResult info = load_persistence_info_single(points_cpu.data_ptr());
        const at::Tensor grad_points =
            compute_vr_gradients(grad_pairs, points_cpu, at::Tensor(), info);
        return grad_points.to(points.device()).to(points.dtype());
    }

    TORCH_CHECK(points_cpu.dim() == 3, "points must be rank-2 or rank-3");
    TORCH_CHECK(grad_cpu.dim() == 3, "grad_diagrams must be rank-3 when points is batched");

    const int64_t batch_size = points_cpu.size(0);
    std::vector<ExtendedPersistenceResult> batched_info =
        load_persistence_info(points_cpu.data_ptr());
    TORCH_CHECK(static_cast<size_t>(batch_size) <= batched_info.size(),
                "persistence backward requires forward metadata for each batch");
    std::vector<at::Tensor> grad_batches;
    grad_batches.reserve(static_cast<size_t>(batch_size));

    for (int64_t b = 0; b < batch_size; ++b)
    {
        ExtendedPersistenceResult info = batched_info[static_cast<size_t>(b)];
        grad_batches.push_back(
            compute_vr_gradients(grad_cpu[b], points_cpu[b], at::Tensor(), info));
    }

    return at::stack(grad_batches, 0).to(points.device()).to(points.dtype());
}

// Convert VRPersistenceResult to Python tuple
py::tuple vr_persistence_forward_py(const at::Tensor &points, int64_t max_dim, double max_radius,
                                    const std::string &metric, int64_t pi_resolution,
                                    double pi_sigma)
{
    auto result =
        vr_persistence_forward_impl(points, max_dim, max_radius, metric, pi_resolution, pi_sigma);
    return py::make_tuple(result.diagrams, result.mask, result.num_pairs);
}

} // anonymous namespace
