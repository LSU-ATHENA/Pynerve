at::Tensor PersistenceDiagram::to_persistence_image(int64_t resolution, double sigma,
                                                    double birth_min, double birth_max,
                                                    double death_min, double death_max) const
{
    validate_positive_count(resolution, "resolution");
    validate_positive_finite_scalar(sigma, "sigma");
    validate_finite_scalar(birth_min, "birth_min");
    validate_finite_scalar(birth_max, "birth_max");
    validate_finite_scalar(death_min, "death_min");
    validate_finite_scalar(death_max, "death_max");

    if (num_pairs() == 0)
    {
        return at::zeros({resolution, resolution}, diagram_.options());
    }

    auto births = diagram_.select(1, 0);
    auto deaths = diagram_.select(1, 1);

    if (birth_max < birth_min)
    {
        birth_min = at::min(births).item<double>();
        birth_max = at::max(births).item<double>();
    }
    if (death_max < death_min)
    {
        auto finite_deaths =
            deaths.masked_select(deaths != std::numeric_limits<double>::infinity());
        if (finite_deaths.numel() > 0)
        {
            death_min = at::min(finite_deaths).item<double>();
            death_max = at::max(finite_deaths).item<double>();
        }
        else
        {
            death_min = birth_min;
            death_max = birth_max;
        }
    }

    auto birth_grid = at::linspace(birth_min, birth_max, resolution, diagram_.options());
    auto death_grid = at::linspace(death_min, death_max, resolution, diagram_.options());
    at::Tensor image = at::zeros({resolution, resolution}, diagram_.options());
    auto finite = get_finite_points();

    for (int64_t i = 0; i < finite.num_pairs(); ++i)
    {
        double b = finite.diagram_[i][0].item<double>();
        double d = finite.diagram_[i][1].item<double>();
        auto birth_diff = birth_grid - b;
        auto death_diff = death_grid - d;
        at::Tensor gaussian =
            at::exp(-(birth_diff.unsqueeze(1).pow(2) + death_diff.unsqueeze(0).pow(2)) /
                    (2 * sigma * sigma));
        image += gaussian * (d - b);
    }

    return image;
}

at::Tensor PersistenceDiagram::to_persistence_landscape(int64_t k, int64_t num_samples,
                                                        double min_val, double max_val) const
{
    validate_positive_count(k, "k");
    validate_positive_count(num_samples, "num_samples");
    validate_finite_scalar(min_val, "min_val");
    validate_finite_scalar(max_val, "max_val");

    auto finite = get_finite_points();
    if (finite.num_pairs() == 0)
    {
        return at::zeros({k, num_samples}, diagram_.options());
    }

    auto births = finite.diagram_.select(1, 0);
    auto deaths = finite.diagram_.select(1, 1);

    if (max_val < min_val)
    {
        min_val = at::min(births).item<double>();
        max_val = at::max(deaths).item<double>();
    }
    if (max_val <= min_val)
    {
        max_val = min_val + 1.0;
    }

    auto x = at::linspace(min_val, max_val, num_samples, diagram_.options());

    // Each persistence pair contributes a triangular tent function.
    auto mids = (births + deaths) * 0.5;
    auto heights = at::clamp_min(deaths - births, 0.0) * 0.5;
    at::Tensor triangles =
        at::clamp_min(heights.unsqueeze(1) - at::abs(x.unsqueeze(0) - mids.unsqueeze(1)), 0.0);

    auto sorted = std::get<0>(at::sort(triangles, 0, true));
    at::Tensor landscape = at::zeros({k, num_samples}, diagram_.options());
    const int64_t rows = std::min<int64_t>(k, sorted.size(0));
    if (rows > 0)
    {
        landscape.slice(0, 0, rows).copy_(sorted.slice(0, 0, rows));
    }
    return landscape;
}

at::Tensor PersistenceDiagram::to_betti_curve(int64_t num_samples, double min_val,
                                              double max_val) const
{
    validate_positive_count(num_samples, "num_samples");
    validate_finite_scalar(min_val, "min_val");
    validate_finite_scalar(max_val, "max_val");

    if (num_pairs() == 0)
    {
        return at::zeros({num_samples}, diagram_.options());
    }

    auto births = diagram_.select(1, 0);
    auto deaths = diagram_.select(1, 1);

    if (max_val < min_val)
    {
        min_val = at::min(births).item<double>();
        auto finite_deaths =
            deaths.masked_select(deaths != std::numeric_limits<double>::infinity());
        max_val = finite_deaths.numel() > 0 ? at::max(finite_deaths).item<double>()
                                            : at::max(births).item<double>();
    }

    auto x = at::linspace(min_val, max_val, num_samples, diagram_.options());
    at::Tensor betti = at::zeros({num_samples}, at::TensorOptions().dtype(at::kLong));

    for (int64_t i = 0; i < num_samples; ++i)
    {
        betti[i] = betti_number(x[i].item<double>());
    }

    return betti.to(diagram_.options().dtype(at::kDouble));
}

at::Tensor PersistenceDiagram::to_vector(int64_t bins_per_dim) const
{
    TORCH_CHECK(bins_per_dim > 0, "bins_per_dim must be positive");
    auto finite = get_finite_points();
    if (finite.num_pairs() == 0)
    {
        return at::zeros({bins_per_dim * bins_per_dim}, diagram_.options());
    }

    auto births = finite.diagram_.select(1, 0);
    auto deaths = finite.diagram_.select(1, 1);

    double b_min = at::min(births).item<double>();
    double b_max = at::max(births).item<double>();
    double d_min = at::min(deaths).item<double>();
    double d_max = at::max(deaths).item<double>();

    at::Tensor vec = at::zeros({bins_per_dim * bins_per_dim}, diagram_.options());
    double b_step = (b_max - b_min) / static_cast<double>(bins_per_dim);
    double d_step = (d_max - d_min) / static_cast<double>(bins_per_dim);
    if (b_step <= 0.0)
    {
        b_step = 1.0;
    }
    if (d_step <= 0.0)
    {
        d_step = 1.0;
    }

    for (int64_t i = 0; i < finite.num_pairs(); ++i)
    {
        double b = births[i].item<double>();
        double d = deaths[i].item<double>();
        int64_t b_bin = static_cast<int64_t>((b - b_min) / b_step);
        int64_t d_bin = static_cast<int64_t>((d - d_min) / d_step);
        b_bin = std::clamp(b_bin, int64_t(0), bins_per_dim - 1);
        d_bin = std::clamp(d_bin, int64_t(0), bins_per_dim - 1);
        vec[b_bin * bins_per_dim + d_bin] += 1.0;
    }

    return vec;
}

double PersistenceDiagram::wasserstein_distance(const PersistenceDiagram &other, double p) const
{
    TORCH_CHECK(p >= 1.0 && std::isfinite(p), "p must be finite and >= 1 for Wasserstein distance");
    const auto d1_finite = get_finite_points();
    const auto d2_finite = other.get_finite_points();
    return diagram_wasserstein(d1_finite.diagram_, d2_finite.diagram_, p);
}

double PersistenceDiagram::bottleneck_distance(const PersistenceDiagram &other) const
{
    const auto d1_finite = get_finite_points();
    const auto d2_finite = other.get_finite_points();
    return diagram_bottleneck(d1_finite.diagram_, d2_finite.diagram_);
}

double PersistenceDiagram::persistence_kernel(const PersistenceDiagram &other, double sigma) const
{
    validate_positive_finite_scalar(sigma, "sigma");

    auto diag1 = diagram_;
    auto diag2 = other.diagram_;
    double kernel_val = 0.0;

    for (int64_t i = 0; i < num_pairs(); ++i)
    {
        for (int64_t j = 0; j < other.num_pairs(); ++j)
        {
            double b1 = diag1[i][0].item<double>();
            double d1 = diag1[i][1].item<double>();
            double b2 = diag2[j][0].item<double>();
            double d2 = diag2[j][1].item<double>();
            double dist_sq = std::pow(b1 - b2, 2) + std::pow(d1 - d2, 2);
            kernel_val += std::exp(-dist_sq / (2 * sigma * sigma));
        }
    }

    return kernel_val;
}

void PersistenceDiagram::append(const PersistenceDiagram &other)
{
    diagram_ = at::cat({diagram_, other.diagram_}, 0);
    dimensions_ = at::cat({dimensions_, other.dimensions_}, 0);
    birth_idx_ = at::cat({birth_idx_, other.birth_idx_}, 0);
    death_idx_ = at::cat({death_idx_, other.death_idx_}, 0);
    if (other.max_dimension_ > max_dimension_)
    {
        max_dimension_ = other.max_dimension_;
    }
}

PersistenceDiagram PersistenceDiagram::threshold(double min_persistence) const
{
    validate_nonnegative_finite_scalar(min_persistence, "min_persistence");
    auto lengths = persistence_lengths();
    auto mask = lengths >= min_persistence;
    auto indices = at::nonzero(mask).reshape({-1});
    if (indices.numel() == 0)
    {
        return PersistenceDiagram();
    }
    return PersistenceDiagram(
        diagram_.index_select(0, indices), dimensions_.index_select(0, indices),
        birth_idx_.index_select(0, indices), death_idx_.index_select(0, indices));
}

PersistenceDiagram PersistenceDiagram::filter_by_dimension(int64_t dim) const
{
    validate_nonnegative_dimension(dim, "dim");
    return get_dimension(dim);
}

void PersistenceDiagram::sort_by_persistence()
{
    auto lengths = persistence_lengths();
    auto sorted = at::sort(lengths, 0, true);
    auto indices = std::get<1>(sorted);
    diagram_ = diagram_.index_select(0, indices);
    dimensions_ = dimensions_.index_select(0, indices);
    birth_idx_ = birth_idx_.index_select(0, indices);
    death_idx_ = death_idx_.index_select(0, indices);
}

void PersistenceDiagram::normalize()
{
    if (num_pairs() == 0)
    {
        return;
    }

    auto births = diagram_.select(1, 0);
    auto deaths = diagram_.select(1, 1);

    double b_min = at::min(births).item<double>();
    double b_max = at::max(births).item<double>();
    double b_range = b_max - b_min;
    if (b_range > 0)
    {
        births.copy_((births - b_min) / b_range);
    }

    auto finite_mask = deaths != std::numeric_limits<double>::infinity();
    auto finite_deaths = deaths.masked_select(finite_mask);
    if (finite_deaths.numel() > 0)
    {
        double d_min = at::min(finite_deaths).item<double>();
        double d_max = at::max(finite_deaths).item<double>();
        double d_range = d_max - d_min;
        if (d_range > 0)
        {
            auto normalized = (finite_deaths - d_min) / d_range;
            deaths.index_put_({finite_mask}, normalized);
        }
    }
}

void PersistenceDiagram::to(at::Device device)
{
    diagram_ = diagram_.to(device);
    dimensions_ = dimensions_.to(device);
    birth_idx_ = birth_idx_.to(device);
    death_idx_ = death_idx_.to(device);
}

PersistenceDiagram PersistenceDiagram::batch(const std::vector<PersistenceDiagram> &diagrams)
{
    if (diagrams.empty())
    {
        return PersistenceDiagram();
    }

    std::vector<at::Tensor> diagram_list;
    std::vector<at::Tensor> dims_list;
    std::vector<at::Tensor> birth_list;
    std::vector<at::Tensor> death_list;

    for (const auto &d : diagrams)
    {
        diagram_list.push_back(d.diagram_);
        dims_list.push_back(d.dimensions_);
        birth_list.push_back(d.birth_idx_);
        death_list.push_back(d.death_idx_);
    }

    PersistenceDiagram result;
    result.diagram_ = at::cat(diagram_list, 0);
    result.dimensions_ = at::cat(dims_list, 0);
    result.birth_idx_ = at::cat(birth_list, 0);
    result.death_idx_ = at::cat(death_list, 0);

    int64_t max_dim = 0;
    for (const auto &d : diagrams)
    {
        max_dim = std::max(max_dim, d.max_dimension_);
    }
    result.max_dimension_ = max_dim;
    std::vector<int64_t> pair_counts;
    pair_counts.reserve(diagrams.size());
    for (const auto &d : diagrams)
    {
        pair_counts.push_back(d.num_pairs());
    }
    result.birth_simplices_ = at::tensor(pair_counts, at::TensorOptions().dtype(at::kLong));
    return result;
}

PersistenceDiagram PersistenceDiagram::get_batch_item(int64_t idx) const
{
    TORCH_CHECK(is_batched(), "diagram is not batched");
    auto counts = birth_simplices_.cpu();
    auto counts_acc = counts.accessor<int64_t, 1>();
    const int64_t bsz = counts.size(0);
    TORCH_CHECK(idx >= 0 && idx < bsz, "batch index out of bounds");

    int64_t offset = 0;
    for (int64_t i = 0; i < idx; ++i)
    {
        offset += counts_acc[i];
    }
    const int64_t count = counts_acc[idx];
    if (count <= 0)
    {
        return PersistenceDiagram();
    }

    return PersistenceDiagram(
        diagram_.narrow(0, offset, count).clone(), dimensions_.narrow(0, offset, count).clone(),
        birth_idx_.narrow(0, offset, count).clone(), death_idx_.narrow(0, offset, count).clone());
}

bool PersistenceDiagram::is_batched() const
{
    if (!birth_simplices_.defined() || birth_simplices_.dim() != 1 || birth_simplices_.numel() == 0)
    {
        return false;
    }
    const int64_t total = at::sum(birth_simplices_).item<int64_t>();
    return total == num_pairs() && birth_simplices_.size(0) > 1;
}

int64_t PersistenceDiagram::batch_size() const
{
    return is_batched() ? birth_simplices_.size(0) : 1;
}

std::vector<std::pair<std::string, at::Tensor>> PersistenceDiagram::state_dict() const
{
    return {{"diagram", diagram_},
            {"dimensions", dimensions_},
            {"birth_idx", birth_idx_},
            {"death_idx", death_idx_}};
}

void PersistenceDiagram::load_state_dict(
    const std::vector<std::pair<std::string, at::Tensor>> &state_dict)
{
    at::Tensor diagram;
    at::Tensor dimensions;
    at::Tensor birth_idx;
    at::Tensor death_idx;
    bool has_diagram = false;
    bool has_dimensions = false;
    bool has_birth_idx = false;
    bool has_death_idx = false;

    for (const auto &[key, tensor] : state_dict)
    {
        if (key == "diagram")
        {
            TORCH_CHECK(!has_diagram, "state_dict contains duplicate diagram");
            diagram = tensor;
            has_diagram = true;
        }
        else if (key == "dimensions")
        {
            TORCH_CHECK(!has_dimensions, "state_dict contains duplicate dimensions");
            dimensions = tensor;
            has_dimensions = true;
        }
        else if (key == "birth_idx")
        {
            TORCH_CHECK(!has_birth_idx, "state_dict contains duplicate birth_idx");
            birth_idx = tensor;
            has_birth_idx = true;
        }
        else if (key == "death_idx")
        {
            TORCH_CHECK(!has_death_idx, "state_dict contains duplicate death_idx");
            death_idx = tensor;
            has_death_idx = true;
        }
        else
        {
            TORCH_CHECK(false, "state_dict contains unknown key: ", key);
        }
    }
    TORCH_CHECK(has_diagram, "state_dict missing diagram");
    TORCH_CHECK(has_dimensions, "state_dict missing dimensions");
    TORCH_CHECK(has_birth_idx, "state_dict missing birth_idx");
    TORCH_CHECK(has_death_idx, "state_dict missing death_idx");

    validate_diagram_state(diagram, dimensions, birth_idx, death_idx);
    diagram_ = std::move(diagram);
    dimensions_ = std::move(dimensions);
    birth_idx_ = std::move(birth_idx);
    death_idx_ = std::move(death_idx);
    birth_simplices_ = at::Tensor();
    death_simplices_ = at::Tensor();
    max_dimension_ = max_dimension_from(dimensions_);
}

void PersistenceDiagram::set_requires_grad(bool requires_grad)
{
    diagram_.set_requires_grad(requires_grad);
}

void PersistenceDiagram::backward() const
{
    if (diagram_.requires_grad())
    {
        diagram_.sum().backward();
    }
}

at::Tensor PersistenceDiagram::births_grad() const
{
    if (!diagram_.requires_grad())
    {
        return at::zeros({num_pairs()}, diagram_.options());
    }
    at::Tensor grad = diagram_.grad();
    if (!grad.defined())
    {
        return at::zeros({num_pairs()}, diagram_.options());
    }
    return grad.select(1, 0).clone();
}

at::Tensor PersistenceDiagram::deaths_grad() const
{
    if (!diagram_.requires_grad())
    {
        return at::zeros({num_pairs()}, diagram_.options());
    }
    at::Tensor grad = diagram_.grad();
    if (!grad.defined())
    {
        return at::zeros({num_pairs()}, diagram_.options());
    }
    return grad.select(1, 1).clone();
}
