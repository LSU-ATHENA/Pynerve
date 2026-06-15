#include "nerve/torch/persistence_diagram.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <string>
#include <tuple>

namespace nerve::torch
{

double diagram_wasserstein(const at::Tensor &input, const at::Tensor &other, double p);
double diagram_bottleneck(const at::Tensor &input, const at::Tensor &other);

namespace
{

void validate_finite_scalar(double value, const char *name)
{
    TORCH_CHECK(std::isfinite(value), name, " must be finite");
}

void validate_positive_finite_scalar(double value, const char *name)
{
    TORCH_CHECK(value > 0.0 && std::isfinite(value), name, " must be finite and positive");
}

void validate_nonnegative_finite_scalar(double value, const char *name)
{
    TORCH_CHECK(value >= 0.0 && std::isfinite(value), name, " must be finite and non-negative");
}

void validate_positive_count(int64_t value, const char *name)
{
    TORCH_CHECK(value > 0, name, " must be positive");
}

void validate_nonnegative_dimension(int64_t dim, const char *name)
{
    TORCH_CHECK(dim >= 0, name, " must be non-negative");
}

int64_t max_dimension_from(const at::Tensor &dimensions)
{
    if (dimensions.numel() == 0)
    {
        return 0;
    }
    return at::max(dimensions).item<int64_t>();
}

void validate_diagram_state(const at::Tensor &diagram, const at::Tensor &dimensions,
                            const at::Tensor &birth_idx, const at::Tensor &death_idx)
{
    TORCH_CHECK(diagram.defined(), "diagram must be defined");
    TORCH_CHECK(dimensions.defined(), "dimensions must be defined");
    TORCH_CHECK(birth_idx.defined(), "birth_idx must be defined");
    TORCH_CHECK(death_idx.defined(), "death_idx must be defined");

    TORCH_CHECK(diagram.dim() == 2 && diagram.size(1) == 2,
                "diagram must be a 2D tensor with shape [N, 2]");
    TORCH_CHECK(diagram.is_floating_point(), "diagram must be a floating-point tensor");
    TORCH_CHECK(dimensions.dim() == 1, "dimensions must be a 1D tensor");
    TORCH_CHECK(birth_idx.dim() == 1, "birth_idx must be a 1D tensor");
    TORCH_CHECK(death_idx.dim() == 1, "death_idx must be a 1D tensor");

    const int64_t num_pairs = diagram.size(0);
    TORCH_CHECK(dimensions.size(0) == num_pairs, "dimensions length must match diagram rows");
    TORCH_CHECK(birth_idx.size(0) == num_pairs, "birth_idx length must match diagram rows");
    TORCH_CHECK(death_idx.size(0) == num_pairs, "death_idx length must match diagram rows");

    TORCH_CHECK(dimensions.scalar_type() == at::kLong, "dimensions must be torch.long");
    TORCH_CHECK(birth_idx.scalar_type() == at::kLong, "birth_idx must be torch.long");
    TORCH_CHECK(death_idx.scalar_type() == at::kLong, "death_idx must be torch.long");

    const auto device = diagram.device();
    TORCH_CHECK(dimensions.device() == device, "dimensions must be on the same device as diagram");
    TORCH_CHECK(birth_idx.device() == device, "birth_idx must be on the same device as diagram");
    TORCH_CHECK(death_idx.device() == device, "death_idx must be on the same device as diagram");

    if (num_pairs == 0)
    {
        return;
    }

    const auto births = diagram.select(1, 0);
    const auto deaths = diagram.select(1, 1);
    TORCH_CHECK(at::isfinite(births).all().item<bool>(), "diagram births must be finite");
    TORCH_CHECK(!at::isnan(deaths).any().item<bool>(), "diagram deaths must not be NaN");
    const auto finite_deaths = at::isfinite(deaths);
    const auto positive_infinite_deaths = deaths == std::numeric_limits<double>::infinity();
    TORCH_CHECK((finite_deaths | positive_infinite_deaths).all().item<bool>(),
                "diagram deaths must be finite or positive infinity");
    if (finite_deaths.any().item<bool>())
    {
        TORCH_CHECK((deaths.masked_select(finite_deaths) >= births.masked_select(finite_deaths))
                        .all()
                        .item<bool>(),
                    "diagram finite deaths must be greater than or equal to births");
    }
    TORCH_CHECK((dimensions >= 0).all().item<bool>(), "dimensions must be non-negative");
    TORCH_CHECK((birth_idx >= 0).all().item<bool>(), "birth_idx must be non-negative");
    TORCH_CHECK((death_idx >= -1).all().item<bool>(),
                "death_idx must be greater than or equal to -1");
}

} // namespace

PersistenceDiagram::PersistenceDiagram()
    : max_dimension_(0)
{
    diagram_ = at::empty({0, 2}, at::TensorOptions().dtype(at::kDouble));
    dimensions_ = at::empty({0}, at::TensorOptions().dtype(at::kLong));
    birth_idx_ = at::empty({0}, at::TensorOptions().dtype(at::kLong));
    death_idx_ = at::empty({0}, at::TensorOptions().dtype(at::kLong));
}

PersistenceDiagram::PersistenceDiagram(at::Tensor diagram, at::Tensor dimensions,
                                       at::Tensor birth_idx, at::Tensor death_idx)
    : diagram_(std::move(diagram))
    , dimensions_(std::move(dimensions))
    , birth_idx_(std::move(birth_idx))
    , death_idx_(std::move(death_idx))
{
    validate_diagram_state(diagram_, dimensions_, birth_idx_, death_idx_);
    max_dimension_ = max_dimension_from(dimensions_);
}

PersistenceDiagram::PersistenceDiagram(int64_t capacity)
    : max_dimension_(0)
{
    TORCH_CHECK(capacity >= 0, "capacity must be non-negative");
    diagram_ = at::zeros({capacity, 2}, at::TensorOptions().dtype(at::kDouble));
    dimensions_ = at::zeros({capacity}, at::TensorOptions().dtype(at::kLong));
    birth_idx_ = at::zeros({capacity}, at::TensorOptions().dtype(at::kLong));
    death_idx_ = at::zeros({capacity}, at::TensorOptions().dtype(at::kLong));
}

at::Tensor PersistenceDiagram::persistence_lengths() const
{
    return diagram_.select(1, 1) - diagram_.select(1, 0);
}

PersistenceDiagram PersistenceDiagram::get_dimension(int64_t dim) const
{
    validate_nonnegative_dimension(dim, "dim");
    auto mask = dimensions_ == dim;
    auto indices = at::nonzero(mask).reshape({-1});

    if (indices.numel() == 0)
    {
        return PersistenceDiagram();
    }

    return PersistenceDiagram(
        diagram_.index_select(0, indices), dimensions_.index_select(0, indices),
        birth_idx_.index_select(0, indices), death_idx_.index_select(0, indices));
}

PersistenceDiagram PersistenceDiagram::get_finite_points() const
{
    auto deaths = diagram_.select(1, 1);
    auto mask = deaths != std::numeric_limits<double>::infinity();

    if (at::sum(mask).item<int64_t>() == 0)
    {
        return PersistenceDiagram();
    }

    auto indices = at::nonzero(mask).reshape({-1});

    return PersistenceDiagram(
        diagram_.index_select(0, indices), dimensions_.index_select(0, indices),
        birth_idx_.index_select(0, indices), death_idx_.index_select(0, indices));
}

PersistenceDiagram PersistenceDiagram::get_infinite_points() const
{
    auto deaths = diagram_.select(1, 1);
    auto mask = deaths == std::numeric_limits<double>::infinity();

    if (at::sum(mask).item<int64_t>() == 0)
    {
        return PersistenceDiagram();
    }

    auto indices = at::nonzero(mask).reshape({-1});

    return PersistenceDiagram(
        diagram_.index_select(0, indices), dimensions_.index_select(0, indices),
        birth_idx_.index_select(0, indices), death_idx_.index_select(0, indices));
}

at::Tensor PersistenceDiagram::total_persistence() const
{
    auto finite = get_finite_points();
    if (finite.num_pairs() == 0)
    {
        return at::tensor({0.0}, diagram_.options());
    }

    return at::sum(finite.persistence_lengths());
}

at::Tensor PersistenceDiagram::persistence_variance() const
{
    auto finite = get_finite_points();
    if (finite.num_pairs() == 0)
    {
        return at::tensor({0.0}, diagram_.options());
    }

    return at::sum(at::pow(finite.persistence_lengths(), 2));
}

at::Tensor PersistenceDiagram::mean_persistence_by_dimension() const
{
    at::Tensor means = at::zeros({max_dimension_ + 1}, diagram_.options());
    at::Tensor counts = at::zeros({max_dimension_ + 1}, at::TensorOptions().dtype(at::kLong));

    auto finite = get_finite_points();
    auto lengths = finite.persistence_lengths();
    auto dims = finite.dimensions_;

    for (int64_t i = 0; i < finite.num_pairs(); ++i)
    {
        int64_t dim = dims[i].item<int64_t>();
        means[dim] += lengths[i];
        counts[dim] += 1;
    }

    // Divide by counts (avoid division by zero)
    for (int64_t d = 0; d <= max_dimension_; ++d)
    {
        int64_t c = counts[d].item<int64_t>();
        if (c > 0)
        {
            means[d] /= c;
        }
    }

    return means;
}

int64_t PersistenceDiagram::betti_number(double threshold, int64_t dim) const
{
    validate_finite_scalar(threshold, "threshold");
    TORCH_CHECK(dim >= -1, "dim must be -1 or non-negative");
    int64_t count = 0;

    for (int64_t i = 0; i < num_pairs(); ++i)
    {
        int64_t pair_dim = dimensions_[i].item<int64_t>();
        if (dim >= 0 && pair_dim != dim)
            continue;

        double birth = diagram_[i][0].item<double>();
        double death = diagram_[i][1].item<double>();

        if (birth <= threshold &&
            (death > threshold || death == std::numeric_limits<double>::infinity()))
        {
            count++;
        }
    }

    return count;
}

double PersistenceDiagram::euler_characteristic(double threshold) const
{
    validate_finite_scalar(threshold, "threshold");
    double chi = 0.0;

    for (int64_t i = 0; i < num_pairs(); ++i)
    {
        double birth = diagram_[i][0].item<double>();
        double death = diagram_[i][1].item<double>();
        int64_t dim = dimensions_[i].item<int64_t>();

        if (birth <= threshold &&
            (death > threshold || death == std::numeric_limits<double>::infinity()))
        {
            chi += (dim % 2 == 0) ? 1.0 : -1.0;
        }
    }

    return chi;
}

double PersistenceDiagram::persistence_entropy() const
{
    auto finite = get_finite_points();
    auto lengths = finite.persistence_lengths();

    if (lengths.numel() == 0)
        return 0.0;

    double total = at::sum(lengths).item<double>();
    if (total == 0.0)
        return 0.0;

    // Entropy = -sum(p_i * log(p_i))
    auto probs = lengths / total;
    auto log_probs = at::log(probs);

    // Filter out zero probabilities
    auto mask = probs > 0;
    probs = probs.masked_select(mask);
    log_probs = log_probs.masked_select(mask);

    double entropy = -at::sum(probs * log_probs).item<double>();
    return entropy;
}

double PersistenceDiagram::landscape_norm(int64_t k) const
{
    TORCH_CHECK(k > 0, "k must be positive");
    auto finite = get_finite_points();
    auto lengths = finite.persistence_lengths();
    if (lengths.numel() == 0)
    {
        return 0.0;
    }
    auto heights = at::clamp_min(lengths, 0.0) * 0.5;
    if (k == 1)
    {
        return at::sum(heights).item<double>();
    }
    auto powered = at::pow(heights, static_cast<double>(k));
    return std::pow(at::sum(powered).item<double>(), 1.0 / static_cast<double>(k));
}

#include "detail/persistence_diagram_ops.inl"

} // namespace nerve::torch
