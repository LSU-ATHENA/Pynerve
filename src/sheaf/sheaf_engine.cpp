#include "nerve/config.hpp"
#include "nerve/sheaf/gpu_sheaf.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <thread>

namespace nerve::sheaf
{

namespace
{

int checkedIntSize(size_t value, const char *label)
{
    if (value > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        throw std::length_error(label);
    }
    return static_cast<int>(value);
}

void validateEngineConfig(const SheafConfig &config)
{
    if (config.num_stalks < 0 || config.stalk_dimension < 0 || config.num_threads < 0 ||
        config.gpu_batch_size <= 0)
    {
        throw std::invalid_argument("SheafConfig contains an invalid negative or zero field");
    }
}

[[nodiscard]] size_t totalDimension(const std::vector<int> &dimensions)
{
    size_t total = 0;
    for (int dim : dimensions)
    {
        if (dim < 0)
        {
            throw std::invalid_argument("Stalk dimensions must be non-negative");
        }
        const size_t add = static_cast<size_t>(dim);
        if (add > std::numeric_limits<size_t>::max() - total)
        {
            throw std::length_error("Total stalk dimension exceeds size_t limits");
        }
        total += add;
    }
    return total;
}

[[nodiscard]] bool allDimensionsEqual(const std::vector<int> &dimensions)
{
    return dimensions.empty() || std::all_of(dimensions.begin(), dimensions.end(),
                                             [&](int dim) { return dim == dimensions.front(); });
}

} // namespace

class SheafEngine::Impl
{
public:
    explicit Impl(const SheafConfig &config)
        : config_(config)
    {
        validateEngineConfig(config_);
    }

    void build(const std::vector<Point> &positions, const std::vector<int> &dimensions)
    {
        if (positions.size() != dimensions.size())
        {
            throw std::invalid_argument("stalk_positions and stalk_dimensions sizes differ");
        }
        if (config_.num_stalks > 0 && positions.size() != static_cast<size_t>(config_.num_stalks))
        {
            throw std::invalid_argument("stalk count does not match SheafConfig::num_stalks");
        }
        if (config_.stalk_dimension > 0 &&
            !std::all_of(dimensions.begin(), dimensions.end(),
                         [&](int dim) { return dim == config_.stalk_dimension; }))
        {
            throw std::invalid_argument(
                "stalk dimensions do not match SheafConfig::stalk_dimension");
        }
        total_dimension_ = totalDimension(dimensions);
        positions_ = positions;
        dimensions_ = dimensions;
        buildStalkStorage();
        built_ = true;
    }

    [[nodiscard]] SheafResult compute(const std::vector<float> &cocycle) const
    {
        requireBuilt();
        const auto start = std::chrono::steady_clock::now();
        if (cocycle.size() != total_dimension_)
        {
            throw std::invalid_argument("cocycle length does not match total stalk dimension");
        }

        /*
         * This API does not accept restriction maps. The represented sheaf is
         * therefore the discrete sheaf with zero coboundary, so every validated
         * 0-cochain is already a cohomology representative.
         */
        SheafResult result;
        result.cohomology = cocycle;
        result.success = true;
        const auto stop = std::chrono::steady_clock::now();
        result.computation_time_ms =
            std::chrono::duration<double, std::milli>(stop - start).count();
        return result;
    }

    void apply(int from, int to, const std::vector<float> &input, std::vector<float> &output) const
    {
        requireBuilt();
        validateStalkId(from);
        validateStalkId(to);
        if (from != to)
        {
            throw std::invalid_argument(
                "No restriction map is registered between the requested stalks");
        }
        const size_t dim = static_cast<size_t>(dimensions_[static_cast<size_t>(from)]);
        if (input.size() != dim)
        {
            throw std::invalid_argument("morphism input length does not match the stalk dimension");
        }
        output = input;
    }

private:
    void buildStalkStorage()
    {
        stalks_.clear();
        if (config_.use_parallel && allDimensionsEqual(dimensions_))
        {
            parallel::ParallelSheafBuilder::SheafConfig builder_config;
            builder_config.num_stalks =
                checkedIntSize(dimensions_.size(), "stalk count exceeds int range");
            builder_config.stalk_dimension = dimensions_.empty() ? 0 : dimensions_.front();
            builder_config.num_threads = config_.num_threads;
            parallel::ParallelSheafBuilder builder(builder_config);
            builder.build();
            stalks_ = builder.getStalks();
            return;
        }

        stalks_.reserve(dimensions_.size());
        for (size_t i = 0; i < dimensions_.size(); ++i)
        {
            stalks_.emplace_back(checkedIntSize(i, "stalk id exceeds int range"), dimensions_[i]);
        }
    }

    void requireBuilt() const
    {
        if (!built_)
        {
            throw std::runtime_error("SheafEngine::buildSheaf must be called before computation");
        }
    }

    void validateStalkId(int id) const
    {
        if (id < 0 || static_cast<size_t>(id) >= dimensions_.size())
        {
            throw std::out_of_range("stalk id is out of range");
        }
    }

    SheafConfig config_;
    std::vector<Point> positions_;
    std::vector<int> dimensions_;
    std::vector<parallel::StalkData> stalks_;
    size_t total_dimension_ = 0;
    bool built_ = false;
};

SheafEngine::SheafEngine(const SheafConfig &config)
    : impl_(std::make_unique<Impl>(config))
{}

SheafEngine::~SheafEngine() = default;

void SheafEngine::buildSheaf(const std::vector<Point> &stalk_positions,
                             const std::vector<int> &stalk_dimensions)
{
    impl_->build(stalk_positions, stalk_dimensions);
}

SheafResult SheafEngine::computeCohomology(const std::vector<float> &cocycle)
{
    return impl_->compute(cocycle);
}

void SheafEngine::applyMorphism(int from_stalk, int to_stalk, const std::vector<float> &input,
                                std::vector<float> &output)
{
    impl_->apply(from_stalk, to_stalk, input, output);
}

SheafHardwareInfo detectSheafHardware()
{
    SheafHardwareInfo info;
#if defined(NERVE_HAS_CUDA)
    info.has_gpu = true;
#endif
#if defined(__AVX512F__) || defined(NERVE_HAS_AVX512)
    info.has_avx512 = true;
#endif
#if defined(__AVX2__)
    info.has_avx2 = true;
#endif
    info.num_cores = checkedIntSize(std::thread::hardware_concurrency(),
                                    "hardware concurrency exceeds int range");
    return info;
}

} // namespace nerve::sheaf
