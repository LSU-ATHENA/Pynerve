#include "nerve/cpu/x86_intrinsics.hpp"
#include "nerve/sheaf/gpu_sheaf.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace nerve::sheaf::morphism
{
namespace
{
constexpr size_t kDefaultMorphismPoolBytes = 1024ULL * 1024ULL;

struct PairHash
{
    size_t operator()(const std::pair<int, int> &p) const noexcept
    {
        const uint64_t a = static_cast<uint32_t>(p.first);
        const uint64_t b = static_cast<uint32_t>(p.second);
        return static_cast<size_t>((a * 11400714819323198485ULL) ^ (b + 0x9e3779b97f4a7c15ULL));
    }
};

int checkedIntSize(size_t value, const char *label)
{
    if (value > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        throw std::length_error(label);
    }
    return static_cast<int>(value);
}
void validateSparseMorphism(const SparseMorphism &morphism)
{
    if (morphism.from_dim < 0 || morphism.to_dim < 0)
    {
        throw std::invalid_argument("SparseMorphism dimensions must be non-negative");
    }
    if (morphism.row_ptr.size() != static_cast<size_t>(morphism.to_dim + 1))
    {
        throw std::invalid_argument("SparseMorphism row_ptr size must equal to_dim + 1");
    }
    if (morphism.col_idx.size() != morphism.values.size())
    {
        throw std::invalid_argument("SparseMorphism col_idx and values sizes differ");
    }
    if (morphism.row_ptr.empty())
    {
        return;
    }
    if (morphism.row_ptr.front() != 0 ||
        morphism.row_ptr.back() !=
            checkedIntSize(morphism.values.size(), "SparseMorphism nnz exceeds int range"))
    {
        throw std::invalid_argument("SparseMorphism row_ptr endpoints are inconsistent");
    }
    for (size_t row = 0; row + 1 < morphism.row_ptr.size(); ++row)
    {
        const int begin = morphism.row_ptr[row];
        const int end = morphism.row_ptr[row + 1];
        if (begin > end || begin < 0 || static_cast<size_t>(end) > morphism.values.size())
        {
            throw std::invalid_argument("SparseMorphism row_ptr must be monotonic and in range");
        }
        for (int k = begin; k < end; ++k)
        {
            const int col = morphism.col_idx[static_cast<size_t>(k)];
            if (col < 0 || col >= morphism.from_dim)
            {
                throw std::invalid_argument("SparseMorphism column index is out of range");
            }
        }
    }
}
void accumulateResult(std::vector<float> &dst, const std::vector<float> &src)
{
    if (dst.empty())
    {
        dst = src;
        return;
    }
    if (dst.size() != src.size())
    {
        throw std::invalid_argument(
            "Batched morphism outputs for a stalk have inconsistent dimensions");
    }
    for (size_t i = 0; i < src.size(); ++i)
    {
        dst[i] += src[i];
    }
}
} // namespace
void SparseMorphism::apply(const std::vector<float> &input, std::vector<float> &output) const
{
    validateSparseMorphism(*this);
    if (input.size() < static_cast<size_t>(from_dim))
    {
        throw std::invalid_argument("SparseMorphism input is smaller than from_dim");
    }
    output.assign(static_cast<size_t>(to_dim), 0.0f);
    for (int row = 0; row < to_dim; ++row)
    {
        float sum = 0.0f;
        for (int k = row_ptr[static_cast<size_t>(row)]; k < row_ptr[static_cast<size_t>(row + 1)];
             ++k)
        {
            const size_t idx = static_cast<size_t>(k);
            sum += values[idx] * input[static_cast<size_t>(col_idx[idx])];
        }
        output[static_cast<size_t>(row)] = sum;
    }
}

void SparseMorphism::applySIMD(const float *input, float *output) const
{
    validateSparseMorphism(*this);
    if ((input == nullptr || output == nullptr) && to_dim > 0)
    {
        throw std::invalid_argument(
            "SparseMorphism SIMD input and output pointers must be non-null");
    }
#if defined(__AVX512F__)
    constexpr int kWidth = 16;
    alignas(64) int indices[kWidth];
    for (int row = 0; row < to_dim; ++row)
    {
        __m512 sum = _mm512_setzero_ps();
        int k = row_ptr[static_cast<size_t>(row)];
        const int end = row_ptr[static_cast<size_t>(row + 1)];
        for (; k + kWidth <= end; k += kWidth)
        {
            for (int lane = 0; lane < kWidth; ++lane)
            {
                indices[lane] = col_idx[static_cast<size_t>(k + lane)];
            }
            const __m512 vals = _mm512_loadu_ps(&values[static_cast<size_t>(k)]);
            const __m512i cols = _mm512_load_si512(reinterpret_cast<const __m512i *>(indices));
            const __m512 gathered = _mm512_i32gather_ps(cols, input, sizeof(float));
            sum = _mm512_add_ps(sum, _mm512_mul_ps(vals, gathered));
        }
        float total = _mm512_reduce_add_ps(sum);
        for (; k < end; ++k)
        {
            const size_t idx = static_cast<size_t>(k);
            total += values[idx] * input[static_cast<size_t>(col_idx[idx])];
        }
        output[static_cast<size_t>(row)] = total;
    }
#else
    for (int row = 0; row < to_dim; ++row)
    {
        float sum = 0.0f;
        for (int k = row_ptr[static_cast<size_t>(row)]; k < row_ptr[static_cast<size_t>(row + 1)];
             ++k)
        {
            const size_t idx = static_cast<size_t>(k);
            sum += values[idx] * input[static_cast<size_t>(col_idx[idx])];
        }
        output[static_cast<size_t>(row)] = sum;
    }
#endif
}

class MorphismMemoryPool::Impl
{
public:
    explicit Impl(size_t initial_size)
    {
        pool_.resize(std::max<size_t>(initial_size / sizeof(float), 1));
    }

    float *allocate(int nnz)
    {
        if (nnz < 0)
        {
            throw std::invalid_argument("Morphism allocation size must be non-negative");
        }
        const size_t required = static_cast<size_t>(nnz);
        if (offset_ + required > pool_.size())
        {
            pool_.resize(std::max(pool_.size() * 2, offset_ + required));
        }
        float *ptr = pool_.data() + offset_;
        offset_ += required;
        return ptr;
    }

    void reset() noexcept { offset_ = 0; }

private:
    std::vector<float> pool_;
    size_t offset_ = 0;
};

MorphismMemoryPool::MorphismMemoryPool(size_t initial_size)
    : impl_(std::make_unique<Impl>(initial_size == 0 ? kDefaultMorphismPoolBytes : initial_size))
{}

MorphismMemoryPool::~MorphismMemoryPool() = default;

float *MorphismMemoryPool::allocateMorphism(int nnz)
{
    return impl_->allocate(nnz);
}

void MorphismMemoryPool::reset()
{
    impl_->reset();
}

class BatchedMorphismComputer::Impl
{
public:
    explicit Impl(int cache_block_size)
        : cache_block_size_(std::max(cache_block_size, 1))
    {}

    void add(int from_stalk, int to_stalk, const SparseMorphism &morphism)
    {
        validateSparseMorphism(morphism);
        morphisms_[{from_stalk, to_stalk}] = morphism;
    }

    void compute(const std::vector<int> &order, const std::vector<std::vector<float>> &data,
                 std::vector<std::vector<float>> &output) const
    {
        if (order.size() != data.size())
        {
            throw std::invalid_argument("stalk_order and stalk_data sizes differ");
        }
        output.assign(order.size(), {});
        const auto index = buildIndex(order);
        for (size_t block = 0; block < order.size();
             block += static_cast<size_t>(cache_block_size_))
        {
            const size_t last =
                std::min(order.size(), block + static_cast<size_t>(cache_block_size_));
            for (size_t target_idx = block; target_idx < last; ++target_idx)
            {
                computeTarget(order[target_idx], target_idx, index, data, output);
            }
        }
    }

    void computeSIMD(const std::vector<int> &order, const std::vector<float *> &data,
                     std::vector<float *> &output, int num_stalks) const
    {
        if (num_stalks < 0 || order.size() < static_cast<size_t>(num_stalks) ||
            data.size() < static_cast<size_t>(num_stalks) ||
            output.size() < static_cast<size_t>(num_stalks))
        {
            throw std::invalid_argument("SIMD batch inputs are smaller than num_stalks");
        }
        const auto index = buildIndex(order);
        for (int i = 0; i < num_stalks; ++i)
        {
            const int stalk_id = order[static_cast<size_t>(i)];
            const size_t dim = inferOutputDim(stalk_id);
            if (dim == 0)
            {
                continue;
            }
            if (output[static_cast<size_t>(i)] == nullptr)
            {
                throw std::invalid_argument("SIMD batch output pointer is null");
            }
            std::fill_n(output[static_cast<size_t>(i)], dim, 0.0f);
            computeTargetSIMD(stalk_id, index, data, output[static_cast<size_t>(i)], dim);
        }
    }

private:
    using Index = std::unordered_map<int, size_t>;

    static Index buildIndex(const std::vector<int> &order)
    {
        Index index;
        index.reserve(order.size());
        for (size_t i = 0; i < order.size(); ++i)
        {
            index.emplace(order[i], i);
        }
        return index;
    }

    void computeTarget(int target_id, size_t target_idx, const Index &index,
                       const std::vector<std::vector<float>> &data,
                       std::vector<std::vector<float>> &output) const
    {
        for (const auto &[key, morphism] : morphisms_)
        {
            if (key.second != target_id)
            {
                continue;
            }
            auto source = index.find(key.first);
            if (source == index.end())
            {
                continue;
            }
            std::vector<float> result;
            morphism.apply(data[source->second], result);
            accumulateResult(output[target_idx], result);
        }
    }

    void computeTargetSIMD(int target_id, const Index &index, const std::vector<float *> &data,
                           float *output, size_t output_dim) const
    {
        for (const auto &[key, morphism] : morphisms_)
        {
            if (key.second != target_id)
            {
                continue;
            }
            auto source = index.find(key.first);
            if (source == index.end())
            {
                continue;
            }
            if (data[source->second] == nullptr)
            {
                throw std::invalid_argument("SIMD batch input pointer is null");
            }
            std::vector<float> temp(static_cast<size_t>(morphism.to_dim), 0.0f);
            morphism.applySIMD(data[source->second], temp.data());
            for (size_t i = 0; i < std::min(output_dim, temp.size()); ++i)
            {
                output[i] += temp[i];
            }
        }
    }

    size_t inferOutputDim(int stalk_id) const
    {
        size_t dim = 0;
        for (const auto &[key, morphism] : morphisms_)
        {
            if (key.second == stalk_id)
            {
                dim = std::max(dim, static_cast<size_t>(morphism.to_dim));
            }
        }
        return dim;
    }

    int cache_block_size_;
    std::unordered_map<std::pair<int, int>, SparseMorphism, PairHash> morphisms_;
};

BatchedMorphismComputer::BatchedMorphismComputer(int cache_block_size)
    : impl_(std::make_unique<Impl>(cache_block_size))
{}

BatchedMorphismComputer::~BatchedMorphismComputer() = default;

void BatchedMorphismComputer::addMorphism(int from_stalk, int to_stalk,
                                          const SparseMorphism &morphism)
{
    impl_->add(from_stalk, to_stalk, morphism);
}

void BatchedMorphismComputer::computeBatch(const std::vector<int> &stalk_order,
                                           const std::vector<std::vector<float>> &stalk_data,
                                           std::vector<std::vector<float>> &output_data)
{
    impl_->compute(stalk_order, stalk_data, output_data);
}

void BatchedMorphismComputer::computeBatchSIMD(const std::vector<int> &stalk_order,
                                               const std::vector<float *> &stalk_data,
                                               std::vector<float *> &output_data, int num_stalks)
{
    impl_->computeSIMD(stalk_order, stalk_data, output_data, num_stalks);
}

class MorphismCompositionOptimizer::Impl
{
public:
    void add(int from, int to, const SparseMorphism &morphism)
    {
        validateSparseMorphism(morphism);
        morphisms_[{from, to}] = morphism;
    }

    void registerChain(const std::vector<int> &chain)
    {
        if (chain.size() < 2)
        {
            return;
        }
        auto current = getSingle(chain[0], chain[1]);
        for (size_t i = 1; i + 1 < chain.size(); ++i)
        {
            current = compose(current, getSingle(chain[i], chain[i + 1]));
        }
        composed_[{chain.front(), chain.back()}] = std::move(current);
    }

    [[nodiscard]] SparseMorphism get(int from, int to) const
    {
        auto composed = composed_.find({from, to});
        if (composed != composed_.end())
        {
            return composed->second;
        }
        auto single = morphisms_.find({from, to});
        return single == morphisms_.end() ? SparseMorphism{} : single->second;
    }

private:
    SparseMorphism getSingle(int from, int to) const
    {
        auto it = morphisms_.find({from, to});
        if (it == morphisms_.end())
        {
            throw std::invalid_argument("Cannot compose a chain with a missing morphism");
        }
        return it->second;
    }

    static SparseMorphism compose(const SparseMorphism &f, const SparseMorphism &g)
    {
        validateSparseMorphism(f);
        validateSparseMorphism(g);
        if (f.to_dim != g.from_dim)
        {
            throw std::invalid_argument("Morphism composition dimension mismatch");
        }

        SparseMorphism result;
        result.from_dim = f.from_dim;
        result.to_dim = g.to_dim;
        result.row_ptr.reserve(static_cast<size_t>(g.to_dim + 1));
        result.row_ptr.push_back(0);

        for (int row = 0; row < g.to_dim; ++row)
        {
            std::unordered_map<int, float> accum;
            for (int kg = g.row_ptr[static_cast<size_t>(row)];
                 kg < g.row_ptr[static_cast<size_t>(row + 1)]; ++kg)
            {
                const int mid = g.col_idx[static_cast<size_t>(kg)];
                const float gv = g.values[static_cast<size_t>(kg)];
                for (int kf = f.row_ptr[static_cast<size_t>(mid)];
                     kf < f.row_ptr[static_cast<size_t>(mid + 1)]; ++kf)
                {
                    accum[f.col_idx[static_cast<size_t>(kf)]] +=
                        gv * f.values[static_cast<size_t>(kf)];
                }
            }

            std::vector<std::pair<int, float>> entries(accum.begin(), accum.end());
            std::sort(entries.begin(), entries.end());
            for (const auto &[col, value] : entries)
            {
                if (value != 0.0f)
                {
                    result.col_idx.push_back(col);
                    result.values.push_back(value);
                }
            }
            result.row_ptr.push_back(
                checkedIntSize(result.values.size(), "composed morphism nnz exceeds int range"));
        }
        return result;
    }

    std::unordered_map<std::pair<int, int>, SparseMorphism, PairHash> morphisms_;
    std::unordered_map<std::pair<int, int>, SparseMorphism, PairHash> composed_;
};

MorphismCompositionOptimizer::MorphismCompositionOptimizer()
    : impl_(std::make_unique<Impl>())
{}

MorphismCompositionOptimizer::~MorphismCompositionOptimizer() = default;

void MorphismCompositionOptimizer::registerChain(const std::vector<int> &stalk_chain)
{
    impl_->registerChain(stalk_chain);
}

SparseMorphism MorphismCompositionOptimizer::getComposed(int from, int to) const
{
    return impl_->get(from, to);
}

void MorphismCompositionOptimizer::addMorphism(int from, int to, const SparseMorphism &morphism)
{
    impl_->add(from, to, morphism);
}

} // namespace nerve::sheaf::morphism
