#include "nerve/streaming/streaming_laplacian.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <thread>

#ifdef NERVE_HAS_CUDA
#include "nerve/spectral/laplacian.hpp"
#include "nerve/spectral/persistent_laplacian.hpp"

#include <Eigen/Sparse>
#endif

namespace nerve::streaming
{

std::vector<double> lanczosIteration(const LaplacianState &L, Size k, double tolerance);

IncrementalGraphLaplacian::IncrementalGraphLaplacian(const IncrementalLaplacianConfig &config)
    : config_(config)
    , state_{}
{
    state_.is_valid = false;
    state_.dimension = 0;
    state_.matrix_size = 0;
    state_.birth_time = 0.0;
}

IncrementalGraphLaplacian::IncrementalGraphLaplacian()
    : IncrementalGraphLaplacian(IncrementalLaplacianConfig{})
{}

IncrementalGraphLaplacian::~IncrementalGraphLaplacian() = default;

void IncrementalGraphLaplacian::addVertex(Index vertex_id)
{
    std::lock_guard<std::mutex> lock(state_mutex_);
    while (vertex_id >= static_cast<Index>(vertex_weights_.size()))
    {
        vertex_weights_.push_back(0.0);
    }
    degree_map_.try_emplace(vertex_id, 0.0);
}

void IncrementalGraphLaplacian::removeVertex(Index vertex_id)
{
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (vertex_id >= static_cast<Index>(vertex_weights_.size()))
        return;
    vertex_weights_[vertex_id] = 0.0;
    degree_map_.erase(vertex_id);
    state_.is_valid = false;
}

Size IncrementalGraphLaplacian::edgeIndex(Index u, Index v) const
{
    Size base = state_.column_start[u];
    for (Size i = 0; i < static_cast<Size>(state_.column_start[u + 1] - state_.column_start[u]);
         ++i)
    {
        if (state_.row_index[base + i] == static_cast<Index>(v))
            return base + i;
    }
    return static_cast<Size>(-1);
}

void IncrementalGraphLaplacian::updateDiagonal(Index idx, double delta)
{
    state_.diagonal[idx] += delta;
}

void IncrementalGraphLaplacian::updateOffDiagonal(Index u, Index v, double delta)
{
    Size e = edgeIndex(u, v);
    if (e < state_.values.size())
    {
        state_.values[e] += delta;
    }
    e = edgeIndex(v, u);
    if (e < state_.values.size())
    {
        state_.values[e] += delta;
    }
}

void IncrementalGraphLaplacian::addEdge(Index u, Index v, double weight)
{
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (u == v)
        return;
    if (!state_.is_valid)
    {
        rebuildMatrix();
        state_.is_valid = true;
    }
    updateDiagonal(u, weight);
    updateDiagonal(v, weight);
    updateOffDiagonal(u, v, -weight);
    degree_map_[u] += weight;
    degree_map_[v] += weight;
}

void IncrementalGraphLaplacian::removeEdge(Index u, Index v)
{
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (u == v || !state_.is_valid)
        return;
    double w = 0.0;
    Size e = edgeIndex(u, v);
    if (e < state_.values.size())
    {
        w = -state_.values[e];
    }
    updateDiagonal(u, -w);
    updateDiagonal(v, -w);
    updateOffDiagonal(u, v, w);
    degree_map_[u] -= w;
    degree_map_[v] -= w;
}

void IncrementalGraphLaplacian::rebuildMatrix()
{
    state_.matrix_size = vertex_weights_.size();
    state_.dimension = static_cast<Index>(state_.matrix_size);
    state_.diagonal.assign(state_.matrix_size, 0.0);
    Size nnz = 0;
    state_.column_start.assign(state_.matrix_size + 1, 0);
    for (Size i = 0; i < state_.matrix_size; ++i)
    {
        state_.column_start[i] = nnz;
        for (Size j = 0; j < state_.matrix_size; ++j)
        {
            if (i != j)
            {
                state_.row_index.push_back(static_cast<Index>(j));
                state_.values.push_back(0.0);
                ++nnz;
            }
        }
    }
    state_.column_start[state_.matrix_size] = nnz;
}

const LaplacianState &IncrementalGraphLaplacian::getState() const
{
    return state_;
}

LaplacianSpectrum IncrementalGraphLaplacian::computeSpectrum() const
{
    LaplacianSpectrum result;
    if (!state_.is_valid || state_.matrix_size == 0)
        return result;

#ifdef NERVE_HAS_CUDA
    if (state_.matrix_size >= 256)
    {
        Eigen::SparseMatrix<double> L(state_.matrix_size, state_.matrix_size);
        std::vector<Eigen::Triplet<double>> triplets;
        triplets.reserve(state_.matrix_size + state_.values.size());
        for (int i = 0; i < static_cast<int>(state_.matrix_size); ++i)
            triplets.emplace_back(i, i, state_.diagonal[i]);
        for (int i = 0; i < static_cast<int>(state_.matrix_size); ++i)
        {
            for (int j = state_.column_start[i]; j < state_.column_start[i + 1]; ++j)
                triplets.emplace_back(i, state_.row_index[j], state_.values[j]);
        }
        L.setFromTriplets(triplets.begin(), triplets.end());

        auto &stack = nerve::spectral::SpectralStackManager::instance();
        auto spec = stack.computeSpectrum(L);
        if (!spec.eigenvalues.empty())
        {
            result.eigenvalues = spec.eigenvalues;
            result.trace = std::accumulate(state_.diagonal.begin(), state_.diagonal.end(), 0.0);
            double fn_sq = 0.0;
            for (double v : state_.diagonal)
                fn_sq += v * v;
            for (double v : state_.values)
                fn_sq += v * v;
            result.frobenius_norm = std::sqrt(fn_sq);
            result.rank = 0;
            for (double ev : result.eigenvalues)
                if (std::fabs(ev) > config_.eigenvalue_tolerance)
                    ++result.rank;
            return result;
        }
    }
#endif

    Size k = std::min(state_.matrix_size, Size{16});
    result.eigenvalues = lanczosIteration(state_, k, config_.eigenvalue_tolerance);
    std::sort(result.eigenvalues.begin(), result.eigenvalues.end());
    result.trace = std::accumulate(state_.diagonal.begin(), state_.diagonal.end(), 0.0);
    double fn_sq = 0.0;
    for (double v : state_.diagonal)
        fn_sq += v * v;
    for (double v : state_.values)
        fn_sq += v * v;
    result.frobenius_norm = std::sqrt(fn_sq);
    result.rank = 0;
    for (double ev : result.eigenvalues)
    {
        if (std::fabs(ev) > config_.eigenvalue_tolerance)
            ++result.rank;
    }
    return result;
}

double IncrementalGraphLaplacian::computeSpectralGap() const
{
    auto spectrum = computeSpectrum();
    if (spectrum.eigenvalues.size() < 2)
        return 0.0;
    return spectrum.eigenvalues[1] - spectrum.eigenvalues[0];
}

std::vector<double> IncrementalGraphLaplacian::computeFiedlerVector() const
{
    if (!state_.is_valid || state_.matrix_size == 0)
        return {};
    Size n = state_.matrix_size;
    std::vector<double> v(n, 1.0 / std::sqrt(static_cast<double>(n)));
    std::vector<double> w(n);
    constexpr Size max_iters = 1000;
    constexpr double tol = 1e-8;
    double lambda = 0.0;
    for (Size iter = 0; iter < max_iters; ++iter)
    {
        w.assign(n, 0.0);
        for (Size i = 0; i < n; ++i)
        {
            double row_sum = state_.diagonal[i] * v[i];
            Size begin = state_.column_start[i];
            Size end = state_.column_start[i + 1];
            for (Size k = begin; k < end; ++k)
            {
                row_sum += state_.values[k] * v[state_.row_index[k]];
            }
            w[i] = row_sum;
        }
        double mean = std::accumulate(w.begin(), w.end(), 0.0) / n;
        for (Size i = 0; i < n; ++i)
            w[i] -= mean;
        double norm = 0.0;
        for (double x : w)
            norm += x * x;
        norm = std::sqrt(norm);
        if (norm < tol)
            break;
        for (Size i = 0; i < n; ++i)
            w[i] /= norm;
        lambda = 0.0;
        for (Size i = 0; i < n; ++i)
            lambda += w[i] * v[i];
        v.swap(w);
        if (iter > 0 && std::fabs(lambda - lambda) < tol)
            break;
    }
    return v;
}

double IncrementalGraphLaplacian::computeAlgebraicConnectivity() const
{
    auto spectrum = computeSpectrum();
    if (spectrum.eigenvalues.empty())
        return 0.0;
    return spectrum.eigenvalues.size() > 1 ? spectrum.eigenvalues[1] : 0.0;
}

double IncrementalGraphLaplacian::computeTotalEffectiveResistance() const
{
    if (!state_.is_valid || state_.matrix_size < 2)
        return 0.0;
    auto spectrum = computeSpectrum();
    double resistance = 0.0;
    for (Size i = 1; i < spectrum.eigenvalues.size(); ++i)
    {
        if (spectrum.eigenvalues[i] > config_.eigenvalue_tolerance)
        {
            resistance += static_cast<double>(state_.matrix_size) / spectrum.eigenvalues[i];
        }
    }
    return resistance;
}

void IncrementalGraphLaplacian::reset()
{
    std::lock_guard<std::mutex> lock(state_mutex_);
    state_ = LaplacianState{};
    vertex_weights_.clear();
    degree_map_.clear();
}

Size IncrementalGraphLaplacian::getVertexCount() const
{
    return state_.matrix_size;
}

Size IncrementalGraphLaplacian::getEdgeCount() const
{
    Size count = 0;
    for (double v : state_.values)
    {
        if (std::fabs(v) > config_.eigenvalue_tolerance)
            ++count;
    }
    return count / 2;
}

} // namespace nerve::streaming
