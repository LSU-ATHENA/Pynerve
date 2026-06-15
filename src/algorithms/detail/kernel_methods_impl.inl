#include "nerve/algorithms/distance.hpp"
#include "nerve/math/constants.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <tuple>

namespace {
constexpr double kDefaultGaussianSigma = nerve::math::kernel::kDefaultGaussianSigma;
constexpr double kDefaultWassersteinP = nerve::math::kernel::kDefaultWassersteinP;
constexpr int kDefaultNProjections = nerve::math::kernel::kDefaultNProjections;
constexpr double kPi = nerve::math::kPi;

size_t checked_kernel_matrix_size(size_t n_diagrams) {
    if (n_diagrams != 0 && n_diagrams > std::numeric_limits<size_t>::max() / n_diagrams) {
        throw std::invalid_argument("kernel matrix size overflows size_t");
    }
    return n_diagrams * n_diagrams;
}

template <typename T>
size_t checked_byte_count(size_t count, const char* context) {
    constexpr size_t kElementSize = sizeof(T);
    if (count != 0 && count > std::numeric_limits<size_t>::max() / kElementSize) {
        throw std::invalid_argument(context);
    }
    return count * kElementSize;
}

template <nerve::algorithms::Numeric T>
void validate_diagram_span(std::span<const T> diagram, size_t pairs) {
    if (pairs > std::numeric_limits<size_t>::max() / 2) {
        throw std::invalid_argument("diagram pair count overflows size_t");
    }
    if (diagram.size() < pairs * 2) {
        throw std::invalid_argument("diagram span is smaller than pair count");
    }
    for (size_t i = 0; i < pairs; ++i) {
        const T birth = diagram[i * 2];
        const T death = diagram[i * 2 + 1];
        if (!std::isfinite(birth) || !std::isfinite(death)) {
            throw std::invalid_argument("diagram coordinates must be finite");
        }
        if (death < birth) {
            throw std::invalid_argument("diagram death must be greater than or equal to birth");
        }
    }
}

template <nerve::algorithms::Numeric T>
void validate_kernel_matrix_inputs(const std::vector<std::span<const T>>& diagrams,
                                   const std::vector<size_t>& sizes) {
    if (diagrams.size() != sizes.size()) {
        throw std::invalid_argument("diagrams and sizes must have matching lengths");
    }
    static_cast<void>(checked_kernel_matrix_size(diagrams.size()));
    for (size_t i = 0; i < diagrams.size(); ++i) {
        validate_diagram_span(diagrams[i], sizes[i]);
    }
}

template <nerve::algorithms::Numeric T>
std::vector<std::span<const T>> make_diagram_spans(const T* const* diagrams,
                                                   const size_t* sizes,
                                                   size_t n_diagrams) {
    if (n_diagrams == 0) {
        return {};
    }
    if (diagrams == nullptr || sizes == nullptr) {
        throw std::invalid_argument("diagram and size arrays must be non-null");
    }

    std::vector<std::span<const T>> spans;
    spans.reserve(n_diagrams);
    for (size_t i = 0; i < n_diagrams; ++i) {
        if (sizes[i] > std::numeric_limits<size_t>::max() / 2) {
            throw std::invalid_argument("diagram pair count overflows size_t");
        }
        if (sizes[i] > 0 && diagrams[i] == nullptr) {
            throw std::invalid_argument("non-empty diagram pointer must be non-null");
        }
        if (sizes[i] == 0) {
            spans.emplace_back();
            continue;
        }
        spans.emplace_back(diagrams[i], sizes[i] * 2);
    }
    return spans;
}

std::vector<size_t> copy_sizes(const size_t* sizes, size_t n_diagrams) {
    if (n_diagrams == 0) {
        return {};
    }
    return std::vector<size_t>(sizes, sizes + n_diagrams);
}
}

namespace nerve::algorithms {

template<typename T = float>
class PersistenceDiagramKernel {
public:
    virtual ~PersistenceDiagramKernel() = default;

    virtual T compute(std::span<const T> d1, size_t n1,
                     std::span<const T> d2, size_t n2) const = 0;

    virtual std::vector<T> compute_matrix(const std::vector<std::span<const T>>& diagrams,
                                         const std::vector<size_t>& sizes) const = 0;
};

template<typename T = float>
class GaussianKernel : public PersistenceDiagramKernel<T> {
public:
    struct Config {
        T sigma = T(1.0);
        enum class DistanceMetric { EUCLIDEAN, WASSERSTEIN, BOTTLENECK } metric = DistanceMetric::EUCLIDEAN;
        T p = static_cast<T>(kDefaultWassersteinP);
    };

    explicit GaussianKernel(const Config& config) : config_(config) {
        if (config_.sigma <= T(0) || !std::isfinite(config_.sigma)) {
            throw std::invalid_argument("sigma must be positive and finite");
        }
        if (config_.p <= T(0) || !std::isfinite(config_.p)) {
            throw std::invalid_argument("p must be positive and finite");
        }
    }

    T compute(std::span<const T> d1, size_t n1,
             std::span<const T> d2, size_t n2) const override {
        validate_diagram_span(d1, n1);
        validate_diagram_span(d2, n2);
        T dist = compute_distance(d1, n1, d2, n2);
        return std::exp(-dist * dist / (T(2) * config_.sigma * config_.sigma));
    }

    std::vector<T> compute_matrix(const std::vector<std::span<const T>>& diagrams,
                                 const std::vector<size_t>& sizes) const override {
        validate_kernel_matrix_inputs(diagrams, sizes);
        size_t n = diagrams.size();
        std::vector<T> kernel_matrix(checked_kernel_matrix_size(n));

        #ifdef NERVE_USE_OPENMP
        #pragma omp parallel for schedule(dynamic)
        #endif
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i; j < n; ++j) {
                T k = compute(diagrams[i], sizes[i], diagrams[j], sizes[j]);
                kernel_matrix[i * n + j] = k;
                kernel_matrix[j * n + i] = k;
            }
        }

        return kernel_matrix;
    }

private:
    Config config_;

    T compute_distance(std::span<const T> d1, size_t n1,
                      std::span<const T> d2, size_t n2) const {
        switch (config_.metric) {
            case Config::DistanceMetric::EUCLIDEAN:
                return euclidean_distance(d1, n1, d2, n2);
            case Config::DistanceMetric::WASSERSTEIN:
                return wasserstein_distance(d1, n1, d2, n2, config_.p);
            case Config::DistanceMetric::BOTTLENECK:
                return bottleneck_distance(d1, n1, d2, n2);
            default:
                return euclidean_distance(d1, n1, d2, n2);
        }
    }

    T euclidean_distance(std::span<const T> d1, size_t n1,
                        std::span<const T> d2, size_t n2) const {
        T total_dist = T(0);
        size_t count = 0;

        for (size_t i = 0; i < n1; ++i) {
            for (size_t j = 0; j < n2; ++j) {
                T db = d1[i * 2] - d2[j * 2];
                T dd = d1[i * 2 + 1] - d2[j * 2 + 1];
                total_dist += std::sqrt(db * db + dd * dd);
                count++;
            }
        }

        return count > 0 ? total_dist / static_cast<T>(count) : T(0);
    }

    T wasserstein_distance(std::span<const T> d1, size_t n1,
                          std::span<const T> d2, size_t n2, T p) const {
        if (n1 == 0 && n2 == 0) return T(0);
        if (n1 == 0 || n2 == 0) {
            auto& d = (n1 == 0) ? d2 : d1;
            size_t n = (n1 == 0) ? n2 : n1;
            T sum = T(0);
            for (size_t i = 0; i < n; ++i) {
                T pers = d[i * 2 + 1] - d[i * 2];
                sum += std::pow(pers / std::sqrt(T(2)), p);
            }
            return std::pow(sum, T(1) / p);
        }

        std::vector<bool> matched1(n1, false);
        std::vector<bool> matched2(n2, false);
        T total_cost = T(0);

        std::vector<std::tuple<T, size_t, size_t>> pairs;
        for (size_t i = 0; i < n1; ++i) {
            for (size_t j = 0; j < n2; ++j) {
                T db = d1[i * 2] - d2[j * 2];
                T dd = d1[i * 2 + 1] - d2[j * 2 + 1];
                T dist = std::pow(std::abs(db), p) + std::pow(std::abs(dd), p);
                pairs.emplace_back(std::pow(dist, T(1) / p), i, j);
            }
        }

        std::sort(pairs.begin(), pairs.end());

        for (const auto& [dist, i, j] : pairs) {
            if (!matched1[i] && !matched2[j]) {
                matched1[i] = true;
                matched2[j] = true;
                total_cost += std::pow(dist, p);
            }
        }

        for (size_t i = 0; i < n1; ++i) {
            if (!matched1[i]) {
                T pers = d1[i * 2 + 1] - d1[i * 2];
                total_cost += std::pow(pers / std::sqrt(T(2)), p);
            }
        }
        for (size_t j = 0; j < n2; ++j) {
            if (!matched2[j]) {
                T pers = d2[j * 2 + 1] - d2[j * 2];
                total_cost += std::pow(pers / std::sqrt(T(2)), p);
            }
        }

        return std::pow(total_cost, T(1) / p);
    }

    T bottleneck_distance(std::span<const T> d1, size_t n1,
                         std::span<const T> d2, size_t n2) const {
        if (n1 == 0 && n2 == 0) return T(0);
        if (n1 == 0 || n2 == 0) {
            auto& d = (n1 == 0) ? d2 : d1;
            size_t n = (n1 == 0) ? n2 : n1;
            T max_pers = T(0);
            for (size_t i = 0; i < n; ++i) {
                max_pers = std::max(max_pers, (d[i * 2 + 1] - d[i * 2]) / std::sqrt(T(2)));
            }
            return max_pers;
        }

        std::vector<bool> matched1(n1, false);
        std::vector<bool> matched2(n2, false);
        T max_dist = T(0);

        std::vector<std::tuple<T, size_t, size_t>> pairs;
        for (size_t i = 0; i < n1; ++i) {
            for (size_t j = 0; j < n2; ++j) {
                T db = std::abs(d1[i * 2] - d2[j * 2]);
                T dd = std::abs(d1[i * 2 + 1] - d2[j * 2 + 1]);
                pairs.emplace_back(std::max(db, dd), i, j);
            }
        }

        std::sort(pairs.begin(), pairs.end());

        for (const auto& [dist, i, j] : pairs) {
            if (!matched1[i] && !matched2[j]) {
                matched1[i] = true;
                matched2[j] = true;
                max_dist = std::max(max_dist, dist);
            }
        }

        for (size_t i = 0; i < n1; ++i) {
            if (!matched1[i]) {
                max_dist = std::max(max_dist, (d1[i * 2 + 1] - d1[i * 2]) / std::sqrt(T(2)));
            }
        }
        for (size_t j = 0; j < n2; ++j) {
            if (!matched2[j]) {
                max_dist = std::max(max_dist, (d2[j * 2 + 1] - d2[j * 2]) / std::sqrt(T(2)));
            }
        }

        return max_dist;
    }
};

template<typename T = float>
class PersistenceScaleSpaceKernel : public PersistenceDiagramKernel<T> {
public:
    struct Config {
        T sigma = T(0.5);
        T C = T(0.0);
    };

    explicit PersistenceScaleSpaceKernel(const Config& config) : config_(config) {
        if (config_.sigma <= T(0) || !std::isfinite(config_.sigma)) {
            throw std::invalid_argument("sigma must be positive and finite");
        }
        if (!std::isfinite(config_.C)) {
            throw std::invalid_argument("C must be finite");
        }
    }

    T compute(std::span<const T> d1, size_t n1,
             std::span<const T> d2, size_t n2) const override {
        validate_diagram_span(d1, n1);
        validate_diagram_span(d2, n2);
        T k_xy = T(0);
        for (size_t i = 0; i < n1; ++i) {
            for (size_t j = 0; j < n2; ++j) {
                T db = d1[i * 2] - d2[j * 2];
                T dd = d1[i * 2 + 1] - d2[j * 2 + 1];
                T dist_sq = db * db + dd * dd;
                k_xy += std::exp(-dist_sq / (T(8) * config_.sigma));
            }
        }
        k_xy /= (T(8) * static_cast<T>(kPi) * config_.sigma);

        T k_x_yref = T(0);
        for (size_t i = 0; i < n1; ++i) {
            for (size_t j = 0; j < n2; ++j) {
                T db = d1[i * 2] - d2[j * 2 + 1];
                T dd = d1[i * 2 + 1] - d2[j * 2];
                T dist_sq = db * db + dd * dd;
                k_x_yref += std::exp(-dist_sq / (T(8) * config_.sigma));
            }
        }
        k_x_yref /= (T(8) * static_cast<T>(kPi) * config_.sigma);

        return k_xy - k_x_yref + config_.C;
    }

    std::vector<T> compute_matrix(const std::vector<std::span<const T>>& diagrams,
                                 const std::vector<size_t>& sizes) const override {
        validate_kernel_matrix_inputs(diagrams, sizes);
        size_t n = diagrams.size();
        std::vector<T> kernel_matrix(checked_kernel_matrix_size(n));

        #ifdef NERVE_USE_OPENMP
        #pragma omp parallel for schedule(dynamic)
        #endif
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i; j < n; ++j) {
                T k = compute(diagrams[i], sizes[i], diagrams[j], sizes[j]);
                kernel_matrix[i * n + j] = k;
                kernel_matrix[j * n + i] = k;
            }
        }

        return kernel_matrix;
    }

private:
    Config config_;
};

template<typename T = float>
class SlicedWassersteinKernel : public PersistenceDiagramKernel<T> {
public:
    struct Config {
        T sigma = static_cast<T>(kDefaultGaussianSigma);
        int n_projections = kDefaultNProjections;
    };

    explicit SlicedWassersteinKernel(const Config& config) : config_(config) {
        if (config_.sigma <= T(0) || !std::isfinite(config_.sigma)) {
            throw std::invalid_argument("sigma must be positive and finite");
        }
        if (config_.n_projections <= 0) {
            throw std::invalid_argument("n_projections must be positive");
        }
    }

    T compute(std::span<const T> d1, size_t n1,
             std::span<const T> d2, size_t n2) const override {
        validate_diagram_span(d1, n1);
        validate_diagram_span(d2, n2);
        T total_dist = T(0);

        for (int p = 0; p < config_.n_projections; ++p) {
            T theta = static_cast<T>(p) * static_cast<T>(kPi) / config_.n_projections;
            T cos_t = std::cos(theta);
            T sin_t = std::sin(theta);

            std::vector<T> proj1;
            std::vector<T> proj2;
            proj1.reserve(n1 + n2);
            proj2.reserve(n1 + n2);

            for (size_t i = 0; i < n1; ++i) {
                const T birth = d1[i * 2];
                const T death = d1[i * 2 + 1];
                const T diagonal = (birth + death) / T(2);
                proj1.push_back(cos_t * birth + sin_t * death);
                proj2.push_back((cos_t + sin_t) * diagonal);
            }
            for (size_t j = 0; j < n2; ++j) {
                const T birth = d2[j * 2];
                const T death = d2[j * 2 + 1];
                const T diagonal = (birth + death) / T(2);
                proj2.push_back(cos_t * birth + sin_t * death);
                proj1.push_back((cos_t + sin_t) * diagonal);
            }

            std::sort(proj1.begin(), proj1.end());
            std::sort(proj2.begin(), proj2.end());

            T dist = T(0);
            for (size_t i = 0; i < proj1.size(); ++i) {
                dist += std::abs(proj1[i] - proj2[i]);
            }

            total_dist += dist;
        }

        T avg_dist = total_dist / config_.n_projections;
        return std::exp(-avg_dist / (T(2) * config_.sigma * config_.sigma));
    }

    std::vector<T> compute_matrix(const std::vector<std::span<const T>>& diagrams,
                                 const std::vector<size_t>& sizes) const override {
        validate_kernel_matrix_inputs(diagrams, sizes);
        size_t n = diagrams.size();
        std::vector<T> kernel_matrix(checked_kernel_matrix_size(n));

        #ifdef NERVE_USE_OPENMP
        #pragma omp parallel for schedule(dynamic)
        #endif
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i; j < n; ++j) {
                T k = compute(diagrams[i], sizes[i], diagrams[j], sizes[j]);
                kernel_matrix[i * n + j] = k;
                kernel_matrix[j * n + i] = k;
            }
        }

        return kernel_matrix;
    }

private:
    Config config_;
};

extern "C" {

void nerve_gaussian_kernel_matrix_f32(const float** diagrams, const size_t* sizes,
                                       size_t n_diagrams, float sigma, float* output) {
    const size_t output_size = checked_kernel_matrix_size(n_diagrams);
    if (output_size != 0 && output == nullptr) {
        throw std::invalid_argument("output pointer must be non-null");
    }
    GaussianKernel<float>::Config config;
    config.sigma = sigma;
    GaussianKernel<float> kernel(config);

    auto spans = make_diagram_spans<float>(diagrams, sizes, n_diagrams);
    auto sizes_vec = copy_sizes(sizes, n_diagrams);

    auto result = kernel.compute_matrix(spans, sizes_vec);
    if (!result.empty()) {
        std::memcpy(output, result.data(),
                    checked_byte_count<float>(result.size(), "kernel output bytes overflow"));
    }
}

void nerve_gaussian_kernel_matrix_f64(const double** diagrams, const size_t* sizes,
                                       size_t n_diagrams, double sigma, double* output) {
    const size_t output_size = checked_kernel_matrix_size(n_diagrams);
    if (output_size != 0 && output == nullptr) {
        throw std::invalid_argument("output pointer must be non-null");
    }
    GaussianKernel<double>::Config config;
    config.sigma = sigma;
    GaussianKernel<double> kernel(config);

    auto spans = make_diagram_spans<double>(diagrams, sizes, n_diagrams);
    auto sizes_vec = copy_sizes(sizes, n_diagrams);

    auto result = kernel.compute_matrix(spans, sizes_vec);
    if (!result.empty()) {
        std::memcpy(output, result.data(),
                    checked_byte_count<double>(result.size(), "kernel output bytes overflow"));
    }
}

}  // extern "C"

template class GaussianKernel<float>;
template class GaussianKernel<double>;
template class PersistenceScaleSpaceKernel<float>;
template class PersistenceScaleSpaceKernel<double>;
template class SlicedWassersteinKernel<float>;
template class SlicedWassersteinKernel<double>;

}  // namespace nerve::algorithms
