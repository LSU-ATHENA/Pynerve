// Distance metric dispatch helpers shared by dense/sparse/knn code paths.
// This file is intended to be included inside an anonymous namespace in
// distance.cpp so template symbols remain TU-local.

constexpr size_t kSimdWidthFloats = nerve::math::simd::kAvx512Floats;
constexpr size_t kSimdWidthDoubles = nerve::math::simd::kAvx512Doubles;
constexpr size_t kDistanceBlockSize = nerve::math::distance::kDefaultBlockSize;
constexpr double kDefaultEpsilon = nerve::math::kEpsilonDouble;

template<nerve::algorithms::Numeric T>
[[nodiscard]] constexpr size_t simd_lane_width() noexcept {
    if constexpr (sizeof(T) == sizeof(float)) {
        return kSimdWidthFloats;
    }
    return kSimdWidthDoubles;
}

template<nerve::algorithms::Numeric T>
[[nodiscard]] T euclidean_distance(const T* a, const T* b, size_t dim) {
    T sum_sq = nerve::math::Constants<T>::kZero;
    for (size_t d = 0; d < dim; ++d) {
        const T diff = a[d] - b[d];
        sum_sq += diff * diff;
    }
    return checked_distance_result(std::sqrt(sum_sq), "euclidean distance");
}

template<nerve::algorithms::Numeric T>
[[nodiscard]] T manhattan_distance(const T* a, const T* b, size_t dim) {
    T sum = nerve::math::Constants<T>::kZero;
    for (size_t d = 0; d < dim; ++d) {
        sum += std::abs(a[d] - b[d]);
    }
    return checked_distance_result(sum, "manhattan distance");
}

template<nerve::algorithms::Numeric T>
[[nodiscard]] T chebyshev_distance(const T* a, const T* b, size_t dim) {
    T max_diff = nerve::math::Constants<T>::kZero;
    for (size_t d = 0; d < dim; ++d) {
        max_diff = std::max(max_diff, std::abs(a[d] - b[d]));
    }
    return checked_distance_result(max_diff, "chebyshev distance");
}

template<nerve::algorithms::Numeric T>
[[nodiscard]] T cosine_distance(const T* a, const T* b, size_t dim) {
    T dot = nerve::math::Constants<T>::kZero;
    T norm_a_sq = nerve::math::Constants<T>::kZero;
    T norm_b_sq = nerve::math::Constants<T>::kZero;
    for (size_t d = 0; d < dim; ++d) {
        dot += a[d] * b[d];
        norm_a_sq += a[d] * a[d];
        norm_b_sq += b[d] * b[d];
    }
    if (!std::isfinite(dot) || !std::isfinite(norm_a_sq) || !std::isfinite(norm_b_sq)) {
        throw std::overflow_error("cosine distance overflowed");
    }

    // Numerical policy for zero vectors:
    // - both zero: distance 0 (indistinguishable)
    // - one zero: distance 1 (neutral dissimilarity in [0, 2] cosine range)
    constexpr T kEpsilon = static_cast<T>(kDefaultEpsilon);
    if (norm_a_sq <= kEpsilon || norm_b_sq <= kEpsilon) {
        return (norm_a_sq <= kEpsilon && norm_b_sq <= kEpsilon)
                 ? nerve::math::Constants<T>::kZero
                 : nerve::math::Constants<T>::kOne;
    }

    const T denom = checked_distance_result(std::sqrt(norm_a_sq * norm_b_sq), "cosine distance");
    const T raw_similarity = dot / denom;
    const T similarity = std::clamp(
        checked_distance_result(raw_similarity, "cosine distance"),
        -nerve::math::Constants<T>::kOne,
        nerve::math::Constants<T>::kOne
    );
    return nerve::math::Constants<T>::kOne - similarity;
}

template<nerve::algorithms::Numeric T>
[[nodiscard]] T distance_from_matrix_metric(
    const T* a,
    const T* b,
    size_t dim,
    typename nerve::algorithms::DistanceMatrixComputer<T>::Config::Metric metric
) {
    using Metric = typename nerve::algorithms::DistanceMatrixComputer<T>::Config::Metric;
    switch (metric) {
        case Metric::EUCLIDEAN:
            return euclidean_distance(a, b, dim);
        case Metric::MANHATTAN:
            return manhattan_distance(a, b, dim);
        case Metric::COSINE:
            return cosine_distance(a, b, dim);
        case Metric::CHEBYSHEV:
            return chebyshev_distance(a, b, dim);
        case Metric::MINKOWSKI:
        case Metric::CANBERRA:
        case Metric::BRAYCURTIS:
        case Metric::CORRELATION:
            break;
    }
    return euclidean_distance(a, b, dim);
}

template<nerve::algorithms::Numeric T>
[[nodiscard]] T distance_from_knn_metric(
    const T* a,
    const T* b,
    size_t dim,
    typename nerve::algorithms::KNNComputer<T>::Config::Metric metric
) {
    using Metric = typename nerve::algorithms::KNNComputer<T>::Config::Metric;
    switch (metric) {
        case Metric::EUCLIDEAN:
            return euclidean_distance(a, b, dim);
        case Metric::MANHATTAN:
            return manhattan_distance(a, b, dim);
        case Metric::COSINE:
            return cosine_distance(a, b, dim);
    }
    return euclidean_distance(a, b, dim);
}
