template <Numeric T>
std::vector<T> DistanceMatrixComputer<T>::compute_blocked(std::span<const T> points,
                                                          size_t n_points, size_t dim,
                                                          size_t block_size) const
{
    validate_flat_matrix(points, n_points, dim, "points");
    const size_t matrix_size = checked_square_count(n_points, "distance matrix");
    std::vector<T> distances(matrix_size, nerve::math::Constants<T>::kZero);
    if (block_size == 0)
    {
        block_size = kDistanceBlockSize;
    }

    for (size_t ii = 0; ii < n_points; ii += block_size)
    {
        size_t i_end = std::min(ii + block_size, n_points);

        for (size_t jj = ii; jj < n_points; jj += block_size)
        {
            size_t j_end = std::min(jj + block_size, n_points);

#ifdef NERVE_USE_OPENMP
#pragma omp parallel for schedule(dynamic) if (config_.use_openmp)
#endif
            for (size_t i = ii; i < i_end; ++i)
            {
                for (size_t j = std::max(jj, i); j < j_end; ++j)
                {
                    T sum = 0;
                    for (size_t d = 0; d < dim; ++d)
                    {
                        T diff = points[i * dim + d] - points[j * dim + d];
                        sum += diff * diff;
                    }
                    T dist = checked_distance_result(std::sqrt(sum), "euclidean distance");
                    distances[i * n_points + j] = dist;
                    if (i != j)
                    {
                        distances[j * n_points + i] = dist;
                    }
                }
            }
        }
    }

    return distances;
}

template <Numeric T>
std::vector<T> DistanceMatrixComputer<T>::compute_euclidean_simd(std::span<const T> points,
                                                                 size_t n_points, size_t dim) const
{
    // Delegate to compute_euclidean which already uses the dispatch table
    // (simd_euclidean for double, simd_euclidean_f32 for float) via if constexpr,
    // with a scalar fallback for other types. The dispatch table selects the
    // optimal SIMD backend at runtime, making the old AVX-512-specific path obsolete.
    return compute_euclidean(points, n_points, dim);
}

template <Numeric T>
typename KNNComputer<T>::Result KNNComputer<T>::compute(std::span<const T> points, size_t n_points,
                                                        size_t dim) const
{
    validate_flat_matrix(points, n_points, dim, "points");
    switch (config_.algorithm)
    {
        case Config::Algorithm::BRUTE_FORCE:
            return compute_brute_force(points, n_points, dim);
        case Config::Algorithm::HNSW:
            break;
    }
    return compute_brute_force(points, n_points, dim);
}

template <Numeric T>
void KNNComputer<T>::find_k_smallest(std::span<T> distances, std::span<size_t> indices, size_t k)
{
    k = std::min({k, distances.size(), indices.size()});
    if (k == 0)
    {
        return;
    }
    std::vector<size_t> idx(distances.size());
    std::iota(idx.begin(), idx.end(), 0);

    using Difference = std::vector<size_t>::difference_type;
    const auto selected_end = std::next(idx.begin(), static_cast<Difference>(k));
    std::partial_sort(idx.begin(), selected_end, idx.end(),
                      [&distances](size_t a, size_t b) { return distances[a] < distances[b]; });

    for (size_t i = 0; i < k; ++i)
    {
        indices[i] = idx[i];
    }
}

template <Numeric T>
typename KNNComputer<T>::Result
KNNComputer<T>::compute_brute_force(std::span<const T> points, size_t n_points, size_t dim) const
{
    validate_flat_matrix(points, n_points, dim, "points");

    Result result;
    result.n_points = n_points;
    result.k = n_points == 0 ? 0 : std::min(config_.k, n_points - 1);
    const size_t result_size = checked_product(n_points, result.k, "KNN result");
    result.distances.resize(result_size);
    result.indices.resize(result_size);
    if (result.k == 0)
    {
        return result;
    }

#ifdef NERVE_USE_OPENMP
#pragma omp parallel for schedule(dynamic) if (config_.use_openmp)
#endif
    for (size_t i = 0; i < n_points; ++i)
    {
        std::vector<std::pair<T, size_t>> nearest;
        nearest.reserve(result.k);
        for (size_t j = 0; j < n_points; ++j)
        {
            if (i == j)
            {
                continue;
            }
            const T distance = distance_from_knn_metric(
                points.data() + i * dim, points.data() + j * dim, dim, config_.metric);
            const std::pair<T, size_t> candidate{distance, j};
            if (nearest.size() < result.k)
            {
                nearest.push_back(candidate);
                std::push_heap(nearest.begin(), nearest.end());
            }
            else if (candidate < nearest.front())
            {
                std::pop_heap(nearest.begin(), nearest.end());
                nearest.back() = candidate;
                std::push_heap(nearest.begin(), nearest.end());
            }
        }
        std::sort_heap(nearest.begin(), nearest.end());

        for (size_t k_idx = 0; k_idx < result.k; ++k_idx)
        {
            result.distances[i * result.k + k_idx] = nearest[k_idx].first;
            result.indices[i * result.k + k_idx] = nearest[k_idx].second;
        }
    }

    return result;
}

template <Numeric T>
typename KNNComputer<T>::Result
KNNComputer<T>::compute_query(std::span<const T> reference, size_t n_ref,
                              std::span<const T> queries, size_t n_query, size_t dim) const
{
    validate_flat_matrix(reference, n_ref, dim, "reference");
    validate_flat_matrix(queries, n_query, dim, "queries");
    Result result;
    result.n_points = n_query;
    result.k = std::min(config_.k, n_ref);
    const size_t result_size = checked_product(n_query, result.k, "KNN query result");
    result.distances.resize(result_size);
    result.indices.resize(result_size);
    if (result.k == 0)
    {
        return result;
    }

#ifdef NERVE_USE_OPENMP
#pragma omp parallel for schedule(dynamic) if (config_.use_openmp)
#endif
    for (size_t i = 0; i < n_query; ++i)
    {
        std::vector<std::pair<T, size_t>> nearest;
        nearest.reserve(result.k);
        for (size_t j = 0; j < n_ref; ++j)
        {
            const T distance = distance_from_knn_metric(
                queries.data() + i * dim, reference.data() + j * dim, dim, config_.metric);
            const std::pair<T, size_t> candidate{distance, j};
            if (nearest.size() < result.k)
            {
                nearest.push_back(candidate);
                std::push_heap(nearest.begin(), nearest.end());
            }
            else if (candidate < nearest.front())
            {
                std::pop_heap(nearest.begin(), nearest.end());
                nearest.back() = candidate;
                std::push_heap(nearest.begin(), nearest.end());
            }
        }
        std::sort_heap(nearest.begin(), nearest.end());
        for (size_t k_idx = 0; k_idx < result.k; ++k_idx)
        {
            result.distances[i * result.k + k_idx] = nearest[k_idx].first;
            result.indices[i * result.k + k_idx] = nearest[k_idx].second;
        }
    }

    return result;
}

template <Numeric T>
typename SparseDistanceMatrixComputer<T>::SparseMatrix
SparseDistanceMatrixComputer<T>::compute(std::span<const T> points, size_t n_points,
                                         size_t dim) const
{
    validate_flat_matrix(points, n_points, dim, "points");
    switch (config_.mode)
    {
        case Config::Mode::EPSILON_NEIGHBORHOOD:
            return compute_epsilon(points, n_points, dim);
        case Config::Mode::KNN:
            return compute_knn(points, n_points, dim);
    }
    throw std::invalid_argument("unsupported sparse distance matrix mode");
}

template <Numeric T>
typename SparseDistanceMatrixComputer<T>::SparseMatrix
SparseDistanceMatrixComputer<T>::compute_epsilon(std::span<const T> points, size_t n_points,
                                                 size_t dim) const
{
    validate_flat_matrix(points, n_points, dim, "points");
    static_cast<void>(checked_square_count(n_points, "sparse epsilon traversal"));
    size_t nnz = 0;
    std::vector<size_t> row_counts(n_points, 0);

#ifdef NERVE_USE_OPENMP
#pragma omp parallel for schedule(dynamic) reduction(+ : nnz)
#endif
    for (size_t i = 0; i < n_points; ++i)
    {
        for (size_t j = 0; j < n_points; ++j)
        {
            if (i == j)
                continue;

            T dist = 0;
            for (size_t d = 0; d < dim; ++d)
            {
                T diff = points[i * dim + d] - points[j * dim + d];
                dist += diff * diff;
            }
            dist = std::sqrt(dist);

            if (dist < config_.epsilon)
            {
                row_counts[i]++;
                nnz++;
            }
        }
    }

    SparseMatrix sparse;
    sparse.n_rows = n_points;
    sparse.n_cols = n_points;
    sparse.nnz = nnz;
    sparse.values.resize(nnz);
    sparse.col_idx.resize(nnz);
    sparse.row_ptr.resize(checked_plus_one(n_points, "sparse row_ptr"));

    sparse.row_ptr[0] = 0;
    for (size_t i = 0; i < n_points; ++i)
    {
        sparse.row_ptr[i + 1] = sparse.row_ptr[i] + row_counts[i];
    }

    std::vector<size_t> current_pos = sparse.row_ptr;

    for (size_t i = 0; i < n_points; ++i)
    {
        for (size_t j = 0; j < n_points; ++j)
        {
            if (i == j)
                continue;

            T dist = 0;
            for (size_t d = 0; d < dim; ++d)
            {
                T diff = points[i * dim + d] - points[j * dim + d];
                dist += diff * diff;
            }
            dist = std::sqrt(dist);

            if (dist < config_.epsilon)
            {
                size_t pos = current_pos[i]++;
                sparse.values[pos] = dist;
                sparse.col_idx[pos] = j;
            }
        }
    }

    return sparse;
}

template <Numeric T>
typename SparseDistanceMatrixComputer<T>::SparseMatrix
SparseDistanceMatrixComputer<T>::compute_knn(std::span<const T> points, size_t n_points,
                                             size_t dim) const
{
    validate_flat_matrix(points, n_points, dim, "points");
    typename KNNComputer<T>::Config knn_cfg;
    knn_cfg.k = config_.k_max;
    knn_cfg.algorithm = KNNComputer<T>::Config::Algorithm::BRUTE_FORCE;
    KNNComputer<T> knn(knn_cfg);
    auto knn_result = knn.compute(points, n_points, dim);

    SparseMatrix sparse;
    sparse.n_rows = n_points;
    sparse.n_cols = n_points;
    sparse.row_ptr.resize(checked_plus_one(n_points, "sparse row_ptr"), 0);
    sparse.values.reserve(knn_result.distances.size());
    sparse.col_idx.reserve(knn_result.indices.size());

    for (size_t i = 0; i < n_points; ++i)
    {
        sparse.row_ptr[i] = sparse.values.size();
        for (size_t k = 0; k < knn_result.k; ++k)
        {
            const size_t flat = i * knn_result.k + k;
            const size_t col = knn_result.indices[flat];
            const T dist = knn_result.distances[flat];
            if (col != i)
            {
                sparse.values.push_back(dist);
                sparse.col_idx.push_back(col);
            }
        }
    }
    sparse.row_ptr[n_points] = sparse.values.size();
    sparse.nnz = sparse.values.size();
    return sparse;
}

template <Numeric T>
typename SparseDistanceMatrixComputer<T>::SparseMatrix
SparseDistanceMatrixComputer<T>::from_dense(std::span<const T> dense, size_t n_rows, size_t n_cols,
                                            T threshold)
{
    validate_flat_matrix(dense, n_rows, n_cols, "dense");
    SparseMatrix sparse;
    sparse.n_rows = n_rows;
    sparse.n_cols = n_cols;
    sparse.row_ptr.resize(checked_plus_one(n_rows, "sparse row_ptr"), 0);

    for (size_t i = 0; i < n_rows; ++i)
    {
        sparse.row_ptr[i] = sparse.values.size();
        for (size_t j = 0; j < n_cols; ++j)
        {
            const T value = dense[i * n_cols + j];
            if (value <= threshold)
            {
                sparse.values.push_back(value);
                sparse.col_idx.push_back(j);
            }
        }
    }
    sparse.row_ptr[n_rows] = sparse.values.size();
    sparse.nnz = sparse.values.size();
    return sparse;
}

template <Numeric T>
std::vector<T> SparseDistanceMatrixComputer<T>::to_dense(const SparseMatrix &sparse)
{
    if (sparse.values.size() != sparse.col_idx.size() || sparse.nnz != sparse.values.size())
    {
        throw std::invalid_argument("sparse values, columns, and nnz must agree");
    }
    const size_t expected_row_ptr_size = checked_plus_one(sparse.n_rows, "sparse row_ptr");
    if (sparse.row_ptr.size() != expected_row_ptr_size)
    {
        throw std::invalid_argument("sparse row_ptr must have n_rows + 1 entries");
    }
    const size_t dense_size = checked_product(sparse.n_rows, sparse.n_cols, "dense sparse matrix");
    std::vector<T> dense(dense_size, std::numeric_limits<T>::infinity());
    for (size_t i = 0; i < sparse.n_rows; ++i)
    {
        if (sparse.row_ptr[i] > sparse.row_ptr[i + 1] ||
            sparse.row_ptr[i + 1] > sparse.values.size())
        {
            throw std::invalid_argument("sparse row_ptr entries are inconsistent with values");
        }
        if (i < sparse.n_cols)
        {
            dense[i * sparse.n_cols + i] = nerve::math::Constants<T>::kZero;
        }
        for (size_t p = sparse.row_ptr[i]; p < sparse.row_ptr[i + 1]; ++p)
        {
            if (p >= sparse.col_idx.size() || sparse.col_idx[p] >= sparse.n_cols)
            {
                throw std::invalid_argument("sparse column index is out of bounds");
            }
            dense[i * sparse.n_cols + sparse.col_idx[p]] = sparse.values[p];
        }
    }
    return dense;
}
