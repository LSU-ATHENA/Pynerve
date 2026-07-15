// C Interface

template <Numeric T>
std::span<const T> make_c_input_span(const T *points, size_t rows, size_t dim,
                                     std::string_view name)
{
    const size_t input_size = checked_product(rows, dim, name);
    if (input_size != 0 && points == nullptr)
    {
        throw std::invalid_argument(std::string(name) + " pointer is null");
    }
    if (input_size == 0)
    {
        // Return an empty span with a non-null sentinel to avoid UBSAN
        // warning about reference binding to null pointer in
        // std::span(nullptr, 0).
        static const T sentinel{};
        return std::span<const T>(&sentinel, 0);
    }
    return std::span<const T>(points, input_size);
}

template <typename T>
void validate_c_output(T *output, size_t output_size, std::string_view name)
{
    if (output_size != 0 && output == nullptr)
    {
        throw std::invalid_argument(std::string(name) + " pointer is null");
    }
}

template <typename T>
size_t checked_byte_count(size_t count, std::string_view name)
{
    return checked_product(count, sizeof(T), name);
}

extern "C"
{
    void nerve_pairwise_distances_f32(const float *points, size_t n, size_t dim, float *output)
    {
        const size_t output_size = checked_square_count(n, "C pairwise distance output");
        validate_c_output(output, output_size, "C pairwise distance output");
        nerve::algorithms::DistanceMatrixComputer<float> computer;
        auto result = computer.compute(
            make_c_input_span(points, n, dim, "C pairwise distance input"), n, dim);
        if (!result.empty())
        {
            std::memcpy(
                output, result.data(),
                checked_byte_count<float>(result.size(), "C pairwise distance output bytes"));
        }
    }

    void nerve_pairwise_distances_f64(const double *points, size_t n, size_t dim, double *output)
    {
        const size_t output_size = checked_square_count(n, "C pairwise distance output");
        validate_c_output(output, output_size, "C pairwise distance output");
        nerve::algorithms::DistanceMatrixComputer<double> computer;
        auto result = computer.compute(
            make_c_input_span(points, n, dim, "C pairwise distance input"), n, dim);
        if (!result.empty())
        {
            std::memcpy(
                output, result.data(),
                checked_byte_count<double>(result.size(), "C pairwise distance output bytes"));
        }
    }

    void nerve_knn_f32(const float *points, size_t n, size_t dim, size_t k, float *out_distances,
                       size_t *out_indices)
    {
        const size_t result_k = n == 0 ? 0 : std::min(k, n - 1);
        const size_t output_size = checked_product(n, result_k, "C KNN result");
        validate_c_output(out_distances, output_size, "C KNN distances output");
        validate_c_output(out_indices, output_size, "C KNN indices output");
        nerve::algorithms::KNNComputer<float>::Config config;
        config.k = k;
        nerve::algorithms::KNNComputer<float> computer(config);

        auto result = computer.compute(make_c_input_span(points, n, dim, "C KNN input"), n, dim);
        if (!result.distances.empty())
        {
            std::memcpy(out_distances, result.distances.data(),
                        checked_byte_count<float>(result.distances.size(), "C KNN distance bytes"));
            std::memcpy(out_indices, result.indices.data(),
                        checked_byte_count<size_t>(result.indices.size(), "C KNN index bytes"));
        }
    }

    void nerve_knn_f64(const double *points, size_t n, size_t dim, size_t k, double *out_distances,
                       size_t *out_indices)
    {
        const size_t result_k = n == 0 ? 0 : std::min(k, n - 1);
        const size_t output_size = checked_product(n, result_k, "C KNN result");
        validate_c_output(out_distances, output_size, "C KNN distances output");
        validate_c_output(out_indices, output_size, "C KNN indices output");
        nerve::algorithms::KNNComputer<double>::Config config;
        config.k = k;
        nerve::algorithms::KNNComputer<double> computer(config);

        auto result = computer.compute(make_c_input_span(points, n, dim, "C KNN input"), n, dim);
        if (!result.distances.empty())
        {
            std::memcpy(
                out_distances, result.distances.data(),
                checked_byte_count<double>(result.distances.size(), "C KNN distance bytes"));
            std::memcpy(out_indices, result.indices.data(),
                        checked_byte_count<size_t>(result.indices.size(), "C KNN index bytes"));
        }
    }

} // extern "C"

#include "distance_matrix_knn_sparse.inl"

// Explicit instantiations
template class EuclideanMetric<float>;
template class EuclideanMetric<double>;
template class EuclideanMetric<long double>;
template class ManhattanMetric<float>;
template class ManhattanMetric<double>;
template class ManhattanMetric<long double>;
template class CosineMetric<float>;
template class CosineMetric<double>;
template class CosineMetric<long double>;
template class DistanceMatrixComputer<float>;
template class DistanceMatrixComputer<double>;
template class DistanceMatrixComputer<long double>;
template class KNNComputer<float>;
template class KNNComputer<double>;
template class KNNComputer<long double>;
template class SparseDistanceMatrixComputer<float>;
template class SparseDistanceMatrixComputer<double>;
template class SparseDistanceMatrixComputer<long double>;
