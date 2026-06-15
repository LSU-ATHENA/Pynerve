namespace {

[[nodiscard]] size_t saturatingSize(long double value) {
    if (value <= 0.0L) {
        return 0;
    }
    const long double limit =
        static_cast<long double>(std::numeric_limits<size_t>::max());
    return value >= limit ? std::numeric_limits<size_t>::max()
                          : static_cast<size_t>(value);
}

}  // namespace

AutomaticMemoryOptimizer::AutomaticMemoryOptimizer(size_t max_ram_bytes)
    : max_ram_(max_ram_bytes == 0 ? DEFAULT_EXTREME_MEMORY_LIMIT : max_ram_bytes) {}

MemoryTier AutomaticMemoryOptimizer::selectTier(
    size_t n_points, size_t point_dim, double max_distance) const {
    const size_t estimated = estimateMemoryBytes(n_points, point_dim, max_distance);
    return estimated <= max_ram_ ? MemoryTier::IN_MEMORY : MemoryTier::STREAMING;
}

size_t AutomaticMemoryOptimizer::estimateMemoryBytes(
    size_t n_points, size_t point_dim, double max_distance) const {
    if (n_points == 0 || point_dim == 0 || !std::isfinite(max_distance) ||
        max_distance < 0.0) {
        return 0;
    }

    const long double n = static_cast<long double>(n_points);
    const long double dim = static_cast<long double>(point_dim);
    const long double radius = static_cast<long double>(max_distance);
    const long double radius_density =
        radius <= 0.0L ? 0.0L : radius / (radius + std::sqrt(dim));
    const long double edge_density = std::clamp(radius_density, 0.0L, 1.0L);
    const long double possible_edges = n * (n - 1.0L) / 2.0L;
    const long double estimated_edges = possible_edges * edge_density;
    const long double coface_factor =
        std::min(n / 3.0L, (8.0L + dim) * edge_density * edge_density);
    const long double estimated_simplices = n + estimated_edges * (1.0L + coface_factor);
    const long double bytes_per_simplex =
        static_cast<long double>(sizeof(int)) * (dim + 2.0L) +
        static_cast<long double>(sizeof(double)) * 2.0L;

    return saturatingSize(estimated_simplices * bytes_per_simplex);
}

PersistenceDiagram AutomaticMemoryOptimizer::compute(
    std::span<const double> points,
    size_t n_points,
    size_t point_dim,
    double max_distance,
    int max_dim
) {
    PersistenceDiagram result;
    if (max_dim < 0 || point_dim == 0 || n_points == 0 ||
        n_points > points.size() / point_dim || !std::isfinite(max_distance) ||
        max_distance < 0.0) {
        return result;
    }

    const MemoryTier tier = selectTier(n_points, point_dim, max_distance);
    streaming::StreamingColumnGenerator generator(points, n_points, point_dim, max_distance);
    if (generator.getNumSimplices() == 0) {
        return result;
    }

    streaming::StreamingReducer reducer(tier == MemoryTier::IN_MEMORY);
    result = reducer.reduce(generator);

    const auto max_output_dim = static_cast<int>(max_dim);
    std::erase_if(result.pairs, [max_output_dim](const PersistencePair& pair) {
        return pair.dimension > max_output_dim;
    });
    return result;
}

PersistenceDiagram computeExtremeMemory(
    std::span<const double> points,
    size_t n_points,
    size_t point_dim,
    double max_distance,
    int max_dim,
    size_t max_ram_bytes
) {
    AutomaticMemoryOptimizer optimizer(max_ram_bytes);
    return optimizer.compute(points, n_points, point_dim, max_distance, max_dim);
}
